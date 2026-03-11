"""Transpile a Stan model to PyMC via an agentic Claude loop.

Takes Stan code as input and produces equivalent PyMC Python code.
Optionally validates by comparing logp values between BridgeStan and
the generated PyMC model at identical unconstrained parameter points.
"""

from __future__ import annotations

import functools
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


_SKILLS_DIR = Path(__file__).parent / "skills"


SYSTEM_PROMPT = """\
You are an expert statistical modeler who translates Stan models to PyMC (v5+).

You produce clean, idiomatic PyMC code that is functionally equivalent to the
input Stan model. The generated code should run correctly and produce the same
posterior when sampled.

You have access to tools to iteratively write and validate PyMC code.
Your workflow:
1. Analyze the Stan code, data, and parameter structure
2. Write the PyMC model using `write_pymc_code`
3. Validate with `validate_model` — this checks that the PyMC model's logp
   matches BridgeStan's logp at multiple test points
4. If validation fails, analyze the errors, fix the code, and revalidate
5. Iterate until validation passes

You can also use `read_stan_code` to re-read the original Stan source.

CRITICAL RULES:
1. The output must be a complete, self-contained Python function that takes a
   data dict and returns a `pm.Model` instance.
2. Use PyMC v5+ API — `import pymc as pm` and `import pytensor.tensor as pt`.
3. Do NOT include sampling code — only model specification.
4. Map Stan constraints to appropriate PyMC distributions:
   - `real<lower=0>` with `normal(0, s)` prior → `pm.HalfNormal("x", sigma=s)`
   - `real<lower=0>` with `cauchy(0, s)` prior → `pm.HalfCauchy("x", beta=s)`
   - `real<lower=0>` with `student_t(nu, 0, s)` prior → `pm.HalfStudentT("x", nu=nu, sigma=s)`
   - `real<lower=0>` with other prior → use the prior distribution directly
     if it has positive support, or wrap with a bound
   NOTE: The validator automatically corrects for the log(2) normalization
   difference between PyMC's Half* distributions and Stan's convention.
   Do NOT add manual log(2) correction potentials — just use the Half* distributions.
5. Translate Stan's 1-based indexing to Python's 0-based indexing.
6. Use numpy arrays for data, pytensor.tensor for symbolic math operations.
7. Vectorize where possible — avoid Python loops over observations.
8. Stan's `target +=` becomes `pm.Potential(...)`.
9. Include all priors. If Stan specifies no prior (improper), use `pm.Flat()`.
10. Include `observed=` for likelihood terms.
11. Prefer standard PyMC distributions over manual `pm.Potential` when possible.

OUTPUT FORMAT:
Generate a Python function with this signature:
```python
def make_model(data: dict) -> pm.Model:
    \"\"\"PyMC model transpiled from Stan.\"\"\"
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # ... model specification ...
        pass

    return model
```

The `data` dict contains the same keys as the Stan data block, with numpy arrays
for vectors/matrices and Python ints/floats for scalars.
"""


TOOLS = [
    {
        "name": "write_pymc_code",
        "description": (
            "Write the complete PyMC model code. This should be a self-contained "
            "Python module defining a `make_model(data: dict) -> pm.Model` function. "
            "The code must be valid Python that runs without errors."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Complete Python source code defining make_model(data). "
                        "Must include all necessary imports."
                    ),
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "validate_model",
        "description": (
            "Validate the generated PyMC model by comparing its logp against "
            "BridgeStan reference values at multiple test points. Returns detailed "
            "comparison including per-point logp differences."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "read_stan_code",
        "description": (
            "Re-read the original Stan source code. Useful if you need to "
            "double-check details of the Stan model."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]


@dataclass
class StanToPyMCResult:
    """Result of transpiling a Stan model to PyMC."""

    pymc_code: str
    logp_validated: bool
    validation_errors: list[str]
    n_attempts: int
    timings: dict[str, float]
    n_tool_calls: int = 0
    conversation_turns: int = 0
    token_usage: dict[str, int] = field(default_factory=lambda: {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
    })

    @property
    def success(self) -> bool:
        return self.logp_validated

    def save(self, path: str | Path):
        """Save the generated PyMC code to a file."""
        Path(path).write_text(self.pymc_code)

    def get_model(self, data: dict):
        """Execute the generated code and return the pm.Model instance."""
        namespace = {}
        exec(self.pymc_code, namespace)
        return namespace["make_model"](data)


@dataclass
class _AgentState:
    """Mutable state for the agent loop."""

    stan_code: str
    data: dict | None
    pymc_code: str
    reference_points: list[dict]  # [{point, logp, dlogp}, ...]
    n_unc_params: int
    unc_param_names: list[str]
    tool_calls: int = 0
    validations: int = 0
    validated: bool = False
    timings: dict = field(default_factory=dict)
    messages: list = field(default_factory=list)


@functools.lru_cache(maxsize=None)
def _load_skill(name: str) -> str:
    """Load a skill file by name."""
    path = _SKILLS_DIR / f"{name}.md"
    if not path.exists():
        return ""
    return path.read_text()


def _build_system_prompt() -> str:
    """Build system prompt with Stan→PyMC skill."""
    prompt = SYSTEM_PROMPT
    content = _load_skill("stan_to_pymc")
    if content:
        prompt += f"\n\n{'='*60}\n{content}"
    return prompt


def _build_user_prompt(
    stan_code: str,
    data: dict | None,
    reference_points: list[dict],
    unc_param_names: list[str],
) -> str:
    """Build the initial user message with Stan code and context."""
    parts = []

    parts.append("Translate this Stan model to PyMC.\n")
    parts.append(f"## Stan Model Code\n```stan\n{stan_code}\n```\n")

    # Data summary
    if data:
        parts.append("## Data\n")
        parts.append("The `data` dict passed to `make_model()` will contain:\n")
        for name, value in data.items():
            arr = np.asarray(value)
            if arr.ndim == 0:
                parts.append(f"- `{name}`: scalar = {value}")
            else:
                parts.append(
                    f"- `{name}`: shape={list(arr.shape)}, dtype={arr.dtype}"
                )
        parts.append("")

    # Parameter info
    parts.append("## Unconstrained Parameter Names (from BridgeStan)")
    parts.append(f"Total unconstrained parameters: {len(unc_param_names)}")
    for i, name in enumerate(unc_param_names):
        parts.append(f"- [{i}] {name}")
    parts.append("")

    # Validation targets
    parts.append("## Validation Targets")
    parts.append(
        "Your PyMC model's logp (evaluated at unconstrained parameter values) "
        "must match these BridgeStan reference values.\n"
    )

    for i, pt in enumerate(reference_points):
        label = "initial (all zeros)" if i == 0 else f"test point {i}"
        parts.append(f"At {label}:")
        parts.append(f"  unconstrained point = {pt['point']}")
        parts.append(f"  logp = {pt['logp']:.10f}\n")

    parts.append(
        "NOTE: The logp comparison uses the unconstrained parameterization. "
        "Both BridgeStan and PyMC include the Jacobian adjustment for transforms. "
        "The validation tool handles mapping unconstrained values to your PyMC model.\n"
    )

    parts.append(
        "Generate the `make_model(data)` function using `write_pymc_code`, "
        "then call `validate_model` to check correctness."
    )

    return "\n".join(parts)


def transpile_stan_to_pymc(
    stan_code: str,
    data: dict | str | None = None,
    api_key: str | None = None,
    max_turns: int = 20,
    model_name: str = "claude-sonnet-4-20250514",
    verbose: bool = True,
) -> StanToPyMCResult:
    """Transpile a Stan model to PyMC via an agentic Claude loop.

    Args:
        stan_code: Stan model code as a string.
        data: Data dict or JSON string for the Stan model.
        api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var).
        max_turns: Max agent turns before giving up.
        model_name: Claude model to use.
        verbose: Print progress.

    Returns:
        StanToPyMCResult with generated PyMC code and validation status.
    """
    import anthropic

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("No API key. Set ANTHROPIC_API_KEY or pass api_key=")

    client = anthropic.Anthropic(api_key=api_key)
    timings: dict[str, float] = {}

    # Parse data if JSON string
    if isinstance(data, str):
        data = json.loads(data)

    # Step 1: Extract reference values via BridgeStan
    if verbose:
        print("Extracting reference values via BridgeStan...")
    t0 = time.time()
    from pymc_rust_compiler.stan_exporter import StanModelExporter

    exporter = StanModelExporter(stan_code, data=data)
    ctx = exporter.context
    timings["extract"] = time.time() - t0

    # Build reference points
    reference_points = [
        {"point": ctx.initial_point.point, "logp": ctx.initial_point.logp,
         "dlogp": ctx.initial_point.dlogp},
    ]
    for pt in ctx.extra_points:
        reference_points.append(
            {"point": pt.point, "logp": pt.logp, "dlogp": pt.dlogp}
        )

    if verbose:
        print(
            f"  {ctx.n_params} unconstrained params, "
            f"{len(reference_points)} validation points"
        )

    # Step 2: Build prompts
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(
        stan_code, data, reference_points, list(ctx.unc_param_names),
    )

    # Step 3: Agent loop
    state = _AgentState(
        stan_code=stan_code,
        data=data,
        pymc_code="",
        reference_points=reference_points,
        n_unc_params=ctx.n_params,
        unc_param_names=list(ctx.unc_param_names),
        messages=[{"role": "user", "content": user_prompt}],
        timings=timings,
    )

    if verbose:
        print("\nStarting agent loop...")

    total_input_tokens = 0
    total_output_tokens = 0
    turn = 0

    for turn in range(max_turns):
        t0 = time.time()
        response = client.messages.create(
            model=model_name,
            max_tokens=8192,
            system=system_prompt,
            tools=TOOLS,
            messages=state.messages,
        )
        timings[f"api_turn_{turn}"] = time.time() - t0

        if hasattr(response, "usage") and response.usage:
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens
            if verbose:
                print(
                    f"  Turn {turn}: "
                    f"{response.usage.input_tokens} in / "
                    f"{response.usage.output_tokens} out tokens"
                )

        if response.stop_reason == "end_turn":
            if verbose:
                for block in response.content:
                    if hasattr(block, "text"):
                        print(f"  Agent: {block.text[:200]}")
            break

        if response.stop_reason != "tool_use":
            if verbose:
                print(f"  Unexpected stop_reason: {response.stop_reason}")
            break

        # Process tool calls
        state.messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "text" and block.text.strip() and verbose:
                print(f"  Agent: {block.text[:200]}")
            elif block.type == "tool_use":
                state.tool_calls += 1
                result = _execute_tool(block.name, block.input, state, verbose)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

                if state.validated:
                    break

        state.messages.append({"role": "user", "content": tool_results})

        if state.validated:
            break

    validation_errors = []
    if not state.validated:
        validation_errors.append(
            f"Agent did not achieve validation after {state.tool_calls} tool calls"
        )

    return StanToPyMCResult(
        pymc_code=state.pymc_code,
        logp_validated=state.validated,
        validation_errors=validation_errors,
        n_attempts=state.validations,
        timings=timings,
        n_tool_calls=state.tool_calls,
        conversation_turns=turn + 1,
        token_usage={
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        },
    )


def _execute_tool(
    name: str, input_data: dict, state: _AgentState, verbose: bool,
) -> str:
    """Execute a tool and return the result string."""
    if name == "write_pymc_code":
        return _tool_write_pymc_code(input_data, state, verbose)
    elif name == "validate_model":
        return _tool_validate_model(state, verbose)
    elif name == "read_stan_code":
        return _tool_read_stan_code(state, verbose)
    else:
        return f"Unknown tool: {name}"


def _tool_write_pymc_code(
    input_data: dict, state: _AgentState, verbose: bool,
) -> str:
    code = input_data.get("code", "")
    if not code:
        return "Error: no code provided"

    # Quick syntax check
    try:
        compile(code, "<pymc_model>", "exec")
    except SyntaxError as e:
        if verbose:
            print(f"  [write_pymc_code] Syntax error: {e}")
        return f"Syntax error in generated code: {e}"

    state.pymc_code = code
    if verbose:
        print(f"  [write_pymc_code] Wrote {len(code)} chars")
    return f"Written {len(code)} chars. Code compiles syntactically. Use `validate_model` to test logp."


def _tool_validate_model(state: _AgentState, verbose: bool) -> str:
    state.validations += 1

    if not state.pymc_code:
        return "Error: no PyMC code written yet. Use `write_pymc_code` first."

    # Execute the code to get make_model
    try:
        namespace = {}
        exec(state.pymc_code, namespace)
    except Exception as e:
        if verbose:
            print(f"  [validate_model] Code execution error: {e}")
        return f"Error executing generated code: {type(e).__name__}: {e}"

    if "make_model" not in namespace:
        return "Error: generated code does not define `make_model(data)` function."

    # Build the model
    try:
        model = namespace["make_model"](state.data or {})
    except Exception as e:
        if verbose:
            print(f"  [validate_model] Model construction error: {e}")
        return f"Error building PyMC model: {type(e).__name__}: {e}"

    import pymc as pm

    # Get the logp function in unconstrained space
    try:
        logp_fn = model.compile_logp()
    except Exception as e:
        if verbose:
            print(f"  [validate_model] compile_logp error: {e}")
        return f"Error compiling logp function: {type(e).__name__}: {e}"

    # Count Half* distributions to compute log(2) correction.
    # Stan evaluates the full symmetric density on lower-bounded params without
    # normalizing to the half-line, while PyMC's Half* distributions add log(2)
    # per parameter for proper normalization. This constant offset doesn't affect
    # sampling but causes logp comparison mismatches.
    _HALF_RV_OPS = {"HalfNormalRV", "HalfCauchyRV", "HalfStudentTRV", "HalfFlatRV"}
    n_half = sum(
        1 for rv in model.free_RVs
        if type(rv.owner.op).__name__ in _HALF_RV_OPS
    )
    half_logp_correction = n_half * np.log(2)
    if verbose and n_half > 0:
        print(f"  [validate_model] Found {n_half} Half* distribution(s), "
              f"applying log(2) correction of {half_logp_correction:.4f}")

    # Map unconstrained point to PyMC's internal variable order
    # We need to understand how PyMC orders its unconstrained parameters
    # vs how BridgeStan does it
    try:
        ip = model.initial_point()
        unc_var_names = [v.name for v in model.value_vars]
    except Exception as e:
        return f"Error getting model variable info: {type(e).__name__}: {e}"

    # Build the report
    report_lines = []
    errors = []

    # Evaluate at each reference point
    # The tricky part: BridgeStan and PyMC may order unconstrained params differently.
    # We'll try to evaluate PyMC's logp at its own initial point first, then compare.

    for i, ref in enumerate(state.reference_points):
        label = "initial (all zeros)" if i == 0 else f"test point {i}"
        unc_point = np.array(ref["point"])

        # Build a point dict for PyMC: map unconstrained values to PyMC variables
        try:
            point_dict = _map_unc_point_to_pymc(
                model, unc_point, state.unc_param_names,
            )
        except Exception as e:
            report_lines.append(f"{label}: ERROR mapping point: {e}")
            errors.append(f"{label}: point mapping error: {e}")
            continue

        # Evaluate logp
        try:
            pymc_logp = logp_fn(point_dict)
            pymc_logp = float(pymc_logp)
        except Exception as e:
            report_lines.append(f"{label}: ERROR evaluating logp: {e}")
            errors.append(f"{label}: logp evaluation error: {e}")
            continue

        ref_logp = ref["logp"]
        # Apply Half* correction: subtract log(2) per Half* distribution
        # to match Stan's convention of not normalizing symmetric priors
        # on lower-bounded parameters.
        corrected_logp = pymc_logp - half_logp_correction
        rel_err = abs(corrected_logp - ref_logp) / max(abs(ref_logp), 1.0)
        status = "OK" if rel_err <= 1e-2 else "MISMATCH"

        correction_note = (
            f" (corrected from {pymc_logp:.6f}, {n_half} Half* dists)"
            if n_half > 0 else ""
        )
        report_lines.append(
            f"{label}: logp BridgeStan={ref_logp:.6f} PyMC={corrected_logp:.6f}"
            f"{correction_note} rel_err={rel_err:.2e} [{status}]"
        )
        if rel_err > 1e-2:
            errors.append(
                f"{label}: logp mismatch: BridgeStan={ref_logp:.6f}, "
                f"PyMC={corrected_logp:.6f}{correction_note}, rel_err={rel_err:.2e}"
            )

    report = "\n".join(report_lines)

    if not errors:
        state.validated = True
        if verbose:
            print("  [validate_model] PASSED!")
        return f"VALIDATION PASSED!\n\n{report}"
    else:
        if verbose:
            print(f"  [validate_model] FAILED ({len(errors)} errors)")
        return (
            f"VALIDATION FAILED ({len(errors)} errors):\n\n{report}\n\n"
            f"Errors:\n" + "\n".join(errors)
            + "\n\nHints:\n"
            "- Check that all prior distributions match (including parameter names)\n"
            "- Ensure Stan constraints map to correct PyMC distributions "
            "(e.g. real<lower=0> with normal prior → HalfNormal)\n"
            "- Verify observed data is passed correctly\n"
            "- Check for missing Jacobian terms (PyMC handles these automatically "
            "for built-in transforms, but custom Potentials may need manual adjustment)\n"
            "- BridgeStan uses propto=True, so constant terms may differ — "
            "a relative error up to 1e-2 is acceptable\n"
            "- NOTE: The validator automatically corrects for the log(2) offset from "
            "Half* distributions (HalfNormal, HalfCauchy, etc.) — you do NOT need to "
            "add manual correction potentials for this.\n"
            f"\nPyMC model variables (unconstrained): {unc_var_names}\n"
            f"BridgeStan parameter names: {state.unc_param_names}"
        )


def _map_unc_point_to_pymc(
    model, unc_point: np.ndarray, stan_param_names: list[str],
) -> dict:
    """Map a BridgeStan unconstrained point to a PyMC point dict.

    Both BridgeStan and PyMC work in unconstrained space. We need to match
    parameters by name (stripping Stan's dot-index notation).
    """
    import re

    # Get PyMC's unconstrained variable info
    point_dict = {}
    pymc_vars = model.value_vars
    pymc_offset = 0

    # Map Stan param names to positions
    # Stan names like "mu", "sigma", "theta.1", "theta.2" etc.
    # PyMC names like "mu", "sigma_log__", "theta"

    # Group Stan params by base name
    stan_groups: dict[str, list[int]] = {}
    for i, name in enumerate(stan_param_names):
        base = re.sub(r"\.\d+$", "", name)
        if base not in stan_groups:
            stan_groups[base] = []
        stan_groups[base].append(i)

    # Try to match each PyMC variable to Stan params
    for var in pymc_vars:
        var_name = var.name
        # Strip PyMC transform suffixes
        base_name = re.sub(r"_(log|logodds|interval|circular|ordered|simplex)__$", "", var_name)

        # Determine size of this variable
        var_size = int(np.prod(var.type.shape) if hasattr(var.type, 'shape') and var.type.shape else 1)

        # Find matching Stan param group
        matched = False
        for stan_base, indices in stan_groups.items():
            if stan_base == base_name or stan_base.replace(".", "_") == base_name:
                values = np.array([unc_point[j] for j in indices])
                point_dict[var_name] = values if len(values) > 1 else values[0]
                matched = True
                break

        if not matched:
            # Fallback: use positional mapping
            end = pymc_offset + var_size
            if end <= len(unc_point):
                values = unc_point[pymc_offset:end]
                point_dict[var_name] = values if var_size > 1 else float(values[0])
            else:
                point_dict[var_name] = np.zeros(var_size) if var_size > 1 else 0.0

        pymc_offset += var_size

    return point_dict


def _tool_read_stan_code(state: _AgentState, verbose: bool) -> str:
    if verbose:
        print(f"  [read_stan_code] {len(state.stan_code)} chars")
    return state.stan_code
