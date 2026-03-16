"""Transpile between JAX and PyTorch via an agentic Claude loop.

Supports both directions:
- JAX → PyTorch: takes a JAX pure function + params, generates nn.Module
- PyTorch → JAX: takes an nn.Module, generates a JAX pure function + params

Validates by comparing forward pass outputs and gradients at multiple test points.
"""

from __future__ import annotations

import functools
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from transpailer.formatting import format_python_code as _format_python
from transpailer.jax_exporter import ModelContext

_SKILLS_DIR = Path(__file__).parent / "skills"


# ── System prompts ──────────────────────────────────────────────────────────

JAX_TO_PYTORCH_SYSTEM = """\
You are an expert deep learning engineer who translates JAX models to PyTorch.

You produce clean, idiomatic PyTorch code that is functionally equivalent to the
input JAX model. The generated code should run correctly and produce the same
outputs and gradients.

You have access to tools to iteratively write and validate PyTorch code.
Your workflow:
1. Analyze the JAX code, parameter structure, and input/output shapes
2. Write the PyTorch module using `write_code`
3. Validate with `validate_model` — this checks forward pass + gradients
4. If validation fails, analyze the errors, fix the code, and revalidate
5. Iterate until validation passes

You can also use `read_source` to re-read the original source code.

CRITICAL RULES:
1. Output must be a complete Python module defining a `make_model(params)` function
   that returns a `torch.nn.Module` with the same architecture.
2. The `params` dict maps names to numpy arrays — use them to initialize weights.
3. Use PyTorch v2+ API: `import torch`, `import torch.nn as nn`, `import torch.nn.functional as F`.
4. Preserve parameter names and shapes exactly.
5. Handle JAX's functional patterns → PyTorch's stateful module pattern:
   - `jax.nn.relu(x)` → `F.relu(x)` or `nn.ReLU()`
   - `jnp.dot(x, w)` → `F.linear(x, w)` or `x @ w`
   - `jax.lax.scan` → `for` loop or `nn.Sequential`
   - Tree-structured params → `nn.Module` with sub-modules
6. Ensure dtypes match (float32 by default).
7. The module's `forward()` should accept the same input shape and return the same output shape.

OUTPUT FORMAT:
```python
def make_model(params: dict) -> torch.nn.Module:
    \"\"\"PyTorch module transpiled from JAX.\"\"\"
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np

    class TranspiledModel(nn.Module):
        def __init__(self, params):
            super().__init__()
            # Initialize parameters from JAX params dict
            ...

        def forward(self, x):
            ...
            return output

    model = TranspiledModel(params)
    return model
```
"""

PYTORCH_TO_JAX_SYSTEM = """\
You are an expert deep learning engineer who translates PyTorch models to JAX.

You produce clean, idiomatic JAX code that is functionally equivalent to the
input PyTorch model. The generated code should run correctly and produce the same
outputs and gradients.

You have access to tools to iteratively write and validate JAX code.
Your workflow:
1. Analyze the PyTorch code, parameter structure, and input/output shapes
2. Write the JAX function using `write_code`
3. Validate with `validate_model` — this checks forward pass + gradients
4. If validation fails, analyze the errors, fix the code, and revalidate
5. Iterate until validation passes

You can also use `read_source` to re-read the original source code.

CRITICAL RULES:
1. Output must be a complete Python module defining:
   - `init_params(param_data)` → dict of jnp.arrays (from numpy param data)
   - `forward(params, x)` → output (pure function, no side effects)
2. Use JAX idioms: pure functions, explicit params, `jnp` operations.
3. Use JAX v0.4+ API: `import jax`, `import jax.numpy as jnp`.
4. Preserve parameter names and shapes exactly.
5. Handle PyTorch's stateful patterns → JAX's functional pattern:
   - `F.relu(x)` → `jax.nn.relu(x)`
   - `F.linear(x, w, b)` → `x @ w.T + b`
   - `nn.BatchNorm` → manual running stats or `flax.linen.BatchNorm`
   - `nn.Sequential` → function composition
6. Ensure dtypes match (float32 by default).
7. The `forward` function should accept the same input shape and return the same output shape.

OUTPUT FORMAT:
```python
import jax
import jax.numpy as jnp
import numpy as np

def init_params(param_data: dict) -> dict:
    \"\"\"Initialize JAX params from numpy arrays.\"\"\"
    return {k: jnp.array(v) for k, v in param_data.items()}

def forward(params: dict, x):
    \"\"\"JAX forward pass transpiled from PyTorch.\"\"\"
    ...
    return output
```
"""


# ── Tools ───────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "write_code",
        "description": (
            "Write the complete transpiled model code. Must be valid Python that "
            "defines the required functions/classes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete Python source code for the transpiled model.",
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "validate_model",
        "description": (
            "Validate the transpiled model by comparing forward pass outputs and "
            "gradients against reference values at multiple test points."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "read_source",
        "description": ("Re-read the original source code of the model being transpiled."),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ── Result types ────────────────────────────────────────────────────────────


@dataclass
class TranspileResult:
    """Result of transpiling between JAX and PyTorch."""

    source_framework: str  # "jax" or "pytorch"
    target_framework: str  # "pytorch" or "jax"
    generated_code: str
    validated: bool
    validation_errors: list[str]
    n_attempts: int
    n_tool_calls: int = 0
    conversation_turns: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    token_usage: dict[str, int] = field(
        default_factory=lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    )

    @property
    def success(self) -> bool:
        return self.validated

    def save(self, path: str | Path):
        Path(path).write_text(self.generated_code)

    def get_model(self, params: dict | None = None):
        """Execute the generated code and return the model.

        For JAX→PyTorch: returns nn.Module (call make_model(params))
        For PyTorch→JAX: returns (init_params, forward) tuple
        """
        namespace = {}
        exec(self.generated_code, namespace)

        if self.target_framework == "pytorch":
            return namespace["make_model"](params or {})
        else:
            init_fn = namespace["init_params"]
            forward_fn = namespace["forward"]
            if params is not None:
                return init_fn(params), forward_fn
            return init_fn, forward_fn


# ── Agent state ─────────────────────────────────────────────────────────────


@dataclass
class _AgentState:
    """Mutable state for the agent loop."""

    direction: str  # "jax_to_pytorch" or "pytorch_to_jax"
    source_context: ModelContext
    generated_code: str
    tool_calls: int = 0
    validations: int = 0
    validated: bool = False
    timings: dict = field(default_factory=dict)
    messages: list = field(default_factory=list)


# ── Skill loading ───────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=None)
def _load_skill(name: str) -> str:
    path = _SKILLS_DIR / f"{name}.md"
    if not path.exists():
        return ""
    return path.read_text()


# ── Prompt building ─────────────────────────────────────────────────────────


def _build_system_prompt(direction: str) -> str:
    if direction == "jax_to_pytorch":
        prompt = JAX_TO_PYTORCH_SYSTEM
        skill = _load_skill("jax_to_pytorch")
    else:
        prompt = PYTORCH_TO_JAX_SYSTEM
        skill = _load_skill("pytorch_to_jax")

    if skill:
        prompt += f"\n\n{'=' * 60}\n{skill}"
    return prompt


def _build_user_prompt(ctx: ModelContext, direction: str) -> str:
    parts = []

    source = "JAX" if direction == "jax_to_pytorch" else "PyTorch"
    target = "PyTorch" if direction == "jax_to_pytorch" else "JAX"
    parts.append(f"Translate this {source} model to {target}.\n")

    if ctx.source_code:
        parts.append(f"## Source Code\n```python\n{ctx.source_code}\n```\n")

    # Parameter info
    parts.append("## Parameters\n")
    for p in ctx.params:
        parts.append(f"- `{p.name}`: shape={p.shape}, dtype={p.dtype}, size={p.size}")
    parts.append("")

    # Input info
    parts.append("## Inputs\n")
    for i in ctx.inputs:
        parts.append(f"- `{i.name}`: shape={i.shape}, dtype={i.dtype}")
    parts.append("")

    # Output info
    parts.append("## Outputs\n")
    for o in ctx.outputs:
        parts.append(f"- `{o.name}`: shape={o.shape}, dtype={o.dtype}")
    parts.append("")

    # Validation targets
    parts.append("## Validation Targets")
    parts.append(
        f"Your {target} model's forward pass must produce the same outputs "
        f"and gradients as the {source} original at these test points.\n"
    )

    for i, vp in enumerate(ctx.validation_points):
        label = "original params" if i == 0 else f"perturbed params {i}"
        parts.append(f"### At {label}:")

        # Show param values (truncated for large tensors)
        for name, val in vp.params.items():
            val_arr = np.asarray(val)
            if val_arr.size <= 20:
                parts.append(f"  params['{name}'] = {val}")
            else:
                parts.append(
                    f"  params['{name}']: shape={list(val_arr.shape)}, "
                    f"mean={val_arr.mean():.6f}, std={val_arr.std():.6f}"
                )

        # Show expected output (truncated)
        out_arr = np.asarray(vp.output)
        if out_arr.size <= 20:
            parts.append(f"  expected output = {vp.output}")
        else:
            parts.append(
                f"  expected output: shape={list(out_arr.shape)}, mean={out_arr.mean():.6f}, std={out_arr.std():.6f}"
            )

        # Show expected gradients (truncated)
        for name, grad in vp.grad_params.items():
            grad_arr = np.asarray(grad)
            if grad_arr.size <= 20:
                parts.append(f"  grad['{name}'] = {grad}")
            else:
                parts.append(
                    f"  grad['{name}']: shape={list(grad_arr.shape)}, "
                    f"mean={grad_arr.mean():.6f}, std={grad_arr.std():.6f}"
                )
        parts.append("")

    parts.append(f"Generate the {target} code using `write_code`, then call `validate_model` to check correctness.")

    return "\n".join(parts)


# ── Tool execution ──────────────────────────────────────────────────────────


def _execute_tool(
    name: str,
    input_data: dict,
    state: _AgentState,
    verbose: bool,
) -> str:
    if name == "write_code":
        return _tool_write_code(input_data, state, verbose)
    elif name == "validate_model":
        return _tool_validate(state, verbose)
    elif name == "read_source":
        return _tool_read_source(state, verbose)
    else:
        return f"Unknown tool: {name}"


def _tool_write_code(
    input_data: dict,
    state: _AgentState,
    verbose: bool,
) -> str:
    code = input_data.get("code", "")
    if not code:
        return "Error: no code provided"

    try:
        compile(code, "<transpiled_model>", "exec")
    except SyntaxError as e:
        if verbose:
            print(f"  [write_code] Syntax error: {e}")
        return f"Syntax error in generated code: {e}"

    code = _format_python(code)
    state.generated_code = code
    if verbose:
        print(f"  [write_code] Wrote {len(code)} chars")
    return f"Written {len(code)} chars. Code is syntactically valid. Use `validate_model` to test."


def _tool_validate(state: _AgentState, verbose: bool) -> str:
    state.validations += 1

    if not state.generated_code:
        return "Error: no code written yet. Use `write_code` first."

    # Execute the generated code
    try:
        namespace = {}
        exec(state.generated_code, namespace)
    except Exception as e:
        if verbose:
            print(f"  [validate] Code execution error: {e}")
        return f"Error executing generated code: {type(e).__name__}: {e}"

    if state.direction == "jax_to_pytorch":
        return _validate_pytorch(namespace, state, verbose)
    else:
        return _validate_jax(namespace, state, verbose)


def _validate_pytorch(namespace: dict, state: _AgentState, verbose: bool) -> str:
    """Validate a generated PyTorch model against reference values."""
    import torch

    if "make_model" not in namespace:
        return "Error: generated code does not define `make_model(params)` function."

    report_lines = []
    errors = []

    for i, vp in enumerate(state.source_context.validation_points):
        label = "original params" if i == 0 else f"perturbed {i}"

        # Build model with these params
        try:
            model = namespace["make_model"](vp.params)
        except Exception as e:
            errors.append(f"{label}: make_model error: {e}")
            report_lines.append(f"{label}: ERROR building model: {e}")
            continue

        # Forward pass
        try:
            model.eval()
            model.zero_grad()
            model.train()

            # Prepare input
            inp = vp.inputs
            if len(inp) == 1 and "x" in inp:
                x = torch.tensor(np.array(inp["x"]), dtype=torch.float32)
                output = model(x)
            else:
                tensors = {k: torch.tensor(np.array(v), dtype=torch.float32) for k, v in inp.items()}
                output = model(**tensors)

            out_np = output.detach().cpu().numpy()
            ref_out = np.array(vp.output)

            # Check output
            if out_np.shape != ref_out.shape:
                errors.append(f"{label}: shape mismatch: got {out_np.shape}, expected {ref_out.shape}")
                report_lines.append(f"{label}: SHAPE MISMATCH {out_np.shape} vs {ref_out.shape}")
                continue

            max_diff = float(np.max(np.abs(out_np - ref_out)))
            rel_err = float(np.max(np.abs(out_np - ref_out) / np.maximum(np.abs(ref_out), 1e-8)))
            out_ok = rel_err <= 1e-4

            report_lines.append(
                f"{label}: output max_diff={max_diff:.2e} rel_err={rel_err:.2e} [{'OK' if out_ok else 'MISMATCH'}]"
            )
            if not out_ok:
                errors.append(f"{label}: output mismatch: max_diff={max_diff:.2e}, rel_err={rel_err:.2e}")

            # Check gradients
            loss = output.sum()
            loss.backward()

            for pname, ref_grad in vp.grad_params.items():
                ref_g = np.array(ref_grad)
                found = False
                for name, param in model.named_parameters():
                    if name == pname and param.grad is not None:
                        got_g = param.grad.detach().cpu().numpy()
                        grad_diff = float(np.max(np.abs(got_g - ref_g)))
                        grad_rel = float(np.max(np.abs(got_g - ref_g) / np.maximum(np.abs(ref_g), 1e-8)))
                        grad_ok = grad_rel <= 1e-3
                        report_lines.append(
                            f"  grad['{name}']: max_diff={grad_diff:.2e} rel_err={grad_rel:.2e} "
                            f"[{'OK' if grad_ok else 'MISMATCH'}]"
                        )
                        if not grad_ok:
                            errors.append(f"{label}: grad['{name}'] mismatch: rel_err={grad_rel:.2e}")
                        found = True
                        break
                if not found:
                    errors.append(f"{label}: gradient for '{pname}' not found in model")

        except Exception as e:
            errors.append(f"{label}: forward/grad error: {e}")
            report_lines.append(f"{label}: ERROR: {e}")

    report = "\n".join(report_lines)

    if not errors:
        state.validated = True
        if verbose:
            print("  [validate] PASSED!")
        return f"VALIDATION PASSED!\n\n{report}"
    else:
        if verbose:
            print(f"  [validate] FAILED ({len(errors)} errors)")
        return f"VALIDATION FAILED ({len(errors)} errors):\n\n{report}\n\nErrors:\n" + "\n".join(
            f"- {e}" for e in errors
        )


def _validate_jax(namespace: dict, state: _AgentState, verbose: bool) -> str:
    """Validate a generated JAX model against reference values."""
    import jax
    import jax.numpy as jnp

    if "forward" not in namespace:
        return "Error: generated code does not define `forward(params, x)` function."
    if "init_params" not in namespace:
        return "Error: generated code does not define `init_params(param_data)` function."

    forward_fn = namespace["forward"]
    init_fn = namespace["init_params"]

    # Build gradient function
    def scalar_fn(params, x):
        return jnp.sum(forward_fn(params, x))

    grad_fn = jax.grad(scalar_fn, argnums=0)

    report_lines = []
    errors = []

    for i, vp in enumerate(state.source_context.validation_points):
        label = "original params" if i == 0 else f"perturbed {i}"

        try:
            params = init_fn(vp.params)
        except Exception as e:
            errors.append(f"{label}: init_params error: {e}")
            report_lines.append(f"{label}: ERROR init: {e}")
            continue

        try:
            # Prepare input
            inp = vp.inputs
            if len(inp) == 1 and "x" in inp:
                x = jnp.array(inp["x"])
            else:
                x = {k: jnp.array(v) for k, v in inp.items()}

            output = forward_fn(params, x)
            out_np = np.asarray(output)
            ref_out = np.array(vp.output)

            if out_np.shape != ref_out.shape:
                errors.append(f"{label}: shape mismatch: got {out_np.shape}, expected {ref_out.shape}")
                report_lines.append(f"{label}: SHAPE MISMATCH {out_np.shape} vs {ref_out.shape}")
                continue

            max_diff = float(np.max(np.abs(out_np - ref_out)))
            rel_err = float(np.max(np.abs(out_np - ref_out) / np.maximum(np.abs(ref_out), 1e-8)))
            out_ok = rel_err <= 1e-4

            report_lines.append(
                f"{label}: output max_diff={max_diff:.2e} rel_err={rel_err:.2e} [{'OK' if out_ok else 'MISMATCH'}]"
            )
            if not out_ok:
                errors.append(f"{label}: output mismatch: max_diff={max_diff:.2e}, rel_err={rel_err:.2e}")

            # Check gradients
            grads = grad_fn(params, x)

            for pname, ref_grad in vp.grad_params.items():
                ref_g = np.array(ref_grad)
                if pname in grads:
                    got_g = np.asarray(grads[pname])
                    grad_diff = float(np.max(np.abs(got_g - ref_g)))
                    grad_rel = float(np.max(np.abs(got_g - ref_g) / np.maximum(np.abs(ref_g), 1e-8)))
                    grad_ok = grad_rel <= 1e-3
                    report_lines.append(
                        f"  grad['{pname}']: max_diff={grad_diff:.2e} rel_err={grad_rel:.2e} "
                        f"[{'OK' if grad_ok else 'MISMATCH'}]"
                    )
                    if not grad_ok:
                        errors.append(f"{label}: grad['{pname}'] mismatch: rel_err={grad_rel:.2e}")
                else:
                    errors.append(f"{label}: gradient for '{pname}' not found")

        except Exception as e:
            errors.append(f"{label}: forward/grad error: {e}")
            report_lines.append(f"{label}: ERROR: {e}")

    report = "\n".join(report_lines)

    if not errors:
        state.validated = True
        if verbose:
            print("  [validate] PASSED!")
        return f"VALIDATION PASSED!\n\n{report}"
    else:
        if verbose:
            print(f"  [validate] FAILED ({len(errors)} errors)")
        return f"VALIDATION FAILED ({len(errors)} errors):\n\n{report}\n\nErrors:\n" + "\n".join(
            f"- {e}" for e in errors
        )


def _tool_read_source(state: _AgentState, verbose: bool) -> str:
    source = state.source_context.source_code or "(no source code available)"
    if verbose:
        print(f"  [read_source] {len(source)} chars")
    return source


# ── Main transpiler functions ───────────────────────────────────────────────


def _run_agent_loop(
    state: _AgentState,
    system_prompt: str,
    api_key: str,
    max_turns: int,
    model_name: str,
    verbose: bool,
) -> tuple[int, dict[str, int]]:
    """Run the agentic Claude loop. Returns (turns, token_usage)."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

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
        state.timings[f"api_turn_{turn}"] = time.time() - t0

        if hasattr(response, "usage") and response.usage:
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens
            if verbose:
                print(f"  Turn {turn}: {response.usage.input_tokens} in / {response.usage.output_tokens} out tokens")

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
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

                if state.validated:
                    break

        state.messages.append({"role": "user", "content": tool_results})

        if state.validated:
            break

    return turn + 1, {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
    }


def transpile_jax_to_pytorch(
    fn: Callable,
    params: dict[str, Any],
    sample_input: Any,
    api_key: str | None = None,
    max_turns: int = 20,
    model_name: str = "claude-sonnet-4-20250514",
    verbose: bool = True,
    loss_fn: Callable | None = None,
    source_code: str | None = None,
) -> TranspileResult:
    """Transpile a JAX model to PyTorch via an agentic Claude loop.

    Args:
        fn: JAX pure function with signature f(params, x) -> output.
        params: Dict of {name: jnp.array} parameter values.
        sample_input: Example input for the model.
        api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var).
        max_turns: Max agent turns before giving up.
        model_name: Claude model to use.
        verbose: Print progress.
        loss_fn: Optional scalar loss function for gradient computation.
        source_code: Optional source code override.

    Returns:
        TranspileResult with generated PyTorch code and validation status.
    """
    from transpailer.jax_exporter import JaxModelExporter

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("No API key. Set ANTHROPIC_API_KEY or pass api_key=")

    timings: dict[str, float] = {}

    # Step 1: Extract model context
    if verbose:
        print("Extracting JAX model context...")
    t0 = time.time()
    exporter = JaxModelExporter(
        fn,
        params,
        sample_input,
        source_code=source_code,
        loss_fn=loss_fn,
    )
    ctx = exporter.context
    timings["extract"] = time.time() - t0

    if verbose:
        n_params = sum(p.size for p in ctx.params)
        print(f"  {n_params} parameters, {len(ctx.validation_points)} validation points")

    # Step 2: Build prompts and run agent
    system_prompt = _build_system_prompt("jax_to_pytorch")
    user_prompt = _build_user_prompt(ctx, "jax_to_pytorch")

    state = _AgentState(
        direction="jax_to_pytorch",
        source_context=ctx,
        generated_code="",
        messages=[{"role": "user", "content": user_prompt}],
        timings=timings,
    )

    if verbose:
        print("\nStarting agent loop...")

    turns, token_usage = _run_agent_loop(
        state,
        system_prompt,
        api_key,
        max_turns,
        model_name,
        verbose,
    )

    validation_errors = []
    if not state.validated:
        validation_errors.append(f"Agent did not achieve validation after {state.tool_calls} tool calls")

    return TranspileResult(
        source_framework="jax",
        target_framework="pytorch",
        generated_code=state.generated_code,
        validated=state.validated,
        validation_errors=validation_errors,
        n_attempts=state.validations,
        n_tool_calls=state.tool_calls,
        conversation_turns=turns,
        timings=timings,
        token_usage=token_usage,
    )


def transpile_pytorch_to_jax(
    module: Any,  # torch.nn.Module
    sample_input: Any,
    api_key: str | None = None,
    max_turns: int = 20,
    model_name: str = "claude-sonnet-4-20250514",
    verbose: bool = True,
    loss_fn: Callable | None = None,
    source_code: str | None = None,
) -> TranspileResult:
    """Transpile a PyTorch model to JAX via an agentic Claude loop.

    Args:
        module: PyTorch nn.Module to transpile.
        sample_input: Example input tensor/dict for the model.
        api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var).
        max_turns: Max agent turns before giving up.
        model_name: Claude model to use.
        verbose: Print progress.
        loss_fn: Optional scalar loss function for gradient computation.
        source_code: Optional source code override.

    Returns:
        TranspileResult with generated JAX code and validation status.
    """
    from transpailer.pytorch_exporter import PytorchModelExporter

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("No API key. Set ANTHROPIC_API_KEY or pass api_key=")

    timings: dict[str, float] = {}

    # Step 1: Extract model context
    if verbose:
        print("Extracting PyTorch model context...")
    t0 = time.time()
    exporter = PytorchModelExporter(
        module,
        sample_input,
        source_code=source_code,
        loss_fn=loss_fn,
    )
    ctx = exporter.context
    timings["extract"] = time.time() - t0

    if verbose:
        n_params = sum(p.size for p in ctx.params)
        print(f"  {n_params} parameters, {len(ctx.validation_points)} validation points")

    # Step 2: Build prompts and run agent
    system_prompt = _build_system_prompt("pytorch_to_jax")
    user_prompt = _build_user_prompt(ctx, "pytorch_to_jax")

    state = _AgentState(
        direction="pytorch_to_jax",
        source_context=ctx,
        generated_code="",
        messages=[{"role": "user", "content": user_prompt}],
        timings=timings,
    )

    if verbose:
        print("\nStarting agent loop...")

    turns, token_usage = _run_agent_loop(
        state,
        system_prompt,
        api_key,
        max_turns,
        model_name,
        verbose,
    )

    validation_errors = []
    if not state.validated:
        validation_errors.append(f"Agent did not achieve validation after {state.tool_calls} tool calls")

    return TranspileResult(
        source_framework="pytorch",
        target_framework="jax",
        generated_code=state.generated_code,
        validated=state.validated,
        validation_errors=validation_errors,
        n_attempts=state.validations,
        n_tool_calls=state.tool_calls,
        conversation_turns=turns,
        timings=timings,
        token_usage=token_usage,
    )
