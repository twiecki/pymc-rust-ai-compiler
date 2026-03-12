"""Compile a Stan model to Rust via an agentic Claude loop.

Same architecture as compiler.py but takes Stan code as input instead of
a PyMC model. Uses BridgeStan for reference logp+gradient values.
"""

from __future__ import annotations

import functools
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pymc_rust_compiler.stan_exporter import StanModelExporter

_SKILLS_DIR = Path(__file__).parent / "skills"


STAN_SYSTEM_PROMPT = """You are an expert Rust code generator for Bayesian statistical models.

You are translating a **Stan model** to optimized Rust. The Stan model is provided
along with reference logp+gradient values from BridgeStan for validation.

You have access to tools to iteratively write, build, and validate Rust code.
Your workflow:
1. Analyze the Stan code, parameter layout, and validation targets
2. Write the Rust code using `write_rust_code`
3. Build with `cargo_build` — if it fails, read the errors, fix the code, rebuild
4. Validate with `validate_logp` — if it fails, analyze which terms are wrong, fix, rebuild, revalidate
5. Iterate until validation passes

You can also use `read_file` to inspect data.rs (pre-generated data arrays),
the current generated.rs, or any other file in the project.

CRITICAL RULES:
1. Parameters are in UNCONSTRAINED space (Stan's internal transforms)
2. Include the Jacobian adjustment for any constraints (lower=0 → exp transform + log_jac)
3. The gradient must be analytically correct — NUTS relies on this
4. Include ALL terms in the logp — match BridgeStan's output exactly
5. Use efficient single-pass computation where possible
6. All data is provided in `data.rs` — use `use crate::data::*;` to access it

USING RUST CRATES (LIBRARIES):
You are ENCOURAGED to use external Rust crates when they help. Do NOT try to implement
everything from scratch — the Rust ecosystem has excellent libraries for many tasks.
Use the `add_cargo_dependency` tool to add crates to Cargo.toml before using them in code.

Good reasons to add a crate:
- Linear algebra (nalgebra, faer, ndarray) — don't hand-roll matrix ops
- Special math functions (statrs, special) — distributions, gamma, beta, erf, etc.
- Numerical routines (argmin, levenberg-marquardt) — optimization, root-finding
- Serialization (serde, serde_json) — data loading
- Any well-maintained crate that simplifies your implementation

If you hit a wall implementing something from scratch (e.g., a Cholesky decomposition,
special function, or complex numerical algorithm), STOP and add an appropriate crate
instead. Getting creative with crate selection is much better than struggling with
a buggy hand-rolled implementation.

STAN → RUST PARAMETER TRANSFORMS:
- `real` → identity (no transform)
- `real<lower=0>` → exp transform: x_constrained = exp(x_unc), jacobian = x_unc
- `real<lower=a,upper=b>` → logit-scaled: x = a + (b-a)*sigmoid(x_unc)
- `real<lower=a>` → shifted exp: x = a + exp(x_unc), jacobian = x_unc

STAN INDEXING: Stan is 1-based, Rust is 0-based. `x[i]` in Stan → `x[i-1]` in Rust.

EXACT LOG-DENSITY FORMULAS (include ALL terms):

Normal(x | mu, sigma):
  logp = -0.5*log(2*pi) - log(sigma) - 0.5*((x-mu)/sigma)^2
  d(logp)/d(x) = -(x-mu)/sigma^2
  d(logp)/d(sigma) = -1/sigma + (x-mu)^2/sigma^3

HalfNormal (sigma with lower=0, exp transform, unconstrained = log_sigma):
  sigma = exp(log_sigma)
  logp_prior = log(2) - 0.5*log(2*pi) - log(tau) - 0.5*(sigma/tau)^2
  jacobian = log_sigma  (i.e. + log_sigma to total logp)

PERFORMANCE OPTIMIZATION RULES:
1. HOIST INVARIANTS OUT OF LOOPS (1/sigma^2, log constants, etc.)
2. USE MULTIPLY INSTEAD OF DIVIDE
3. USE SEPARATE GRADIENT ACCUMULATORS for auto-vectorization
4. `x * x` not `.powi(2)` in hot loops
5. Pre-allocate scratch arrays in the struct (zero allocation in logp)
6. PRECOMPUTE LOG CONSTANTS: `sigma.ln()` = `log_sigma` for log-transformed params

DEBUGGING TIPS:
- Define `const LN_2PI: f64 = 1.8378770664093453;` (this is ln(2π))
- If logp is way off, check that you included the observed likelihood
- If gradient is wrong for a sigma parameter, check the Jacobian term
- Use `read_file` to inspect data.rs for array names and sizes

RUST CODE STRUCTURE:
```rust
use std::collections::HashMap;
use nuts_rs::{CpuLogpFunc, CpuMathError, LogpError, Storable};
use nuts_storable::HasDims;
use thiserror::Error;
use crate::data::*;

#[derive(Debug, Error)]
pub enum SampleError {
    #[error("Recoverable: {0}")]
    Recoverable(String),
}

impl LogpError for SampleError {
    fn is_recoverable(&self) -> bool { true }
}

pub const N_PARAMS: usize = ...;

#[derive(Storable, Clone)]
pub struct Draw {
    #[storable(dims("param"))]
    pub parameters: Vec<f64>,
}

#[derive(Clone, Default)]
pub struct GeneratedLogp;

impl HasDims for GeneratedLogp {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::from([("param".to_string(), N_PARAMS as u64)])
    }
}

impl CpuLogpFunc for GeneratedLogp {
    type LogpError = SampleError;
    type FlowParameters = ();
    type ExpandedVector = Draw;

    fn dim(&self) -> usize { N_PARAMS }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, SampleError> {
        // YOUR IMPLEMENTATION HERE
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}
```
"""

# Tool definitions — same as compiler.py
TOOLS = [
    {
        "name": "write_rust_code",
        "description": (
            "Write the complete generated.rs Rust file. This overwrites the current file. "
            "The code must be a complete, compilable Rust module implementing CpuLogpFunc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete Rust source code for generated.rs",
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "cargo_build",
        "description": (
            "Build the Rust project with `cargo build --release`. "
            "Returns build output including any compiler errors."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "validate_logp",
        "description": (
            "Run the compiled validator against BridgeStan reference values. "
            "Returns per-point logp comparison and gradient errors."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a file from the Rust project. Useful for inspecting data.rs "
            "(pre-generated data arrays), the current generated.rs, Cargo.toml, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path within the build directory",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "add_cargo_dependency",
        "description": (
            "Add a crate dependency to Cargo.toml. Use this when you want to leverage "
            "an external Rust library instead of implementing something from scratch. "
            "The dependency is appended to the [dependencies] section. "
            "Examples: name='nalgebra' version='0.33' or name='statrs' version='0.17' "
            "or name='serde' version='1' features='derive'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Crate name (e.g., 'nalgebra', 'statrs', 'ndarray')",
                },
                "version": {
                    "type": "string",
                    "description": "Version requirement (e.g., '0.33', '1', '0.17')",
                },
                "features": {
                    "type": "string",
                    "description": "Optional comma-separated features to enable (e.g., 'derive' or 'std,alloc')",
                },
            },
            "required": ["name", "version"],
        },
    },
]


@dataclass
class StanCompilationResult:
    """Result of compiling a Stan model to Rust."""

    rust_code: str
    logp_validated: bool
    validation_errors: list[str]
    n_attempts: int
    build_dir: Path | None
    timings: dict[str, float]
    n_tool_calls: int = 0
    conversation_turns: int = 0
    token_usage: dict[str, int] = field(
        default_factory=lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    )
    us_per_eval: float | None = None

    @property
    def success(self) -> bool:
        return self.logp_validated


@dataclass
class _AgentState:
    """Mutable state for the agent loop."""

    build_path: Path
    ctx: object  # StanModelContext
    messages: list[dict]
    tool_calls: int = 0
    builds: int = 0
    validations: int = 0
    validated: bool = False
    timings: dict = field(default_factory=dict)


def _detect_stan_skills(stan_code: str) -> list[str]:
    """Detect which skills are needed based on Stan model structure."""
    skills = ["stan"]  # Always include the Stan skill

    # GP detection
    gp_indicators = [
        "gp_exp_quad_cov",
        "gp_matern",
        "gp_periodic",
        "cov_exp_quad",
        "multi_normal",
        "multi_normal_cholesky",
        "cholesky_decompose",
    ]
    if any(kw in stan_code.lower() for kw in gp_indicators):
        skills.append("gp")

    return skills


@functools.lru_cache(maxsize=None)
def _load_skill(name: str) -> str:
    """Load a skill file by name."""
    path = _SKILLS_DIR / f"{name}.md"
    if not path.exists():
        return ""
    return path.read_text()


def _build_system_prompt(skills: list[str]) -> str:
    """Build system prompt with skill augmentation."""
    prompt = STAN_SYSTEM_PROMPT
    for skill_name in skills:
        content = _load_skill(skill_name)
        if content:
            prompt += f"\n\n{'=' * 60}\n{content}"
    return prompt


# Extra Cargo.toml dependencies per skill
_SKILL_CARGO_DEPS: dict[str, dict[str, str]] = {
    "gp": {"faer": "0.24"},
}


def compile_stan_model(
    stan_code: str,
    data: dict | str | None = None,
    api_key: str | None = None,
    max_turns: int = 30,
    model_name: str = "claude-sonnet-4-20250514",
    build_dir: str | Path | None = None,
    verbose: bool = True,
) -> StanCompilationResult:
    """Compile a Stan model to optimized Rust via an agentic Claude loop.

    Args:
        stan_code: Stan model code as a string.
        data: Data dict or JSON string for the Stan model.
        api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var).
        max_turns: Max agent turns before giving up.
        model_name: Claude model to use.
        build_dir: Where to create the Rust project. Temp dir if None.
        verbose: Print progress.

    Returns:
        StanCompilationResult with generated Rust code and validation status.
    """
    import anthropic

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("No API key. Set ANTHROPIC_API_KEY or pass api_key=")

    client = anthropic.Anthropic(api_key=api_key)
    timings: dict[str, float] = {}

    # Step 1: Extract model context via BridgeStan
    if verbose:
        print("Extracting Stan model context via BridgeStan...")
    t0 = time.time()
    exporter = StanModelExporter(stan_code, data=data)
    ctx = exporter.context
    prompt = exporter.to_prompt()
    timings["extract"] = time.time() - t0
    if verbose:
        print(f"  {ctx.n_params} unconstrained params, {len(prompt)} char prompt")

    # Detect skills
    skills = _detect_stan_skills(stan_code)
    system_prompt = _build_system_prompt(skills)
    if verbose:
        print(f"  Skills loaded: {', '.join(skills)}")

    # Extra cargo deps
    extra_deps: dict[str, str] = {}
    for skill_name in skills:
        extra_deps.update(_SKILL_CARGO_DEPS.get(skill_name, {}))

    # Step 2: Set up build directory
    if build_dir:
        build_path = Path(build_dir)
        build_path.mkdir(parents=True, exist_ok=True)
    else:
        build_path = Path(tempfile.mkdtemp(prefix="stan_rust_"))

    _setup_rust_project(build_path, ctx, data, extra_cargo_deps=extra_deps)

    if verbose:
        print(f"  Build dir: {build_path}")

    # Step 3: Agent loop
    state = _AgentState(
        build_path=build_path,
        ctx=ctx,
        messages=[
            {
                "role": "user",
                "content": (
                    "Generate a Rust logp+gradient implementation for this Stan model.\n\n"
                    "Use your tools to write the code, build it, and validate it. "
                    "Iterate until validation passes.\n\n"
                    f"{prompt}"
                ),
            }
        ],
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
            max_tokens=16384,
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
                    f"  Turn {turn}: {response.usage.input_tokens} in / {response.usage.output_tokens} out tokens"
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

    # Read final code
    gen_path = build_path / "src" / "generated.rs"
    rust_code = gen_path.read_text() if gen_path.exists() else ""

    validation_errors = []
    if not state.validated:
        validation_errors.append(
            f"Agent did not achieve validation after {state.tool_calls} tool calls"
        )

    return StanCompilationResult(
        rust_code=rust_code,
        logp_validated=state.validated,
        validation_errors=validation_errors,
        n_attempts=state.builds,
        build_dir=build_path,
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
    name: str,
    input_data: dict,
    state: _AgentState,
    verbose: bool,
) -> str:
    """Execute a tool and return the result string."""
    if name == "write_rust_code":
        return _tool_write_rust_code(input_data, state, verbose)
    elif name == "cargo_build":
        return _tool_cargo_build(state, verbose)
    elif name == "validate_logp":
        return _tool_validate_logp(state, verbose)
    elif name == "read_file":
        return _tool_read_file(input_data, state, verbose)
    elif name == "add_cargo_dependency":
        return _tool_add_cargo_dependency(input_data, state, verbose)
    else:
        return f"Unknown tool: {name}"


def _tool_write_rust_code(
    input_data: dict,
    state: _AgentState,
    verbose: bool,
) -> str:
    code = input_data.get("code", "")
    if not code:
        return "Error: no code provided"
    gen_path = state.build_path / "src" / "generated.rs"
    gen_path.write_text(code)
    if verbose:
        print(f"  [write_rust_code] Wrote {len(code)} chars to generated.rs")
    return f"Written {len(code)} chars to src/generated.rs"


def _tool_cargo_build(state: _AgentState, verbose: bool) -> str:
    state.builds += 1
    t0 = time.time()
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=state.build_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    elapsed = time.time() - t0
    state.timings[f"build_{state.builds}"] = elapsed

    if result.returncode == 0:
        if verbose:
            print(f"  [cargo_build] OK ({elapsed:.1f}s)")
        return f"Build successful ({elapsed:.1f}s)"
    else:
        if verbose:
            print(f"  [cargo_build] FAILED ({elapsed:.1f}s)")
        errors = result.stderr
        if len(errors) > 4000:
            errors = errors[:4000] + "\n... (truncated)"
        return f"Build FAILED:\n{errors}"


def _tool_validate_logp(state: _AgentState, verbose: bool) -> str:
    state.validations += 1
    ctx = state.ctx

    binary = state.build_path / "target" / "release" / "validate"
    if not binary.exists():
        return "Error: validation binary not found. Run cargo_build first."

    all_points = [("initial", ctx.initial_point)] + [
        (f"extra_{i}", p) for i, p in enumerate(ctx.extra_points)
    ]

    input_lines = []
    for name, vp in all_points:
        input_lines.append(",".join(f"{v:.17e}" for v in vp.point))

    stdin_data = "\n".join(input_lines) + "\n"

    try:
        result = subprocess.run(
            [str(binary)],
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return "Error: validation binary timed out"

    if result.returncode != 0:
        return f"Error: validator crashed: {result.stderr[:500]}"

    output_lines = [line for line in result.stdout.strip().split("\n") if line]

    if len(output_lines) != len(all_points):
        return (
            f"Error: expected {len(all_points)} output lines, got {len(output_lines)}"
        )

    parsed = []
    for output_line in output_lines:
        if output_line.startswith("ERROR:"):
            parsed.append(None)
        else:
            values = [float(v) for v in output_line.split(",")]
            parsed.append((values[0], values[1:]))

    # Build detailed report
    report_lines = []
    errors = []

    for (name, vp), res in zip(all_points, parsed):
        if res is None:
            report_lines.append(f"{name}: ERROR from validator")
            errors.append(f"{name}: ERROR")
            continue

        rust_logp, rust_grad = res
        rel_err = abs(rust_logp - vp.logp) / max(abs(vp.logp), 1.0)
        status = "OK" if rel_err <= 1e-4 else "MISMATCH"
        report_lines.append(
            f"{name}: logp BridgeStan={vp.logp:.10f} Rust={rust_logp:.10f} "
            f"rel_err={rel_err:.2e} [{status}]"
        )
        if rel_err > 1e-4:
            errors.append(
                f"{name}: logp mismatch: BridgeStan={vp.logp:.10f}, "
                f"Rust={rust_logp:.10f}, rel_err={rel_err:.2e}"
            )

        # Gradient comparison
        grad_errors = 0
        for j, (rust_g, ref_g) in enumerate(zip(rust_grad, vp.dlogp)):
            grad_err = abs(rust_g - ref_g) / max(abs(ref_g), 1.0)
            if grad_err > 1e-3:
                grad_errors += 1
                if grad_errors <= 5:
                    errors.append(
                        f"{name}: gradient[{j}] mismatch: BridgeStan={ref_g:.6e}, "
                        f"Rust={rust_g:.6e}, rel_err={grad_err:.2e}"
                    )
        if grad_errors > 0:
            report_lines.append(f"  {grad_errors} gradient mismatches")
        else:
            report_lines.append("  All gradients OK")

    report = "\n".join(report_lines)

    if not errors:
        state.validated = True
        if verbose:
            print("  [validate_logp] PASSED!")
        return f"VALIDATION PASSED!\n\n{report}"
    else:
        if verbose:
            print(f"  [validate_logp] FAILED ({len(errors)} errors)")
        return (
            f"VALIDATION FAILED ({len(errors)} errors):\n\n{report}\n\n"
            f"Errors:\n" + "\n".join(errors)
        )


def _tool_read_file(
    input_data: dict,
    state: _AgentState,
    verbose: bool,
) -> str:
    rel_path = input_data.get("path", "")
    if not rel_path:
        return "Error: no path provided"

    file_path = state.build_path / rel_path
    if not file_path.exists():
        available = []
        for f in state.build_path.iterdir():
            if f.is_file():
                available.append(f.name)
        src = state.build_path / "src"
        if src.exists():
            for f in src.rglob("*"):
                if f.is_file():
                    available.append(str(f.relative_to(state.build_path)))
        return f"File not found: {rel_path}\nAvailable files: {', '.join(available)}"

    content = file_path.read_text()
    if len(content) > 8000:
        content = content[:8000] + "\n... (truncated)"

    if verbose:
        print(f"  [read_file] {rel_path} ({len(content)} chars)")
    return content


def _tool_add_cargo_dependency(
    input_data: dict,
    state: _AgentState,
    verbose: bool,
) -> str:
    """Add a crate dependency to Cargo.toml."""
    name = input_data.get("name", "")
    version = input_data.get("version", "")
    features = input_data.get("features", "")

    if not name or not version:
        return "Error: 'name' and 'version' are required"

    cargo_path = state.build_path / "Cargo.toml"
    if not cargo_path.exists():
        return "Error: Cargo.toml not found"

    cargo_content = cargo_path.read_text()

    # Build the dependency line
    if features:
        feature_list = ", ".join(f'"{f.strip()}"' for f in features.split(","))
        dep_line = f'{name} = {{ version = "{version}", features = [{feature_list}] }}'
    else:
        dep_line = f'{name} = "{version}"'

    # Check if dependency already exists
    if f"\n{name} " in cargo_content or f"\n{name}=" in cargo_content:
        return f"Dependency '{name}' already exists in Cargo.toml"

    # Insert after [dependencies] section
    lines = cargo_content.split("\n")
    insert_idx = None
    in_deps = False
    for i, line in enumerate(lines):
        if line.strip() == "[dependencies]":
            in_deps = True
            continue
        if in_deps:
            if line.strip().startswith("["):
                insert_idx = i
                break
            if not line.strip():
                continue
            insert_idx = i + 1

    if insert_idx is None:
        insert_idx = len(lines)

    lines.insert(insert_idx, dep_line)
    cargo_path.write_text("\n".join(lines))

    if verbose:
        print(f"  [add_cargo_dependency] Added {dep_line}")

    return f"Added dependency: {dep_line}\nRun cargo_build to download and compile it."


def _generate_data_rs(data: dict | None) -> str:
    """Generate a Rust data module from Stan data dict."""
    lines = ["//! Auto-generated data arrays from Stan data block. DO NOT EDIT.\n"]

    if not data or not isinstance(data, dict):
        return "\n".join(lines)

    for name, value in data.items():
        arr = np.asarray(value)

        if arr.ndim == 0:
            # Scalar
            if np.issubdtype(arr.dtype, np.integer):
                lines.append(f"pub const {name.upper()}: usize = {int(arr)};")
            else:
                lines.append(f"pub const {name.upper()}: f64 = {float(arr):.17e};")
        else:
            # Array
            flat = arr.ravel().astype(np.float64)
            n = len(flat)
            lines.append(f"pub const {name.upper()}_N: usize = {n};")
            formatted = ", ".join(f"{v:.17e}" for v in flat)
            lines.append(f"pub const {name.upper()}_DATA: &[f64] = &[{formatted}];\n")

            # Also export shape info for multi-dimensional arrays
            if arr.ndim > 1:
                for dim_i, dim_size in enumerate(arr.shape):
                    lines.append(
                        f"pub const {name.upper()}_DIM{dim_i}: usize = {dim_size};"
                    )
                lines.append("")

    return "\n".join(lines)


def _setup_rust_project(
    build_path: Path,
    ctx,
    data: dict | str | None = None,
    extra_cargo_deps: dict[str, str] | None = None,
):
    """Create the Rust project structure with pre-generated data."""
    src_dir = build_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    deps_lines = [
        'nuts-rs = "0.17"',
        'nuts-storable = "0.2"',
        'rand = "0.10"',
        'thiserror = "2"',
        'anyhow = "1"',
    ]
    for dep_name, dep_version in (extra_cargo_deps or {}).items():
        if dep_version.startswith("{"):
            deps_lines.append(f"{dep_name} = {dep_version}")
        else:
            deps_lines.append(f'{dep_name} = "{dep_version}"')

    deps_block = "\n".join(deps_lines)
    cargo_toml = f"""[package]
name = "stan-compiled-model"
version = "0.1.0"
edition = "2021"

[dependencies]
{deps_block}

[[bin]]
name = "validate"
path = "src/validate.rs"

[[bin]]
name = "bench"
path = "src/bench.rs"
"""
    (build_path / "Cargo.toml").write_text(cargo_toml)

    # Write data module
    data_dict = data if isinstance(data, dict) else None
    data_rs = _generate_data_rs(data_dict)
    (src_dir / "data.rs").write_text(data_rs)

    # lib.rs
    lib_rs = "pub mod data;\npub mod generated;\npub use generated::*;\n"
    (src_dir / "lib.rs").write_text(lib_rs)

    # Placeholder generated.rs
    (src_dir / "generated.rs").write_text(
        "// Placeholder — will be overwritten by the agent\n"
    )

    # Validation binary
    validate_rs = """
use stan_compiled_model::generated::GeneratedLogp;
use nuts_rs::CpuLogpFunc;
use std::io::{self, BufRead, Write};

fn main() {
    let mut logp_fn = GeneratedLogp::default();
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines() {
        let line = line.unwrap();
        if line.is_empty() { continue; }

        let position: Vec<f64> = line.split(',')
            .map(|s| s.trim().parse().unwrap())
            .collect();

        let n = logp_fn.dim();
        let mut gradient = vec![0.0f64; n];
        match logp_fn.logp(&position, &mut gradient) {
            Ok(logp) => {
                write!(out, "{:.17e}", logp).unwrap();
                for g in &gradient {
                    write!(out, ",{:.17e}", g).unwrap();
                }
                writeln!(out).unwrap();
            }
            Err(e) => {
                writeln!(out, "ERROR:{}", e).unwrap();
            }
        }
    }
}
"""
    (src_dir / "validate.rs").write_text(validate_rs)

    # Benchmark binary
    bench_rs = """
use stan_compiled_model::generated::GeneratedLogp;
use nuts_rs::CpuLogpFunc;
use std::io::{self, BufRead};
use std::time::Instant;

fn main() {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    let n_iters: usize = lines.next().unwrap().unwrap().trim().parse().unwrap();
    let param_line = lines.next().unwrap().unwrap();
    let position: Vec<f64> = param_line.split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();

    let mut logp_fn = GeneratedLogp::default();
    let n = logp_fn.dim();
    let mut gradient = vec![0.0f64; n];
    let mut logp_val = 0.0f64;

    // Warmup
    for _ in 0..100 {
        logp_val = logp_fn.logp(&position, &mut gradient).unwrap();
    }

    // Timed loop
    let start = Instant::now();
    for _ in 0..n_iters {
        logp_val = logp_fn.logp(&position, &mut gradient).unwrap();
    }
    let elapsed = start.elapsed();

    let nanos = elapsed.as_nanos() as f64;
    let us_per_eval = nanos / (n_iters as f64) / 1000.0;
    print!("{:.6},{:.17e}", us_per_eval, logp_val);
    for g in &gradient {
        print!(",{:.17e}", g);
    }
    println!();
}
"""
    (src_dir / "bench.rs").write_text(bench_rs)
