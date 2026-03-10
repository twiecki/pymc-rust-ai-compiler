"""Compile a PyMC model to Rust via an agentic Claude loop.

Instead of blind retries, the compiler runs Claude as an *agent* with tools:
- write_rust_code: write generated.rs
- cargo_build: compile the Rust project
- validate_logp: check logp+gradient against PyMC reference values
- read_file: inspect any project file (data.rs, generated.rs, etc.)

The agent iterates autonomously — reading compiler errors, analyzing
validation mismatches, inspecting data, and fixing code — until the
model compiles and validates correctly.
"""

from __future__ import annotations

import functools
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pymc as pm

from pymc_rust_compiler.exporter import RustModelExporter

_SKILLS_DIR = Path(__file__).parent / "skills"


SYSTEM_PROMPT = """You are an expert Rust code generator for Bayesian statistical models.

You have access to tools to iteratively write, build, and validate Rust code.
Your workflow:
1. Analyze the model context (parameter layout, logp graph, validation targets)
2. Write the Rust code using `write_rust_code`
3. Build with `cargo_build` — if it fails, read the errors, fix the code, rebuild
4. Validate with `validate_logp` — if it fails, analyze which terms are wrong, fix, rebuild, revalidate
5. Iterate until validation passes

You can also use `read_file` to inspect data.rs (pre-generated data arrays),
the current generated.rs, or any other file in the project.

CRITICAL RULES:
1. Parameters are in UNCONSTRAINED space (log-transforms for positive parameters)
2. Include the Jacobian adjustment for any transforms (e.g., +log_sigma for LogTransform)
3. The gradient must be analytically correct — NUTS relies on this
4. Include ALL terms in the logp — PyMC includes everything, including -0.5*log(2*pi) and -log(sigma).
   Your output MUST match PyMC's logp exactly. Do NOT drop any terms.
5. Use efficient single-pass computation where possible
6. All data (observed + covariates/predictors) is provided in `data.rs` — use `use crate::data::*;` to access it. Do NOT embed data arrays in your generated code.

IMPORTANT - OBSERVED DATA LIKELIHOOD:
The logp MUST include the log-likelihood of ALL observed data points. This is typically the
dominant term. For example, if y ~ Normal(mu, sigma) is observed with N data points, the logp
must include: sum over i of [-0.5*log(2*pi) - log(sigma) - 0.5 * ((y[i] - mu[i]) / sigma)^2] for all N observations.
Forgetting the observed likelihood is the #1 error — the logp from priors alone is much smaller
than the total logp including observations.

INDEX ARRAYS:
When a data array is labeled as an INTEGER INDEX ARRAY, cast its elements to `usize` for array
indexing. For example: `let group = X_1_DATA[i] as usize;` then use `a[group]` to index into
group-level parameter arrays.

EXACT LOG-DENSITY FORMULAS (include ALL terms, PyMC drops nothing):

Normal(x | mu, sigma):
  logp = -0.5*log(2*pi) - log(sigma) - 0.5*((x-mu)/sigma)^2
  d(logp)/d(x) = -(x-mu)/sigma^2
  d(logp)/d(sigma) = -1/sigma + (x-mu)^2/sigma^3
  d(logp)/d(mu) = (x-mu)/sigma^2

HalfNormal(x | sigma) with LogTransform (unconstrained param = log_x):
  x = exp(log_x)
  logp = log(2) - 0.5*log(2*pi) - log(sigma) - 0.5*(x/sigma)^2 + log_x
  d(logp)/d(log_x) = -x^2/sigma^2 + 1

PERFORMANCE OPTIMIZATION RULES (critical for competitive speed):

1. HOIST INVARIANTS OUT OF THE OBSERVATION LOOP:
   Precompute everything that doesn't depend on the observation index BEFORE the loop.
   ```rust
   // GOOD: precompute once
   let inv_sigma_sq = 1.0 / (sigma * sigma);
   let neg_log_sigma = -sigma.ln();  // or equivalently: -log_sigma_y for log-transformed params
   let log_norm = -0.5 * LN_2PI + neg_log_sigma;  // constant per observation
   for i in 0..N {
       let residual = y[i] - mu[i];
       logp += log_norm - 0.5 * residual * residual * inv_sigma_sq;
   }
   // BAD: recomputing inside loop
   for i in 0..N {
       logp += -0.5 * LN_2PI - sigma.ln() - 0.5 * ((y[i] - mu[i]) / sigma).powi(2);
   }
   ```

2. USE MULTIPLY INSTEAD OF DIVIDE:
   Replace `x / sigma` with `x * inv_sigma` — division is 5-20x slower than multiplication.
   Precompute `inv_sigma = 1.0 / sigma` and `inv_sigma_sq = inv_sigma * inv_sigma`.

3. USE SEPARATE GRADIENT ACCUMULATORS FOR AUTO-VECTORIZATION:
   LLVM can auto-vectorize (SIMD) accumulation loops when the accumulator is a simple local
   variable, not an indexed array write. Use local f64 accumulators, then write to gradient[] once.
   ```rust
   let mut grad_alpha = 0.0f64;
   let mut grad_beta = 0.0f64;
   let mut grad_log_sigma = 0.0f64;
   for i in 0..N {
       let r = (y[i] - mu_i) * inv_sigma_sq;
       grad_alpha += r;
       grad_beta += r * x[i];
       grad_log_sigma += r * r * sigma * sigma - 1.0;  // or equivalent
   }
   gradient[0] += grad_alpha;
   gradient[1] += grad_beta;
   gradient[2] += grad_log_sigma;
   ```

4. FOR GROUP-LEVEL GRADIENT ACCUMULATION (hierarchical models):
   Use a local fixed-size array for group gradient accumulators instead of writing to
   gradient[] inside the loop. This enables LLVM to keep them in registers.
   ```rust
   let mut grad_offset = [0.0f64; N_GROUPS];
   for i in 0..N {
       let g = group_idx[i] as usize;
       grad_offset[g] += residual_scaled;
   }
   for g in 0..N_GROUPS {
       gradient[OFFSET + g] += grad_offset[g] * sigma_a;
   }
   ```

5. AVOID .powi(2) in HOT LOOPS — use `x * x` instead. The compiler may not always optimize
   `.powi(2)` to a simple multiply.

6. PRECOMPUTE LOG CONSTANTS: For log-transformed parameters, `sigma.ln()` equals `log_sigma`
   (the unconstrained parameter). Use `log_sigma` directly instead of calling `.ln()`.

7. PRE-ALLOCATE SCRATCH ARRAYS IN THE STRUCT (zero allocation in logp):
   Any arrays needed during logp evaluation should be stored as fields in `GeneratedLogp`
   and reused across calls. NEVER use `vec![]` or `Vec::new()` inside `logp()` — heap
   allocation per call destroys performance. Use stack-allocated fixed arrays `[0.0; N]`
   for small sizes (< 64 elements) or struct fields for larger ones.
   When using struct fields, implement Default:
   ```rust
   pub struct GeneratedLogp {
       scratch: Vec<f64>,  // reused across calls
   }
   impl Default for GeneratedLogp {
       fn default() -> Self { Self { scratch: vec![0.0; N] } }
   }
   // The validate/bench binaries call GeneratedLogp::default()
   // In logp(): just zero self.scratch and reuse it
   ```
   For simple models with no heap-allocated scratch, use `#[derive(Clone, Default)]`.

DEBUGGING TIPS:
- IMPORTANT: Define `const LN_2PI: f64 = 1.8378770664093453;` (this is ln(2π)).
  Do NOT compute it as `LN_2 * PI` — that gives ln(2)*π ≈ 2.177, which is WRONG.
- If logp is way off, check that you included the observed likelihood (it dominates)
- If gradient is wrong for a sigma parameter, check N_UNCONSTRAINED vs N_CONSTRAINED
- Use `read_file` to inspect data.rs to confirm array names and sizes
- Use `validate_logp` which shows per-RV logp decomposition to isolate which term is wrong

RUST CODE STRUCTURE:
```rust
use std::collections::HashMap;
use nuts_rs::{CpuLogpFunc, CpuMathError, LogpError, Storable};
use nuts_storable::HasDims;
use thiserror::Error;
use crate::data::*;  // Data arrays are pre-generated here

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
        // Access data via DATA, GROUP_IDX, etc. from data.rs
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}
```
"""

# Tool definitions for the Anthropic API
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
            "Run the compiled validator against PyMC reference values. "
            "Returns per-point logp comparison and gradient errors. "
            "Also shows per-RV logp decomposition to help isolate which term is wrong."
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
                    "description": "Relative path within the build directory (e.g., 'src/data.rs', 'src/generated.rs')",
                },
            },
            "required": ["path"],
        },
    },
]


@functools.lru_cache(maxsize=1)
def _cuda_available() -> bool:
    """Check if CUDA is available at runtime."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@functools.lru_cache(maxsize=1)
def _mlx_available() -> bool:
    """Check if Apple Silicon with Metal is available at runtime."""
    import platform
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    # Verify Metal support via system_profiler
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=5,
        )
        return "Metal" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _detect_skills(
    model: pm.Model, ctx, use_cuda: bool | None = None, use_mlx: bool | None = None,
) -> list[str]:
    """Detect which skills are needed based on model structure.

    Returns a list of skill names (matching filenames in skills/).
    If use_cuda/use_mlx is None, auto-detect hardware availability.
    GPU priority: CUDA > MLX > CPU (faer).
    """
    skills = []
    has_gp = False

    # GP: detect MvNormal or gp_ variables in the model
    for rv in list(model.free_RVs) + list(model.observed_RVs):
        op_name = type(rv.owner.op).__name__ if rv.owner else ""
        if "MvNormal" in op_name or "GP" in op_name:
            has_gp = True
            break
    # Also check the logp graph text for GP indicators
    if not has_gp and any(
        kw in ctx.logp_graph.lower()
        for kw in ["cholesky", "mvnormal", "gp_"]
    ):
        has_gp = True

    if has_gp:
        cuda = use_cuda if use_cuda is not None else _cuda_available()
        mlx = use_mlx if use_mlx is not None else _mlx_available()
        if cuda:
            skills.append("gp_cuda")
        elif mlx:
            skills.append("gp_mlx")
        else:
            skills.append("gp")

    # ZeroSumNormal: detect ZeroSumTransform
    for p in ctx.params:
        if p.transform == "ZeroSumTransform":
            skills.append("zerosumnormal")
            break

    return skills


@functools.lru_cache(maxsize=None)
def _load_skill(name: str) -> str:
    """Load a skill file by name (cached after first read)."""
    path = _SKILLS_DIR / f"{name}.md"
    if not path.exists():
        return ""
    return path.read_text()


def _build_system_prompt(skills: list[str]) -> str:
    """Build system prompt with optional skill augmentation."""
    prompt = SYSTEM_PROMPT
    for skill_name in skills:
        content = _load_skill(skill_name)
        if content:
            prompt += f"\n\n{'='*60}\n{content}"
    return prompt


# Extra Cargo.toml dependencies needed per skill
_SKILL_CARGO_DEPS: dict[str, dict[str, str]] = {
    "gp": {"faer": "0.24"},
    "gp_cuda": {"cudarc": '{ version = "0.12", features = ["cublas", "cusolver"] }'},
    "gp_mlx": {"mlx-rs": '{ version = "0.25", features = ["metal"] }'},
}


@dataclass
class CompilationResult:
    """Result of compiling a PyMC model to Rust."""

    rust_code: str
    logp_validated: bool
    validation_errors: list[str]
    n_attempts: int
    build_dir: Path | None
    timings: dict[str, float]
    n_tool_calls: int = 0
    conversation_turns: int = 0
    token_usage: dict[str, int] = field(default_factory=lambda: {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
    })
    us_per_eval: float | None = None  # benchmark result if available

    @property
    def success(self) -> bool:
        return self.logp_validated


@dataclass
class _AgentState:
    """Mutable state for the agent loop."""

    build_path: Path
    ctx: object  # ModelContext
    messages: list[dict]
    tool_calls: int = 0
    builds: int = 0
    validations: int = 0
    validated: bool = False
    timings: dict = field(default_factory=dict)


def compile_model(
    model: pm.Model,
    source_code: str | None = None,
    api_key: str | None = None,
    max_turns: int = 30,
    model_name: str = "claude-sonnet-4-20250514",
    build_dir: str | Path | None = None,
    verbose: bool = True,
    use_cuda: bool | None = None,
    use_mlx: bool | None = None,
) -> CompilationResult:
    """Compile a PyMC model to optimized Rust via an agentic Claude loop.

    Instead of blind retries, Claude acts as an agent with tools to
    iteratively write, build, debug, and validate the generated Rust code.

    Args:
        model: A PyMC model instance.
        source_code: Optional PyMC source code string for better context.
        api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var).
        max_turns: Max agent turns (tool calls) before giving up.
        model_name: Claude model to use.
        build_dir: Where to create the Rust project. Temp dir if None.
        verbose: Print progress.

    Returns:
        CompilationResult with the generated Rust code and validation status.
    """
    import anthropic

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("No API key provided. Set ANTHROPIC_API_KEY or pass api_key=")

    client = anthropic.Anthropic(api_key=api_key)
    timings: dict[str, float] = {}

    # Step 1: Extract model context
    if verbose:
        print("Extracting model context...")
    t0 = time.time()
    exporter = RustModelExporter(model, source_code=source_code)
    ctx = exporter.context
    prompt = exporter.to_prompt()
    timings["extract"] = time.time() - t0
    if verbose:
        print(f"  {ctx.n_params} parameters, {len(prompt)} char prompt")

    # Detect model-specific skills and build augmented system prompt
    skills = _detect_skills(model, ctx, use_cuda=use_cuda, use_mlx=use_mlx)
    system_prompt = _build_system_prompt(skills)
    if verbose and skills:
        print(f"  Skills loaded: {', '.join(skills)}")

    # Collect extra Cargo dependencies from skills
    extra_deps: dict[str, str] = {}
    for skill_name in skills:
        extra_deps.update(_SKILL_CARGO_DEPS.get(skill_name, {}))

    # Step 2: Set up build directory and write data
    if build_dir:
        build_path = Path(build_dir)
        build_path.mkdir(parents=True, exist_ok=True)
    else:
        build_path = Path(tempfile.mkdtemp(prefix="pymc_rust_"))

    _setup_rust_project(build_path, ctx, extra_cargo_deps=extra_deps)

    if verbose:
        print(f"  Build dir: {build_path}")

    # Step 3: Run the agent loop
    state = _AgentState(
        build_path=build_path,
        ctx=ctx,
        messages=[{
            "role": "user",
            "content": (
                "Generate a Rust logp+gradient implementation for this PyMC model.\n\n"
                "Use your tools to write the code, build it, and validate it. "
                "Iterate until validation passes.\n\n"
                f"{prompt}"
            ),
        }],
        timings=timings,
    )

    if verbose:
        print("\nStarting agent loop...")

    total_input_tokens = 0
    total_output_tokens = 0
    turn = 0

    for turn in range(max_turns):
        # Call Claude
        t0 = time.time()
        response = client.messages.create(
            model=model_name,
            max_tokens=16384,
            system=system_prompt,
            tools=TOOLS,
            messages=state.messages,
        )
        timings[f"api_turn_{turn}"] = time.time() - t0

        # Track token usage
        if hasattr(response, "usage") and response.usage:
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens
            if verbose:
                print(f"  Turn {turn}: {response.usage.input_tokens} in / {response.usage.output_tokens} out tokens")

        # Check stop reason
        if response.stop_reason == "end_turn":
            # Agent finished (text response, no more tool calls)
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

                # Check if validation passed
                if state.validated:
                    break

        state.messages.append({"role": "user", "content": tool_results})

        if state.validated:
            break

    # Read final generated code
    gen_path = build_path / "src" / "generated.rs"
    rust_code = gen_path.read_text() if gen_path.exists() else ""

    validation_errors = []
    if not state.validated:
        validation_errors.append(
            f"Agent did not achieve validation after {state.tool_calls} tool calls"
        )

    return CompilationResult(
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
    name: str, input_data: dict, state: _AgentState, verbose: bool
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
    else:
        return f"Unknown tool: {name}"


def _tool_write_rust_code(
    input_data: dict, state: _AgentState, verbose: bool
) -> str:
    """Write the generated.rs file."""
    code = input_data.get("code", "")
    if not code:
        return "Error: no code provided"

    gen_path = state.build_path / "src" / "generated.rs"
    gen_path.write_text(code)

    if verbose:
        print(f"  [write_rust_code] Wrote {len(code)} chars to generated.rs")

    return f"Written {len(code)} chars to src/generated.rs"


def _tool_cargo_build(state: _AgentState, verbose: bool) -> str:
    """Build the Rust project."""
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
        # Return compiler errors (truncated to avoid token explosion)
        errors = result.stderr
        if len(errors) > 4000:
            errors = errors[:4000] + "\n... (truncated)"
        return f"Build FAILED:\n{errors}"


def _tool_validate_logp(state: _AgentState, verbose: bool) -> str:
    """Run the compiled validator against PyMC reference values."""
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
        position = []
        for param_name in ctx.param_order:
            val = vp.point[param_name]
            position.extend(np.asarray(val, dtype=np.float64).ravel())
        input_lines.append(",".join(f"{v:.17e}" for v in position))

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

    output_lines = [l for l in result.stdout.strip().split("\n") if l]

    if len(output_lines) != len(all_points):
        return f"Error: expected {len(all_points)} output lines, got {len(output_lines)}"

    # Parse results
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

    # Check for constant offset
    offsets = []
    for (name, vp), res in zip(all_points, parsed):
        if res is not None:
            offsets.append(res[0] - vp.logp)

    mean_offset = 0.0
    has_constant_offset = False
    if offsets:
        mean_offset = sum(offsets) / len(offsets)
        offset_var = max(abs(o - mean_offset) for o in offsets)
        has_constant_offset = offset_var < 1.0

    for (name, vp), res in zip(all_points, parsed):
        if res is None:
            report_lines.append(f"{name}: ERROR from validator")
            errors.append(f"{name}: ERROR")
            continue

        rust_logp, rust_grad = res

        # logp comparison
        if has_constant_offset:
            adjusted_err = abs((rust_logp - mean_offset) - vp.logp) / max(abs(vp.logp), 1.0)
        else:
            adjusted_err = abs(rust_logp - vp.logp) / max(abs(vp.logp), 1.0)

        status = "OK" if adjusted_err <= 1e-4 else "MISMATCH"
        report_lines.append(
            f"{name}: logp PyMC={vp.logp:.10f} Rust={rust_logp:.10f} "
            f"rel_err={adjusted_err:.2e} [{status}]"
        )
        if adjusted_err > 1e-4:
            errors.append(
                f"{name}: logp mismatch: PyMC={vp.logp:.10f}, Rust={rust_logp:.10f}, "
                f"rel_err={adjusted_err:.2e}"
            )

        # Per-RV logp decomposition (only for initial point)
        if name == "initial" and vp.per_rv_logp:
            report_lines.append("  Per-RV logp (PyMC reference):")
            for rv_name, rv_logp in vp.per_rv_logp.items():
                report_lines.append(f"    {rv_name}: {rv_logp:.10f}")

        # Gradient comparison
        grad_errors = 0
        for j, (rust_g, pymc_g) in enumerate(zip(rust_grad, vp.dlogp)):
            grad_err = abs(rust_g - pymc_g) / max(abs(pymc_g), 1.0)
            if grad_err > 1e-3:
                grad_errors += 1
                if grad_errors <= 5:  # Show first 5
                    errors.append(
                        f"{name}: gradient[{j}] mismatch: PyMC={pymc_g:.6e}, "
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
            print(f"  [validate_logp] PASSED!")
        return f"VALIDATION PASSED!\n\n{report}"
    else:
        if verbose:
            print(f"  [validate_logp] FAILED ({len(errors)} errors)")
        return f"VALIDATION FAILED ({len(errors)} errors):\n\n{report}\n\nErrors:\n" + "\n".join(errors)


def _tool_read_file(
    input_data: dict, state: _AgentState, verbose: bool
) -> str:
    """Read a file from the build directory."""
    rel_path = input_data.get("path", "")
    if not rel_path:
        return "Error: no path provided"

    file_path = state.build_path / rel_path
    if not file_path.exists():
        # List available files
        available = []
        # List root-level files and src/ contents (skip target/ which can be huge)
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


def _generate_data_rs(ctx) -> str:
    """Generate a Rust file with all data arrays (observed + covariates)."""
    lines = ["//! Auto-generated data arrays. DO NOT EDIT.\n"]

    # Combine observed and covariate data
    all_data = {}
    all_data.update(ctx.observed_data)
    all_data.update(ctx.covariate_data)

    for name, info in all_data.items():
        values = info.get("values")
        if values is None:
            continue

        if isinstance(values, list) and len(values) > 0:
            if isinstance(values[0], list):
                # Multi-dimensional: flatten
                flat = [v for sublist in values for v in sublist]
            else:
                flat = values

            n = len(flat)
            lines.append(f"pub const {name.upper()}_N: usize = {n};")

            # Format with full f64 precision
            formatted_values = ", ".join(f"{v:.17e}" for v in flat)
            lines.append(
                f"pub const {name.upper()}_DATA: &[f64] = &[{formatted_values}];\n"
            )

    return "\n".join(lines)


def _setup_rust_project(
    build_path: Path, ctx, extra_cargo_deps: dict[str, str] | None = None,
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
            deps_lines.append(f'{dep_name} = {dep_version}')
        else:
            deps_lines.append(f'{dep_name} = "{dep_version}"')

    deps_block = "\n".join(deps_lines)
    cargo_toml = f"""[package]
name = "pymc-compiled-model"
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

    # Write data module (exact values, no LLM rounding)
    data_rs = _generate_data_rs(ctx)
    (src_dir / "data.rs").write_text(data_rs)

    # Save observed data as .npy for benchmark reproducibility
    all_data = {}
    all_data.update(ctx.observed_data)
    all_data.update(ctx.covariate_data)
    for name, info in all_data.items():
        values = info.get("values")
        if values is not None:
            import numpy as _np

            _np.save(build_path / f"{name}_data.npy", _np.asarray(values))

    # lib.rs to expose modules
    lib_rs = "pub mod data;\npub mod generated;\npub use generated::*;\n"
    (src_dir / "lib.rs").write_text(lib_rs)

    # Placeholder generated.rs so the project structure is valid
    (src_dir / "generated.rs").write_text(
        "// Placeholder — will be overwritten by the agent\n"
    )

    # Validation binary
    validate_rs = """
use pymc_compiled_model::generated::GeneratedLogp;
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

    # Benchmark binary — tight loop logp+dlogp evaluations
    bench_rs = """
use pymc_compiled_model::generated::GeneratedLogp;
use nuts_rs::CpuLogpFunc;
use std::io::{self, BufRead};
use std::time::Instant;

fn main() {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    // First line: number of iterations
    let n_iters: usize = lines.next().unwrap().unwrap().trim().parse().unwrap();

    // Second line: parameter vector
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


# ---------------------------------------------------------------------------
# Optimization loop (Round 2): profile-guided optimization of generated code
# ---------------------------------------------------------------------------

OPTIMIZE_SYSTEM_PROMPT = """You are an expert Rust performance engineer optimizing logp+gradient code.

You are given existing, CORRECT Rust code that computes logp and gradients for a Bayesian model.
Your job is to make it FASTER while keeping the output NUMERICALLY IDENTICAL.

You have the same tools as before: write_rust_code, cargo_build, validate_logp, read_file.
Additionally you have a `bench_logp` tool that measures evaluation speed.

OPTIMIZATION STRATEGIES (in order of typical impact):

1. **SIMD-FRIENDLY LOOPS**: Ensure inner loops are auto-vectorizable:
   - Use local f64 accumulators (not array writes inside the loop)
   - Avoid branches inside hot loops
   - Use `x * x` not `.powi(2)`, avoid `.ln()` / `.exp()` inside loops if possible

2. **LOOP FUSION**: Merge multiple passes over the same data into one.
   If the code loops over observations multiple times (once for logp, once for gradients),
   fuse them into a single pass.

3. **MEMORY ACCESS PATTERNS**: Ensure sequential memory access.
   - Process data arrays in order (no random access if possible)
   - Keep working set in cache (L1 is ~32KB)

4. **PRECOMPUTE EVERYTHING**: Move ALL invariants outside loops.
   - 1/sigma, 1/sigma^2, log constants, etc.
   - If a log-transformed parameter is used, use the unconstrained value directly

5. **UNSAFE HINTS** (when safe): Use `get_unchecked()` for data array access
   in inner loops when you've already verified the bounds. This eliminates
   bounds checks that prevent auto-vectorization.

6. **EXPLICIT SIMD** (last resort): Use `std::arch::x86_64` intrinsics for
   critical inner loops if auto-vectorization isn't happening.

RULES:
- The output MUST be numerically identical — validation must still pass
- Do NOT change the public interface (GeneratedLogp struct, CpuLogpFunc impl)
- Measure before and after with bench_logp
- If an optimization breaks validation, revert it immediately
"""


def _bench_logp_tool(state: _AgentState, verbose: bool) -> str:
    """Run the bench binary and return timing results."""
    build_path = state.build_path.resolve()
    binary = build_path / "target" / "release" / "bench"
    if not binary.exists():
        # Try building it
        build_result = subprocess.run(
            ["cargo", "build", "--release", "--bin", "bench"],
            cwd=build_path,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if build_result.returncode != 0:
            return f"Error: bench build failed: {build_result.stderr[:500]}"

    # Use initial point as test vector
    ctx = state.ctx
    position = []
    for param_name in ctx.param_order:
        val = ctx.initial_point.point[param_name]
        position.extend(np.asarray(val, dtype=np.float64).ravel())

    n_evals = 500_000
    param_str = ",".join(f"{v:.17e}" for v in position)
    stdin_data = f"{n_evals}\n{param_str}\n"

    try:
        result = subprocess.run(
            [str(binary)],
            cwd=build_path,
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "Error: benchmark timed out"

    if result.returncode != 0:
        return f"Error: bench failed: {result.stderr[:500]}"

    parts = result.stdout.strip().split(",")
    us_per_eval = float(parts[0])

    if verbose:
        print(f"  [bench_logp] {us_per_eval:.3f} us/eval ({n_evals:,} evaluations)")

    return f"Benchmark: {us_per_eval:.3f} us/eval ({n_evals:,} evaluations, {1e6/us_per_eval:,.0f} evals/sec)"


def optimize_model(
    compile_result: CompilationResult,
    model: pm.Model,
    api_key: str | None = None,
    max_turns: int = 20,
    model_name: str = "claude-sonnet-4-20250514",
    verbose: bool = True,
    use_cuda: bool | None = None,
    use_mlx: bool | None = None,
) -> CompilationResult:
    """Optimize an already-compiled model's Rust code for speed.

    Takes a successful CompilationResult and runs a second agent pass
    focused on performance optimization. The agent can benchmark,
    rewrite code, and validate that correctness is preserved.

    Args:
        compile_result: A successful CompilationResult from compile_model.
        model: The original PyMC model (for re-extracting context).
        api_key: Anthropic API key.
        max_turns: Max optimization iterations.
        model_name: Claude model to use.
        verbose: Print progress.

    Returns:
        New CompilationResult with optimized code and benchmark timing.
    """
    import anthropic

    if not compile_result.success:
        raise ValueError("Cannot optimize a failed compilation")

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("No API key. Set ANTHROPIC_API_KEY or pass api_key=")

    client = anthropic.Anthropic(api_key=api_key)
    build_path = compile_result.build_dir

    # Re-extract model context for validation
    exporter = RustModelExporter(model)
    ctx = exporter.context

    # Read current generated code
    current_code = (build_path / "src" / "generated.rs").read_text()

    # Tools: same as compile + bench_logp
    optimize_tools = TOOLS + [
        {
            "name": "bench_logp",
            "description": (
                "Benchmark logp+dlogp evaluation speed. Returns microseconds per evaluation. "
                "Run this before and after optimizations to measure improvement."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
    ]

    # Build initial prompt with code and baseline
    state = _AgentState(
        build_path=build_path,
        ctx=ctx,
        messages=[{
            "role": "user",
            "content": (
                "Optimize this Rust logp+gradient implementation for maximum speed.\n\n"
                "The code is CORRECT and passes validation. Your goal is to make it faster "
                "while keeping output numerically identical.\n\n"
                "Steps:\n"
                "1. Run `bench_logp` to get the baseline speed\n"
                "2. Read the current code with `read_file`\n"
                "3. Apply optimizations and write the new code\n"
                "4. Build and validate (correctness must be preserved!)\n"
                "5. Benchmark again to measure improvement\n"
                "6. Iterate if there's more to gain\n\n"
                f"The model has {ctx.n_params} parameters and the code is {len(current_code)} chars.\n"
                f"Build directory: {build_path}\n"
            ),
        }],
    )

    if verbose:
        print(f"\nStarting optimization loop (build_dir={build_path})...")

    # Detect skills for system prompt augmentation
    skills = _detect_skills(model, ctx, use_cuda=use_cuda, use_mlx=use_mlx)
    system_prompt = OPTIMIZE_SYSTEM_PROMPT
    for skill_name in skills:
        content = _load_skill(skill_name)
        if content:
            system_prompt += f"\n\n{'='*60}\n{content}"

    total_input_tokens = 0
    total_output_tokens = 0
    turn = 0
    best_us = None

    for turn in range(max_turns):
        t0 = time.time()
        response = client.messages.create(
            model=model_name,
            max_tokens=16384,
            system=system_prompt,
            tools=optimize_tools,
            messages=state.messages,
        )

        if hasattr(response, "usage") and response.usage:
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens
            if verbose:
                print(f"  Turn {turn}: {response.usage.input_tokens} in / {response.usage.output_tokens} out tokens")

        if response.stop_reason == "end_turn":
            if verbose:
                for block in response.content:
                    if hasattr(block, "text"):
                        print(f"  Agent: {block.text[:300]}")
            break

        if response.stop_reason != "tool_use":
            break

        state.messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "text" and block.text.strip() and verbose:
                print(f"  Agent: {block.text[:300]}")
            elif block.type == "tool_use":
                state.tool_calls += 1
                if block.name == "bench_logp":
                    result = _bench_logp_tool(state, verbose)
                    # Track best timing
                    if "us/eval" in result and "Error" not in result:
                        us = float(result.split()[1])
                        if best_us is None or us < best_us:
                            best_us = us
                else:
                    result = _execute_tool(block.name, block.input, state, verbose)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        state.messages.append({"role": "user", "content": tool_results})

    # Read final code
    final_code = (build_path / "src" / "generated.rs").read_text()

    return CompilationResult(
        rust_code=final_code,
        logp_validated=state.validated or compile_result.logp_validated,
        validation_errors=[],
        n_attempts=state.builds,
        build_dir=build_path,
        timings={},
        n_tool_calls=state.tool_calls,
        conversation_turns=turn + 1,
        token_usage={
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        },
        us_per_eval=best_us,
    )
