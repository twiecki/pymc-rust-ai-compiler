"""Transpile a PyTorch nn.Module to optimized Rust via an agentic Claude loop.

The transpiler:
1. Extracts model parameters, forward pass outputs, and gradients from a PyTorch module
2. Sets up a Rust project with a Cargo template for neural network inference
3. Runs Claude as an agent with tools: write_code, cargo_build, validate_model, read_source
4. The agent iterates until forward pass + gradient outputs match the PyTorch reference

The generated Rust code uses no ML framework — pure Rust with f32 arrays.
This is the path to zero-overhead inference deployment.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from pymc_rust_compiler.jax_exporter import ModelContext, TensorInfo, ValidationPoint


_SKILLS_DIR = Path(__file__).parent / "skills"


# ── Rust project templates ──────────────────────────────────────────────────

CARGO_TOML = """\
[package]
name = "pytorch-to-rust"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "validate"
path = "src/main.rs"
"""

CARGO_TOML_BURN = """\
[package]
name = "pytorch-to-rust"
version = "0.1.0"
edition = "2021"

[dependencies]
burn = { version = "0.20", features = ["ndarray", "autodiff"] }

[[bin]]
name = "validate"
path = "src/main.rs"
"""

SYSTEM_PROMPT_BURN = """\
You are an expert Rust code generator that translates PyTorch neural network models
to Rust using the **Burn deep learning framework**. Burn provides optimized tensor
operations and automatic differentiation.

You have access to tools to iteratively write, build, and validate Rust code.
Your workflow:
1. Analyze the PyTorch model: parameter shapes, forward pass logic, activation functions
2. Write the Rust code using `write_code`
3. Build with `cargo_build` — if it fails, read the errors, fix the code, rebuild
4. Validate with `validate_model` — checks forward pass outputs and gradients against PyTorch reference
5. Iterate until validation passes

You can use `read_source` to re-read the original PyTorch source code, and
`read_file` to inspect any file in the Rust project.

CRITICAL RULES:
1. Generate a complete Rust module (src/generated.rs) with:
   - `pub fn forward(input: &[f32]) -> Vec<f32>` — the forward pass
   - `pub fn forward_with_grad(input: &[f32], param_name: &str) -> (Vec<f32>, Vec<f32>)` —
     forward pass + gradient of sum(output) w.r.t. the named parameter
2. Use Burn's Tensor API for all tensor operations:
   - `use burn::prelude::*;`
   - `use burn::tensor::activation;`
   - `use burn::backend::NdArray;` for forward-only
   - `use burn::backend::Autodiff;` for gradient computation
   - Type alias: `type B = NdArray<f32>;` and `type AB = Autodiff<NdArray<f32>>;`
3. Load weights from data.rs constants into Burn tensors:
   ```rust
   let data = TensorData::new(WEIGHT_CONST.to_vec(), [rows, cols]);
   let w: Tensor<B, 2> = Tensor::from_data(data, &device);
   ```
4. Use Burn operations instead of manual loops:
   - `a.matmul(b)` for matrix multiply
   - `activation::softmax(x, dim)` for softmax
   - `activation::gelu(x)` for GELU
   - `x.reshape([d1, d2, ...])` for reshape
   - `x.swap_dims(a, b)` for transpose
   - `x.narrow(dim, start, len)` to slice along a dimension
   - `x.mask_fill(mask, value)` for masked fill
5. For layer_norm, implement manually:
   ```rust
   fn layer_norm<B: Backend>(input: Tensor<B, 3>, weight: Tensor<B, 1>,
                              bias: Tensor<B, 1>, eps: f64) -> Tensor<B, 3> {
       let mean = input.clone().mean_dim(2);
       let diff = input - mean;
       let var = diff.clone().powf_scalar(2.0).mean_dim(2);
       let inv_std = (var + eps).sqrt().recip();
       let normalized = diff * inv_std;
       normalized * weight.unsqueeze() + bias.unsqueeze()
   }
   ```
6. For `forward_with_grad`, use the Autodiff backend:
   - Load the target parameter with `.require_grad()`
   - Run the forward pass
   - Call `output.sum().backward()` to get gradients
   - Extract gradient with `param.grad(&grads).unwrap()`
7. Use f32 throughout. Device: `let device = Default::default();`
8. Parameter data is provided in `src/data.rs` as `pub const` arrays.
9. Causal attention mask: use `Tensor::ones([t, t]).triu(1)` for upper triangular,
   then `.bool()` and `.mask_fill()` with NEG_INFINITY.

IMPORTANT PyTorch → Burn Operation Mapping:
- `x @ w.T + b` → `x.matmul(w.transpose()).add(b.unsqueeze())`
- `x.view(B,T,C)` → `x.reshape([b, t, c])`
- `x.transpose(1, 2)` → `x.swap_dims(1, 2)`
- `F.softmax(x, dim=-1)` → `activation::softmax(x, last_dim_index)`
- `x.masked_fill(mask == 0, -inf)` → `x.mask_fill(mask.bool_not(), f32::NEG_INFINITY)`
- `torch.tril(ones)` → create with `Tensor::ones().triu(1)` then invert

RUST CODE STRUCTURE:
```rust
use burn::prelude::*;
use burn::tensor::activation;
use burn::backend::NdArray;
use crate::data::*;

type B = NdArray<f32>;

// Helper to load 2D tensor from data.rs
fn load_2d(data: &[f32], rows: usize, cols: usize, device: &<B as Backend>::Device) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(data.to_vec(), [rows, cols]), device)
}

fn load_1d(data: &[f32], len: usize, device: &<B as Backend>::Device) -> Tensor<B, 1> {
    Tensor::from_data(TensorData::new(data.to_vec(), [len]), device)
}

pub fn forward(input: &[f32]) -> Vec<f32> {
    let device = Default::default();
    // Load input as tensor, load weights, run model
    // Return flattened output as Vec<f32>
}

pub fn forward_with_grad(input: &[f32], param_name: &str) -> (Vec<f32>, Vec<f32>) {
    // Same but using Autodiff<NdArray<f32>> backend
    // Return (output, gradient)
}
```
"""

MAIN_RS = """\
mod generated;
mod data;

use std::io::{self, BufRead};

fn main() {
    // Read validation commands from stdin
    // Format: "forward <input_json>" or "gradient <input_json> <param_json>"
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.splitn(2, ' ').collect();
        if parts.len() < 2 {
            println!("ERROR: invalid command");
            continue;
        }

        match parts[0] {
            "forward" => {
                let input: Vec<f32> = parts[1]
                    .split(',')
                    .map(|s| s.trim().parse().unwrap())
                    .collect();
                let output = generated::forward(&input);
                let out_str: Vec<String> = output.iter().map(|v| format!("{:.9e}", v)).collect();
                println!("{}", out_str.join(","));
            }
            "forward_grad" => {
                // Input format: input_values;param_name
                let subparts: Vec<&str> = parts[1].splitn(2, ';').collect();
                let input: Vec<f32> = subparts[0]
                    .split(',')
                    .map(|s| s.trim().parse().unwrap())
                    .collect();
                let param_name = if subparts.len() > 1 { subparts[1] } else { "" };
                let (output, grads) = generated::forward_with_grad(&input, param_name);
                let out_str: Vec<String> = output.iter().map(|v| format!("{:.9e}", v)).collect();
                let grad_str: Vec<String> = grads.iter().map(|v| format!("{:.9e}", v)).collect();
                println!("{}|{}", out_str.join(","), grad_str.join(","));
            }
            _ => {
                println!("ERROR: unknown command {}", parts[0]);
            }
        }
    }
}
"""


# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert Rust code generator that translates PyTorch neural network models
to pure, optimized Rust code — no ML framework dependencies, just raw arrays and math.

You have access to tools to iteratively write, build, and validate Rust code.
Your workflow:
1. Analyze the PyTorch model: parameter shapes, forward pass logic, activation functions
2. Write the Rust code using `write_code`
3. Build with `cargo_build` — if it fails, read the errors, fix the code, rebuild
4. Validate with `validate_model` — checks forward pass outputs and gradients against PyTorch reference
5. Iterate until validation passes

You can use `read_source` to re-read the original PyTorch source code, and
`read_file` to inspect any file in the Rust project.

CRITICAL RULES:
1. Generate a complete Rust module (src/generated.rs) with:
   - `pub fn forward(input: &[f32]) -> Vec<f32>` — the forward pass
   - `pub fn forward_with_grad(input: &[f32], param_name: &str) -> (Vec<f32>, Vec<f32>)` —
     forward pass + gradient of sum(output) w.r.t. the named parameter
   - Parameters should be embedded as `const` arrays (the model is baked in)
2. Start with pure Rust (no external crates) for simple models. But if you struggle with
   a complex operation or keep failing, you are ENCOURAGED to add crates via the
   `add_cargo_dependency` tool. Good candidates: nalgebra (matrix ops), ndarray (N-d arrays),
   or any well-maintained crate that solves your problem. Don't waste iterations on buggy
   hand-rolled implementations when a crate exists.
3. Implement activation functions manually:
   - ReLU: `if x > 0.0 { x } else { 0.0 }`
   - Sigmoid: `1.0 / (1.0 + (-x).exp())`
   - Tanh: `x.tanh()`
   - GELU: `0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh())`
   - SiLU/Swish: `x * (1.0 / (1.0 + (-x).exp()))`
4. Use f32 throughout (matching PyTorch default).
5. For matrix multiply: implement manually with loops. The compiler will auto-vectorize.
6. Preserve the exact computation order to minimize numerical differences.
7. Parameter data is provided in `src/data.rs` as `pub const` arrays.

PERFORMANCE TIPS:
- Use `#[inline]` on small functions
- Use iterators and `.iter().zip()` for vectorizable loops
- Pre-allocate output vectors with `vec![0.0f32; size]`
- For matmul: iterate output rows, then columns, accumulating dot products
- The Rust compiler with `--release` will auto-vectorize clean loop patterns

RUST CODE STRUCTURE:
```rust
use crate::data::*;

#[inline]
fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

/// Forward pass: input shape [batch, in_features] flattened to [batch * in_features]
pub fn forward(input: &[f32]) -> Vec<f32> {
    // Layer 1: linear + activation
    // Layer 2: linear + activation
    // ...
    output
}

/// Forward pass with gradient computation via backpropagation.
/// Returns (output, gradient_of_sum_output_wrt_named_param).
pub fn forward_with_grad(input: &[f32], param_name: &str) -> (Vec<f32>, Vec<f32>) {
    // Forward pass (same as above, but keep intermediates)
    // Backward pass to compute gradient
    // Return gradient for the requested parameter
    (output, grad)
}
```
"""


# ── Tool definitions ─────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "write_code",
        "description": (
            "Write the complete generated.rs Rust file. This overwrites the current file. "
            "Must define `pub fn forward(input: &[f32]) -> Vec<f32>` and "
            "`pub fn forward_with_grad(input: &[f32], param_name: &str) -> (Vec<f32>, Vec<f32>)`."
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
        "name": "validate_model",
        "description": (
            "Validate the compiled Rust model against PyTorch reference values. "
            "Checks forward pass outputs and gradients at multiple test points. "
            "Returns per-point comparison with max absolute and relative errors."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "read_source",
        "description": (
            "Re-read the original PyTorch source code of the model being transpiled."
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
            "(parameter data), generated.rs, or Cargo.toml."
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
    {
        "name": "add_cargo_dependency",
        "description": (
            "Add a crate dependency to Cargo.toml. Use this when you want to leverage "
            "an external Rust library instead of implementing something from scratch. "
            "The dependency is appended to the [dependencies] section. "
            "Examples: name='nalgebra' version='0.33' or name='ndarray' version='0.16' "
            "or name='serde' version='1' features='derive'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Crate name (e.g., 'nalgebra', 'ndarray')",
                },
                "version": {
                    "type": "string",
                    "description": "Version requirement (e.g., '0.33', '1', '0.16')",
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


# ── Result type ──────────────────────────────────────────────────────────────

@dataclass
class RustTranspileResult:
    """Result of transpiling PyTorch to Rust."""
    generated_code: str
    validated: bool
    validation_errors: list[str]
    n_attempts: int
    build_dir: Path | None
    n_tool_calls: int = 0
    conversation_turns: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    token_usage: dict[str, int] = field(default_factory=lambda: {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
    })

    @property
    def success(self) -> bool:
        return self.validated

    def save(self, path: str | Path):
        Path(path).write_text(self.generated_code)

    @property
    def binary_path(self) -> Path | None:
        """Path to the compiled binary, if build succeeded."""
        if self.build_dir is None:
            return None
        binary = self.build_dir / "target" / "release" / "validate"
        return binary if binary.exists() else None


# ── Agent state ──────────────────────────────────────────────────────────────

@dataclass
class _AgentState:
    """Mutable state for the agent loop."""
    build_path: Path
    source_context: ModelContext
    source_code: str | None
    messages: list = field(default_factory=list)
    tool_calls: int = 0
    builds: int = 0
    validations: int = 0
    validated: bool = False
    timings: dict = field(default_factory=dict)


# ── Skill loading ────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def _load_skill(name: str) -> str:
    path = _SKILLS_DIR / f"{name}.md"
    if not path.exists():
        return ""
    return path.read_text()


# ── Rust project setup ───────────────────────────────────────────────────────

def _setup_rust_project(build_path: Path, ctx: ModelContext, backend: str = "pure"):
    """Create a Rust project with parameter data baked in."""
    src = build_path / "src"
    src.mkdir(parents=True, exist_ok=True)

    # Cargo.toml
    cargo = CARGO_TOML_BURN if backend == "burn" else CARGO_TOML
    (build_path / "Cargo.toml").write_text(cargo)

    # main.rs
    (src / "main.rs").write_text(MAIN_RS)

    # data.rs — bake in parameter values from the first validation point
    data_lines = [
        "// Auto-generated parameter data from PyTorch model.",
        "// Do NOT edit — generated by pytorch_rust_transpiler.",
        "",
        "#![allow(dead_code)]",
        "",
    ]

    vp0 = ctx.validation_points[0]
    for param_info in ctx.params:
        name = param_info.name
        safe_name = name.replace(".", "_").upper()
        values = np.array(vp0.params[name], dtype=np.float32).ravel()

        data_lines.append(f"/// Parameter: {name}, shape: {param_info.shape}")
        data_lines.append(f"pub const {safe_name}: &[f32] = &[")

        # Write values in chunks of 8
        for i in range(0, len(values), 8):
            chunk = values[i:i+8]
            vals = ", ".join(f"{v:.9e}" for v in chunk)
            data_lines.append(f"    {vals},")

        data_lines.append("];")
        data_lines.append(f"pub const {safe_name}_SHAPE: &[usize] = &{param_info.shape};")
        data_lines.append("")

    (src / "data.rs").write_text("\n".join(data_lines))

    # Placeholder generated.rs
    (src / "generated.rs").write_text(
        "use crate::data::*;\n\n"
        "pub fn forward(input: &[f32]) -> Vec<f32> { vec![] }\n\n"
        "pub fn forward_with_grad(input: &[f32], _param_name: &str) -> (Vec<f32>, Vec<f32>) {\n"
        "    (vec![], vec![])\n"
        "}\n"
    )


# ── Prompt building ──────────────────────────────────────────────────────────

def _build_user_prompt(ctx: ModelContext) -> str:
    parts = []
    parts.append("Translate this PyTorch model to pure Rust.\n")

    if ctx.source_code:
        parts.append(f"## PyTorch Source Code\n```python\n{ctx.source_code}\n```\n")

    # Parameter info
    parts.append("## Parameters\n")
    for p in ctx.params:
        parts.append(f"- `{p.name}`: shape={p.shape}, dtype={p.dtype}, size={p.size}")
    parts.append("")

    # Data.rs mapping
    parts.append("## Parameter Constants in data.rs\n")
    parts.append("Each parameter is available as a flat `&[f32]` constant plus a shape constant:\n")
    for p in ctx.params:
        safe_name = p.name.replace(".", "_").upper()
        parts.append(f"- `{safe_name}`: &[f32] (len={p.size}), `{safe_name}_SHAPE`: &[usize] = {p.shape}")
    parts.append("")

    # Input info
    parts.append("## Inputs\n")
    for i in ctx.inputs:
        parts.append(f"- `{i.name}`: shape={i.shape}, dtype={i.dtype}")
    parts.append("The `forward()` function receives the input as a flat &[f32] array.\n")

    # Output info
    parts.append("## Outputs\n")
    for o in ctx.outputs:
        parts.append(f"- `{o.name}`: shape={o.shape}, dtype={o.dtype}")
    parts.append("")

    # Validation targets
    parts.append("## Validation Targets")
    parts.append(
        "Your Rust forward pass must produce the same outputs as PyTorch. "
        "Gradients of sum(output) w.r.t. each parameter must also match.\n"
    )

    for i, vp in enumerate(ctx.validation_points[:2]):  # Show first 2 for prompt size
        label = "original params" if i == 0 else f"perturbed params {i}"
        parts.append(f"### At {label}:")

        # Show expected output (truncated)
        out_arr = np.asarray(vp.output)
        if out_arr.size <= 20:
            parts.append(f"  expected output = {vp.output}")
        else:
            parts.append(
                f"  expected output: shape={list(out_arr.shape)}, "
                f"mean={out_arr.mean():.6f}, std={out_arr.std():.6f}"
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

    parts.append(
        "Generate the Rust code using `write_code`, build with `cargo_build`, "
        "then call `validate_model` to check correctness."
    )

    return "\n".join(parts)


# ── Tool execution ───────────────────────────────────────────────────────────

def _execute_tool(
    name: str, input_data: dict, state: _AgentState, verbose: bool,
) -> str:
    if name == "write_code":
        return _tool_write_code(input_data, state, verbose)
    elif name == "cargo_build":
        return _tool_cargo_build(state, verbose)
    elif name == "validate_model":
        return _tool_validate(state, verbose)
    elif name == "read_source":
        return _tool_read_source(state, verbose)
    elif name == "read_file":
        return _tool_read_file(input_data, state, verbose)
    elif name == "add_cargo_dependency":
        return _tool_add_cargo_dependency(input_data, state, verbose)
    else:
        return f"Unknown tool: {name}"


def _tool_write_code(
    input_data: dict, state: _AgentState, verbose: bool,
) -> str:
    code = input_data.get("code", "")
    if not code:
        return "Error: no code provided"

    gen_path = state.build_path / "src" / "generated.rs"
    gen_path.write_text(code)

    if verbose:
        print(f"  [write_code] Wrote {len(code)} chars to generated.rs")

    return f"Written {len(code)} chars to src/generated.rs. Use `cargo_build` to compile."


def _tool_cargo_build(state: _AgentState, verbose: bool) -> str:
    state.builds += 1
    t0 = time.time()

    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=state.build_path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed = time.time() - t0
    state.timings[f"build_{state.builds}"] = elapsed

    if result.returncode == 0:
        if verbose:
            print(f"  [cargo_build] OK ({elapsed:.1f}s)")
        return f"Build successful ({elapsed:.1f}s). Use `validate_model` to test."
    else:
        if verbose:
            print(f"  [cargo_build] FAILED ({elapsed:.1f}s)")
        errors = result.stderr
        if len(errors) > 4000:
            errors = errors[:4000] + "\n... (truncated)"
        return f"Build FAILED:\n{errors}"


def _tool_validate(state: _AgentState, verbose: bool) -> str:
    """Validate the compiled Rust model against PyTorch reference values."""
    state.validations += 1

    binary = state.build_path / "target" / "release" / "validate"
    if not binary.exists():
        return "Error: validation binary not found. Run cargo_build first."

    ctx = state.source_context
    report_lines = []
    errors = []

    for i, vp in enumerate(ctx.validation_points):
        label = "original params" if i == 0 else f"perturbed {i}"

        # For perturbed points, we need to update data.rs and rebuild
        # Instead, only validate the first point (original params baked into data.rs)
        if i > 0:
            # Update data.rs for this validation point
            _update_data_rs(state.build_path, ctx, vp)
            # Rebuild
            build_result = subprocess.run(
                ["cargo", "build", "--release"],
                cwd=state.build_path,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if build_result.returncode != 0:
                errors.append(f"{label}: rebuild failed for perturbed params")
                report_lines.append(f"{label}: REBUILD FAILED")
                continue

        # Prepare flat input
        inp = vp.inputs
        if "x" in inp:
            flat_input = np.array(inp["x"], dtype=np.float32).ravel()
        else:
            flat_input = np.concatenate(
                [np.array(v, dtype=np.float32).ravel() for v in inp.values()]
            )

        input_str = ",".join(f"{v:.9e}" for v in flat_input)

        # Test forward pass
        try:
            result = subprocess.run(
                [str(binary)],
                input=f"forward {input_str}\n",
                capture_output=True,
                text=True,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            errors.append(f"{label}: forward pass timed out")
            report_lines.append(f"{label}: TIMEOUT")
            continue

        if result.returncode != 0 or result.stdout.strip().startswith("ERROR"):
            errors.append(f"{label}: forward error: {result.stderr[:200]}")
            report_lines.append(f"{label}: FORWARD ERROR")
            continue

        # Parse output
        try:
            rust_output = np.array(
                [float(v) for v in result.stdout.strip().split(",")],
                dtype=np.float32,
            )
        except (ValueError, IndexError) as e:
            errors.append(f"{label}: parse error: {e}, stdout={result.stdout[:200]}")
            report_lines.append(f"{label}: PARSE ERROR")
            continue

        ref_output = np.array(vp.output, dtype=np.float32).ravel()

        if rust_output.shape != ref_output.shape:
            errors.append(
                f"{label}: output shape mismatch: got {rust_output.shape}, expected {ref_output.shape}"
            )
            report_lines.append(f"{label}: SHAPE MISMATCH {rust_output.shape} vs {ref_output.shape}")
            continue

        max_diff = float(np.max(np.abs(rust_output - ref_output)))
        rel_err = float(np.max(np.abs(rust_output - ref_output) / np.maximum(np.abs(ref_output), 1e-8)))
        out_ok = rel_err <= 1e-4

        report_lines.append(
            f"{label}: output max_diff={max_diff:.2e} rel_err={rel_err:.2e} "
            f"[{'OK' if out_ok else 'MISMATCH'}]"
        )
        if not out_ok:
            errors.append(f"{label}: output mismatch: max_diff={max_diff:.2e}, rel_err={rel_err:.2e}")

        # Test gradients for each parameter
        for pname, ref_grad in vp.grad_params.items():
            try:
                result = subprocess.run(
                    [str(binary)],
                    input=f"forward_grad {input_str};{pname}\n",
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except subprocess.TimeoutExpired:
                errors.append(f"{label}: gradient for '{pname}' timed out")
                continue

            if result.returncode != 0 or "ERROR" in result.stdout:
                errors.append(f"{label}: gradient error for '{pname}'")
                continue

            parts = result.stdout.strip().split("|")
            if len(parts) != 2:
                errors.append(f"{label}: gradient parse error for '{pname}'")
                continue

            try:
                rust_grad = np.array(
                    [float(v) for v in parts[1].split(",")],
                    dtype=np.float32,
                )
            except (ValueError, IndexError):
                errors.append(f"{label}: gradient parse error for '{pname}'")
                continue

            ref_g = np.array(ref_grad, dtype=np.float32).ravel()

            if rust_grad.shape != ref_g.shape:
                errors.append(
                    f"{label}: grad['{pname}'] shape mismatch: {rust_grad.shape} vs {ref_g.shape}"
                )
                continue

            grad_diff = float(np.max(np.abs(rust_grad - ref_g)))
            grad_rel = float(np.max(np.abs(rust_grad - ref_g) / np.maximum(np.abs(ref_g), 1e-8)))
            grad_ok = grad_rel <= 1e-3

            report_lines.append(
                f"  grad['{pname}']: max_diff={grad_diff:.2e} rel_err={grad_rel:.2e} "
                f"[{'OK' if grad_ok else 'MISMATCH'}]"
            )
            if not grad_ok:
                errors.append(f"{label}: grad['{pname}'] mismatch: rel_err={grad_rel:.2e}")

    # Restore original data.rs if we modified it
    if len(ctx.validation_points) > 1:
        _update_data_rs(state.build_path, ctx, ctx.validation_points[0])

    report = "\n".join(report_lines)

    if not errors:
        state.validated = True
        if verbose:
            print("  [validate] PASSED!")
        return f"VALIDATION PASSED!\n\n{report}"
    else:
        if verbose:
            print(f"  [validate] FAILED ({len(errors)} errors)")
        return (
            f"VALIDATION FAILED ({len(errors)} errors):\n\n{report}\n\n"
            f"Errors:\n" + "\n".join(f"- {e}" for e in errors)
        )


def _update_data_rs(build_path: Path, ctx: ModelContext, vp: ValidationPoint):
    """Update data.rs with parameter values from a validation point."""
    data_lines = [
        "// Auto-generated parameter data from PyTorch model.",
        "// Do NOT edit — generated by pytorch_rust_transpiler.",
        "",
        "#![allow(dead_code)]",
        "",
    ]

    for param_info in ctx.params:
        name = param_info.name
        safe_name = name.replace(".", "_").upper()
        values = np.array(vp.params[name], dtype=np.float32).ravel()

        data_lines.append(f"/// Parameter: {name}, shape: {param_info.shape}")
        data_lines.append(f"pub const {safe_name}: &[f32] = &[")

        for i in range(0, len(values), 8):
            chunk = values[i:i+8]
            vals = ", ".join(f"{v:.9e}" for v in chunk)
            data_lines.append(f"    {vals},")

        data_lines.append("];")
        data_lines.append(f"pub const {safe_name}_SHAPE: &[usize] = &{param_info.shape};")
        data_lines.append("")

    (build_path / "src" / "data.rs").write_text("\n".join(data_lines))


def _tool_read_source(state: _AgentState, verbose: bool) -> str:
    source = state.source_code or state.source_context.source_code or "(no source code available)"
    if verbose:
        print(f"  [read_source] {len(source)} chars")
    return source


def _tool_read_file(
    input_data: dict, state: _AgentState, verbose: bool,
) -> str:
    rel_path = input_data.get("path", "")
    if not rel_path:
        return "Error: no path provided"

    file_path = state.build_path / rel_path
    if not file_path.exists():
        available = []
        for f in state.build_path.rglob("*"):
            if f.is_file():
                available.append(str(f.relative_to(state.build_path)))
        return f"File not found: {rel_path}\nAvailable: {', '.join(sorted(available))}"

    content = file_path.read_text()
    if len(content) > 8000:
        content = content[:8000] + "\n... (truncated)"

    if verbose:
        print(f"  [read_file] {rel_path}: {len(content)} chars")
    return content


def _tool_add_cargo_dependency(
    input_data: dict, state: _AgentState, verbose: bool,
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


# ── Agent loop ───────────────────────────────────────────────────────────────

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
            max_tokens=16384,
            system=system_prompt,
            tools=TOOLS,
            messages=state.messages,
        )
        state.timings[f"api_turn_{turn}"] = time.time() - t0

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

    return turn + 1, {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
    }


# ── Main entry point ─────────────────────────────────────────────────────────

def transpile_pytorch_to_rust(
    module: Any,  # torch.nn.Module
    sample_input: Any,
    api_key: str | None = None,
    max_turns: int = 30,
    model_name: str = "claude-sonnet-4-20250514",
    build_dir: str | Path | None = None,
    verbose: bool = True,
    loss_fn: Callable | None = None,
    source_code: str | None = None,
    backend: str = "pure",
) -> RustTranspileResult:
    """Transpile a PyTorch nn.Module to optimized Rust via an agentic Claude loop.

    Args:
        module: PyTorch nn.Module to transpile.
        sample_input: Example input tensor for the model.
        api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var).
        max_turns: Max agent turns before giving up.
        model_name: Claude model to use.
        build_dir: Where to create the Rust project. Temp dir if None.
        verbose: Print progress.
        loss_fn: Optional scalar loss function for gradient computation.
        source_code: Optional source code override.
        backend: "pure" for zero-dependency Rust, "burn" for Burn framework
                 (optimized matmul + autodiff).

    Returns:
        RustTranspileResult with generated Rust code and validation status.
    """
    if backend not in ("pure", "burn"):
        raise ValueError(f"backend must be 'pure' or 'burn', got {backend!r}")

    from pymc_rust_compiler.pytorch_exporter import PytorchModelExporter

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("No API key. Set ANTHROPIC_API_KEY or pass api_key=")

    timings: dict[str, float] = {}

    # Step 1: Extract model context from PyTorch
    if verbose:
        print(f"Extracting PyTorch model context (backend={backend})...")
    t0 = time.time()
    exporter = PytorchModelExporter(
        module, sample_input,
        source_code=source_code, loss_fn=loss_fn,
    )
    ctx = exporter.context
    timings["extract"] = time.time() - t0

    if verbose:
        n_params = sum(p.size for p in ctx.params)
        print(f"  {n_params} parameters, {len(ctx.validation_points)} validation points")

    # Step 2: Set up Rust build directory
    if build_dir:
        build_path = Path(build_dir)
        build_path.mkdir(parents=True, exist_ok=True)
    else:
        build_path = Path(tempfile.mkdtemp(prefix="pytorch_rust_"))

    _setup_rust_project(build_path, ctx, backend=backend)

    if verbose:
        print(f"  Build dir: {build_path}")

    # Step 3: Build system prompt with skill
    if backend == "burn":
        system_prompt = SYSTEM_PROMPT_BURN
        skill = _load_skill("pytorch_to_rust_burn")
    else:
        system_prompt = SYSTEM_PROMPT
        skill = _load_skill("pytorch_to_rust")
    if skill:
        system_prompt += f"\n\n{'='*60}\n{skill}"

    # Step 4: Build user prompt and run agent
    user_prompt = _build_user_prompt(ctx)

    state = _AgentState(
        build_path=build_path,
        source_context=ctx,
        source_code=source_code,
        messages=[{"role": "user", "content": user_prompt}],
        timings=timings,
    )

    if verbose:
        print("\nStarting agent loop...")

    turns, token_usage = _run_agent_loop(
        state, system_prompt, api_key, max_turns, model_name, verbose,
    )

    # Read final generated code
    gen_path = build_path / "src" / "generated.rs"
    rust_code = gen_path.read_text() if gen_path.exists() else ""

    validation_errors = []
    if not state.validated:
        validation_errors.append(
            f"Agent did not achieve validation after {state.tool_calls} tool calls"
        )

    return RustTranspileResult(
        generated_code=rust_code,
        validated=state.validated,
        validation_errors=validation_errors,
        n_attempts=state.builds,
        build_dir=build_path,
        n_tool_calls=state.tool_calls,
        conversation_turns=turns,
        timings=timings,
        token_usage=token_usage,
    )
