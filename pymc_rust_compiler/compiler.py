"""Compile a PyMC model to Rust via Claude API.

The core compilation loop:
1. Extract model context (params, logp graph, validation points)
2. Write observed data as exact Rust const arrays (no LLM rounding!)
3. Send to Claude API to generate Rust logp+gradient code
4. Build Rust project with generated code
5. Validate against PyMC reference values
6. If validation fails, feed errors back to Claude and retry
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymc as pm

from pymc_rust_compiler.exporter import RustModelExporter


SYSTEM_PROMPT = """You are a Rust code generator for Bayesian statistical models.

Given a PyMC model definition and its PyTensor computational graph, generate
a Rust implementation of the log-probability density function (logp) and its
gradient with respect to all unconstrained parameters.

CRITICAL RULES:
1. Parameters are in UNCONSTRAINED space (log-transforms for positive parameters)
2. Include the Jacobian adjustment for any transforms (e.g., +log_sigma for LogTransform)
3. The gradient must be analytically correct - NUTS relies on this
4. Do NOT include constant terms that don't depend on parameters (e.g. -0.5*log(2*pi) per observation)
5. Use efficient single-pass computation where possible
6. All data (observed + covariates/predictors) is provided in `data.rs` — use `use crate::data::*;` to access it. Do NOT embed data arrays in your generated code.

IMPORTANT: Constant terms like -n/2*log(2*pi) and -log(scale) for priors are DROPPED by PyMC
when computing the logp. Do NOT include them. Only include terms that depend on the parameters.

For LogTransform (HalfNormal, etc.):
- The unconstrained parameter is log(x)
- The constrained value is x = exp(log_x)
- The Jacobian adjustment adds +log_x to the logp
- The prior logp for HalfNormal(scale) in unconstrained space (dropping constants):
  logp = -x^2 / (2 * scale^2) + log_x
  d(logp)/d(log_x) = -x^2 / scale^2 + 1

You MUST output a COMPLETE, compilable Rust file (generated.rs). Use this structure:

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

#[derive(Clone)]
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


@dataclass
class CompilationResult:
    """Result of compiling a PyMC model to Rust."""

    rust_code: str
    logp_validated: bool
    validation_errors: list[str]
    n_attempts: int
    build_dir: Path | None
    timings: dict[str, float]

    @property
    def success(self) -> bool:
        return self.logp_validated


def compile_model(
    model: pm.Model,
    source_code: str | None = None,
    api_key: str | None = None,
    max_attempts: int = 3,
    model_name: str = "claude-sonnet-4-20250514",
    build_dir: str | Path | None = None,
    verbose: bool = True,
) -> CompilationResult:
    """Compile a PyMC model to optimized Rust via Claude API.

    Args:
        model: A PyMC model instance.
        source_code: Optional PyMC source code string for better context.
        api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var).
        max_attempts: Max retry attempts if validation fails.
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
    timings = {}

    # Step 1: Extract model context
    if verbose:
        print("Step 1: Extracting model context...")
    t0 = time.time()
    exporter = RustModelExporter(model, source_code=source_code)
    ctx = exporter.context
    prompt = exporter.to_prompt()
    timings["extract"] = time.time() - t0
    if verbose:
        print(f"  {ctx.n_params} parameters, {len(prompt)} char prompt")

    # Step 2: Set up build directory and write data
    if build_dir:
        build_path = Path(build_dir)
        build_path.mkdir(parents=True, exist_ok=True)
    else:
        build_path = Path(tempfile.mkdtemp(prefix="pymc_rust_"))

    _setup_rust_project(build_path, ctx)

    # Step 3: Generation + validation loop
    rust_code = ""
    validation_errors: list[str] = []
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(1, max_attempts + 1):
        if verbose:
            print(f"\nStep 2: Generating Rust code (attempt {attempt}/{max_attempts})...")

        t0 = time.time()
        response = client.messages.create(
            model=model_name,
            max_tokens=8192,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        rust_code = _extract_rust_code(response.content[0].text)
        timings[f"generate_{attempt}"] = time.time() - t0

        if verbose:
            print(f"  Generated {len(rust_code)} chars of Rust code")

        # Write generated code
        (build_path / "src" / "generated.rs").write_text(rust_code)

        # Step 4: Build
        if verbose:
            print("Step 3: Building Rust project...")
        t0 = time.time()
        build_ok, build_output = _cargo_build(build_path)
        timings[f"build_{attempt}"] = time.time() - t0

        if not build_ok:
            error_msg = f"Build failed:\n{build_output}"
            validation_errors.append(error_msg)
            if verbose:
                print(f"  BUILD FAILED")
                print(f"  {build_output[:500]}")

            messages.append({"role": "assistant", "content": response.content[0].text})
            messages.append({
                "role": "user",
                "content": f"The code failed to compile. Fix the errors:\n\n```\n{build_output}\n```\n\nGenerate the COMPLETE corrected Rust file.",
            })
            continue

        if verbose:
            print("  Build OK")

        # Step 5: Validate
        if verbose:
            print("Step 4: Validating against PyMC reference values...")
        t0 = time.time()
        valid, errors = _validate_logp(build_path, ctx)
        timings[f"validate_{attempt}"] = time.time() - t0

        if valid:
            if verbose:
                print("  VALIDATION PASSED!")
            return CompilationResult(
                rust_code=rust_code,
                logp_validated=True,
                validation_errors=[],
                n_attempts=attempt,
                build_dir=build_path,
                timings=timings,
            )
        else:
            validation_errors.extend(errors)
            if verbose:
                print(f"  VALIDATION FAILED:")
                for e in errors[:3]:
                    print(f"    {e}")

            messages.append({"role": "assistant", "content": response.content[0].text})
            messages.append({
                "role": "user",
                "content": (
                    f"The code compiled but produced incorrect values:\n\n"
                    + "\n".join(errors)
                    + "\n\nFix the logp and gradient computation. "
                    "Generate the COMPLETE corrected Rust file."
                ),
            })

    return CompilationResult(
        rust_code=rust_code,
        logp_validated=False,
        validation_errors=validation_errors,
        n_attempts=max_attempts,
        build_dir=build_path,
        timings=timings,
    )


def _extract_rust_code(response_text: str) -> str:
    """Extract Rust code from Claude's response."""
    pattern = r"```rust\s*\n(.*?)```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        return max(matches, key=len)
    if "fn logp" in response_text and "impl" in response_text:
        return response_text
    return response_text


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


def _setup_rust_project(build_path: Path, ctx):
    """Create the Rust project structure with pre-generated data."""
    src_dir = build_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    cargo_toml = """[package]
name = "pymc-compiled-model"
version = "0.1.0"
edition = "2021"

[dependencies]
nuts-rs = "0.17"
nuts-storable = "0.2"
rand = "0.10"
thiserror = "2"
anyhow = "1"

[[bin]]
name = "validate"
path = "src/validate.rs"
"""
    (build_path / "Cargo.toml").write_text(cargo_toml)

    # Write data module (exact values, no LLM rounding)
    data_rs = _generate_data_rs(ctx)
    (src_dir / "data.rs").write_text(data_rs)

    # lib.rs to expose modules
    lib_rs = "pub mod data;\npub mod generated;\n"
    (src_dir / "lib.rs").write_text(lib_rs)

    # Validation binary
    validate_rs = """
use pymc_compiled_model::generated::GeneratedLogp;
use nuts_rs::CpuLogpFunc;
use std::io::{self, BufRead, Write};

fn main() {
    let mut logp_fn = GeneratedLogp;
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


def _cargo_build(build_path: Path) -> tuple[bool, str]:
    """Build the Rust project."""
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=build_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.returncode == 0, result.stderr


def _validate_logp(build_path: Path, ctx) -> tuple[bool, list[str]]:
    """Run the compiled validator against PyMC reference values."""
    binary = build_path / "target" / "release" / "validate"
    if not binary.exists():
        return False, ["Validation binary not found"]

    all_points = [("initial", ctx.initial_point)] + [
        (f"extra_{i}", p) for i, p in enumerate(ctx.extra_points)
    ]

    input_lines = []
    for name, vp in all_points:
        position = []
        for param_name in ctx.param_order:
            val = vp.point[param_name]
            if isinstance(val, list):
                position.extend(val)
            else:
                position.append(val)
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
        return False, ["Validation binary timed out"]

    if result.returncode != 0:
        return False, [f"Validator crashed: {result.stderr[:500]}"]

    errors = []
    output_lines = [l for l in result.stdout.strip().split("\n") if l]

    if len(output_lines) != len(all_points):
        return False, [f"Expected {len(all_points)} output lines, got {len(output_lines)}"]

    # Parse all results
    parsed = []
    for output_line in output_lines:
        if output_line.startswith("ERROR:"):
            parsed.append(None)
        else:
            values = [float(v) for v in output_line.split(",")]
            parsed.append((values[0], values[1:]))

    # Check for constant logp offset (NUTS doesn't care about constant terms)
    # If logp differs by the same constant across all points, that's fine
    offsets = []
    for i, ((name, vp), result) in enumerate(zip(all_points, parsed)):
        if result is None:
            errors.append(f"{name}: ERROR")
            continue
        rust_logp, _ = result
        offsets.append(rust_logp - vp.logp)

    if offsets and not any(r is None for r in parsed):
        mean_offset = sum(offsets) / len(offsets)
        offset_var = max(abs(o - mean_offset) for o in offsets)
        has_constant_offset = offset_var < 1.0  # constant offsets are consistent

        for i, ((name, vp), result) in enumerate(zip(all_points, parsed)):
            if result is None:
                continue
            rust_logp, rust_grad = result

            # Check logp (accounting for possible constant offset)
            if has_constant_offset:
                adjusted_err = abs((rust_logp - mean_offset) - vp.logp) / max(abs(vp.logp), 1.0)
            else:
                adjusted_err = abs(rust_logp - vp.logp) / max(abs(vp.logp), 1.0)

            if adjusted_err > 1e-4:
                errors.append(
                    f"{name}: logp mismatch: PyMC={vp.logp:.10f}, Rust={rust_logp:.10f}, "
                    f"rel_err={adjusted_err:.2e}"
                )

            # Gradients must be exact (they drive NUTS)
            for j, (rust_g, pymc_g) in enumerate(zip(rust_grad, vp.dlogp)):
                grad_err = abs(rust_g - pymc_g) / max(abs(pymc_g), 1.0)
                if grad_err > 1e-3:
                    errors.append(
                        f"{name}: gradient[{j}] mismatch: PyMC={pymc_g:.6e}, "
                        f"Rust={rust_g:.6e}, rel_err={grad_err:.2e}"
                    )

    return len(errors) == 0, errors
