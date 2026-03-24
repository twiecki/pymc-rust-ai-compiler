# Skill: Enzyme Automatic Differentiation for PyMC Models

You have the option to use **Enzyme** (Rust's built-in autodiff via `#[autodiff]`) instead of writing gradients by hand. This lets you write **only the forward logp computation** and have the compiler generate correct gradients automatically.

## When to Use Enzyme

**Prefer Enzyme when:**
- The model has complex gradient derivations (many chain rules, transforms)
- You are struggling with gradient correctness after multiple attempts
- The model has non-standard distributions or custom likelihoods
- Getting hand-written gradients correct is taking too many iterations

**Prefer hand-written gradients when:**
- The model is simple (Normal, LinReg) with well-known gradient formulas
- You are confident in the gradient derivation
- Maximum performance is critical (hand-tuned SIMD-friendly accumulators)
- The model uses external crates like `faer` (Enzyme may not differentiate through them)

## How Enzyme Works

Enzyme is an LLVM-based autodiff plugin integrated into nightly Rust. You annotate a function with `#[autodiff]` and the compiler generates the derivative function.

```rust
#![feature(autodiff)]
use std::autodiff::autodiff;

// The #[autodiff] attribute generates a new function `d_compute_logp` that
// computes both the value AND gradient via reverse-mode AD.
#[autodiff(d_compute_logp, Reverse, Duplicated, Active)]
fn compute_logp(position: &[f64; N_PARAMS]) -> f64 {
    // ... only the forward logp computation ...
}
```

The generated `d_compute_logp` has signature:
```rust
fn d_compute_logp(
    position: &[f64; N_PARAMS],      // primal input
    d_position: &mut [f64; N_PARAMS], // gradient output (adjoint)
    d_return: f64,                     // seed (always 1.0 for us)
) -> f64;                              // returns logp value
```

## Integration with CpuLogpFunc

The `CpuLogpFunc::logp` method receives `&[f64]` slices, but Enzyme needs fixed-size arrays `&[f64; N]`. Bridge them like this:

```rust
#![feature(autodiff)]
use std::autodiff::autodiff;
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

pub const N_PARAMS: usize = /* number of unconstrained parameters */;

const LN_2PI: f64 = 1.8378770664093453;

// ─── Forward-only logp (Enzyme will differentiate this) ─────────────

#[autodiff(d_compute_logp, Reverse, Duplicated, Active)]
fn compute_logp(position: &[f64; N_PARAMS]) -> f64 {
    let mu = position[0];
    let log_sigma = position[1];
    let sigma = log_sigma.exp();

    let mut logp = 0.0;

    // Prior: mu ~ Normal(0, 10)
    logp += -0.5 * LN_2PI - (10.0_f64).ln() - 0.5 * (mu / 10.0) * (mu / 10.0);

    // Prior: sigma ~ HalfNormal(5) with LogTransform + Jacobian
    let sigma_scaled = sigma / 5.0;
    logp += (2.0_f64).ln() - 0.5 * LN_2PI - (5.0_f64).ln()
          - 0.5 * sigma_scaled * sigma_scaled + log_sigma;

    // Likelihood: y ~ Normal(mu, sigma)
    let inv_sigma = 1.0 / sigma;
    let log_norm = -0.5 * LN_2PI - log_sigma;
    for i in 0..Y_N {
        let residual = Y_DATA[i] - mu;
        logp += log_norm - 0.5 * residual * residual * inv_sigma * inv_sigma;
    }

    logp
}

// ─── CpuLogpFunc impl (bridges slices ↔ fixed arrays) ──────────────

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
        // Convert slice → fixed-size array for Enzyme
        let pos_array: &[f64; N_PARAMS] = position.try_into()
            .map_err(|_| SampleError::Recoverable("wrong position length".into()))?;

        let mut grad_array = [0.0f64; N_PARAMS];

        // Call Enzyme-generated reverse-mode AD
        let logp = d_compute_logp(pos_array, &mut grad_array, 1.0);

        // Copy gradient back to output slice
        gradient[..N_PARAMS].copy_from_slice(&grad_array);

        Ok(logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}
```

## Critical Rules for Enzyme Compatibility

1. **Fixed-size arrays only**: Enzyme cannot differentiate through `Vec<T>` or heap allocations. Use `[f64; N]` with compile-time constants.

2. **No external crate calls in the differentiated function**: Enzyme differentiates at the LLVM IR level. Calls to external crates (like `faer` for Cholesky) may not work. Keep the `compute_logp` function self-contained with only `std` math.

3. **Use `x * x` not `.powi(2)`**: Enzyme handles multiplication better than `.powi()`.

4. **Use explicit loops, not iterators**: Enzyme works best with simple `for` loops and array indexing. Avoid `.iter().map().sum()` chains — use manual accumulation.

5. **Constants are fine**: `const` arrays from `data.rs` (like `Y_DATA`, `X_0_DATA`) work because they're known at compile time.

6. **No early returns or `Result` in the differentiated function**: `compute_logp` must return `f64` directly, no `Result`. Handle errors in the `CpuLogpFunc::logp` wrapper instead.

7. **`#![feature(autodiff)]` at crate root**: The feature flag must be in `lib.rs`, not just `generated.rs`.

## Nightly Rust Requirement

Enzyme requires nightly Rust. The build system will automatically use nightly when it detects `#![feature(autodiff)]` in your code. You do not need to worry about toolchain setup.

## Performance Notes

Enzyme generates LLVM-level derivatives that benefit from the same optimizations as the forward pass (inlining, SIMD, etc.). For most models, Enzyme gradients are within 10-20% of hand-written gradients. The main overhead is:
- Enzyme may not hoist invariants as aggressively as a human would
- It generates generic adjoint code rather than exploiting mathematical simplifications (e.g., Normal gradient has a known closed form that's simpler than generic reverse-mode AD)

For NUTS sampling where `logp` is called millions of times, this difference can matter. But correctness is more important than a 10% speedup — wrong gradients cause NUTS to diverge.

## Debugging

If gradients don't match PyMC reference values:
1. Double-check the forward `compute_logp` is correct first (Enzyme can only give correct gradients for a correct forward pass)
2. Verify the `try_into()` conversion works (position length must match N_PARAMS exactly)
3. Check that `#![feature(autodiff)]` is in `lib.rs`
4. Ensure you're building with nightly: the build tool handles this automatically
