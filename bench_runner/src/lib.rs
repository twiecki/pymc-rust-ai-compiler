//! Rust bench runner for calling Numba cfunc logp+dlogp via C function pointer.
//!
//! This replicates how nutpie calls Numba-compiled logp functions:
//! Rust receives a C function pointer and calls it in a tight loop.

use std::time::Instant;

/// C function signature matching nutpie's logp convention:
///   fn(dim: u64, x: *const f64, grad: *mut f64, logp: *mut f64) -> i64
type LogpFunc = extern "C" fn(u64, *const f64, *mut f64, *mut f64) -> i64;

/// Benchmark a logp+dlogp C function pointer.
///
/// # Arguments
/// * `func_ptr` - Address of a C-callable logp function (e.g. from numba.cfunc)
/// * `dim` - Number of parameters
/// * `x_ptr` - Pointer to input parameter array (length = dim)
/// * `n_warmup` - Number of warmup iterations
/// * `n_iters` - Number of timed iterations
/// * `logp_out` - Output: logp value from last evaluation
/// * `grad_out` - Output: gradient array from last evaluation (length = dim)
///
/// # Returns
/// Microseconds per evaluation
#[no_mangle]
pub extern "C" fn bench_logp_cfunc(
    func_ptr: usize,
    dim: u64,
    x_ptr: *const f64,
    n_warmup: u64,
    n_iters: u64,
    logp_out: *mut f64,
    grad_out: *mut f64,
) -> f64 {
    let func: LogpFunc = unsafe { std::mem::transmute(func_ptr) };
    let mut grad = vec![0.0f64; dim as usize];
    let mut logp = 0.0f64;

    // Warmup
    for _ in 0..n_warmup {
        func(dim, x_ptr, grad.as_mut_ptr(), &mut logp);
    }

    // Timed loop
    let start = Instant::now();
    for _ in 0..n_iters {
        func(dim, x_ptr, grad.as_mut_ptr(), &mut logp);
    }
    let elapsed = start.elapsed();

    // Copy results out
    unsafe {
        *logp_out = logp;
        std::ptr::copy_nonoverlapping(grad.as_ptr(), grad_out, dim as usize);
    }

    elapsed.as_nanos() as f64 / n_iters as f64 / 1000.0
}
