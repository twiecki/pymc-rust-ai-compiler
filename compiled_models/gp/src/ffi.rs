// C FFI wrapper for nutpie integration.
// Exposes logp as a C-callable function via shared library.
use crate::generated::GeneratedLogp;
use nuts_rs::CpuLogpFunc;

/// Thread-local logp function instance (avoids mutex overhead).
thread_local! {
    static LOGP_FN: std::cell::RefCell<GeneratedLogp> = std::cell::RefCell::new(
        GeneratedLogp::default()
    );
}

/// C-callable logp function.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub unsafe extern "C" fn logp_ffi(
    x: *const f64,
    grad: *mut f64,
    logp_out: *mut f64,
    dim: i32,
) -> i32 {
    let dim = dim as usize;
    let position = std::slice::from_raw_parts(x, dim);
    let gradient = std::slice::from_raw_parts_mut(grad, dim);

    LOGP_FN.with(|cell| {
        let mut logp_fn = cell.borrow_mut();
        // Zero gradient
        for g in gradient.iter_mut() {
            *g = 0.0;
        }
        match logp_fn.logp(position, gradient) {
            Ok(logp) => {
                *logp_out = logp;
                0
            }
            Err(_) => {
                *logp_out = f64::NEG_INFINITY;
                -1
            }
        }
    })
}
