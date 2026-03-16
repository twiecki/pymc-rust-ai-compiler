"""Bridge between AI-compiled Rust models and nutpie sampler.

Usage:
    import pymc as pm
    from transpailer import compile_model
    from transpailer.nutpie_bridge import to_nutpie

    with pm.Model() as model:
        ...

    result = compile_model(model)
    compiled = to_nutpie(result, model)

    import nutpie
    idata = nutpie.sample(compiled, draws=1000, tune=500, chains=4)
"""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pymc as pm

from transpailer.compiler import CompilationResult


def _build_shared_lib(build_dir: Path) -> Path:
    """Build the compiled model as a shared library (.so)."""
    build_dir = build_dir.resolve()
    so_path = build_dir / "target" / "release" / "libpymc_compiled_model.so"

    # Check if already built and up to date
    gen_rs = build_dir / "src" / "generated.rs"
    if so_path.exists() and so_path.stat().st_mtime > gen_rs.stat().st_mtime:
        return so_path

    result = subprocess.run(
        ["cargo", "build", "--release", "--lib"],
        cwd=build_dir,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to build shared library:\n{result.stderr}")

    if not so_path.exists():
        raise RuntimeError(
            f'Shared library not found at {so_path}. Ensure Cargo.toml has [lib] crate-type = ["cdylib"]'
        )
    return so_path


def _load_logp_fn(so_path: Path, n_dim: int):
    """Load the shared library and return a Python-callable logp function."""
    lib = ctypes.CDLL(str(so_path))

    # C FFI signature: int logp_ffi(const double* x, double* grad, double* logp_out, int dim)
    lib.logp_ffi.restype = ctypes.c_int
    lib.logp_ffi.argtypes = [
        ctypes.c_void_p,  # x (input)
        ctypes.c_void_p,  # grad (output)
        ctypes.c_void_p,  # logp_out (output)
        ctypes.c_int,  # dim
    ]

    # Pre-allocate output buffers
    grad_buf = np.zeros(n_dim, dtype=np.float64)
    logp_buf = np.zeros(1, dtype=np.float64)

    def logp_fn(x):
        x = np.ascontiguousarray(x, dtype=np.float64)
        grad_buf[:] = 0.0
        ret = lib.logp_ffi(
            x.ctypes.data,
            grad_buf.ctypes.data,
            logp_buf.ctypes.data,
            n_dim,
        )
        if ret != 0:
            return np.float64(-np.inf), np.zeros(n_dim, dtype=np.float64)
        return np.float64(logp_buf[0]), grad_buf.copy()

    return logp_fn, lib  # Return lib to keep it alive


def to_nutpie(
    compile_result: CompilationResult,
    model: pm.Model,
) -> "nutpie.compiled_pyfunc.PyFuncModel":  # noqa: F821
    """Convert a CompilationResult into a nutpie-compatible model for sampling.

    Args:
        compile_result: A successful CompilationResult from compile_model.
        model: The original PyMC model.

    Returns:
        A PyFuncModel that can be passed to nutpie.sample().

    Example:
        result = compile_model(model)
        compiled = to_nutpie(result, model)
        idata = nutpie.sample(compiled, draws=1000, chains=4)
    """
    from nutpie.compiled_pyfunc import from_pyfunc

    if not compile_result.success:
        raise ValueError("Cannot create nutpie model from failed compilation")

    build_dir = compile_result.build_dir

    # Ensure the Cargo.toml has cdylib output and the FFI wrapper exists
    _ensure_ffi_setup(build_dir)

    # Build the shared library
    so_path = _build_shared_lib(build_dir)

    # Get model metadata
    model_fn = model.logp_dlogp_function(ravel_inputs=True)
    ip = model.initial_point()
    from pymc.blocking import DictToArrayBijection

    x0 = DictToArrayBijection.map({v.name: ip[v.name] for v in model_fn._grad_vars}).data
    n_dim = len(x0)

    # Get variable names, shapes, dtypes from the model
    # Use grad_vars which are the unconstrained (transformed) parameters
    var_names = []
    var_shapes = []
    var_dtypes = []
    for v in model_fn._grad_vars:
        name = v.name
        val = ip[name]
        arr = np.atleast_1d(val)
        var_names.append(name)
        var_shapes.append(arr.shape)
        var_dtypes.append(arr.dtype)

    # Keep a reference to the loaded library
    _lib_refs = []

    def make_logp_fn():
        logp_fn, lib = _load_logp_fn(so_path, n_dim)
        _lib_refs.append(lib)
        return logp_fn

    def make_expand_fn(seed1, seed2, chain):
        # Map unconstrained vector to named parameters (unconstrained space)
        def expand_fn(x):
            result = {}
            offset = 0
            for name, shape in zip(var_names, var_shapes):
                size = int(np.prod(shape)) if shape else 1
                result[name] = x[offset : offset + size].reshape(shape)
                offset += size
            return result

        return expand_fn

    def make_initial_point(seed):
        ip_ = model.initial_point()
        return DictToArrayBijection.map({v.name: ip_[v.name] for v in model_fn._grad_vars}).data.astype(np.float64)

    return from_pyfunc(
        ndim=n_dim,
        make_logp_fn=make_logp_fn,
        make_expand_fn=make_expand_fn,
        expanded_dtypes=var_dtypes,
        expanded_shapes=var_shapes,
        expanded_names=var_names,
        make_initial_point_fn=make_initial_point,
    )


# FFI wrapper code that gets added to the Rust project
_FFI_WRAPPER_RS = """\
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
"""


def _ensure_ffi_setup(build_dir: Path):
    """Ensure the build directory has the FFI wrapper and cdylib config."""
    build_dir = Path(build_dir).resolve()
    src_dir = build_dir / "src"

    # Write FFI wrapper
    ffi_path = src_dir / "ffi.rs"
    ffi_path.write_text(_FFI_WRAPPER_RS)

    # Update lib.rs to include ffi module
    lib_rs = src_dir / "lib.rs"
    content = lib_rs.read_text()
    if "pub mod ffi;" not in content:
        content += "\npub mod ffi;\n"
        lib_rs.write_text(content)

    # Update Cargo.toml to produce cdylib
    cargo_toml = build_dir / "Cargo.toml"
    cargo_content = cargo_toml.read_text()
    if "[lib]" not in cargo_content:
        # Add lib section before the first [[bin]]
        lib_section = '\n[lib]\ncrate-type = ["cdylib", "rlib"]\n\n'
        cargo_content = cargo_content.replace("[[bin]]", lib_section + "[[bin]]", 1)
        cargo_toml.write_text(cargo_content)
