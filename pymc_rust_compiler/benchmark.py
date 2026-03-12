"""Benchmark: compare PyMC (nutpie) vs AI-compiled Rust sampler."""

from __future__ import annotations

import ctypes
import subprocess
import time
from pathlib import Path

import numba
import numpy as np
import pymc as pm
from pymc.blocking import DictToArrayBijection, RaveledVars
from pymc.model.transform.optimization import freeze_dims_and_data

# Path to the bench_runner shared library
_BENCH_RUNNER_DIR = Path(__file__).resolve().parent.parent / "bench_runner"


def benchmark_nutpie(
    model: pm.Model, draws: int = 2000, tune: int = 1000, chains: int = 4
) -> dict:
    """Benchmark PyMC sampling with nutpie backend."""
    print(f"  nutpie: {chains} chains x {draws} draws...")
    start = time.time()
    idata = pm.sample(
        draws=draws,
        tune=tune,
        chains=chains,
        nuts_sampler="nutpie",
        model=model,
        random_seed=42,
        progressbar=False,
    )
    elapsed = time.time() - start
    throughput = (chains * draws) / elapsed

    return {
        "backend": "nutpie",
        "elapsed_s": elapsed,
        "throughput": throughput,
        "idata": idata,
    }


def benchmark_rust(
    build_dir: str | Path, draws: int = 2000, tune: int = 1000, chains: int = 4
) -> dict:
    """Benchmark the AI-compiled Rust sampler."""
    build_dir = Path(build_dir)
    binary = build_dir / "target" / "release" / "sample"

    if not binary.exists():
        # Build with the sampler main
        print("  Building Rust sampler...")
        subprocess.run(
            ["cargo", "build", "--release", "--bin", "sample"],
            cwd=build_dir,
            capture_output=True,
            check=True,
        )

    print(f"  Rust: {chains} chains x {draws} draws...")
    start = time.time()
    result = subprocess.run(
        [str(binary)],
        cwd=build_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:500]}")
        return {"backend": "rust", "elapsed_s": elapsed, "error": result.stderr}

    throughput = (chains * draws) / elapsed

    return {
        "backend": "rust-ai",
        "elapsed_s": elapsed,
        "throughput": throughput,
        "output": result.stdout,
    }


# ---------------------------------------------------------------------------
# logp+dlogp evaluation benchmark (no sampling overhead)
# ---------------------------------------------------------------------------


def _make_test_point(
    model: pm.Model, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Generate a random unconstrained test point in model variable order.

    Uses a random point (instead of the initial point) to avoid
    over-specialising benchmarks on a particular parameter vector.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    ip = model.initial_point()
    model_fn = model.logp_dlogp_function(ravel_inputs=True)
    x0 = np.concatenate([np.atleast_1d(ip[v.name]) for v in model_fn._grad_vars])
    return rng.standard_normal(x0.shape).astype(x0.dtype)


def _prepare_frozen_inputs(model, x0_model_order=None):
    """Prepare frozen model, jit_fn, and reordered x0.

    Shared setup for benchmark_logp_pytensor and benchmark_logp_numba_cfunc.
    Returns (jit_fn, x0, frozen_rv, model_fn).
    """
    frozen_model = freeze_dims_and_data(model)
    logp_dlogp_wrapper = frozen_model.logp_dlogp_function(
        ravel_inputs=True, mode="NUMBA"
    )
    logp_dlogp_fn = logp_dlogp_wrapper._pytensor_function
    logp_dlogp_fn.trust_input = True

    ip = model.initial_point()
    frozen_vars = [v.name for v in logp_dlogp_wrapper._grad_vars]

    model_fn = model.logp_dlogp_function(ravel_inputs=True)

    if x0_model_order is None:
        x0_model_order = _make_test_point(model)

    # Reorder x0 from model order → frozen order for this function
    model_ip = {v.name: ip[v.name] for v in model_fn._grad_vars}
    model_rv = DictToArrayBijection.map(model_ip)
    x0_dict = DictToArrayBijection.rmap(
        RaveledVars(x0_model_order, model_rv.point_map_info)
    )
    frozen_ip = {name: x0_dict[name] for name in frozen_vars}
    frozen_rv = DictToArrayBijection.map(frozen_ip)
    x0 = frozen_rv.data

    # Trigger numba compilation via a regular call, then grab the jit_fn
    logp_dlogp_fn(x0)
    jit_fn = logp_dlogp_fn.vm.jit_fn

    return jit_fn, x0, frozen_rv, model_fn


def _reorder_dlogp(dlogp_val, frozen_rv, model_fn):
    """Reorder dlogp from frozen order → model (= Rust) variable order."""
    dlogp_dict = DictToArrayBijection.rmap(
        RaveledVars(np.asarray(dlogp_val, dtype=np.float64), frozen_rv.point_map_info)
    )
    return DictToArrayBijection.map(
        {v.name: dlogp_dict[v.name] for v in model_fn._grad_vars}
    ).data


def benchmark_logp_pytensor(
    model: pm.Model,
    n_evals: int = 10_000,
    x0_model_order: np.ndarray | None = None,
) -> dict:
    """Benchmark PyTensor's compiled logp+dlogp function (what nutpie calls).

    Calls the numba jit_fn directly to avoid Python wrapper overhead,
    giving a fair apples-to-apples comparison against the Rust implementation.
    """
    jit_fn, x0, frozen_rv, model_fn = _prepare_frozen_inputs(model, x0_model_order)

    # Warmup the direct path
    for _ in range(200):
        jit_fn(x0)

    # Timed run – call numba directly, no Python wrapper overhead
    start = time.perf_counter()
    for _ in range(n_evals):
        logp_val, dlogp_val = jit_fn(x0)
    elapsed = time.perf_counter() - start

    us_per_eval = (elapsed / n_evals) * 1e6

    dlogp_val = _reorder_dlogp(dlogp_val, frozen_rv, model_fn)

    return {
        "backend": "pytensor (numba)",
        "n_evals": n_evals,
        "elapsed_s": elapsed,
        "us_per_eval": us_per_eval,
        "logp": float(logp_val),
        "dlogp": dlogp_val,
    }


# ---------------------------------------------------------------------------
# Numba cfunc benchmark: Rust calling Numba via C function pointer (like nutpie)
# ---------------------------------------------------------------------------


def _build_bench_runner() -> ctypes.CDLL:
    """Build and load the bench_runner shared library."""
    so_path = _BENCH_RUNNER_DIR / "target" / "release" / "libbench_runner.so"
    if not so_path.exists():
        print("  Building bench_runner shared library...")
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=_BENCH_RUNNER_DIR,
            capture_output=True,
            check=True,
        )
    lib = ctypes.CDLL(str(so_path))
    lib.bench_logp_cfunc.restype = ctypes.c_double
    lib.bench_logp_cfunc.argtypes = [
        ctypes.c_size_t,  # func_ptr (usize)
        ctypes.c_uint64,  # dim
        ctypes.c_void_p,  # x_ptr
        ctypes.c_uint64,  # n_warmup
        ctypes.c_uint64,  # n_iters
        ctypes.c_void_p,  # logp_out
        ctypes.c_void_p,  # grad_out
    ]
    return lib


def _make_numba_cfunc(jit_fn, n_dim: int):
    """Wrap a PyTensor Numba jit_fn in a C-callable function, like nutpie does.

    Returns a numba CFunc whose .address is a raw C function pointer that Rust
    can call directly with zero Python overhead.
    """
    c_sig = numba.types.int64(
        numba.types.uint64,  # dim
        numba.types.CPointer(numba.types.float64),  # x (input)
        numba.types.CPointer(numba.types.float64),  # grad (output)
        numba.types.CPointer(numba.types.float64),  # logp (output)
    )

    @numba.cfunc(c_sig)
    def logp_cfunc(dim, x_ptr, grad_ptr, logp_ptr):
        x = numba.carray(x_ptr, (dim,))
        logp_val, grad_val = jit_fn(x)
        logp_out = numba.carray(logp_ptr, (1,))
        grad_out = numba.carray(grad_ptr, (dim,))
        logp_out[0] = logp_val
        grad_out[:] = grad_val
        return 0

    return logp_cfunc


def benchmark_logp_numba_cfunc(
    model: pm.Model,
    n_evals: int = 10_000,
    x0_model_order: np.ndarray | None = None,
) -> dict:
    """Benchmark Numba logp+dlogp called from Rust via C function pointer.

    This replicates how nutpie actually calls PyTensor's compiled logp:
    Rust receives a C function pointer from numba.cfunc and calls it directly,
    with zero Python interpreter overhead.
    """
    jit_fn, x0, frozen_rv, model_fn = _prepare_frozen_inputs(model, x0_model_order)

    # Create C-callable wrapper (like nutpie does)
    n_dim = len(x0)
    print(f"  Compiling numba cfunc wrapper (dim={n_dim})...")
    cfunc = _make_numba_cfunc(jit_fn, n_dim)

    # Load Rust bench runner
    lib = _build_bench_runner()

    # Prepare buffers
    x_arr = np.ascontiguousarray(x0, dtype=np.float64)
    logp_arr = np.zeros(1, dtype=np.float64)
    grad_arr = np.zeros(n_dim, dtype=np.float64)

    # Call Rust: it calls the Numba cfunc in a tight loop
    us_per_eval = lib.bench_logp_cfunc(
        cfunc.address,
        n_dim,
        x_arr.ctypes.data,
        200,  # warmup
        n_evals,
        logp_arr.ctypes.data,
        grad_arr.ctypes.data,
    )

    elapsed = us_per_eval * n_evals / 1e6

    dlogp_val = _reorder_dlogp(grad_arr, frozen_rv, model_fn)

    return {
        "backend": "numba cfunc (rust loop)",
        "n_evals": n_evals,
        "elapsed_s": elapsed,
        "us_per_eval": us_per_eval,
        "logp": float(logp_arr[0]),
        "dlogp": dlogp_val,
    }


def benchmark_logp_rust(
    build_dir: str | Path,
    model: pm.Model,
    n_evals: int = 10_000,
    x0_model_order: np.ndarray | None = None,
) -> dict:
    """Benchmark the AI-compiled Rust logp+dlogp function."""
    build_dir = Path(build_dir).resolve()
    binary = build_dir / "target" / "release" / "bench"

    # Always rebuild to pick up bench.rs changes
    print("  Building Rust bench binary...")
    result = subprocess.run(
        ["cargo", "build", "--release", "--bin", "bench"],
        cwd=build_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {"backend": "rust-ai", "error": f"Build failed: {result.stderr[:500]}"}

    # Use provided x0 (model order) or generate a random one
    if x0_model_order is None:
        x0_model_order = _make_test_point(model)
    x0 = x0_model_order

    # Prepare stdin: first line = n_iters, second line = param vector
    param_str = ",".join(f"{v:.17e}" for v in x0)
    stdin_data = f"{n_evals}\n{param_str}\n"

    result = subprocess.run(
        [str(binary)],
        cwd=build_dir,
        input=stdin_data,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        return {"backend": "rust-ai", "error": f"Run failed: {result.stderr[:500]}"}

    # Parse output: us_per_eval,logp,grad[0],grad[1],...
    parts = result.stdout.strip().split(",")
    us_per_eval = float(parts[0])
    logp = float(parts[1])
    dlogp = np.array([float(v) for v in parts[2:]], dtype=np.float64)

    elapsed = us_per_eval * n_evals / 1e6

    return {
        "backend": "rust-ai",
        "n_evals": n_evals,
        "elapsed_s": elapsed,
        "us_per_eval": us_per_eval,
        "logp": logp,
        "dlogp": dlogp,
    }


def print_logp_comparison(
    pytensor_result: dict, rust_result: dict, model_name: str = ""
):
    """Print logp+dlogp evaluation benchmark comparison."""
    header = f"LOGP+DLOGP BENCHMARK{f': {model_name}' if model_name else ''}"
    print("\n" + "=" * 65)
    print(header)
    print("=" * 65)

    print(f"\n{'Backend':<20} {'us/eval':<12} {'evals/sec':<14} {'Speedup':<10}")
    print("-" * 56)

    pt = pytensor_result
    if "error" in pt:
        print(f"{pt['backend']:<20} {'ERROR':<12}")
    else:
        evals_per_sec_pt = 1e6 / pt["us_per_eval"]
        print(
            f"{pt['backend']:<20} {pt['us_per_eval']:<12.2f} {evals_per_sec_pt:<14,.0f} {'1.00x':<10}"
        )

    rs = rust_result
    if "error" in rs:
        print(f"{'rust-ai':<20} {'ERROR':<12}  {rs['error'][:60]}")
    else:
        evals_per_sec_rs = 1e6 / rs["us_per_eval"]
        speedup = pt["us_per_eval"] / rs["us_per_eval"] if "error" not in pt else 0
        print(
            f"{'rust-ai':<20} {rs['us_per_eval']:<12.2f} {evals_per_sec_rs:<14,.0f} {speedup:<10.2f}x"
        )

        # Check logp and dlogp agreement
        if "error" not in pt:
            logp_diff = abs(pt["logp"] - rs["logp"])
            logp_rel_err = logp_diff / max(abs(pt["logp"]), 1e-10)
            logp_ok = logp_rel_err < 1e-4
            logp_status = (
                "MATCH" if logp_ok else f"MISMATCH (rel_err={logp_rel_err:.2e})"
            )
            print(
                f"\n  logp check:  pytensor={pt['logp']:.8f}  rust={rs['logp']:.8f}  [{logp_status}]"
            )

            pt_dlogp = pt["dlogp"]
            rs_dlogp = rs["dlogp"]
            dlogp_abs_err = np.max(np.abs(pt_dlogp - rs_dlogp))
            dlogp_rel_err = dlogp_abs_err / max(np.max(np.abs(pt_dlogp)), 1e-10)
            dlogp_ok = dlogp_rel_err < 1e-4
            dlogp_status = (
                "MATCH" if dlogp_ok else f"MISMATCH (rel_err={dlogp_rel_err:.2e})"
            )
            print(
                f"  dlogp check: pytensor={pt_dlogp}  rust={rs_dlogp}  [{dlogp_status}]"
            )

            assert logp_ok, (
                f"logp mismatch: pytensor={pt['logp']:.10f} rust={rs['logp']:.10f} "
                f"rel_err={logp_rel_err:.2e}"
            )
            assert dlogp_ok, (
                f"dlogp mismatch: pytensor={pt_dlogp} rust={rs_dlogp} "
                f"rel_err={dlogp_rel_err:.2e}"
            )

    print()


def print_comparison(nutpie_result: dict, rust_result: dict):
    """Print a nice comparison table."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\n{'Backend':<20} {'Time (s)':<12} {'Draws/sec':<12} {'Speedup':<10}")
    print("-" * 54)

    nt = nutpie_result["elapsed_s"]
    print(
        f"{'nutpie':<20} {nt:<12.2f} {nutpie_result['throughput']:<12.0f} {'1.00x':<10}"
    )

    if "error" not in rust_result:
        rt = rust_result["elapsed_s"]
        speedup = nt / rt
        print(
            f"{'rust-ai':<20} {rt:<12.2f} {rust_result['throughput']:<12.0f} {speedup:<10.2f}x"
        )
    else:
        print(f"{'rust-ai':<20} {'FAILED':<12}")

    print()
