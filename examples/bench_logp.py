"""Benchmark logp+dlogp evaluation: PyTensor vs AI-compiled Rust.

This measures raw logp+gradient computation speed without any sampling overhead.
It's the fairest comparison of what the Rust compiler actually produces vs
what nutpie calls under the hood (pytensor's compiled logp_dlogp_function).

Usage:
    cd pymc-rust-ai-compiler
    uv run python examples/bench_logp.py
"""

from pathlib import Path

import numpy as np
import pymc as pm

from pymc_rust_compiler.benchmark import (
    _make_test_point,
    benchmark_logp_pytensor,
    benchmark_logp_rust,
    print_logp_comparison,
)

N_EVALS = 10_000


def make_normal_model():
    """Simple 2-parameter model."""
    build_dir = Path("compiled_models/normal")
    y_obs = np.load(build_dir / "y_data.npy")
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=5)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)
    return model, str(build_dir)


def make_linreg_model():
    """Linear regression, 3 parameters."""
    build_dir = Path("compiled_models/linreg")
    y_obs = np.load(build_dir / "y_data.npy")
    x = np.load(build_dir / "x_0_data.npy")
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=5)
        mu = alpha + beta * x
        pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)
    return model, str(build_dir)


def make_hierarchical_model():
    """Hierarchical model, 12 unconstrained parameters."""
    build_dir = Path("compiled_models/hierarchical")
    y_obs = np.load(build_dir / "y_data.npy")
    x = np.load(build_dir / "x_0_data.npy")         # binary covariate
    group_idx = np.load(build_dir / "x_1_data.npy").astype(int)  # group indices
    n_groups = int(group_idx.max()) + 1
    with pm.Model() as model:
        mu_a = pm.Normal("mu_a", mu=0, sigma=10)
        sigma_a = pm.HalfNormal("sigma_a", sigma=5)
        a_offset = pm.Normal("a_offset", mu=0, sigma=1, shape=n_groups)
        a = pm.Deterministic("a", mu_a + sigma_a * a_offset)
        b = pm.Normal("b", mu=0, sigma=10)
        sigma_y = pm.HalfNormal("sigma_y", sigma=5)
        mu_y = a[group_idx] + b * x
        pm.Normal("y", mu=mu_y, sigma=sigma_y, observed=y_obs)
    return model, str(build_dir)


def main():
    models = [
        ("Normal (2 params)", make_normal_model),
        ("LinReg (3 params)", make_linreg_model),
        ("Hierarchical (12 params)", make_hierarchical_model),
    ]

    results = []
    for name, make_fn in models:
        print(f"\n{'='*65}")
        print(f"  {name}")
        print(f"{'='*65}")

        model, build_dir = make_fn()
        n_evals = N_EVALS
        x0 = _make_test_point(model)

        print(f"  Running {n_evals:,} logp+dlogp evaluations...")

        pt_result = benchmark_logp_pytensor(model, n_evals=n_evals, x0_model_order=x0)
        print(f"    pytensor: {pt_result['us_per_eval']:.2f} us/eval")

        rs_result = benchmark_logp_rust(build_dir, model, n_evals=n_evals, x0_model_order=x0)
        if "error" in rs_result:
            print(f"    rust-ai: ERROR - {rs_result['error'][:100]}")
        else:
            print(f"    rust-ai:  {rs_result['us_per_eval']:.2f} us/eval")

        print_logp_comparison(pt_result, rs_result, model_name=name)
        results.append((name, pt_result, rs_result))

    # Summary table
    print("\n" + "=" * 65)
    print("SUMMARY: logp+dlogp evaluation speed")
    print("=" * 65)
    print(f"\n{'Model':<30} {'pytensor':<14} {'rust-ai':<14} {'Speedup':<10}")
    print("-" * 68)
    for name, pt, rs in results:
        pt_us = f"{pt['us_per_eval']:.2f} us" if "error" not in pt else "ERROR"
        if "error" not in rs:
            rs_us = f"{rs['us_per_eval']:.2f} us"
            speedup = f"{pt['us_per_eval'] / rs['us_per_eval']:.1f}x" if "error" not in pt else "?"
        else:
            rs_us = "ERROR"
            speedup = "-"
        print(f"  {name:<28} {pt_us:<14} {rs_us:<14} {speedup:<10}")
    print()


if __name__ == "__main__":
    main()
