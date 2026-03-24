"""Compare MLX vs CPU (faer) vs PyTensor for GP model."""

import numpy as np
import pymc as pm

from transalchemy import compile_model
from transalchemy.benchmark import (
    benchmark_logp_pytensor,
    benchmark_logp_rust,
)


def make_gp_model(N=200):
    np.random.seed(42)
    x = np.sort(np.random.uniform(0, 10, N))
    K = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / 4.0) + 0.09 * np.eye(N)
    y_obs = np.random.multivariate_normal(np.zeros(N), K)

    with pm.Model() as model:
        ls = pm.HalfNormal("ls", sigma=5)
        eta = pm.HalfNormal("eta", sigma=5)
        sigma = pm.HalfNormal("sigma", sigma=5)
        cov = eta**2 * pm.gp.cov.ExpQuad(1, ls=ls)
        gp = pm.gp.Marginal(cov_func=cov)
        gp.marginal_likelihood("y", X=x[:, None], y=y_obs, sigma=sigma)
    return model


model = make_gp_model(N=200)

# Compile CPU (faer) version
print("=== Compiling CPU (faer) ===")
result_cpu = compile_model(
    model,
    build_dir="compiled_models/gp_cpu_200",
    use_mlx=False,
    verbose=True,
)
if not result_cpu.success:
    print("CPU compilation FAILED")
    exit(1)

# Benchmark all three
N_EVALS = 5000
print("\n=== Benchmarking ===")
pt = benchmark_logp_pytensor(model, n_evals=N_EVALS)
cpu = benchmark_logp_rust("compiled_models/gp_cpu_200", model, n_evals=N_EVALS)
mlx = benchmark_logp_rust("compiled_models/gp_mlx_test", model, n_evals=N_EVALS)

print("\n" + "=" * 65)
print("GP N=200: PyTensor vs Rust (faer CPU) vs Rust (MLX)")
print("=" * 65)
print(f"\n{'Backend':<25} {'µs/eval':<12} {'Speedup vs Numba':<18}")
print("-" * 55)
print(f"{'PyTensor (Numba)':<25} {pt['us_per_eval']:<12.2f} {'1.00x':<18}")
if "error" not in cpu:
    print(f"{'Rust + faer (CPU)':<25} {cpu['us_per_eval']:<12.2f} {pt['us_per_eval'] / cpu['us_per_eval']:<18.2f}x")
else:
    print(f"{'Rust + faer (CPU)':<25} {'ERROR':<12}")
if "error" not in mlx:
    print(
        f"{'Rust + MLX (Metal GPU)':<25} {mlx['us_per_eval']:<12.2f} {pt['us_per_eval'] / mlx['us_per_eval']:<18.2f}x"
    )
else:
    print(f"{'Rust + MLX (Metal GPU)':<25} {'ERROR':<12}")
print()
