"""Test MLX compilation for GP model and benchmark against CPU (faer)."""

import numpy as np
import pymc as pm

from transalchemy import compile_model
from transalchemy.benchmark import (
    benchmark_logp_pytensor,
    benchmark_logp_rust,
    print_logp_comparison,
)
from transalchemy.compiler import _mlx_available


def make_gp_model(N=50):
    """GP regression with ExpQuad kernel."""
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


print(f"MLX available: {_mlx_available()}")

model = make_gp_model(N=200)
print(f"Free RVs: {[rv.name for rv in model.free_RVs]}")

# Compile with MLX
print("\n=== Compiling with MLX ===")
result_mlx = compile_model(
    model,
    build_dir="compiled_models/gp_mlx_test",
    use_mlx=True,
    verbose=True,
)

if not result_mlx.success:
    print("MLX compilation FAILED")
    exit(1)

print(f"\nMLX compilation succeeded in {result_mlx.n_attempts} attempt(s)")

# Benchmark
print("\n=== Benchmarking ===")
pt_result = benchmark_logp_pytensor(model, n_evals=5000)
rust_mlx_result = benchmark_logp_rust("compiled_models/gp_mlx_test", model, n_evals=5000)

print_logp_comparison(pt_result, rust_mlx_result, "GP (MLX)")
