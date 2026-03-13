"""End-to-end benchmark: compile 5 models and compare nutpie vs AI-compiled Rust.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/run_benchmark.py

This will:
1. Define 5 PyMC models (Normal, LinReg, Hierarchical, GP, ZeroSumNormal)
2. Compile each to Rust via Claude API
3. Benchmark nutpie vs compiled Rust sampler
4. Print comparison table
"""

import numpy as np
import pymc as pm

from transpailer import compile_model
from transpailer.benchmark import (
    benchmark_nutpie,
    benchmark_rust,
    print_comparison,
)


def make_normal_model():
    """Simple Normal(mu, sigma) with 100 observations."""
    np.random.seed(42)
    y_obs = np.random.normal(5.0, 1.2, size=100)
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=5)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)
    return model, "mu ~ N(0,10), sigma ~ HN(5), y ~ N(mu, sigma)"


def make_linreg_model():
    """Linear regression with 200 observations."""
    np.random.seed(42)
    N = 200
    x = np.linspace(0, 10, N)
    y_obs = 2.5 - 1.3 * x + np.random.normal(0, 0.8, N)
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=5)
        pm.Normal("y", mu=alpha + beta * x, sigma=sigma, observed=y_obs)
    return model, "alpha, beta ~ N(0,10), sigma ~ HN(5), y ~ N(a+b*x, s)"


def make_hierarchical_model():
    """Hierarchical model with 8 groups, ~150 observations."""
    np.random.seed(42)
    n_groups = 8
    n_per_group = np.random.randint(10, 30, size=n_groups)
    N = n_per_group.sum()
    true_a = np.random.normal(1.5, 0.7, n_groups)
    group_idx = np.repeat(np.arange(n_groups), n_per_group)
    x = np.random.binomial(1, 0.5, N).astype(float)
    y_obs = true_a[group_idx] - 0.8 * x + np.random.normal(0, 0.5, N)

    with pm.Model() as model:
        mu_a = pm.Normal("mu_a", mu=0, sigma=10)
        sigma_a = pm.HalfNormal("sigma_a", sigma=5)
        a_offset = pm.Normal("a_offset", mu=0, sigma=1, shape=n_groups)
        a = pm.Deterministic("a", mu_a + sigma_a * a_offset)
        b = pm.Normal("b", mu=0, sigma=10)
        sigma_y = pm.HalfNormal("sigma_y", sigma=5)
        pm.Normal("y", mu=a[group_idx] + b * x, sigma=sigma_y, observed=y_obs)
    return model, f"Hierarchical: {n_groups} groups, {N} obs, 12 params"


def make_gp_model():
    """GP regression with ExpQuad kernel, 50 observations, 3 params."""
    np.random.seed(42)
    N = 50
    x = np.sort(np.random.uniform(0, 10, N))
    # True GP draw with ls=2, eta=1, sigma=0.3
    K = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / 4.0) + 0.09 * np.eye(N)
    y_obs = np.random.multivariate_normal(np.zeros(N), K)

    with pm.Model() as model:
        ls = pm.HalfNormal("ls", sigma=5)
        eta = pm.HalfNormal("eta", sigma=5)
        sigma = pm.HalfNormal("sigma", sigma=5)
        cov = eta**2 * pm.gp.cov.ExpQuad(1, ls=ls)
        gp = pm.gp.Marginal(cov_func=cov)
        gp.marginal_likelihood("y", X=x[:, None], y=y_obs, sigma=sigma)
    return model, f"GP regression: ExpQuad kernel, {N} obs, 3 params"


def make_zerosumnormal_model():
    """Saturated ANOVA with ZeroSumNormal effects, 124 params."""
    np.random.seed(314)
    n_stores = 6
    n_days = 7
    n_categories = 4

    # True effects
    true_grand_mean = 8.0
    true_sigma_store = 0.4
    true_sigma_cat = 0.5
    true_sigma_y = 0.6

    raw_store = np.random.normal(0, true_sigma_store, n_stores)
    true_store_effect = raw_store - raw_store.mean()
    raw_day = np.array([-0.2, -0.1, 0.0, 0.05, 0.15, 0.35, 0.25])
    raw_day += np.random.normal(0, 0.05, n_days)
    true_day_effect = raw_day - raw_day.mean()
    raw_interaction = np.random.normal(
        0, true_sigma_cat, (n_stores, n_days, n_categories)
    )
    raw_interaction -= raw_interaction.mean(axis=-1, keepdims=True)
    raw_interaction -= raw_interaction.mean(axis=-2, keepdims=True)

    # Generate observations
    records = []
    for s in range(n_stores):
        for d in range(n_days):
            for c in range(n_categories):
                n = np.random.poisson(5) + 1
                mu = (
                    true_grand_mean
                    + true_store_effect[s]
                    + true_day_effect[d]
                    + raw_interaction[s, d, c]
                )
                y_vals = np.random.normal(mu, true_sigma_y, n)
                for y in y_vals:
                    records.append((s, d, c, y))

    data = np.array(records)
    store_idx = data[:, 0].astype(int)
    day_idx = data[:, 1].astype(int)
    cat_idx = data[:, 2].astype(int)
    y_obs = data[:, 3]
    N = len(y_obs)

    with pm.Model() as model:
        grand_mean = pm.Normal("grand_mean", mu=0, sigma=10)
        sigma_store = pm.HalfNormal("sigma_store", sigma=2)
        sigma_day = pm.HalfNormal("sigma_day", sigma=2)
        sigma_cat = pm.HalfNormal("sigma_cat", sigma=2)
        store_effect = pm.ZeroSumNormal(
            "store_effect", sigma=sigma_store, shape=n_stores
        )
        day_effect = pm.ZeroSumNormal("day_effect", sigma=sigma_day, shape=n_days)
        interaction = pm.ZeroSumNormal(
            "interaction",
            sigma=sigma_cat,
            shape=(n_stores, n_days, n_categories),
            n_zerosum_axes=2,
        )
        mu_y = (
            grand_mean
            + store_effect[store_idx]
            + day_effect[day_idx]
            + interaction[store_idx, day_idx, cat_idx]
        )
        sigma_y = pm.HalfNormal("sigma_y", sigma=5)
        pm.Normal("y", mu=mu_y, sigma=sigma_y, observed=y_obs)
    return (
        model,
        f"ZeroSumNormal ANOVA: {n_stores}×{n_days}×{n_categories}, {N} obs, 124 params",
    )


MODELS = [
    ("Normal", make_normal_model),
    ("LinReg", make_linreg_model),
    ("Hierarchical", make_hierarchical_model),
    ("GP", make_gp_model),
    ("ZeroSumNormal", make_zerosumnormal_model),
]


def main():
    print("=" * 70)
    print("PyMC Rust AI Compiler — Benchmark Suite")
    print("=" * 70)

    results = []

    for name, make_model in MODELS:
        print(f"\n{'─' * 70}")
        print(f"Model: {name}")
        print(f"{'─' * 70}")

        model, desc = make_model()
        print(f"  {desc}")
        print(f"  Free RVs: {[rv.name for rv in model.free_RVs]}")
        print()

        # Step 1: Compile to Rust
        print("Compiling to Rust...")
        result = compile_model(
            model,
            build_dir=f"compiled_models/{name.lower()}",
            verbose=True,
        )

        if not result.success:
            print("  FAILED — skipping benchmark")
            results.append((name, None, None))
            continue

        # Step 2: Benchmark nutpie
        print("\nBenchmarking...")
        nutpie_result = benchmark_nutpie(model)

        # Step 3: Benchmark compiled Rust
        rust_result = benchmark_rust(result.build_dir)

        print_comparison(nutpie_result, rust_result)
        results.append((name, nutpie_result, rust_result))

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<20} {'nutpie (s)':<12} {'rust-ai (s)':<12} {'Speedup':<10}")
    print("-" * 54)
    for name, nutpie_r, rust_r in results:
        if nutpie_r is None:
            print(f"{name:<20} {'FAILED':<12}")
        elif "error" in (rust_r or {}):
            print(f"{name:<20} {nutpie_r['elapsed_s']:<12.2f} {'FAILED':<12}")
        else:
            nt = nutpie_r["elapsed_s"]
            rt = rust_r["elapsed_s"]
            print(f"{name:<20} {nt:<12.2f} {rt:<12.2f} {nt / rt:<10.2f}x")


if __name__ == "__main__":
    main()
