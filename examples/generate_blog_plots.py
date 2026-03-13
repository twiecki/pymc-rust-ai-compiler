"""Generate blog-quality plots by running real compilation + optimization pipelines.

Runs:
1. PyMC → Rust compilation (Normal + Hierarchical models)
2. Optimization loop (autoresearcher-style)
3. logp benchmark comparison (PyTensor/Numba vs AI-compiled Rust)

Generates plots using the built-in analysis.py plotting functions.

Usage:
    cd projects/bayes-ai-compiler
    ANTHROPIC_API_KEY=... uv run python examples/generate_blog_plots.py
"""

import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Model factory functions
# ---------------------------------------------------------------------------

def make_normal_model():
    build_dir = Path("compiled_models/normal")
    y_obs = np.load(build_dir / "y_data.npy")
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=5)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)
    return model


def make_linreg_model():
    build_dir = Path("compiled_models/linreg")
    y_obs = np.load(build_dir / "y_data.npy")
    x = np.load(build_dir / "x_0_data.npy")
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=5)
        mu = alpha + beta * x
        pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)
    return model


def make_hierarchical_model():
    build_dir = Path("compiled_models/hierarchical")
    y_obs = np.load(build_dir / "y_data.npy")
    x = np.load(build_dir / "x_0_data.npy")
    group_idx = np.load(build_dir / "x_1_data.npy").astype(int)
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
    return model


def make_gp_model():
    build_dir = Path("compiled_models/gp")
    y_obs = np.load(build_dir / "y_data.npy")
    x = np.load(build_dir / "x_1_data.npy")
    with pm.Model() as model:
        ls = pm.HalfNormal("ls", sigma=5)
        eta = pm.HalfNormal("eta", sigma=5)
        sigma = pm.HalfNormal("sigma", sigma=5)
        cov = eta**2 * pm.gp.cov.ExpQuad(1, ls=ls)
        gp = pm.gp.Latent(cov_func=cov)
        f = gp.prior("f", X=x[:, None])
        pm.Normal("y", mu=f, sigma=sigma, observed=y_obs)
    return model


# ---------------------------------------------------------------------------
# Compile + Optimize
# ---------------------------------------------------------------------------

def run_compile_and_optimize(name, make_model_fn, source_code, build_dir):
    """Compile a PyMC model to Rust, then optimize it."""
    from pymc_rust_compiler import compile_model, optimize_model
    from pymc_rust_compiler.analysis import (
        plot_optimization_progress,
        plot_waterfall,
        plot_timeline,
        print_summary,
    )

    model = make_model_fn()
    slug = name.lower().replace(" ", "_")

    print("\n" + "=" * 65)
    print(f"  PyMC → Rust: {name} (compile)")
    print("=" * 65)

    compile_result = compile_model(
        model,
        source_code=source_code,
        build_dir=build_dir,
        verbose=True,
    )

    if not compile_result.success:
        print(f"Compilation FAILED for {name}")
        return compile_result, None

    print(f"\nCompilation successful in {compile_result.n_attempts} attempt(s)")

    # Optimize
    print("\n" + "=" * 65)
    print(f"  PyMC → Rust: {name} (optimize)")
    print("=" * 65)

    opt_result = optimize_model(
        compile_result,
        model,
        verbose=True,
        max_turns=15,
    )

    # Save results TSV
    tsv_path = opt_result.write_results_tsv()
    print(f"\nResults saved to: {tsv_path}")

    # Generate plots from TSV (avoids object-level bugs)
    for plot_fn, plot_name in [
        (plot_optimization_progress, "opt_progress"),
        (plot_waterfall, "opt_waterfall"),
        (plot_timeline, "opt_timeline"),
    ]:
        try:
            fig = plot_fn(tsv_path, title=f"{plot_name.replace('_', ' ').title()}: {name}")
            fig.savefig(OUTPUT_DIR / f"{plot_name}_{slug}.png", dpi=150)
            plt.close(fig)
            print(f"Saved: {plot_name}_{slug}.png")
        except Exception as e:
            print(f"Could not generate {plot_name} plot: {e}")

    print_summary(tsv_path)

    return compile_result, opt_result


# ---------------------------------------------------------------------------
# Benchmark comparison plot
# ---------------------------------------------------------------------------

def run_benchmarks_and_plot():
    """Run logp benchmarks on compiled models and generate comparison bar chart."""
    from pymc_rust_compiler.benchmark import (
        _make_test_point,
        benchmark_logp_pytensor,
        benchmark_logp_rust,
    )

    models_info = [
        ("Normal\n(2 params)", make_normal_model, "compiled_models/normal"),
        ("LinReg\n(3 params)", make_linreg_model, "compiled_models/linreg"),
        ("Hierarchical\n(12 params)", make_hierarchical_model, "compiled_models/hierarchical"),
        ("GP\n(3 params)", make_gp_model, "compiled_models/gp"),
    ]

    n_evals = 200_000
    results = []

    print("\n" + "=" * 65)
    print("  Benchmark: logp+dlogp speed comparison")
    print("=" * 65)

    for name, make_fn, build_dir in models_info:
        label = name.replace(chr(10), " ")
        print(f"\n  {label}:")
        try:
            model = make_fn()
            x0 = _make_test_point(model)

            pt_result = benchmark_logp_pytensor(model, n_evals=n_evals, x0_model_order=x0)
            rs_result = benchmark_logp_rust(build_dir, model, n_evals=n_evals, x0_model_order=x0)

            pt_us = pt_result["us_per_eval"]
            rs_us = rs_result["us_per_eval"] if "error" not in rs_result else None
            speedup = pt_us / rs_us if rs_us else None

            print(f"    PyTensor: {pt_us:.2f} us/eval")
            if rs_us:
                print(f"    Rust:     {rs_us:.2f} us/eval  ({speedup:.1f}x faster)")
            else:
                print(f"    Rust:     ERROR - {rs_result.get('error', 'unknown')[:80]}")

            results.append((name, pt_us, rs_us, speedup))
        except Exception as e:
            print(f"    SKIPPED: {e}")
            continue

    if not results:
        print("No benchmark results to plot")
        return

    # Generate comparison bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    names = [r[0] for r in results]
    pt_vals = [r[1] for r in results]
    rs_vals = [r[2] if r[2] else 0 for r in results]
    speedups = [r[3] if r[3] else 0 for r in results]

    x = np.arange(len(names))
    width = 0.35

    # Bar chart: us/eval comparison
    bars1 = ax1.bar(x - width / 2, pt_vals, width, label="PyTensor (Numba)",
                    color="#3498db", edgecolor="#2c3e50", linewidth=0.5)
    bars2 = ax1.bar(x + width / 2, rs_vals, width, label="AI-compiled Rust",
                    color="#e74c3c", edgecolor="#2c3e50", linewidth=0.5)

    ax1.set_ylabel("us/eval (lower is better)", fontsize=11)
    ax1.set_title("logp+gradient Evaluation Speed", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9)
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis="y")

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        if bar.get_height() > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    # Speedup chart
    colors = ["#2ecc71" if s >= 3 else "#f39c12" if s >= 2 else "#e74c3c" for s in speedups]
    bars3 = ax2.bar(x, speedups, width * 1.5, color=colors, edgecolor="#2c3e50", linewidth=0.5)
    ax2.axhline(y=1, color="#2c3e50", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.set_ylabel("Speedup (higher is better)", fontsize=11)
    ax2.set_title("Rust vs PyTensor Speedup", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    for bar, s in zip(bars3, speedups):
        if s > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{s:.1f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "benchmark_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: benchmark_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.chdir(Path(__file__).parent.parent)
    print(f"Working dir: {Path.cwd()}")
    print(f"Output dir: {OUTPUT_DIR.resolve()}")

    # Part 1: PyMC → Rust compile + optimize (Normal)
    print("\n" + "#" * 65)
    print("#  PART 1: PyMC → Rust Compilation + Optimization")
    print("#" * 65)

    normal_source = """
mu ~ Normal(0, 10)
sigma ~ HalfNormal(5)
y ~ Normal(mu, sigma), observed
"""
    normal_compile, normal_opt = run_compile_and_optimize(
        "Normal",
        make_normal_model,
        source_code=normal_source,
        build_dir="compiled_models/normal",
    )

    # Part 2: PyMC → Rust compile + optimize (Hierarchical)
    hier_source = """
mu_a ~ Normal(0, 10)
sigma_a ~ HalfNormal(5)
a_offset ~ Normal(0, 1, shape=8)
a = mu_a + sigma_a * a_offset
b ~ Normal(0, 10)
sigma_y ~ HalfNormal(5)
y ~ Normal(a[group_idx] + b * x, sigma_y), observed
"""
    hier_compile, hier_opt = run_compile_and_optimize(
        "Hierarchical",
        make_hierarchical_model,
        source_code=hier_source,
        build_dir="compiled_models/hierarchical",
    )

    # Part 2b: PyMC → Rust compile + optimize (GP)
    gp_source = """
ls ~ HalfNormal(5)
eta ~ HalfNormal(5)
sigma ~ HalfNormal(5)
K = eta^2 * ExpQuad(x, ls) + sigma^2 * I
y ~ MvNormal(0, K), observed
"""
    gp_compile, gp_opt = run_compile_and_optimize(
        "GP",
        make_gp_model,
        source_code=gp_source,
        build_dir="compiled_models/gp",
    )

    # Part 3: Benchmark comparison
    print("\n" + "#" * 65)
    print("#  PART 3: Benchmark Comparison")
    print("#" * 65)

    run_benchmarks_and_plot()

    # Summary
    print("\n" + "=" * 65)
    print("  All plots saved to:", OUTPUT_DIR.resolve())
    print("=" * 65)
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
