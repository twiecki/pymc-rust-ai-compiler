"""Comprehensive benchmarks: PyMC (nutpie/numba) vs Stan (cmdstanpy) vs Rust.

Benchmarks only models that have a compiled Rust implementation.
Measures compilation and sampling times separately, runs multiple iterations,
and stores convergence diagnostics (ESS, Rhat).

Usage:
    cd /path/to/transalchemy
    uv run python benchmarks/run_benchmarks.py [--models normal,linreg] [--n-repeats 3] [--output results.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm

# ── Project root ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPILED_MODELS_DIR = PROJECT_ROOT / "compiled_models"


# ── Result dataclasses ────────────────────────────────────────────────────


@dataclass
class SamplingResult:
    """Result from a single sampling run."""

    backend: str
    model_name: str
    compile_time_s: float | None  # None if pre-compiled
    sampling_time_s: float  # wall-clock time for sampling only
    total_time_s: float  # compile + sample
    draws: int
    tune: int
    chains: int
    # Convergence diagnostics (per-variable)
    ess_bulk: dict[str, float] = field(default_factory=dict)
    ess_tail: dict[str, float] = field(default_factory=dict)
    rhat: dict[str, float] = field(default_factory=dict)
    divergences: int = 0
    max_treedepth_hits: int = 0
    # Derived
    draws_per_sec: float = 0.0
    error: str | None = None


@dataclass
class BenchmarkResult:
    """All results for one model across backends and repeats."""

    model_name: str
    n_params: int
    description: str
    pymc_runs: list[SamplingResult] = field(default_factory=list)
    stan_runs: list[SamplingResult] = field(default_factory=list)
    rust_runs: list[SamplingResult] = field(default_factory=list)


# ── Model definitions ─────────────────────────────────────────────────────
# Each returns (pm.Model, stan_code, stan_data_dict, description)


def _make_normal():
    np.random.seed(42)
    y_obs = np.random.normal(5.0, 1.2, size=100)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=5)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)

    stan_code = """
data {
  int<lower=0> N;
  array[N] real y;
}
parameters {
  real mu;
  real<lower=0> sigma;
}
model {
  mu ~ normal(0, 10);
  sigma ~ normal(0, 5);
  y ~ normal(mu, sigma);
}
"""
    stan_data = {"N": len(y_obs), "y": y_obs.tolist()}
    return model, stan_code, stan_data, "Normal: mu, sigma, 100 obs, 2 params"


def _make_linreg():
    np.random.seed(42)
    N = 200
    x = np.linspace(0, 10, N)
    y_obs = 2.5 - 1.3 * x + np.random.normal(0, 0.8, N)

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=5)
        pm.Normal("y", mu=alpha + beta * x, sigma=sigma, observed=y_obs)

    stan_code = """
data {
  int<lower=0> N;
  array[N] real x;
  array[N] real y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, 10);
  beta ~ normal(0, 10);
  sigma ~ normal(0, 5);
  for (n in 1:N)
    y[n] ~ normal(alpha + beta * x[n], sigma);
}
"""
    stan_data = {"N": N, "x": x.tolist(), "y": y_obs.tolist()}
    return model, stan_code, stan_data, "LinReg: alpha, beta, sigma, 200 obs, 3 params"


def _make_hierarchical():
    np.random.seed(42)
    n_groups = 8
    n_per_group = np.random.randint(10, 30, size=n_groups)
    N = int(n_per_group.sum())
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

    stan_code = """
data {
  int<lower=0> N;
  int<lower=1> J;
  array[N] int<lower=1,upper=J> group;
  array[N] real x;
  array[N] real y;
}
parameters {
  real mu_a;
  real<lower=0> sigma_a;
  array[J] real a_offset;
  real b;
  real<lower=0> sigma_y;
}
transformed parameters {
  array[J] real a;
  for (j in 1:J)
    a[j] = mu_a + sigma_a * a_offset[j];
}
model {
  mu_a ~ normal(0, 10);
  sigma_a ~ normal(0, 5);
  a_offset ~ normal(0, 1);
  b ~ normal(0, 10);
  sigma_y ~ normal(0, 5);
  for (n in 1:N)
    y[n] ~ normal(a[group[n]] + b * x[n], sigma_y);
}
"""
    stan_data = {
        "N": N,
        "J": n_groups,
        "group": (group_idx + 1).tolist(),  # 1-based
        "x": x.tolist(),
        "y": y_obs.tolist(),
    }
    return (
        model,
        stan_code,
        stan_data,
        f"Hierarchical: {n_groups} groups, {N} obs, 12 params",
    )


def _make_gp():
    np.random.seed(42)
    N = 50
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

    # Stan GP with squared exponential kernel
    stan_code = """
data {
  int<lower=1> N;
  array[N] real x;
  vector[N] y;
}
parameters {
  real<lower=0> ls;
  real<lower=0> eta;
  real<lower=0> sigma;
}
model {
  matrix[N, N] K;
  matrix[N, N] L;

  ls ~ normal(0, 5);
  eta ~ normal(0, 5);
  sigma ~ normal(0, 5);

  for (i in 1:N)
    for (j in 1:N)
      K[i, j] = square(eta) * exp(-0.5 * square(x[i] - x[j]) / square(ls));

  for (i in 1:N)
    K[i, i] = K[i, i] + square(sigma);

  L = cholesky_decompose(K);
  y ~ multi_normal_cholesky(rep_vector(0, N), L);
}
"""
    stan_data = {"N": N, "x": x.tolist(), "y": y_obs.tolist()}
    return model, stan_code, stan_data, f"GP: ExpQuad kernel, {N} obs, 3 params"


def _make_zerosumnormal():
    np.random.seed(314)
    n_stores, n_days, n_categories = 6, 7, 4

    raw_store = np.random.normal(0, 0.4, n_stores)
    true_store_effect = raw_store - raw_store.mean()
    raw_day = np.array([-0.2, -0.1, 0.0, 0.05, 0.15, 0.35, 0.25])
    raw_day += np.random.normal(0, 0.05, n_days)
    true_day_effect = raw_day - raw_day.mean()
    raw_interaction = np.random.normal(0, 0.5, (n_stores, n_days, n_categories))
    raw_interaction -= raw_interaction.mean(axis=-1, keepdims=True)
    raw_interaction -= raw_interaction.mean(axis=-2, keepdims=True)

    records = []
    for s in range(n_stores):
        for d in range(n_days):
            for c in range(n_categories):
                n = np.random.poisson(5) + 1
                mu = 8.0 + true_store_effect[s] + true_day_effect[d] + raw_interaction[s, d, c]
                for y in np.random.normal(mu, 0.6, n):
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
        store_effect = pm.ZeroSumNormal("store_effect", sigma=sigma_store, shape=n_stores)
        day_effect = pm.ZeroSumNormal("day_effect", sigma=sigma_day, shape=n_days)
        interaction = pm.ZeroSumNormal(
            "interaction",
            sigma=sigma_cat,
            shape=(n_stores, n_days, n_categories),
            n_zerosum_axes=2,
        )
        mu_y = grand_mean + store_effect[store_idx] + day_effect[day_idx] + interaction[store_idx, day_idx, cat_idx]
        sigma_y = pm.HalfNormal("sigma_y", sigma=5)
        pm.Normal("y", mu=mu_y, sigma=sigma_y, observed=y_obs)

    # No Stan equivalent for ZeroSumNormal — skip Stan benchmark
    return (
        model,
        None,  # no Stan code
        None,
        f"ZeroSumNormal ANOVA: {n_stores}x{n_days}x{n_categories}, {N} obs, 124 params",
    )


MODEL_REGISTRY: dict[str, tuple] = {
    "normal": (_make_normal, 2),
    "linreg": (_make_linreg, 3),
    "hierarchical": (_make_hierarchical, 12),
    "gp": (_make_gp, 3),
    "zerosumnormal": (_make_zerosumnormal, 124),
}


# ── Diagnostics extraction ───────────────────────────────────────────────


def _extract_diagnostics(idata: az.InferenceData, result: SamplingResult):
    """Extract ESS, Rhat, divergences from an InferenceData object."""
    try:
        summary = az.summary(idata, kind="diagnostics")
        for var_name in summary.index:
            result.ess_bulk[var_name] = float(summary.loc[var_name, "ess_bulk"])
            result.ess_tail[var_name] = float(summary.loc[var_name, "ess_tail"])
            result.rhat[var_name] = float(summary.loc[var_name, "r_hat"])
    except Exception as e:
        result.error = f"Diagnostics extraction failed: {e}"

    # Divergences
    if hasattr(idata, "sample_stats"):
        try:
            divs = idata.sample_stats.get("diverging", None)
            if divs is not None:
                result.divergences = int(divs.values.sum())
        except Exception:
            pass


def _min_ess(result: SamplingResult) -> float:
    """Return the minimum bulk ESS across all variables."""
    if result.ess_bulk:
        return min(result.ess_bulk.values())
    return float("nan")


def _max_rhat(result: SamplingResult) -> float:
    """Return the maximum Rhat across all variables."""
    if result.rhat:
        return max(result.rhat.values())
    return float("nan")


# ── PyMC benchmark (nutpie + numba) ──────────────────────────────────────


def benchmark_pymc(
    model: pm.Model,
    model_name: str,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
) -> SamplingResult:
    """Benchmark PyMC with nutpie sampler and numba backend."""
    result = SamplingResult(
        backend="pymc-nutpie-numba",
        model_name=model_name,
        compile_time_s=None,
        sampling_time_s=0.0,
        total_time_s=0.0,
        draws=draws,
        tune=tune,
        chains=chains,
    )

    t_start = time.perf_counter()
    try:
        # Compilation: first call to compile the numba function
        t_compile_start = time.perf_counter()
        try:
            from pymc.model.transform.optimization import freeze_dims_and_data

            frozen = freeze_dims_and_data(model)
            logp_fn = frozen.logp_dlogp_function(ravel_inputs=True, mode="NUMBA")
            fn = logp_fn._pytensor_function
            fn.trust_input = True
            ip = model.initial_point()
            x0 = np.concatenate([np.atleast_1d(ip[v.name]) for v in logp_fn._grad_vars])
            fn(x0)
            t_compile_end = time.perf_counter()
            result.compile_time_s = t_compile_end - t_compile_start
        except Exception:
            # Compilation timing failed (e.g., ZeroSumNormal),
            # will be measured as part of total sampling time
            result.compile_time_s = None

        # Sampling
        t_sample_start = time.perf_counter()
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            nuts_sampler="nutpie",
            nuts_sampler_kwargs={"backend": "numba"},
            model=model,
            random_seed=42,
            progressbar=False,
        )
        t_sample_end = time.perf_counter()
        result.sampling_time_s = t_sample_end - t_sample_start
        if result.compile_time_s is not None:
            result.total_time_s = result.compile_time_s + result.sampling_time_s
        else:
            result.total_time_s = t_sample_end - t_start
        result.draws_per_sec = (chains * draws) / result.sampling_time_s

        _extract_diagnostics(idata, result)

    except Exception as e:
        result.error = str(e)
        result.total_time_s = time.perf_counter() - t_start

    return result


# ── Stan benchmark (cmdstanpy) ────────────────────────────────────────────


def benchmark_stan(
    stan_code: str,
    stan_data: dict,
    model_name: str,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
) -> SamplingResult:
    """Benchmark Stan via cmdstanpy."""
    result = SamplingResult(
        backend="stan-cmdstanpy",
        model_name=model_name,
        compile_time_s=None,
        sampling_time_s=0.0,
        total_time_s=0.0,
        draws=draws,
        tune=tune,
        chains=chains,
    )

    try:
        from cmdstanpy import CmdStanModel

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write Stan code
            stan_file = Path(tmpdir) / f"{model_name}.stan"
            stan_file.write_text(stan_code)

            # Write data
            data_file = Path(tmpdir) / f"{model_name}.json"
            data_file.write_text(json.dumps(stan_data))

            # Compilation
            t_compile_start = time.perf_counter()
            stan_model = CmdStanModel(stan_file=str(stan_file))
            t_compile_end = time.perf_counter()
            result.compile_time_s = t_compile_end - t_compile_start

            # Sampling
            t_sample_start = time.perf_counter()
            fit = stan_model.sample(
                data=str(data_file),
                iter_warmup=tune,
                iter_sampling=draws,
                chains=chains,
                seed=42,
                show_progress=False,
            )
            t_sample_end = time.perf_counter()
            result.sampling_time_s = t_sample_end - t_sample_start
            result.total_time_s = result.compile_time_s + result.sampling_time_s
            result.draws_per_sec = (chains * draws) / result.sampling_time_s

            # Extract diagnostics via arviz
            idata = az.from_cmdstanpy(fit)
            _extract_diagnostics(idata, result)

            # Divergences from cmdstanpy
            diag = fit.diagnose()
            if "divergent" in diag.lower():
                # Parse divergence count from diagnose output
                for line in diag.split("\n"):
                    if "divergent" in line.lower():
                        try:
                            result.divergences = int("".join(c for c in line if c.isdigit()) or "0")
                        except ValueError:
                            pass

    except Exception as e:
        result.error = str(e)

    return result


# ── Rust benchmark (via nutpie bridge) ────────────────────────────────────


def _build_shared_lib_portable(build_dir: Path) -> Path:
    """Build the compiled model as a shared library, handling macOS .dylib."""
    import platform

    build_dir = build_dir.resolve()
    ext = ".dylib" if platform.system() == "Darwin" else ".so"
    lib_path = build_dir / "target" / "release" / f"libpymc_compiled_model{ext}"

    gen_rs = build_dir / "src" / "generated.rs"
    if lib_path.exists() and lib_path.stat().st_mtime > gen_rs.stat().st_mtime:
        return lib_path

    result = subprocess.run(
        ["cargo", "build", "--release", "--lib"],
        cwd=build_dir,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to build shared library:\n{result.stderr}")

    if not lib_path.exists():
        raise RuntimeError(
            f'Shared library not found at {lib_path}. Ensure Cargo.toml has [lib] crate-type = ["cdylib"]'
        )
    return lib_path


def benchmark_rust(
    model: pm.Model,
    model_name: str,
    build_dir: Path,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
) -> SamplingResult:
    """Benchmark the AI-compiled Rust model via nutpie bridge."""
    result = SamplingResult(
        backend="rust-nutpie",
        model_name=model_name,
        compile_time_s=None,
        sampling_time_s=0.0,
        total_time_s=0.0,
        draws=draws,
        tune=tune,
        chains=chains,
    )

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from transalchemy.nutpie_bridge import _ensure_ffi_setup, _load_logp_fn

        # Compilation: build the shared library
        t_compile_start = time.perf_counter()
        _ensure_ffi_setup(build_dir)
        so_path = _build_shared_lib_portable(build_dir)
        t_compile_end = time.perf_counter()
        result.compile_time_s = t_compile_end - t_compile_start

        import nutpie
        from nutpie.compiled_pyfunc import from_pyfunc
        from pymc.blocking import DictToArrayBijection

        model_fn = model.logp_dlogp_function(ravel_inputs=True)
        ip = model.initial_point()
        x0 = DictToArrayBijection.map({v.name: ip[v.name] for v in model_fn._grad_vars}).data
        n_dim = len(x0)

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

        _lib_refs = []

        def make_logp_fn():
            logp_fn, lib = _load_logp_fn(so_path, n_dim)
            _lib_refs.append(lib)
            return logp_fn

        def make_expand_fn(seed1, seed2, chain):
            def expand_fn(x):
                result_dict = {}
                offset = 0
                for name, shape in zip(var_names, var_shapes):
                    size = int(np.prod(shape)) if shape else 1
                    result_dict[name] = x[offset : offset + size].reshape(shape)
                    offset += size
                return result_dict

            return expand_fn

        def make_initial_point(seed):
            ip_ = model.initial_point()
            return DictToArrayBijection.map({v.name: ip_[v.name] for v in model_fn._grad_vars}).data.astype(np.float64)

        compiled = from_pyfunc(
            ndim=n_dim,
            make_logp_fn=make_logp_fn,
            make_expand_fn=make_expand_fn,
            expanded_dtypes=var_dtypes,
            expanded_shapes=var_shapes,
            expanded_names=var_names,
            make_initial_point_fn=make_initial_point,
        )

        # Sampling
        t_sample_start = time.perf_counter()
        idata = nutpie.sample(
            compiled,
            draws=draws,
            tune=tune,
            chains=chains,
            seed=42,
            progress_bar=False,
        )
        t_sample_end = time.perf_counter()
        result.sampling_time_s = t_sample_end - t_sample_start
        result.total_time_s = result.compile_time_s + result.sampling_time_s
        result.draws_per_sec = (chains * draws) / result.sampling_time_s

        _extract_diagnostics(idata, result)

    except Exception as e:
        result.error = str(e)
        import traceback

        traceback.print_exc()

    return result


# ── Reporting ─────────────────────────────────────────────────────────────


def print_run_summary(run: SamplingResult, indent: str = "  "):
    """Print a single run's results."""
    if run.error:
        print(f"{indent}ERROR: {run.error[:120]}")
        return

    compile_str = f"{run.compile_time_s:.2f}s" if run.compile_time_s else "N/A"
    print(
        f"{indent}compile={compile_str}  sample={run.sampling_time_s:.2f}s  "
        f"total={run.total_time_s:.2f}s  draws/s={run.draws_per_sec:.0f}"
    )
    print(f"{indent}min_ess_bulk={_min_ess(run):.0f}  max_rhat={_max_rhat(run):.4f}  divergences={run.divergences}")


def print_model_summary(bench: BenchmarkResult):
    """Print summary for one model across all backends."""
    print(f"\n{'=' * 72}")
    print(f"  {bench.model_name} ({bench.n_params} params) — {bench.description}")
    print(f"{'=' * 72}")

    for label, runs in [
        ("PyMC (nutpie/numba)", bench.pymc_runs),
        ("Stan (cmdstanpy)", bench.stan_runs),
        ("Rust (nutpie)", bench.rust_runs),
    ]:
        if not runs:
            print(f"\n  {label}: skipped")
            continue

        valid_runs = [r for r in runs if r.error is None]
        print(f"\n  {label}: {len(valid_runs)}/{len(runs)} successful runs")

        for i, run in enumerate(runs):
            print(f"    Run {i + 1}:")
            print_run_summary(run, indent="      ")

        if len(valid_runs) >= 2:
            sample_times = [r.sampling_time_s for r in valid_runs]
            print(f"    Mean sample time: {np.mean(sample_times):.2f}s (+/- {np.std(sample_times):.2f}s)")

    # Speedup comparison
    pymc_valid = [r for r in bench.pymc_runs if r.error is None]
    rust_valid = [r for r in bench.rust_runs if r.error is None]
    stan_valid = [r for r in bench.stan_runs if r.error is None]

    if pymc_valid and rust_valid:
        pymc_mean = np.mean([r.sampling_time_s for r in pymc_valid])
        rust_mean = np.mean([r.sampling_time_s for r in rust_valid])
        print(f"\n  Rust vs PyMC sampling speedup: {pymc_mean / rust_mean:.2f}x")

    if stan_valid and rust_valid:
        stan_mean = np.mean([r.sampling_time_s for r in stan_valid])
        rust_mean = np.mean([r.sampling_time_s for r in rust_valid])
        print(f"  Rust vs Stan sampling speedup: {stan_mean / rust_mean:.2f}x")

    if pymc_valid and stan_valid:
        pymc_mean = np.mean([r.sampling_time_s for r in pymc_valid])
        stan_mean = np.mean([r.sampling_time_s for r in stan_valid])
        print(f"  PyMC vs Stan sampling speedup: {stan_mean / pymc_mean:.2f}x")


def print_final_summary(results: list[BenchmarkResult]):
    """Print final comparison table."""
    print(f"\n{'=' * 90}")
    print("FINAL SUMMARY")
    print(f"{'=' * 90}")

    header = (
        f"{'Model':<18} {'Backend':<20} {'Compile (s)':<13} "
        f"{'Sample (s)':<12} {'Draws/s':<10} {'Min ESS':<9} {'Max Rhat':<10} {'Divs':<5}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for bench in results:
        for label, runs in [
            ("pymc-nutpie", bench.pymc_runs),
            ("stan", bench.stan_runs),
            ("rust-nutpie", bench.rust_runs),
        ]:
            valid = [r for r in runs if r.error is None]
            if not valid:
                if runs:
                    print(f"{bench.model_name:<18} {label:<20} {'FAILED'}")
                continue

            # Use median run
            by_time = sorted(valid, key=lambda r: r.sampling_time_s)
            median_run = by_time[len(by_time) // 2]

            compile_str = f"{median_run.compile_time_s:.2f}" if median_run.compile_time_s is not None else "—"
            print(
                f"{bench.model_name:<18} {label:<20} {compile_str:<13} "
                f"{median_run.sampling_time_s:<12.2f} "
                f"{median_run.draws_per_sec:<10.0f} "
                f"{_min_ess(median_run):<9.0f} "
                f"{_max_rhat(median_run):<10.4f} "
                f"{median_run.divergences:<5}"
            )


def save_results(results: list[BenchmarkResult], output_path: Path):
    """Save results to JSON."""

    def _serialize(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.dtype):
            return str(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    data = []
    for bench in results:
        entry = {
            "model_name": bench.model_name,
            "n_params": bench.n_params,
            "description": bench.description,
            "pymc_runs": [asdict(r) for r in bench.pymc_runs],
            "stan_runs": [asdict(r) for r in bench.stan_runs],
            "rust_runs": [asdict(r) for r in bench.rust_runs],
        }
        data.append(entry)

    output_path.write_text(json.dumps(data, indent=2, default=_serialize))
    print(f"\nResults saved to {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyMC vs Stan vs Rust")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names (default: all with Rust impl)",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=3,
        help="Number of sampling runs per backend (default: 3)",
    )
    parser.add_argument("--draws", type=int, default=2000, help="Number of draws (default: 2000)")
    parser.add_argument("--tune", type=int, default=1000, help="Number of tuning draws (default: 1000)")
    parser.add_argument("--chains", type=int, default=4, help="Number of chains (default: 4)")
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results.json",
        help="Output JSON file (default: benchmarks/results.json)",
    )
    parser.add_argument(
        "--skip-stan",
        action="store_true",
        help="Skip Stan benchmarks",
    )
    parser.add_argument(
        "--skip-pymc",
        action="store_true",
        help="Skip PyMC benchmarks",
    )
    parser.add_argument(
        "--skip-rust",
        action="store_true",
        help="Skip Rust benchmarks",
    )
    args = parser.parse_args()

    # Determine which models to benchmark
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    else:
        # Only models with compiled Rust implementations
        model_names = [
            name for name in MODEL_REGISTRY if (COMPILED_MODELS_DIR / name / "src" / "generated.rs").exists()
        ]

    print("=" * 72)
    print("  Transalchemy Benchmark Suite: PyMC vs Stan vs Rust")
    print("=" * 72)
    print(f"  Models: {', '.join(model_names)}")
    print(f"  Config: {args.draws} draws, {args.tune} tune, {args.chains} chains")
    print(f"  Repeats: {args.n_repeats} per backend")
    print(f"  Output: {args.output}")
    print()

    all_results: list[BenchmarkResult] = []

    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            print(f"WARNING: Unknown model '{model_name}', skipping")
            continue

        make_fn, n_params = MODEL_REGISTRY[model_name]
        model, stan_code, stan_data, description = make_fn()
        build_dir = COMPILED_MODELS_DIR / model_name

        bench = BenchmarkResult(
            model_name=model_name,
            n_params=n_params,
            description=description,
        )

        print(f"\n{'─' * 72}")
        print(f"  Model: {model_name} — {description}")
        print(f"{'─' * 72}")

        # PyMC benchmarks
        if not args.skip_pymc:
            print(f"\n  [PyMC nutpie/numba] {args.n_repeats} runs...")
            for i in range(args.n_repeats):
                print(f"    Run {i + 1}/{args.n_repeats}...", end=" ", flush=True)
                run = benchmark_pymc(model, model_name, args.draws, args.tune, args.chains)
                bench.pymc_runs.append(run)
                if run.error:
                    print(f"FAILED: {run.error[:80]}")
                else:
                    print(
                        f"sample={run.sampling_time_s:.2f}s  "
                        f"draws/s={run.draws_per_sec:.0f}  "
                        f"min_ess={_min_ess(run):.0f}"
                    )

        # Stan benchmarks
        if not args.skip_stan and stan_code is not None:
            print(f"\n  [Stan cmdstanpy] {args.n_repeats} runs...")
            for i in range(args.n_repeats):
                print(f"    Run {i + 1}/{args.n_repeats}...", end=" ", flush=True)
                run = benchmark_stan(stan_code, stan_data, model_name, args.draws, args.tune, args.chains)
                bench.stan_runs.append(run)
                if run.error:
                    print(f"FAILED: {run.error[:80]}")
                else:
                    print(
                        f"compile={run.compile_time_s:.2f}s  "
                        f"sample={run.sampling_time_s:.2f}s  "
                        f"draws/s={run.draws_per_sec:.0f}  "
                        f"min_ess={_min_ess(run):.0f}"
                    )
        elif stan_code is None:
            print(f"\n  [Stan] skipped (no Stan equivalent for {model_name})")

        # Rust benchmarks
        if not args.skip_rust and build_dir.exists():
            print(f"\n  [Rust nutpie] {args.n_repeats} runs...")
            for i in range(args.n_repeats):
                print(f"    Run {i + 1}/{args.n_repeats}...", end=" ", flush=True)
                run = benchmark_rust(model, model_name, build_dir, args.draws, args.tune, args.chains)
                bench.rust_runs.append(run)
                if run.error:
                    print(f"FAILED: {run.error[:80]}")
                else:
                    print(
                        f"compile={run.compile_time_s:.2f}s  "
                        f"sample={run.sampling_time_s:.2f}s  "
                        f"draws/s={run.draws_per_sec:.0f}  "
                        f"min_ess={_min_ess(run):.0f}"
                    )
        elif not build_dir.exists():
            print(f"\n  [Rust] skipped (no compiled model at {build_dir})")

        print_model_summary(bench)
        all_results.append(bench)

    # Final summary
    print_final_summary(all_results)

    # Save results
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(all_results, output_path)


if __name__ == "__main__":
    main()
