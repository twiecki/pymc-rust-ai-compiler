import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # PyMC → Rust AI Compiler

    **Compile PyMC models to optimized Rust via Claude API — automatically.**

    This project takes a standard `pm.Model()`, extracts everything the LLM needs
    (parameters, transforms, computational graph, validation data), sends it to Claude,
    and gets back a complete, validated Rust implementation of `logp` + gradients.

    The generated Rust code implements the `CpuLogpFunc` trait from
    [`nuts-rs`](https://github.com/pymc-devs/nuts-rs), so it plugs directly into
    the NUTS sampler — no Python overhead during sampling.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## How It Works

    ```
    ┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
    │  pm.Model()  │────▶│   Exporter    │────▶│  Claude API  │────▶│  Rust Code   │
    │              │     │              │     │              │     │              │
    │ - free_RVs   │     │ - params     │     │ - system     │     │ - logp()     │
    │ - transforms │     │ - logp graph │     │   prompt     │     │ - gradient   │
    │ - logp()     │     │ - data       │     │ - model      │     │ - CpuLogpFunc│
    │ - observed   │     │ - test pts   │     │   context    │     │   trait impl  │
    └─────────────┘     └──────────────┘     └─────────────┘     └──────┬───────┘
                                                                        │
                                              ┌──────────────┐          │
                                              │   Validate    │◀────────┘
                                              │              │
                                              │ logp match?  │──── yes ──▶ Done!
                                              │ grad match?  │──── no ───▶ Retry with
                                              └──────────────┘            error feedback
    ```

    **Key insight**: `pm.Model()` already contains *everything* needed — parameters,
    transforms, shapes, the full computational graph, and logp functions.
    We just need to read it out and present it to the LLM in the right format.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 1: Define a PyMC Model

    Nothing special needed — just a standard PyMC model. Here's a linear regression:
    """)
    return


@app.cell
def _():
    import numpy as np
    import pymc as pm

    # Generate synthetic data
    np.random.seed(42)
    N = 200
    true_alpha, true_beta, true_sigma = 2.5, -1.3, 0.8

    x_data = np.linspace(0, 1, N)
    y_data = true_alpha + true_beta * x_data + np.random.normal(0, true_sigma, N)

    with pm.Model() as linreg_model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=5)
        mu = alpha + beta * x_data
        pm.Normal("y", mu=mu, sigma=sigma, observed=y_data)

    print(f"Model: {len(linreg_model.free_RVs)} free parameters, {len(linreg_model.observed_RVs)} observed")
    print(f"Parameters: {[rv.name for rv in linreg_model.free_RVs]}")
    print(f"Transforms: {[type(linreg_model.rvs_to_transforms.get(rv)).__name__ for rv in linreg_model.free_RVs]}")
    return (linreg_model,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 2: Extract Model Context

    The `RustModelExporter` reads the PyMC model and extracts:

    - **Parameter layout**: names, transforms, shapes, position in the unconstrained vector
    - **Computational graph**: PyTensor's `debugprint` of the full logp
    - **Observed data**: `y` values with exact f64 precision
    - **Covariate data**: `x` arrays, group indices, etc.
    - **Validation points**: logp + gradient values at multiple test points
    """)
    return


@app.cell
def _(linreg_model):
    from transpailer.exporter import RustModelExporter

    SOURCE = """alpha ~ Normal(0, 10)
    beta ~ Normal(0, 10)
    sigma ~ HalfNormal(5)
    y ~ Normal(alpha + beta * x, sigma), observed"""

    exporter = RustModelExporter(linreg_model, source_code=SOURCE)
    ctx = exporter.context

    print(f"Parameters ({ctx.n_params} unconstrained):")
    for p in ctx.params:
        transform_str = f" [{p.transform}]" if p.transform else ""
        print(f"  {p.name}: shape={p.shape}, size={p.size}{transform_str}")

    print("\nObserved data:")
    for name2, info2 in ctx.observed_data.items():
        print(f"  {name2}: n={info2['n']}, range=[{info2['min']:.3f}, {info2['max']:.3f}]")

    print("\nCovariates:")
    for name3, info3 in ctx.covariate_data.items():
        is_idx = info3.get("is_index_array", False)
        label = f" [INDEX ARRAY, {info3['n_groups']} groups]" if is_idx else ""
        print(f"  {name3}: n={info3['n']}, range=[{info3['min']:.3f}, {info3['max']:.3f}]{label}")

    print(f"\nValidation points: 1 initial + {len(ctx.extra_points)} extra")
    print(f"  Initial logp = {ctx.initial_point.logp:.6f}")
    return ctx, exporter


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 3: The LLM Prompt

    Everything gets assembled into a single prompt for Claude. It includes:

    1. The PyMC source code
    2. Parameter layout (which index maps to which parameter)
    3. Data constants available in `data.rs`
    4. The full PyTensor computational graph
    5. Exact validation values the generated code must match
    """)
    return


@app.cell
def _(exporter, mo):
    prompt = exporter.to_prompt()
    # Show first ~2000 chars
    preview = prompt[:2000] + "\n\n... [truncated] ...\n"

    mo.md(f"**Prompt length**: {len(prompt):,} characters\n\n```\n{preview}\n```")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 4: Generated Rust Code

    Claude generates a complete Rust file implementing `CpuLogpFunc`.
    Here's what the linear regression output looks like:
    """)
    return


@app.cell
def _(mo):
    with open("compiled_models/linreg/src/generated.rs") as f:
        rust_code = f.read()

    mo.md(f"```rust\n{rust_code}\n```")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Notice:

    - **Single-pass computation**: residuals, their sum, and sum-of-squares are
      accumulated in one loop — cache-friendly, no extra allocations
    - **Analytical gradients**: no autograd overhead, every derivative is hand-derived
    - **Transform handling**: `sigma` lives in log-space (`sigma_log = position[2]`),
      with the Jacobian adjustment `+ sigma_log` added to logp
    - **Data from `data.rs`**: the `Y_DATA` and `X_0_DATA` arrays are pre-generated
      with full f64 precision — the LLM never sees or rounds the actual numbers
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 5: Validation

    Before declaring success, the compiled binary is tested against PyMC reference values:

    - **logp**: must match within `rel_err < 1e-4` (constant offsets are OK since NUTS doesn't care)
    - **gradients**: must match within `rel_err < 1e-3` (NUTS depends on correct gradients)

    If validation fails, the errors are fed back to Claude for a retry (up to 3 attempts).
    """)
    return


@app.cell
def _(ctx, mo):
    rows = []
    rows.append("| Point | PyMC logp | Gradient[0] | Gradient[1] | Gradient[2] |")
    rows.append("|-------|-----------|-------------|-------------|-------------|")

    pts = [("initial", ctx.initial_point)] + [(f"extra_{i}", p) for i, p in enumerate(ctx.extra_points)]
    for name, vp in pts:
        g = vp.dlogp
        rows.append(f"| {name} | {vp.logp:.4f} | {g[0]:.4f} | {g[1]:.4f} | {g[2]:.4f} |")

    mo.md("**Validation reference values:**\n\n" + "\n".join(rows))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Sampling with nutpie

    Let's actually run the sampler on both models — the linear regression and
    a hierarchical model. `nutpie` is PyMC's Rust-based NUTS backend.
    """)
    return


@app.cell
def _(linreg_model):
    import pymc as _pm

    linreg_idata = _pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        nuts_sampler="nutpie",
        model=linreg_model,
        random_seed=42,
        progressbar=False,
    )
    print(f"Sampling done: {linreg_idata.posterior.dims}")
    return (linreg_idata,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Posterior — Linear Regression

    The true values were `alpha=2.5`, `beta=-1.3`, `sigma=0.8` — marked as
    reference lines below.
    """)
    return


@app.cell
def _(linreg_idata):
    import arviz as az
    import matplotlib.pyplot as plt

    az.style.use("arviz-darkgrid")
    az.plot_trace(linreg_idata, var_names=["alpha", "beta", "sigma"], compact=True)
    plt.tight_layout()
    plt.gcf()
    return az, plt


@app.cell
def _(az, linreg_idata, plt):
    az.plot_posterior(
        linreg_idata,
        var_names=["alpha", "beta", "sigma"],
        ref_val=[2.5, -1.3, 0.8],
    )
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Hierarchical Model

    Non-centered parameterization with 8 groups — the harder test case.
    """)
    return


@app.cell
def _():
    import numpy as _np
    import pymc as _pm

    _np.random.seed(123)
    n_groups = 8
    n_per_group = 20
    _N = n_groups * n_per_group

    true_mu_a, true_sigma_a = 3.0, 1.5
    true_b, true_sigma_y = -0.8, 0.5

    group_idx = _np.repeat(_np.arange(n_groups), n_per_group)
    true_a = _np.random.normal(true_mu_a, true_sigma_a, n_groups)
    x_h = _np.random.uniform(0, 1, _N)
    y_h = true_a[group_idx] + true_b * x_h + _np.random.normal(0, true_sigma_y, _N)

    with _pm.Model() as hierarchical_model:
        mu_a = _pm.Normal("mu_a", mu=0, sigma=10)
        sigma_a = _pm.HalfNormal("sigma_a", sigma=5)
        a_offset = _pm.Normal("a_offset", mu=0, sigma=1, shape=n_groups)
        a = mu_a + sigma_a * a_offset
        b = _pm.Normal("b", mu=0, sigma=10)
        sigma_y = _pm.HalfNormal("sigma_y", sigma=5)
        _pm.Normal("y_obs", mu=a[group_idx] + b * x_h, sigma=sigma_y, observed=y_h)

    print(f"Hierarchical model: {len(hierarchical_model.free_RVs)} free params, {_N} observations, {n_groups} groups")
    return (hierarchical_model,)


@app.cell
def _(hierarchical_model):
    import pymc as _pm

    hier_idata = _pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        nuts_sampler="nutpie",
        model=hierarchical_model,
        random_seed=42,
        progressbar=False,
    )
    print(f"Sampling done: {hier_idata.posterior.dims}")
    return (hier_idata,)


@app.cell
def _(az, hier_idata, plt):
    az.plot_posterior(
        hier_idata,
        var_names=["mu_a", "sigma_a", "b", "sigma_y"],
        ref_val=[3.0, 1.5, -0.8, 0.5],
    )
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(az, hier_idata, plt):
    az.plot_forest(
        hier_idata,
        var_names=["a_offset"],
        combined=True,
    )
    plt.suptitle("Group-level offsets (a_offset)", y=1.02)
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Hierarchical Models: The Hard Part

    The real test is multi-level models with group indexing. Consider this non-centered
    parameterization with 8 groups:

    ```python
    mu_a ~ Normal(0, 10)
    sigma_a ~ HalfNormal(5)
    a_offset ~ Normal(0, 1, shape=8)
    a = mu_a + sigma_a * a_offset        # deterministic
    b ~ Normal(0, 10)
    sigma_y ~ HalfNormal(5)
    y ~ Normal(a[group_idx] + b * x, sigma_y), observed
    ```

    **Challenges for the LLM:**

    1. **12 unconstrained parameters** (mu_a, log_sigma_a, 8 offsets, b, log_sigma_y)
    2. **Group indexing**: `a[group_idx]` requires casting float indices to `usize`
    3. **Non-centered parameterization**: `a = mu_a + sigma_a * a_offset` means
       gradients flow through multiple parameters
    4. **Observed likelihood**: 170 observations contribute the dominant logp term

    The compiler detects integer index arrays automatically and provides explicit
    mapping hints in the prompt (`group_idx` in source → `X_1_DATA`).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Architecture

    ```
    transpailer/
    ├── exporter.py      # Extract: pm.Model() → ModelContext → LLM prompt
    ├── compiler.py      # Generate + Build + Validate loop
    └── benchmark.py     # Compare: nutpie vs AI-compiled Rust

    compiled_models/
    ├── normal/          # 2 params  → ~100 lines of Rust
    ├── linreg/          # 3 params  → ~120 lines of Rust
    └── hierarchical/    # 12 params → ~200 lines of Rust
    ```

    The generated Rust code implements `CpuLogpFunc` from `nuts-rs`, which means it
    plugs directly into the same NUTS sampler that `nutpie` uses — but without any
    Python/PyTensor overhead during sampling.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why This Works

    1. **`pm.Model()` is self-describing**: parameters, transforms, shapes, the full
       logp computation graph — it's all there, no instrumentation needed

    2. **LLMs are good at code generation**: given the right context (computational graph,
       parameter layout, data mapping, validation targets), Claude reliably generates
       correct logp + gradient code

    3. **Validation closes the loop**: if the generated code is wrong, we know immediately
       (logp/gradient mismatch) and can retry with error feedback

    4. **Data separation**: observed data and covariates live in `data.rs` with full f64
       precision — the LLM only generates the *logic*, never touches the numbers

    5. **The reward is huge**: pure Rust logp+gradient means no GIL, no Python overhead,
       no autograd tape — just raw math at native speed
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Try It

    ```python
    from transpailer import compile_model

    result = compile_model(
        model,
        source_code="...",       # optional but helps
        build_dir="compiled/",   # where to put the Rust project
        verbose=True,
    )

    if result.success:
        print(f"Done in {result.n_attempts} attempt(s)!")
        # Run: cargo run --release --bin sample
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
