"""Example 4: ZeroSumNormal with rank-3 arrays.

A saturated ANOVA model for retail sales data across
stores × weekdays × product categories.

The ZeroSumNormal effects sum to zero along constrained axes,
giving identifiable contrasts without a reference category.

PyMC model:
    grand_mean ~ Normal(0, 10)
    sigma_store ~ HalfNormal(2)
    sigma_day ~ HalfNormal(2)
    sigma_cat ~ HalfNormal(2)
    store_effect ~ ZeroSumNormal(sigma_store, shape=n_stores)
    day_effect ~ ZeroSumNormal(sigma_day, shape=n_days)
    interaction ~ ZeroSumNormal(sigma_cat, shape=(n_stores, n_days, n_categories), n_zerosum_axes=2)
    sigma_y ~ HalfNormal(5)
    y ~ Normal(grand_mean + store_effect[store] + day_effect[day]
               + interaction[store, day, category], sigma_y)  [observed]

Unconstrained parameters:
  - grand_mean: 1
  - log_sigma_store, log_sigma_day, log_sigma_cat, log_sigma_y: 4
  - store_effect: n_stores - 1 = 5
  - day_effect: n_days - 1 = 6
  - interaction: n_stores * (n_days - 1) * (n_categories - 1) = 6 * 6 * 3 = 108
  Total: 124 unconstrained parameters
"""

import numpy as np
import pymc as pm

from transpailer import compile_model

# --- Synthetic retail sales data ---
np.random.seed(314)

n_stores = 6
n_days = 7  # Mon-Sun
n_categories = 4  # e.g., Electronics, Clothing, Food, Home

store_names = [f"store_{i}" for i in range(n_stores)]
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
cat_names = ["Electronics", "Clothing", "Food", "Home"]

# True effects (sum-to-zero by construction)
true_grand_mean = 8.0  # log-scale baseline (~$3000/day)
true_sigma_store = 0.4
true_sigma_day = 0.3
true_sigma_cat = 0.5
true_sigma_y = 0.6

# Store effects (sum to zero)
raw_store = np.random.normal(0, true_sigma_store, n_stores)
true_store_effect = raw_store - raw_store.mean()

# Day effects (sum to zero): weekend boost
raw_day = np.array([-0.2, -0.1, 0.0, 0.05, 0.15, 0.35, 0.25])
raw_day += np.random.normal(0, 0.05, n_days)
true_day_effect = raw_day - raw_day.mean()

# Interaction: (stores, days, categories), zero-sum on last 2 axes
raw_interaction = np.random.normal(0, true_sigma_cat, (n_stores, n_days, n_categories))
# Remove means along day and category axes to enforce zero-sum
raw_interaction -= raw_interaction.mean(axis=-1, keepdims=True)
raw_interaction -= raw_interaction.mean(axis=-2, keepdims=True)
true_interaction = raw_interaction

# Generate observations: ~5 observations per cell
n_obs_per_cell = 5
records = []
for s in range(n_stores):
    for d in range(n_days):
        for c in range(n_categories):
            n = np.random.poisson(n_obs_per_cell) + 1
            mu = (
                true_grand_mean
                + true_store_effect[s]
                + true_day_effect[d]
                + true_interaction[s, d, c]
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

print(f"Dataset: {N} sales observations")
print(f"  {n_stores} stores × {n_days} days × {n_categories} categories")
print(f"  Interaction tensor shape: ({n_stores}, {n_days}, {n_categories})")
print(f"  y range: [{y_obs.min():.1f}, {y_obs.max():.1f}]")
print()

SOURCE = f"""
# Saturated ANOVA with ZeroSumNormal effects
# {n_stores} stores × {n_days} weekdays × {n_categories} product categories
# {N} total observations
grand_mean ~ Normal(0, 10)
sigma_store ~ HalfNormal(2)
sigma_day ~ HalfNormal(2)
sigma_cat ~ HalfNormal(2)
store_effect ~ ZeroSumNormal(sigma_store, shape={n_stores})
day_effect ~ ZeroSumNormal(sigma_day, shape={n_days})
interaction ~ ZeroSumNormal(sigma_cat, shape=({n_stores}, {n_days}, {n_categories}), n_zerosum_axes=2)
sigma_y ~ HalfNormal(5)
y ~ Normal(grand_mean + store_effect[store_idx] + day_effect[day_idx]
           + interaction[store_idx, day_idx, cat_idx], sigma_y), observed
"""

with pm.Model() as model:
    grand_mean = pm.Normal("grand_mean", mu=0, sigma=10)

    sigma_store = pm.HalfNormal("sigma_store", sigma=2)
    sigma_day = pm.HalfNormal("sigma_day", sigma=2)
    sigma_cat = pm.HalfNormal("sigma_cat", sigma=2)

    # 1D zero-sum effects
    store_effect = pm.ZeroSumNormal("store_effect", sigma=sigma_store, shape=n_stores)
    day_effect = pm.ZeroSumNormal("day_effect", sigma=sigma_day, shape=n_days)

    # Rank-3 interaction: (stores, days, categories)
    # n_zerosum_axes=2 → sums to zero along both day and category axes
    interaction = pm.ZeroSumNormal(
        "interaction",
        sigma=sigma_cat,
        shape=(n_stores, n_days, n_categories),
        n_zerosum_axes=2,
    )

    mu = (
        grand_mean
        + store_effect[store_idx]
        + day_effect[day_idx]
        + interaction[store_idx, day_idx, cat_idx]
    )

    sigma_y = pm.HalfNormal("sigma_y", sigma=5)
    pm.Normal("y", mu=mu, sigma=sigma_y, observed=y_obs)

n_free = sum(v.size for v in model.initial_point().values())
print(f"Free RVs: {[rv.name for rv in model.free_RVs]}")
print(f"Unconstrained parameters: {n_free}")
print(
    f"Transforms: {[(rv.name, type(model.rvs_to_transforms.get(rv)).__name__) for rv in model.free_RVs]}"
)
print()

result = compile_model(
    model,
    source_code=SOURCE,
    build_dir="compiled_models/zerosumnormal",
    verbose=True,
)

if result.success:
    print(f"\nCompilation successful in {result.n_attempts} attempt(s)!")

    # Sample with nutpie for comparison
    print("\nSampling with nutpie...")
    idata = pm.sample(
        draws=1000,
        tune=500,
        chains=4,
        nuts_sampler="nutpie",
        model=model,
        random_seed=42,
        progressbar=True,
    )

    import arviz as az

    print("\n--- Posterior summary (hyperparameters) ---")
    print(
        az.summary(
            idata,
            var_names=[
                "grand_mean",
                "sigma_store",
                "sigma_day",
                "sigma_cat",
                "sigma_y",
            ],
        )
    )

    print(
        f"\nTrue values: grand_mean={true_grand_mean}, "
        f"sigma_store={true_sigma_store}, sigma_day={true_sigma_day}, "
        f"sigma_cat={true_sigma_cat}, sigma_y={true_sigma_y}"
    )

    # Posterior plots
    axes = az.plot_posterior(
        idata,
        var_names=["grand_mean", "sigma_store", "sigma_day", "sigma_cat", "sigma_y"],
        ref_val={
            "grand_mean": [{"ref_val": true_grand_mean}],
            "sigma_store": [{"ref_val": true_sigma_store}],
            "sigma_day": [{"ref_val": true_sigma_day}],
            "sigma_cat": [{"ref_val": true_sigma_cat}],
            "sigma_y": [{"ref_val": true_sigma_y}],
        },
    )
    import matplotlib.pyplot as plt

    plt.tight_layout()
    plt.savefig("compiled_models/zerosumnormal/posterior_hyperparams.png", dpi=150)
    print("Saved: compiled_models/zerosumnormal/posterior_hyperparams.png")

    # Forest plot for store and day effects
    az.plot_forest(
        idata,
        var_names=["store_effect", "day_effect"],
        combined=True,
    )
    plt.tight_layout()
    plt.savefig("compiled_models/zerosumnormal/posterior_effects.png", dpi=150)
    print("Saved: compiled_models/zerosumnormal/posterior_effects.png")

else:
    print(f"\nCompilation FAILED after {result.n_attempts} attempts")
    for err in result.validation_errors[:5]:
        print(f"  - {err}")
