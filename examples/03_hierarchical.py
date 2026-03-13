"""Example 3: Hierarchical model (non-centered parameterization).

This is the real test — a multi-level model with group-level
and observation-level parameters, non-centered parameterization,
and transforms.

PyMC model:
    mu_a ~ Normal(0, 10)
    sigma_a ~ HalfNormal(5)
    a_offset ~ Normal(0, 1, shape=n_groups)
    a = mu_a + sigma_a * a_offset  (deterministic)
    b ~ Normal(0, 10)
    sigma_y ~ HalfNormal(5)
    y ~ Normal(a[group_idx] + b * x, sigma_y)  [observed]

12 unconstrained parameters for 8 groups.
"""

import numpy as np
import pymc as pm

from transpailer import compile_model

# Generate synthetic hierarchical data
np.random.seed(42)
n_groups = 8
n_per_group = np.random.randint(10, 30, size=n_groups)
N = n_per_group.sum()

true_mu_a = 1.5
true_sigma_a = 0.7
true_b = -0.8
true_sigma_y = 0.5

# Group intercepts
true_a = np.random.normal(true_mu_a, true_sigma_a, n_groups)

# Generate observations
group_idx = np.repeat(np.arange(n_groups), n_per_group)
x = np.random.binomial(1, 0.5, N).astype(float)
y_obs = true_a[group_idx] + true_b * x + np.random.normal(0, true_sigma_y, N)

SOURCE = f"""
# Hierarchical model (non-centered)
# {n_groups} groups, {N} total observations
mu_a ~ Normal(0, 10)
sigma_a ~ HalfNormal(5)
a_offset ~ Normal(0, 1, shape={n_groups})
a = mu_a + sigma_a * a_offset
b ~ Normal(0, 10)
sigma_y ~ HalfNormal(5)
y ~ Normal(a[group_idx] + b * x, sigma_y), observed
"""

with pm.Model() as model:
    mu_a = pm.Normal("mu_a", mu=0, sigma=10)
    sigma_a = pm.HalfNormal("sigma_a", sigma=5)
    a_offset = pm.Normal("a_offset", mu=0, sigma=1, shape=n_groups)
    a = pm.Deterministic("a", mu_a + sigma_a * a_offset)
    b = pm.Normal("b", mu=0, sigma=10)
    sigma_y = pm.HalfNormal("sigma_y", sigma=5)
    mu_y = a[group_idx] + b * x
    y = pm.Normal("y", mu=mu_y, sigma=sigma_y, observed=y_obs)

print(
    f"True: mu_a={true_mu_a}, sigma_a={true_sigma_a}, b={true_b}, sigma_y={true_sigma_y}"
)
print(f"Data: {n_groups} groups, {N} observations")
print(f"Group sizes: {n_per_group}")
print()

result = compile_model(
    model,
    source_code=SOURCE,
    build_dir="compiled_models/hierarchical",
    verbose=True,
)

if result.success:
    print(f"\nCompilation successful in {result.n_attempts} attempt(s)!")
    print("\nNow you can benchmark:")
    print("  python -c 'from transpailer.benchmark import *; ...'")
else:
    print(f"\nCompilation FAILED after {result.n_attempts} attempts")
    for err in result.validation_errors[:5]:
        print(f"  - {err}")
