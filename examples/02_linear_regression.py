"""Example 2: Linear regression — y = alpha + beta * x + noise.

PyMC model:
    alpha ~ Normal(0, 10)
    beta ~ Normal(0, 10)
    sigma ~ HalfNormal(5)
    y ~ Normal(alpha + beta * x, sigma)  [observed]

3 unconstrained parameters: alpha, beta, log(sigma).
"""

import numpy as np
import pymc as pm

from transpailer import compile_model

# Generate synthetic data
np.random.seed(42)
N = 200
true_alpha, true_beta, true_sigma = 2.5, -1.3, 0.8
x = np.linspace(0, 10, N)
y_obs = true_alpha + true_beta * x + np.random.normal(0, true_sigma, N)

SOURCE = """
alpha ~ Normal(0, 10)
beta ~ Normal(0, 10)
sigma ~ HalfNormal(5)
y ~ Normal(alpha + beta * x, sigma), observed
"""

with pm.Model() as model:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)
    mu = alpha + beta * x
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)

print(f"True values: alpha={true_alpha}, beta={true_beta}, sigma={true_sigma}")
print(f"Data: n={N}")
print()

result = compile_model(
    model,
    source_code=SOURCE,
    build_dir="compiled_models/linreg",
    verbose=True,
)

if result.success:
    print(f"\nCompilation successful in {result.n_attempts} attempt(s)!")
else:
    print(f"\nCompilation FAILED after {result.n_attempts} attempts")
    for err in result.validation_errors[:5]:
        print(f"  - {err}")
