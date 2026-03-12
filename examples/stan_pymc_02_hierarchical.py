"""Stan → PyMC transpilation example: hierarchical model.

Transpiles a Stan hierarchical (partial pooling) model to PyMC.
"""

import numpy as np

from pymc_rust_compiler import transpile_stan_to_pymc

STAN_CODE = """
data {
  int<lower=0> N;
  int<lower=1> J;
  array[N] int<lower=1,upper=J> group;
  array[N] real y;
}
parameters {
  real mu;
  real<lower=0> sigma;
  real<lower=0> tau;
  array[J] real theta;
}
model {
  mu ~ normal(0, 10);
  sigma ~ normal(0, 5);
  tau ~ normal(0, 5);
  theta ~ normal(mu, tau);
  for (n in 1:N)
    y[n] ~ normal(theta[group[n]], sigma);
}
"""

if __name__ == "__main__":
    # Generate hierarchical data
    rng = np.random.default_rng(42)
    J = 5
    N = 200
    true_mu = 3.0
    true_tau = 1.5
    true_sigma = 0.8

    true_theta = rng.normal(true_mu, true_tau, size=J)
    group = rng.integers(1, J + 1, size=N)  # 1-based for Stan
    y = rng.normal(true_theta[group - 1], true_sigma)

    data = {
        "N": N,
        "J": J,
        "group": group.tolist(),
        "y": y.tolist(),
    }

    result = transpile_stan_to_pymc(
        STAN_CODE,
        data=data,
        verbose=True,
    )

    print(f"\n{'=' * 60}")
    print(f"Success: {result.success}")
    print(f"Tool calls: {result.n_tool_calls}")
    print(f"Validation attempts: {result.n_attempts}")
    print(f"Tokens: {result.token_usage}")

    if result.success:
        print("\n--- Generated PyMC Code ---")
        print(result.pymc_code)

        model = result.get_model(data)
        print(f"\nModel free RVs: {[rv.name for rv in model.free_RVs]}")
        print(f"Model observed RVs: {[rv.name for rv in model.observed_RVs]}")
    else:
        print(f"\nValidation errors: {result.validation_errors}")
