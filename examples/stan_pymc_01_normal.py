"""Stan → PyMC transpilation example: simple Normal model.

Transpiles a Stan normal model to equivalent PyMC code,
validates that logp values match via BridgeStan.
"""

import numpy as np

from pymc_rust_compiler import transpile_stan_to_pymc

STAN_CODE = """
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

if __name__ == "__main__":
    # Generate some data
    rng = np.random.default_rng(42)
    y = rng.normal(3.0, 1.5, size=100)

    data = {"N": int(len(y)), "y": y.tolist()}

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

        # Test that the model actually works
        model = result.get_model(data)
        print(f"\nModel free RVs: {[rv.name for rv in model.free_RVs]}")
        print(f"Model observed RVs: {[rv.name for rv in model.observed_RVs]}")
    else:
        print(f"\nValidation errors: {result.validation_errors}")
        if result.pymc_code:
            print("\n--- Last Generated Code ---")
            print(result.pymc_code)
