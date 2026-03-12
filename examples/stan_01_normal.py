"""Stan → Rust compilation example: simple Normal model.

Equivalent to examples/01_normal.py but starting from Stan code.
"""

import numpy as np

from pymc_rust_compiler import compile_stan_model

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

    result = compile_stan_model(
        STAN_CODE,
        data=data,
        verbose=True,
    )

    print(f"\n{'=' * 60}")
    print(f"Success: {result.success}")
    print(f"Tool calls: {result.n_tool_calls}")
    print(f"Build attempts: {result.n_attempts}")
    print(f"Tokens: {result.token_usage}")
    if result.build_dir:
        print(f"Build dir: {result.build_dir}")
