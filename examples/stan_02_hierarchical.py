"""Stan → Rust compilation example: hierarchical (Eight Schools) model.

Classic Eight Schools example — non-centered parameterization.
"""

from pymc_rust_compiler import compile_stan_model

STAN_CODE = """
data {
  int<lower=0> J;
  array[J] real y;
  array[J] real<lower=0> sigma;
}
parameters {
  real mu;
  real<lower=0> tau;
  array[J] real eta;
}
transformed parameters {
  array[J] real theta;
  for (j in 1:J)
    theta[j] = mu + tau * eta[j];
}
model {
  mu ~ normal(0, 5);
  tau ~ cauchy(0, 5);
  eta ~ std_normal();
  y ~ normal(theta, sigma);
}
"""

if __name__ == "__main__":
    # Eight Schools data
    data = {
        "J": 8,
        "y": [28, 8, -3, 7, -1, 1, 18, 12],
        "sigma": [15, 10, 16, 11, 9, 11, 10, 18],
    }

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
