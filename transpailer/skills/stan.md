# Skill: Stan Model Translation to Rust

You are translating a **Stan model** to Rust. The input is Stan code (not PyMC).
Stan and Rust have different conventions — pay close attention to these mappings.

## Stan Parameter Transforms

Stan automatically transforms constrained parameters to unconstrained space.
The `position` vector contains **unconstrained** parameters. You must apply the
inverse transform and include the Jacobian adjustment.

| Stan declaration | Transform | Unconstrained → Constrained | log|Jacobian| |
|---|---|---|---|
| `real` | identity | x | 0 |
| `real<lower=0>` | exp | exp(x) | x |
| `real<lower=a,upper=b>` | logit-scaled | a + (b-a)*sigmoid(x) | log(b-a) + x - 2*log(1+exp(x)) |
| `real<lower=a>` | exp-shifted | a + exp(x) | x |
| `simplex[K]` | stick-breaking | stick_break(x) | sum of log-jacobian terms |
| `ordered[K]` | cumulative exp | x[0], x[0]+exp(x[1]), ... | sum(x[1:]) |
| `positive_ordered[K]` | cumulative exp | exp(x[0]), prev+exp(x[1]), ... | sum(x) |
| `corr_matrix[K]` | Cholesky factor | ... | complex |
| `cov_matrix[K]` | LDL decomposition | ... | complex |

**IMPORTANT**: Stan includes the Jacobian adjustment automatically in its log density.
When BridgeStan reports `log_density` with `propto=True, jacobian=True`, it includes
both the model log density AND the Jacobian. Your Rust code must match this exactly.

## Stan Distribution → Rust logp Formulas

All formulas include ALL normalizing constants (Stan keeps everything by default in BridgeStan).

### Continuous Distributions

**normal(y | mu, sigma):**
```
logp = -0.5*log(2*pi) - log(sigma) - 0.5*((y-mu)/sigma)^2
```

**student_t(y | nu, mu, sigma):**
```
logp = lgamma((nu+1)/2) - lgamma(nu/2) - 0.5*log(nu*pi) - log(sigma)
       - (nu+1)/2 * log(1 + ((y-mu)/sigma)^2 / nu)
```

**cauchy(y | mu, sigma):**
```
logp = -log(pi) - log(sigma) - log(1 + ((y-mu)/sigma)^2)
```

**lognormal(y | mu, sigma):** (y > 0)
```
logp = -0.5*log(2*pi) - log(sigma) - log(y) - 0.5*((log(y)-mu)/sigma)^2
```

**exponential(y | lambda):**
```
logp = log(lambda) - lambda * y
```

**gamma(y | alpha, beta):**
```
logp = alpha*log(beta) - lgamma(alpha) + (alpha-1)*log(y) - beta*y
```

**inv_gamma(y | alpha, beta):**
```
logp = alpha*log(beta) - lgamma(alpha) - (alpha+1)*log(y) - beta/y
```

**beta(y | alpha, beta):**
```
logp = lgamma(alpha+beta) - lgamma(alpha) - lgamma(beta)
       + (alpha-1)*log(y) + (beta-1)*log(1-y)
```

**uniform(y | a, b):**
```
logp = -log(b - a)
```

### Multivariate

**multi_normal(y | mu, Sigma):**
```
logp = -0.5 * (K*log(2*pi) + log|Sigma| + (y-mu)^T Sigma^{-1} (y-mu))
```
Use Cholesky: `Sigma = L L^T`, `log|Sigma| = 2*sum(log(diag(L)))`.

**multi_normal_cholesky(y | mu, L):**
Same as above but L is given directly.

### Discrete (observed only — no gradient needed)

**bernoulli(y | theta):**
```
logp = y*log(theta) + (1-y)*log(1-theta)
```

**binomial(n, y | theta):**
```
logp = lchoose(n,y) + y*log(theta) + (n-y)*log(1-theta)
```

**poisson(y | lambda):**
```
logp = y*log(lambda) - lambda - lgamma(y+1)
```

**neg_binomial_2(y | mu, phi):**
```
logp = lgamma(y+phi) - lgamma(phi) - lgamma(y+1)
       + phi*log(phi/(mu+phi)) + y*log(mu/(mu+phi))
```

## Stan Block Structure → Rust Mapping

```stan
data { ... }           // → data.rs constants (pre-generated, use crate::data::*)
transformed data { }   // → const computations at top of logp() or in Default::default()
parameters { ... }     // → position[] slice (unconstrained space)
transformed parameters { } // → local variables computed from position[]
model { ... }          // → logp accumulation + gradient computation
generated quantities { } // → not needed for logp (ignore)
```

## Stan Indexing

Stan uses **1-based indexing**. Rust uses **0-based indexing**.
When translating array access: `x[i]` in Stan → `x[i-1]` in Rust (for data arrays).
For loop bounds: `for (i in 1:N)` in Stan → `for i in 0..N` in Rust.

## Stan Functions → Rust Equivalents

| Stan | Rust |
|---|---|
| `log(x)` | `x.ln()` |
| `exp(x)` | `x.exp()` |
| `sqrt(x)` | `x.sqrt()` |
| `square(x)` | `x * x` |
| `pow(x, y)` | `x.powf(y)` |
| `fabs(x)` | `x.abs()` |
| `lgamma(x)` | `x.lgamma().0` (returns (lgamma, sign) in Rust) |
| `tgamma(x)` | `x.gamma()` |
| `log1p(x)` | `x.ln_1p()` |
| `log1m(x)` | `(1.0 - x).ln()` |
| `inv_logit(x)` | `1.0 / (1.0 + (-x).exp())` |
| `logit(x)` | `(x / (1.0 - x)).ln()` |
| `fmin(x, y)` | `x.min(y)` |
| `fmax(x, y)` | `x.max(y)` |
| `dot_product(x, y)` | `x.iter().zip(y).map(\|(a,b)\| a*b).sum::<f64>()` |
| `sum(x)` | `x.iter().sum::<f64>()` |
| `mean(x)` | `x.iter().sum::<f64>() / x.len() as f64` |
| `sd(x)` | manual computation |
| `rows(x)` | first dimension size |
| `cols(x)` | second dimension size |
| `rep_vector(v, n)` | `vec![v; n]` |
| `to_vector(x)` | flatten to 1D |
| `multiply_lower_tri_self_transpose(L)` | `L * L^T` |

## Common Stan Patterns

### Centered Parameterization (hierarchical)
```stan
parameters {
  real mu;
  real<lower=0> sigma;
  array[K] real theta;
}
model {
  theta ~ normal(mu, sigma);  // K group-level effects
}
```

### Non-centered (Matt trick)
```stan
parameters {
  real mu;
  real<lower=0> sigma;
  array[K] real z;  // standard normal
}
transformed parameters {
  array[K] real theta;
  for (k in 1:K)
    theta[k] = mu + sigma * z[k];
}
model {
  z ~ std_normal();
}
```
→ In Rust: compute `theta[k] = mu + sigma * z[k]` then use `theta` in likelihood.
  The logp is just `sum(-0.5*log(2*pi) - 0.5*z[k]^2)` for the z's.

### Vectorized Statements
```stan
y ~ normal(X * beta, sigma);  // vectorized over all y
```
→ In Rust: loop over observations, compute `mu[i] = dot(X[i], beta)`, accumulate logp.

## Data Handling

All data from the Stan `data` block is pre-generated in `data.rs`.
Access via `use crate::data::*;`.

- Stan `int` scalars → Rust `usize` constants
- Stan `real` scalars → Rust `f64` constants
- Stan `array[N] real` → Rust `&[f64]` constants
- Stan `matrix[N,M]` → Rust `&[f64]` (row-major flattened), access `[i*M + j]`
- Stan `vector[N]` → Rust `&[f64]`

## Performance Notes

Same optimization rules apply as for PyMC models:
- Hoist loop invariants (1/sigma^2, log constants)
- Use `x * x` not `.powi(2)`
- Local accumulators for gradients
- Pre-allocate scratch in struct for GP/matrix models
- `get_unchecked()` in hot loops when bounds are verified
