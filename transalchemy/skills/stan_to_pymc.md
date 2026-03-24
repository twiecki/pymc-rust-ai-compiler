# Skill: Stan → PyMC Model Translation

You are translating a **Stan model** to **PyMC (v5+)**. Pay close attention to
the differences between Stan and PyMC conventions.

## Core Translation Principles

These rules are **mandatory** — violating them produces slow, unidiomatic, or broken models:

1. **Always vectorize.** Never use Python `for` loops to create distributions or build up
   likelihood terms. Use `shape`/`dims`, array indexing, and numpy-like broadcasting instead.
   A Stan `for` loop over independent iterations should become a single vectorized distribution
   call in PyMC.

2. **Use `pytensor.scan` for sequential dependencies.** When step `t` depends on step `t-1`
   (AR, MA, state-space, ODEs), use `pytensor.scan` — never a Python `for` loop. Python loops
   unroll the graph and create one node per iteration; `scan` compiles the loop into a single
   graph node. Use `pm.AR` for standard AR(p) models.

3. **Use `pt.switch`/`ifelse` for branching.** You **cannot** use Python `if/else` on symbolic
   PyTensor variables — they have no concrete value at graph construction time. Use `pt.switch`
   (element-wise, like `np.where`) for most cases, or `pytensor.ifelse.ifelse` for scalar
   conditions with expensive branches.

4. **Prefer numpy-like syntax throughout.** Use `pt.dot`, `pt.sum`, `pt.stack`, `pt.concatenate`,
   broadcasting, and advanced indexing. Write array expressions the way you would in NumPy.
   Avoid building results element-by-element or accumulating in Python lists.

## Stan Block → PyMC Mapping

| Stan Block | PyMC Equivalent |
|---|---|
| `data { ... }` | Function arguments or module-level constants (numpy arrays) |
| `transformed data { ... }` | Python computations before `pm.Model()` |
| `parameters { ... }` | `pm.Distribution(...)` calls inside `with pm.Model()` |
| `transformed parameters { ... }` | `pm.Deterministic(...)` or pytensor expressions |
| `model { ... }` | Prior + likelihood specifications |
| `generated quantities { ... }` | `pm.Deterministic(...)` or post-sampling transforms |

## Parameter Type Mapping

| Stan Declaration | PyMC Equivalent |
|---|---|
| `real mu;` | `mu = pm.Flat("mu")` or use a prior |
| `real<lower=0> sigma;` | `sigma = pm.HalfFlat("sigma")` or `pm.HalfNormal(...)` etc. |
| `real<lower=a,upper=b> x;` | `x = pm.Uniform("x", lower=a, upper=b)` |
| `real<lower=0> x;` | Use positive distribution or `pm.HalfFlat("x")` |
| `vector[K] x;` | `x = pm.Normal("x", ..., shape=K)` |
| `array[K] real x;` | `x = pm.Normal("x", ..., shape=K)` |
| `ordered[K] x;` | `x = pm.Normal("x", ..., shape=K, transform=pm.distributions.transforms.ordered)` |
| `simplex[K] x;` | `x = pm.Dirichlet("x", a=np.ones(K))` |
| `corr_matrix[K] R;` | `pm.LKJCorr("R", n=2, eta=K)` or `pm.LKJCholeskyCov(...)` |
| `cov_matrix[K] S;` | `pm.LKJCholeskyCov(...)` or manual construction |
| `cholesky_factor_corr[K] L;` | `pm.LKJCholeskyCov(...)` |
| `positive_ordered[K] x;` | ordered transform on positive support |

## Distribution Mapping

### Continuous

| Stan | PyMC |
|---|---|
| `normal(mu, sigma)` | `pm.Normal("x", mu=mu, sigma=sigma)` |
| `std_normal()` | `pm.Normal("x", mu=0, sigma=1)` |
| `student_t(nu, mu, sigma)` | `pm.StudentT("x", nu=nu, mu=mu, sigma=sigma)` |
| `cauchy(mu, sigma)` | `pm.Cauchy("x", alpha=mu, beta=sigma)` |
| `double_exponential(mu, sigma)` | `pm.Laplace("x", mu=mu, b=sigma)` |
| `lognormal(mu, sigma)` | `pm.LogNormal("x", mu=mu, sigma=sigma)` |
| `exponential(lambda)` | `pm.Exponential("x", lam=lambda_)` |
| `gamma(alpha, beta)` | `pm.Gamma("x", alpha=alpha, beta=beta)` |
| `inv_gamma(alpha, beta)` | `pm.InverseGamma("x", alpha=alpha, beta=beta)` |
| `beta(alpha, beta)` | `pm.Beta("x", alpha=alpha, beta=beta)` |
| `uniform(a, b)` | `pm.Uniform("x", lower=a, upper=b)` |
| `half_normal(sigma)` | `pm.HalfNormal("x", sigma=sigma)` (**Stan: `normal(0, sigma)` on `<lower=0>`**) |
| `half_cauchy(0, sigma)` | `pm.HalfCauchy("x", beta=sigma)` |
| `pareto(ymin, alpha)` | `pm.Pareto("x", alpha=alpha, m=ymin)` |
| `weibull(alpha, sigma)` | `pm.Weibull("x", alpha=alpha, beta=sigma)` |
| `von_mises(mu, kappa)` | `pm.VonMises("x", mu=mu, kappa=kappa)` |
| `multi_normal(mu, Sigma)` | `pm.MvNormal("x", mu=mu, cov=Sigma)` |
| `multi_normal_cholesky(mu, L)` | `pm.MvNormal("x", mu=mu, chol=L)` |
| `dirichlet(alpha)` | `pm.Dirichlet("x", a=alpha)` |
| `lkj_corr(eta)` | `pm.LKJCorr("R", n=dim, eta=eta)` |
| `wishart(nu, S)` | `pm.Wishart("x", nu=nu, V=S)` |

### Discrete

| Stan | PyMC |
|---|---|
| `bernoulli(theta)` | `pm.Bernoulli("x", p=theta, observed=...)` |
| `bernoulli_logit(alpha)` | `pm.Bernoulli("x", logit_p=alpha, observed=...)` |
| `binomial(n, theta)` | `pm.Binomial("x", n=n, p=theta, observed=...)` |
| `poisson(lambda)` | `pm.Poisson("x", mu=lambda_, observed=...)` |
| `neg_binomial(alpha, beta)` | `pm.NegativeBinomial("x", alpha=alpha, beta=beta, observed=...)` |
| `neg_binomial_2(mu, phi)` | `pm.NegativeBinomial("x", mu=mu, alpha=phi, observed=...)` |
| `categorical(theta)` | `pm.Categorical("x", p=theta, observed=...)` |
| `multinomial(n, theta)` | `pm.Multinomial("x", n=n, p=theta, observed=...)` |
| `ordered_logistic(eta, c)` | `pm.OrderedLogistic("x", eta=eta, cutpoints=c, observed=...)` |

## Stan → PyMC Idiom Translation

### Vectorized Sampling Statements

```stan
// Stan: vectorized
y ~ normal(mu, sigma);
```
```python
# PyMC: observed kwarg
y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y_data)
```

### Loop-based Sampling

```stan
// Stan: loop
for (i in 1:N)
    y[i] ~ normal(mu[i], sigma);
```
```python
# PyMC: vectorized (preferred)
y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y_data)
```

### Hierarchical (Centered)

```stan
parameters {
  real mu_group;
  real<lower=0> sigma_group;
  array[K] real theta;
}
model {
  mu_group ~ normal(0, 10);
  sigma_group ~ half_normal(5);
  theta ~ normal(mu_group, sigma_group);
  y ~ normal(theta[group_idx], sigma_y);
}
```
```python
with pm.Model() as model:
    mu_group = pm.Normal("mu_group", mu=0, sigma=10)
    sigma_group = pm.HalfNormal("sigma_group", sigma=5)
    theta = pm.Normal("theta", mu=mu_group, sigma=sigma_group, shape=K)
    y_obs = pm.Normal("y", mu=theta[group_idx], sigma=sigma_y, observed=y_data)
```

### Non-centered (Matt Trick)

```stan
parameters {
  real mu_group;
  real<lower=0> sigma_group;
  array[K] real z;
}
transformed parameters {
  array[K] real theta;
  for (k in 1:K)
    theta[k] = mu_group + sigma_group * z[k];
}
model {
  z ~ std_normal();
}
```
```python
with pm.Model() as model:
    mu_group = pm.Normal("mu_group", mu=0, sigma=10)
    sigma_group = pm.HalfNormal("sigma_group", sigma=5)
    theta = pm.Normal("theta", mu=mu_group, sigma=sigma_group, shape=K)
    # PyMC handles non-centering automatically — just use centered form
    # Or explicitly:
    z = pm.Normal("z", mu=0, sigma=1, shape=K)
    theta = pm.Deterministic("theta", mu_group + sigma_group * z)
```

### Linear Regression

```stan
data {
  int<lower=0> N;
  int<lower=0> K;
  matrix[N, K] X;
  vector[N] y;
}
parameters {
  vector[K] beta;
  real<lower=0> sigma;
}
model {
  y ~ normal(X * beta, sigma);
}
```
```python
with pm.Model() as model:
    beta = pm.Normal("beta", mu=0, sigma=10, shape=K)
    sigma = pm.HalfNormal("sigma", sigma=5)
    mu = pm.math.dot(X_data, beta)
    y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y_data)
```

### Target += (Custom log-probability)

```stan
model {
  target += -0.5 * square(x);  // custom logp term
}
```
```python
# PyMC: use pm.Potential
pm.Potential("custom", -0.5 * x**2)
```

### Truncated Distributions

```stan
real<lower=0> y;
y ~ normal(mu, sigma) T[0,];  // truncated below at 0
```
```python
y = pm.TruncatedNormal("y", mu=mu, sigma=sigma, lower=0)
```

## Stan Functions → PyMC/PyTensor Equivalents

| Stan | PyMC/PyTensor |
|---|---|
| `log(x)` | `pt.log(x)` |
| `exp(x)` | `pt.exp(x)` |
| `sqrt(x)` | `pt.sqrt(x)` |
| `square(x)` | `x**2` |
| `pow(x, y)` | `x**y` |
| `fabs(x)` | `pt.abs(x)` |
| `inv_logit(x)` | `pm.math.invlogit(x)` |
| `logit(x)` | `pm.math.logit(x)` |
| `log1p(x)` | `pt.log1p(x)` |
| `dot_product(x, y)` | `pt.dot(x, y)` |
| `sum(x)` | `pt.sum(x)` or `x.sum()` |
| `mean(x)` | `pt.mean(x)` |
| `sd(x)` | `pt.std(x)` |
| `rows(x)` | `x.shape[0]` |
| `cols(x)` | `x.shape[1]` |
| `rep_vector(v, n)` | `pt.full(n, v)` |
| `to_vector(x)` | `x.ravel()` |
| `append_row(x, y)` | `pt.concatenate([x, y])` |
| `segment(x, i, n)` | `x[i-1:i-1+n]` (adjust for 0-based) |
| `cholesky_decompose(M)` | `pt.linalg.cholesky(M)` |
| `inverse(M)` | `pt.linalg.inv(M)` |
| `determinant(M)` | `pt.linalg.det(M)` |
| `quad_form(A, b)` | `pt.dot(b.T, pt.dot(A, b))` |
| `multiply(A, B)` | `A * B` (element-wise) |

## Critical: Half-Distribution log(2) Mismatch (Stan vs PyMC)

**This is the #1 source of logp validation failures when transpiling Stan → PyMC.**

When Stan declares `real<lower=0> sigma; sigma ~ cauchy(0, 5);`, it evaluates the
**full Cauchy density** on the positive half-line WITHOUT normalizing. PyMC's
`pm.HalfCauchy('sigma', beta=5)` uses a properly normalized half-distribution that
adds `log(2)` to the logp (since `pdf_half(x) = 2 * pdf_full(x)` for `x > 0`).

This creates an **exact `log(2)` offset per Half* parameter** in logp comparisons.

### How to handle it

**Option A (recommended for logp validation):** Use the Half distribution and add a
correction potential:

```python
n_half_params = 0  # count how many Half* distributions you use

sigma = pm.HalfCauchy("sigma", beta=5)  # Stan: real<lower=0> sigma ~ cauchy(0,5)
n_half_params += 1

tau = pm.HalfNormal("tau", sigma=1)  # Stan: real<lower=0> tau ~ normal(0,1)
n_half_params += 1

# Correct the log(2) offset for exact logp match with Stan
pm.Potential("half_dist_correction", -n_half_params * pt.log(2.0))
```

**Option B:** Use Gamma/InverseGamma/Exponential distributions that naturally have
positive support (no `log(2)` issue since they're not "half" versions of symmetric
distributions).

### Affected patterns

| Stan Pattern | Naive PyMC (WRONG logp) | Corrected PyMC |
|---|---|---|
| `real<lower=0> x ~ cauchy(0, s)` | `pm.HalfCauchy(beta=s)` | + correction potential |
| `real<lower=0> x ~ normal(0, s)` | `pm.HalfNormal(sigma=s)` | + correction potential |
| `real<lower=0> x ~ student_t(nu, 0, s)` | `pm.HalfStudentT(nu=nu, sigma=s)` | + correction potential |
| `real<lower=0> x` (no prior) | `pm.HalfFlat()` | + correction potential |

See: https://github.com/pymc-devs/pymc/issues/8186

## Transformed Data Block

Stan's `transformed data` block computes derived quantities from data BEFORE the model.
These are **constants**, not random variables. The most common mistake is forgetting to
use the transformed variable as `observed` instead of the raw data.

```stan
transformed data {
    vector[N] log_earn;
    log_earn = log(earn);  // THIS is what gets used as observed
}
model {
    log_earn ~ normal(mu, sigma);  // NOT earn!
}
```

```python
# CORRECT: compute transformation, use as observed
log_earn = np.log(data['earn'])

with pm.Model() as model:
    # ...
    pm.Normal("log_earn", mu=mu, sigma=sigma, observed=log_earn)
```

## Stan sd() vs NumPy std()

Stan's `sd()` function uses **population standard deviation** (divides by N, equivalent
to `ddof=0`). NumPy's default `np.std()` also uses `ddof=0`, but be careful not to
accidentally use `ddof=1`.

```python
# CORRECT: matches Stan's sd()
z_height = (height - np.mean(height)) / np.std(height)  # ddof=0, same as Stan

# WRONG: sample std dev, does NOT match Stan
z_height = (height - np.mean(height)) / np.std(height, ddof=1)
```

## Indexing: Stan 1-based → Python 0-based

Stan uses 1-based indexing. **All index arrays from posteriordb data are 1-based** and
must be adjusted with `- 1` when used as Python array indices.

```stan
// Stan: county is 1-based (values 1..J)
y[i] ~ normal(alpha[county[i]], sigma);
```

```python
# CORRECT: subtract 1 for 0-based indexing
county_idx = data['county'] - 1  # convert 1-based → 0-based
pm.Normal("y", mu=alpha[county_idx], sigma=sigma, observed=y)
```

**Do NOT subtract 1 from size/count variables** like `N`, `J`, `K` — only from index
arrays that are used for array subscripting.

## Bounded Parameters Without Explicit Prior

When Stan declares `real<lower=0, upper=100> sigma` with NO explicit prior in the model
block, this is an **implicit uniform prior** on `[0, 100]`.

```python
# CORRECT
sigma = pm.Uniform("sigma", lower=0, upper=100)

# WRONG: TruncatedNormal has a non-uniform density shape
sigma = pm.TruncatedNormal("sigma", mu=50, sigma=1000, lower=0, upper=100)
```

## Dependent Parameter Bounds

Stan supports bounds that depend on other parameters:
```stan
real<lower=0, upper=1> alpha1;
real<lower=0, upper=(1-alpha1)> beta1;  // upper bound depends on alpha1
```

PyMC does not directly support dependent bounds. Use a manual transform:
```python
alpha1 = pm.Uniform("alpha1", lower=0, upper=1)
# beta1 lives in (0, 1-alpha1), so use a helper variable in (0,1)
beta1_raw = pm.Uniform("beta1_raw", lower=0, upper=1)
beta1 = pm.Deterministic("beta1", beta1_raw * (1 - alpha1))
# Add Jacobian: log|d(beta1)/d(beta1_raw)| = log(1 - alpha1)
pm.Potential("beta1_jacobian", pt.log(1 - alpha1))
```

## Automatic Transforms in PyMC

A key conceptual difference: Stan explicitly defines parameters separately from their
distributions, requiring manual constraint specification. PyMC implicitly creates
parameters when named variables are defined and **automatically applies appropriate
transforms** based on distribution domains.

For example, `pm.Exponential("x", 1)` automatically applies a log transform because
exponential requires positive values.

```python
# Inspect what transform PyMC chose
pymc_model.rvs_to_transforms[x]  # e.g. LogTransform
pymc_model.rvs_to_values[x]      # internal parameter name

# Override the default transform
x = pm.Exponential("x", 1, default_transform=None)
```

### Jacobian Corrections

Both libraries apply Jacobian corrections by default when transforming parameters. This
accounts for the change-of-variable effect on the prior density. You can disable this:

**Stan:**
```python
stan_model.log_prob(values, adjust_transform=False)
```

**PyMC:**
```python
pymc_model.compile_logp(jacobian=False)
```

## Logp Validation: Compare Differences, Not Absolutes

Stan drops normalizing constants by default; PyMC does not. When validating that a
translated model matches, **do not compare absolute logp values**. Instead, evaluate at
several different parameter points and check that the **differences** are the same:

```python
# Evaluate logp at two points
logp_stan_a, logp_stan_b = ...
logp_pymc_a, logp_pymc_b = ...

# These should match (up to numerical precision)
assert np.isclose(logp_stan_a - logp_stan_b, logp_pymc_a - logp_pymc_b)
```

## Multiple Densities on One Parameter (CustomDist and Potential)

Stan allows assigning multiple `~` statements to the same parameter (adding logp terms).
PyMC's one-parameter-one-distribution design requires workarounds.

### Option 1: CustomDist (for component-wise different distributions)

```python
def custom_dist_graph(_):
    return pm.math.concatenate([
        pm.Exponential.dist(1, shape=(2,)),
        pm.HalfNormal.dist(1, shape=(1,))
    ])

x = pm.CustomDist("x", dist=custom_dist_graph, transform=pm.distributions.transforms.log)
```

### Option 2: Potential (add arbitrary logp terms)

```python
pm.Potential("extra_density", pm.logp(pm.Normal.dist(0, 1), x))
```

**Caveat:** Potentials are ignored during prior/posterior predictive sampling. This can
lead to invalid or biased predictive samples.

### Stan-style Model in PyMC (Flat + Potentials)

For complex models where the Stan pattern is clearer, you can replicate it directly:

```python
with pm.Model() as model:
    x = pm.Flat("x", shape=(3,), default_transform=pm.distributions.transforms.log)
    pm.Potential("prior1", pm.logp(pm.Exponential.dist(1), x[:2]))
    pm.Potential("prior2", pm.logp(pm.Normal.dist(0, 1), x[2]))
    pm.Potential("likelihood", pm.logp(pm.Normal.dist(0, x + 1), data))
```

## Preferred Way to Compute logp

Use `pm.logp()` rather than calling `Distribution.logp()` directly — it works
universally across all distributions and parametrizations:

```python
from pymc import logp

# Preferred
logp(pm.Normal.dist(0, 1), x)

# Also works but less universal
pm.Normal.logp(x, 0, 1)
```

## Forward Sampling with draw()

PyMC can draw forward samples from the random graph without running a sampler. Stan
requires implementing equivalent sampling manually in `generated_quantities` blocks.

```python
from pymc import draw

draw(x)                                    # single variable
draw([x, y])                               # multiple variables
draw(y, givens={x: posterior_sample})       # posterior predictive
```

## Censored Distributions

Stan's truncation syntax `T[lower, upper]` maps to `pm.Censored` or
`pm.TruncatedNormal` etc. Use `pm.Censored` for the general case:

```stan
y ~ normal(mu, sigma) T[0,];
```
```python
y = pm.Censored("y", pm.Normal.dist(mu=mu, sigma=sigma), lower=0, upper=None,
                observed=y_data)
```

Prefer `pm.Censored` over `pm.Potential` with manual logp for censored data — it
handles forward sampling correctly.

## Use Idiomatic PyMC Distributions

When translating, prefer PyMC's built-in distributions over manual `pm.Potential`
constructs:

- Use `pm.HalfNormal` instead of `pm.Normal` + lower bound potential
- Use `pm.Censored` for censored/truncated data
- Use `pm.ZeroInflatedPoisson`, `pm.Hurdle*`, etc. for mixture-type likelihoods
- Only fall back to `pm.Potential` when there is no clean distribution equivalent

## Critical Anti-Pattern: Never Use pm.Flat + pm.Potential for Standard Distributions

**This is the #1 source of poor sampling performance in transpiled models.**

Benchmarks show that models using `pm.Flat` priors with manual `pm.Potential` log-probability
terms are dramatically slower and sometimes fail to converge, compared to using proper PyMC
distributions. This is because:

1. **Potentials break prior/posterior predictive sampling** — `pm.sample_prior_predictive()`
   and `pm.sample_posterior_predictive()` ignore Potentials entirely
2. **Potentials bypass PyMC's automatic transform handling** — proper distributions get
   automatic Jacobian corrections and domain-appropriate transforms
3. **Manual logp formulas are error-prone** — missing normalizing terms, wrong signs, or
   omitted Jacobians can cause divergences

### Rule 1: ALWAYS use proper distributions for Stan priors

When Stan has an explicit prior like `mu ~ normal(0, 1000)`, translate it directly to the
corresponding PyMC distribution. **NEVER** use `pm.Flat` + `pm.Potential`.

```stan
// Stan
real mu;
real<lower=0> sigmasq;
vector[N] b;
mu ~ normal(0, 1000);
sigmasq ~ inv_gamma(0.001, 0.001);
b ~ normal(mu, sigma);
```
```python
# WRONG — causes divergences and slow sampling
mu = pm.Flat("mu")
sigmasq = pm.HalfFlat("sigmasq")
b = pm.Flat("b", shape=N)
pm.Potential("mu_prior", -0.5 * (mu / 1000)**2)
pm.Potential("sigmasq_prior", (0.001 - 1) * pt.log(sigmasq) - 0.001 / sigmasq)
pm.Potential("b_prior", pt.sum(-0.5 * ((b - mu) / sigma)**2))

# CORRECT — clean, fast, works with predictive sampling
mu = pm.Normal("mu", mu=0, sigma=1000)
sigmasq = pm.InverseGamma("sigmasq", alpha=0.001, beta=0.001)
b = pm.Normal("b", mu=mu, sigma=sigma, shape=N)
```

### Rule 2: ALWAYS use observed distributions for Stan likelihoods

When Stan has a sampling statement for observed data like `y ~ normal(mu, sigma)`, use
`pm.Distribution(..., observed=y_data)`. **NEVER** compute the logp manually as a Potential.

```stan
// Stan
y ~ normal(mu, sigma);
r ~ binomial_logit(n, b);
```
```python
# WRONG — manual logp computation
lik_logp = pt.sum(r * b - n * pt.log1p(pt.exp(b)))
pm.Potential("likelihood", lik_logp)

# CORRECT — proper distributions with observed data
pm.Normal("y", mu=mu, sigma=sigma, observed=y_data)
pm.Binomial("r_obs", n=n, logit_p=b, observed=r)
```

### Rule 3: Use pm.Dirichlet for simplex, pm.LKJCholeskyCov for correlations

```stan
// Stan
simplex[K] theta;
cholesky_factor_corr[2] L_Omega;
L_Omega ~ lkj_corr_cholesky(4);
```
```python
# WRONG — manual ILR transform with Jacobians
theta_raw = pm.Flat("theta_raw")
theta = custom_simplex_transform(theta_raw)
pm.Potential("theta_jacobian", manual_jacobian_computation)

# CORRECT — PyMC handles transforms automatically
theta = pm.Dirichlet("theta", a=np.ones(K))
L_cov = pm.LKJCholeskyCov("L_cov", eta=4, n=2, sd_dist=pm.Exponential.dist(0.1))
```

### When pm.Potential IS appropriate

Only use `pm.Potential` for:
- **Custom constraint terms** that have no distribution equivalent (e.g., stationarity constraints)
- **Forward algorithm likelihoods** in HMMs (no PyMC HMM primitive)
- **Multiple density terms on one parameter** (Stan's multiple `~` statements)
- **Jacobian corrections** for manual parameter transforms with dependent bounds

## Coords and Dims: Named Dimensions for All Parameters

**Always set up `coords` and `dims` when translating Stan models.** This is idiomatic PyMC
and produces cleaner InferenceData with labeled axes instead of integer indices.

### Why this matters

- **Readable traces**: `idata.posterior["theta"].sel(group="A")` vs `idata.posterior["theta"][:, :, 0]`
- **Self-documenting models**: dims make the shape semantics explicit
- **Better plotting**: ArviZ automatically labels axes from coords
- **Error catching**: shape mismatches are caught earlier with named dims

### Setting up coords from Stan data

Stan models declare array sizes in the `data` block (`int<lower=1> K`, `int<lower=1> N`,
etc.). Map these to coords in the `pm.Model()` constructor:

```stan
data {
  int<lower=1> N;        // number of observations
  int<lower=1> K;        // number of predictors
  int<lower=1> J;        // number of groups
  array[N] int<lower=1,upper=J> group;
  matrix[N, K] X;
  vector[N] y;
}
parameters {
  vector[K] beta;
  vector[J] alpha;
  real<lower=0> sigma;
}
```
```python
# Define coords from data dimensions
coords = {
    "predictor": [f"x{k}" for k in range(K)],  # or actual feature names if available
    "group": group_names,                        # e.g. ["A", "B", "C", ...] or range(J)
    "obs": np.arange(N),
}

with pm.Model(coords=coords) as model:
    beta = pm.Normal("beta", mu=0, sigma=10, dims="predictor")
    alpha = pm.Normal("alpha", mu=0, sigma=10, dims="group")
    sigma = pm.HalfNormal("sigma", sigma=5)

    mu = pm.math.dot(X_data, beta) + alpha[group_idx]
    pm.Normal("y", mu=mu, sigma=sigma, observed=y_data, dims="obs")
```

### Multi-dimensional parameters

For parameters with multiple dimensions, pass a tuple of dim names:

```stan
// Stan: matrix parameter
matrix[J, K] beta;
```
```python
# PyMC: use a tuple of dims
beta = pm.Normal("beta", mu=0, sigma=10, dims=("group", "predictor"))
```

### Coords for hierarchical models

```stan
data {
  int<lower=1> J;      // counties
  int<lower=1> N;      // observations
  array[N] int county;
}
parameters {
  vector[J] alpha;
}
```
```python
coords = {
    "county": county_names,  # use meaningful labels, not just range(J)
    "obs": np.arange(N),
}

with pm.Model(coords=coords) as model:
    alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, dims="county")
    pm.Normal("y", mu=alpha[county_idx], sigma=sigma_y, observed=y_data, dims="obs")
```

### Freezing coords and data with `freeze_dims_and_data`

To freeze mutable coords and data (e.g., to simplify the computation graph or generate
more efficient compiled code), use the standalone function:

```python
from pymc.model.transform.optimization import freeze_dims_and_data

frozen_model = freeze_dims_and_data(model)
```

This returns a new model where all mutable coords and data containers are replaced with
fixed constants, enabling additional graph optimizations.

### Always use `dims`, never `shape`

- **Always use `dims`** for every parameter, observed variable, and `pm.Deterministic`.
  `shape` is never needed — `dims` provides the same sizing while adding labeled axes.
- Named dims produce cleaner InferenceData, better ArviZ plots, and catch shape
  mismatches earlier.

## Important Conventions

1. **Indexing**: Stan is 1-based, Python/PyMC is 0-based. Adjust all index arrays.
2. **Column-major vs row-major**: Stan uses column-major matrices, NumPy uses row-major.
   When converting matrix data, transpose if needed.
3. **Half distributions**: Stan uses `normal(0, s)` with `<lower=0>` constraint. PyMC has
   dedicated `pm.HalfNormal`, `pm.HalfCauchy`, etc. **BUT they differ by `log(2)` in logp!**
   See the "Half-Distribution log(2) Mismatch" section above.
4. **Improper priors**: Stan's `real` with no prior = improper uniform. In PyMC use
   `pm.Flat()` or (better) specify an explicit prior.
5. **Observed data**: Pass as `observed=` kwarg to the likelihood distribution. Data should
   be numpy arrays.
6. **Coords and dims**: **Always** set up coords and dims. See the "Coords and Dims" section
   above for full guidance. This produces labeled InferenceData and self-documenting models.
7. **PyTensor**: PyMC uses PyTensor (formerly Aesara/Theano) for symbolic math. Import as
   `import pytensor.tensor as pt` for operations not directly in `pm.math`.
8. **Transformed parameters**: Use `pm.Deterministic(...)` or plain pytensor expressions.
   These are NOT new distributions — do not use `pm.Normal` for a derived quantity like
   `sigma = sqrt(sigmasq)`.
9. **binomial_logit**: Stan's `binomial_logit(n, alpha)` = `pm.Binomial("x", n=n, logit_p=alpha)`.
   Do NOT manually apply `invlogit` — use the `logit_p` parameter directly.

## Vectorization: Prefer Array Operations Over Python Loops

**This is critical for performance.** Stan models often use `for` loops; PyMC models should
use vectorized numpy-like operations instead. Python `for` loops at model construction time
unroll into enormous computation graphs (potentially 20x+ more nodes), causing slow
compilation, slow sampling, and slow gradient computation.

### Use `shape` or `dims` instead of creating distributions in a loop

```python
# WRONG — creates K separate graph nodes
for k in range(K):
    theta[k] = pm.Normal(f"theta_{k}", mu=mu_group, sigma=sigma_group)

# CORRECT — single vectorized distribution
theta = pm.Normal("theta", mu=mu_group, sigma=sigma_group, shape=K)

# BEST — use dims for labeled axes
theta = pm.Normal("theta", mu=mu_group, sigma=sigma_group, dims="group")
```

### Use array indexing instead of conditionals or loops to select parameters

```stan
// Stan
for (n in 1:N)
    y[n] ~ normal(alpha[group[n]], sigma);
```
```python
# PyMC — vectorized, no loop
pm.Normal("y", mu=alpha[group_idx], sigma=sigma, observed=y_data)
```

### Use `pt.dot`, `pt.sum`, `pt.stack` for linear algebra

```stan
// Stan
real total = 0;
for (k in 1:K)
    total += beta[k] * X[n, k];
```
```python
# PyMC — single dot product
mu = pt.dot(X_data, beta)
```

### Never accumulate in Python lists — use array operations

```python
# WRONG — builds list then stacks, creating unnecessary graph nodes
mu_list = []
for t in range(K, T):
    mu_t = alpha
    for k in range(K):
        mu_t = mu_t + beta[k] * y_data[t - k - 1]
    mu_list.append(mu_t)
mu = pt.stack(mu_list)

# CORRECT — vectorized with array slicing and dot product
# Build a lagged matrix and use a single dot product
Y_lagged = pt.stack([y_data[K-k-1:T-k-1] for k in range(K)], axis=1)
mu = alpha + pt.dot(Y_lagged, beta)
```

### Common vectorization patterns

| Stan pattern | PyMC equivalent |
|---|---|
| `for (n in 1:N) y[n] ~ dist(f(n))` | `pm.Dist("y", f_vectorized, observed=y_data)` |
| `for (n in 1:N) target += g(y[n])` | `pm.Potential("name", pt.sum(g(y_data)))` |
| Accumulate in a loop | `pt.dot`, `pt.sum`, `pt.cumsum`, `pt.cumprod` |
| Build array element-by-element | `pt.stack`, `pt.concatenate`, broadcasting |
| Conditional per element | `pt.switch(condition_array, a, b)` |
| Index with loop variable | Advanced indexing: `params[index_array]` |

## Looping with `pytensor.scan` (Sequential Dependencies)

When a computation has **true sequential dependencies** (step `t` depends on step `t-1`),
use `pytensor.scan` instead of a Python `for` loop. This compiles the loop into the
computation graph rather than unrolling it.

### When to use `scan` vs. vectorization

- **Vectorize** when observations are conditionally independent given parameters (the vast majority of models)
- **Use `scan`** only for true recurrences: AR/MA/state-space models, ODEs, reinforcement learning
- **Use `pm.AR`** for standard AR(p) models — it wraps scan for you

### Stan for-loop → pytensor.scan

```stan
// Stan: AR(1) in transformed parameters
transformed parameters {
  vector[T] mu;
  mu[1] = alpha;
  for (t in 2:T)
    mu[t] = alpha + rho * y[t-1];
}
```
```python
# PyMC: use scan for sequential dependency
import pytensor

def ar_step(y_prev, alpha, rho):
    return alpha + rho * y_prev

mu, _ = pytensor.scan(
    fn=ar_step,
    sequences=y_data[:-1],       # iterate over lagged data
    outputs_info=None,           # no recurrent state needed here
    non_sequences=[alpha, rho],  # parameters passed at every step
)
```

### Scan argument ordering

The inner function receives arguments in a **fixed order**:
1. **Sequences** (time slices from `sequences`)
2. **Previous outputs** (recurrent values from `outputs_info`)
3. **Non-sequences** (constants)

### AR(p) with recurrent state (multi-lag)

```python
def ar_step(x_tm2, x_tm1, rho):
    mu = x_tm1 * rho[0] + x_tm2 * rho[1]
    return mu

mu_steps, _ = pytensor.scan(
    fn=ar_step,
    outputs_info=[{"initial": ar_init, "taps": range(-2, 0)}],
    non_sequences=[rho],
    n_steps=T - 2,
)
```

### `outputs_info` patterns

| Pattern | `outputs_info` | Meaning |
|---------|---------------|---------|
| No recurrence (map) | `None` | Pure map over sequences |
| Single previous step | `initial_value` | `fn` receives `x_{t-1}` |
| Multiple taps AR(p) | `{"initial": init, "taps": range(-p, 0)}` | `fn` receives `x_{t-p}, ..., x_{t-1}` |

### Scan gotchas

- **Argument order is rigid**: sequences → recurrent outputs → non-sequences. Mismatching is the most common bug.
- **Shape/dtype of `outputs_info`** must match the return value of `fn`.
- Use `strict=True` to catch undeclared shared variables.
- For stochastic steps inside scan (e.g., random innovations), wrap with `pm.CustomDist` and use `collect_default_updates` from `pymc`.

## Conditional Logic: `pt.switch` and `ifelse`

**You cannot use Python `if/else` on symbolic PyTensor variables.** They are not concrete
values at graph construction time. Use `pt.switch` or `ifelse` instead.

```python
# WRONG — evaluates at graph construction time, not sampling time
x = pm.Normal("x", 0, 1)
if x > 0:        # ERROR: x is symbolic, this doesn't work as intended
    y = x ** 2
else:
    y = -x

# CORRECT — stays in the computation graph
y = pt.switch(x > 0, x**2, -x)
```

### `pt.switch` (element-wise, like `np.where`)

Use for most conditional logic in PyMC models. Works element-wise on tensors.
Evaluates both branches (not lazy).

```stan
// Stan: conditional in loop
for (n in 1:N) {
  if (y[n] == 0)
    target += log_sum_exp(log(psi), log1m(psi) + poisson_lpmf(0 | lambda));
  else
    target += log1m(psi) + poisson_lpmf(y[n] | lambda);
}
```
```python
# PyMC: vectorized with pt.switch (no loop needed)
logp_zero = pt.log(psi + (1 - psi) * pt.exp(pm.logp(pm.Poisson.dist(mu=lam), value)))
logp_nonzero = pt.log(1 - psi) + pm.logp(pm.Poisson.dist(mu=lam), value)
pm.Potential("lik", pt.sum(pt.switch(pt.eq(value, 0), logp_zero, logp_nonzero)))
```

### `ifelse` (scalar condition, lazy evaluation)

Use only when you have a **scalar** boolean condition and the branches are expensive.
Evaluates only the taken branch.

```python
from pytensor.ifelse import ifelse

# Only evaluates the taken branch
z = ifelse(pt.lt(a, b), expensive_branch_a, expensive_branch_b)
```

### Stan → PyMC conditional mapping

| Stan idiom | PyMC equivalent |
|---|---|
| `condition ? a : b` (element-wise) | `pt.switch(condition, a, b)` |
| `if (cond) { ... } else { ... }` | `pt.switch(cond, ...)` or `ifelse(cond, ...)` for scalar |
| `target += (cond) ? lp1 : lp2` | `pm.Potential("name", pt.switch(cond, lp1, lp2))` |
| Stan `for` loop (independent iterations) | Vectorize with `shape`/`dims` and array indexing |
| Stan `for` loop (sequential dependency) | `pytensor.scan(...)` |
