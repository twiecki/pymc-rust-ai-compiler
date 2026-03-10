# Skill: Stan → PyMC Model Translation

You are translating a **Stan model** to **PyMC (v5+)**. Pay close attention to
the differences between Stan and PyMC conventions.

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

## Important Conventions

1. **Indexing**: Stan is 1-based, Python/PyMC is 0-based. Adjust all index arrays.
2. **Column-major vs row-major**: Stan uses column-major matrices, NumPy uses row-major.
   When converting matrix data, transpose if needed.
3. **Half distributions**: Stan uses `normal(0, s)` with `<lower=0>` constraint. PyMC has
   dedicated `pm.HalfNormal`, `pm.HalfCauchy`, etc.
4. **Improper priors**: Stan's `real` with no prior = improper uniform. In PyMC use
   `pm.Flat()` or (better) specify an explicit prior.
5. **Observed data**: Pass as `observed=` kwarg to the likelihood distribution. Data should
   be numpy arrays.
6. **Coords and dims**: PyMC supports named dimensions. Use `coords=` in `pm.Model()` and
   `dims=` in distributions for better InferenceData output.
7. **PyTensor**: PyMC uses PyTensor (formerly Aesara/Theano) for symbolic math. Import as
   `import pytensor.tensor as pt` for operations not directly in `pm.math`.
