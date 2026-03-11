# Skill: PyMC Model Optimization

You are optimizing a **PyMC model** for performance. The primary goal is to
eliminate Python-level for-loops and replace them with vectorized NumPy/PyTensor
operations. Every Python for-loop in a PyMC model creates a separate graph node
per iteration, leading to massive compilation overhead and slow sampling.

## Golden Rule

**If it loops over data dimensions, it must be vectorized.**

PyMC models are compiled to computational graphs. A Python for-loop unrolls into
N separate graph nodes. A vectorized operation is a single node operating on a
tensor. The difference can be 10-1000x in compilation time and 2-10x in sampling
speed.

## Anti-Pattern Catalog

### 1. Cumulative Sums via Loops

**BAD:**
```python
n_avoid = np.zeros((n_dogs, n_trials))
for j in range(n_dogs):
    for t in range(1, n_trials):
        n_avoid[j, t] = n_avoid[j, t-1] + 1 - y[j, t-1]
```

**GOOD:**
```python
n_avoid = np.cumsum(1 - y[:, :-1], axis=1)
n_avoid = np.hstack([np.zeros((n_dogs, 1)), n_avoid])
```

### 2. AR(K) Models via Loops

**BAD:**
```python
mu_list = []
for t in range(K, T):
    mu_t = alpha
    for k in range(K):
        mu_t = mu_t + beta[k] * y_data[t - k - 1]
    mu_list.append(mu_t)
mu = pt.stack(mu_list)
```

**GOOD:**
```python
# Build lag matrix: shape (T-K, K) where row t has [y[t+K-1], y[t+K-2], ..., y[t]]
y_arr = np.array(y_data)
lag_matrix = np.column_stack([y_arr[K-k-1:T-k-1] for k in range(K)])
mu = alpha + pt.dot(lag_matrix, beta)
```

### 3. Individual Potentials per Observation

**BAD:**
```python
for i in range(M):
    if s[i] > 0:
        pm.Potential(f"likelihood_{i}", log_prob_detected(i))
    else:
        pm.Potential(f"likelihood_{i}", log_prob_undetected(i))
```

**GOOD:**
```python
observed_mask = s > 0
# Compute both cases for all individuals vectorized
logp_detected = compute_detected_logp_vectorized(...)   # shape (M,)
logp_undetected = compute_undetected_logp_vectorized(...)  # shape (M,)
# Select appropriate logp per individual
logp = pt.where(observed_mask, logp_detected, logp_undetected)
pm.Potential("likelihood", pt.sum(logp))
```

### 4. Sequential/Recursive Computations (ARMA, GARCH, State Space)

**BAD:**
```python
sigma_list = [sigma1]
for t in range(1, T):
    sigma_curr = pt.sqrt(alpha0 + alpha1 * (y[t-1] - mu)**2 + beta1 * sigma_list[-1]**2)
    sigma_list.append(sigma_curr)
sigma = pt.stack(sigma_list)
```

**GOOD (pytensor.scan):**
```python
def step(y_prev, sigma_prev, alpha0, alpha1, beta1, mu):
    return pt.sqrt(alpha0 + alpha1 * (y_prev - mu)**2 + beta1 * sigma_prev**2)

sigmas, _ = pytensor.scan(
    fn=step,
    sequences=[pt.as_tensor_variable(y[:-1])],
    outputs_info=[pt.as_tensor_variable(np.float64(sigma1))],
    non_sequences=[alpha0, alpha1, beta1, mu],
)
sigma = pt.concatenate([pt.atleast_1d(pt.as_tensor_variable(np.float64(sigma1))), sigmas])
```

### 5. Expanding/Repeating Data via Loops

**BAD:**
```python
r = np.zeros((T, N), dtype=int)
for i in range(R):
    for j in range(culm[i-1], culm[i]):
        for k in range(T):
            r[k, j] = response[i, k]
```

**GOOD:**
```python
counts = np.diff(np.concatenate([[0], culm]))
r = np.repeat(response, counts, axis=0).T
```

### 6. Changepoint/Indicator Matrices via Loops

**BAD:**
```python
A = np.zeros((T, S))
for i in range(T):
    for j in range(S):
        if t[i] >= t_change[j]:
            A[i, j] = 1.0
```

**GOOD:**
```python
A = (np.asarray(t)[:, None] >= np.asarray(t_change)[None, :]).astype(float)
```

### 7. Piecewise Logistic Gamma via Loops

**BAD:**
```python
gamma = pt.zeros(S)
m_pr = m
for i in range(S):
    gamma_i = (t_change[i] - m_pr) * (1 - k_s[i] / k_s[i + 1])
    gamma = pt.set_subtensor(gamma[i], gamma_i)
    m_pr = m_pr + gamma_i
```

**GOOD (pytensor.scan for truly sequential):**
```python
def gamma_step(tc_i, k_i, k_ip1, m_prev):
    gamma_i = (tc_i - m_prev) * (1 - k_i / k_ip1)
    return gamma_i, m_prev + gamma_i

(gammas, _), _ = pytensor.scan(
    fn=gamma_step,
    sequences=[pt.as_tensor_variable(t_change), k_s[:-1], k_s[1:]],
    outputs_info=[None, m],
)
gamma = gammas
```

### 8. Per-Timestep Potentials for Random Walks/Seasonal

**BAD:**
```python
for t in range(1, n):
    pm.Potential(f"mu_constraint_{t}",
                 pm.logp(pm.Normal.dist(mu=mu[t-1], sigma=sigma[1]), mu[t]))
```

**GOOD:**
```python
mu_diff = mu[1:] - mu[:-1]
pm.Normal("mu_rw", mu=0, sigma=sigma[1], observed=mu_diff)
# Or equivalently:
pm.Potential("mu_rw", pm.logp(pm.Normal.dist(mu=0, sigma=sigma[1]), mu[1:] - mu[:-1]).sum())
```

### 9. Nested Loops Building Manual Likelihood (e.g., Ordinal Models)

**BAD:**
```python
total_log_prob = 0.0
for i in range(nChild):
    for j in range(nInd):
        if grade[i, j] != -1:
            Q_list = []
            for k in range(n_cat - 1):
                Q_list.append(pm.math.invlogit(delta[j] * (theta[i] - gamma[j, k])))
            # ... manual category probability computation
            total_log_prob += pt.log(p_list[grade_idx])
```

**GOOD:** Vectorize using broadcasting and advanced indexing:
```python
# Compute all Q values at once: shape (nChild, nInd, max_ncat-1)
# theta[:, None] broadcasts with gamma[None, :, :]
diff = theta[:, None, None] - gamma[None, :, :]  # (nChild, nInd, max_ncat-1)
Q = pm.math.invlogit(delta[None, :, None] * diff)

# Build category probabilities vectorized
# p[..., 0] = 1 - Q[..., 0]
# p[..., k] = Q[..., k-1] - Q[..., k]  for 1 <= k < ncat-1
# p[..., ncat-1] = Q[..., ncat-2]
# Then index with grade to get log_prob per observation
```

### 10. MVN Structure via Loops

**BAD:**
```python
for i in range(S):
    # Manual MVN density for each species
    u1, u2 = uv1[i], uv2[i]
    quad_form = inv_11 * u1**2 + inv_22 * u2**2 + 2 * inv_12 * u1 * u2
    mvn_potential += -0.5 * quad_form - 0.5 * pt.log(det_sigma)
```

**GOOD:**
```python
# Stack into matrix: shape (S, 2)
uv = pt.stack([uv1, uv2], axis=1)
# Use MvNormal or compute vectorized quadratic form
cov = pt.stack([[var1, cov12], [cov12, var2]])
inv_cov = pt.linalg.inv(cov)
quad_forms = pt.sum(pt.dot(uv, inv_cov) * uv, axis=1)  # (S,)
mvn_potential = -0.5 * pt.sum(quad_forms) - 0.5 * S * pt.log(pt.linalg.det(cov)) - S * np.log(2 * np.pi)
# Subtract already-counted standard normal densities
mvn_potential -= -0.5 * pt.sum(uv1**2) - 0.5 * pt.sum(uv2**2) - S * np.log(2 * np.pi)
```

## Capture-Recapture Model Pattern

Many posteriordb models follow the capture-recapture pattern where observed (z=1)
and unobserved (marginalized) individuals are handled differently. The key
vectorization strategy:

```python
# Separate observed and unobserved
observed_mask = s > 0  # numpy boolean array, known at graph build time
n_observed = int(np.sum(observed_mask))
n_unobserved = M - n_observed

# Vectorized logp for observed individuals (z=1 certain)
logp_obs = pt.log(omega) + bernoulli_logit_lpmf_vectorized(y[observed_mask], logit_p[observed_mask])

# Vectorized logp for unobserved (marginalize over z)
logp_z1 = pt.log(omega) + bernoulli_logit_lpmf_vectorized(y[~observed_mask], logit_p[~observed_mask])
logp_z0 = pt.log(1 - omega)
logp_unobs = pm.math.logaddexp(logp_z1, logp_z0)  # broadcasts scalar with vector

pm.Potential("likelihood", pt.sum(logp_obs) + pt.sum(logp_unobs))
```

## Validation

After optimization, verify that the model produces the same log probability:

```python
import pymc as pm
point = model.initial_point()
logp_original = original_model.compile_logp()(point)
logp_optimized = optimized_model.compile_logp()(point)
assert abs(logp_original - logp_optimized) < 1e-6
```

## Checklist

Before declaring a model optimized, verify:
- [ ] No Python `for` loops iterate over data dimensions
- [ ] No per-observation `pm.Potential(f"name_{i}", ...)` calls
- [ ] Cumulative operations use `np.cumsum`/`pt.cumsum`
- [ ] Matrix operations use `pt.dot`/`@` instead of element-wise loops
- [ ] Broadcasting is used instead of explicit loops for tensor operations
- [ ] Sequential dependencies use `pytensor.scan` instead of Python loops
- [ ] Random walks use differenced observations, not per-step Potentials
- [ ] The optimized model produces the same logp as the original
