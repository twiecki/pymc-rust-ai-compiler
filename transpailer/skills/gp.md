# Skill: Gaussian Process Models

This model contains a Gaussian Process. GP models require matrix operations
(Cholesky decomposition, solves, inverses) which need special handling in Rust.

## Dependencies

`faer = "0.24"` is already added to Cargo.toml automatically. faer is a pure-Rust
linear algebra library with excellent performance (comparable to LAPACK).

## MANDATORY: Pre-allocated Struct with Default

GP models MUST pre-allocate all matrices and scratch buffers as struct fields.
Creating Mat::zeros() or MemBuffer inside logp() allocates on every call — this is
catastrophic for performance. The validate/bench binaries call `GeneratedLogp::default()`,
so you MUST implement `Default`.

**DO NOT use a unit struct.** Always use this pattern:

```rust
use faer::{Mat, Par, Spec};
use faer::linalg::cholesky::llt::{factor, solve, inverse};
use faer::linalg::cholesky::llt::factor::LltParams;
use faer::dyn_stack::{MemBuffer, MemStack};

const LN_2PI: f64 = 1.8378770664093453;  // ln(2π) — use this EXACT value
const JITTER: f64 = 1e-6;

pub struct GeneratedLogp {
    k_mat: Mat<f64>,      // N×N kernel matrix (overwritten each call)
    alpha: Mat<f64>,      // N×1 for K^{-1} y
    kinv: Mat<f64>,       // N×N for inverse
    dk_dls: Vec<f64>,     // N*N flat storage for dK/d(log_ls)
    chol_buf: MemBuffer,  // scratch for cholesky_in_place
    inv_buf: MemBuffer,   // scratch for inverse
}

impl Default for GeneratedLogp {
    fn default() -> Self {
        let chol_scratch = factor::cholesky_in_place_scratch::<f64>(
            N, Par::Seq, Spec::<LltParams, f64>::default(),
        );
        let inv_scratch = inverse::inverse_scratch::<f64>(N, Par::Seq);
        Self {
            k_mat: Mat::zeros(N, N),
            alpha: Mat::zeros(N, 1),
            kinv: Mat::zeros(N, N),
            dk_dls: vec![0.0; N * N],
            chol_buf: MemBuffer::new(chol_scratch),
            inv_buf: MemBuffer::new(inv_scratch),
        }
    }
}
```

## Cholesky Decomposition (in-place)

```rust
// Overwrites k_mat lower triangle with L where K = L L^T
factor::cholesky_in_place(
    self.k_mat.as_mut(),
    Default::default(),
    Par::Seq,
    MemStack::new(&mut self.chol_buf),
    Spec::<LltParams, f64>::default(),
).map_err(|_| {
    SampleError::Recoverable("Cholesky failed: not positive definite".to_string())
})?;
```

## Solving K x = b (after Cholesky)

```rust
// alpha starts as y, becomes K^{-1} y in-place
for i in 0..N {
    self.alpha[(i, 0)] = Y_DATA[i];
}
solve::solve_in_place(
    self.k_mat.as_ref(),  // L from Cholesky
    self.alpha.as_mut(),
    Par::Seq,
    MemStack::new(&mut []),
);
```

## Log-determinant

```rust
// After Cholesky: log|K| = 2 * sum(log(L_ii))
let mut log_det = 0.0;
for i in 0..N {
    log_det += self.k_mat[(i, i)].ln();
}
log_det *= 2.0;
```

## Computing K^{-1} (for gradients)

```rust
inverse::inverse(
    self.kinv.as_mut(),
    self.k_mat.as_ref(),  // L from Cholesky
    Par::Seq,
    MemStack::new(&mut self.inv_buf),
);
// Note: only lower triangle is filled. Access symmetrically:
let kinv_ij = if i >= j { self.kinv[(i, j)] } else { self.kinv[(j, i)] };
```

## GP Log-likelihood

For y ~ MvNormal(0, K):
```
logp = -0.5 * (N * LN_2PI + log|K| + y^T K^{-1} y)
```
where LN_2PI = 1.8378770664093453 (use this exact constant).

## GP Gradients

For a kernel hyperparameter θ with derivative dK/dθ:
```
d(logp)/dθ = -0.5 * tr((K^{-1} - α α^T) dK/dθ)
```
where α = K^{-1} y. Compute element-wise:
```rust
for i in 0..N {
    let alpha_i = self.alpha[(i, 0)];
    for j in 0..N {
        let alpha_j = self.alpha[(j, 0)];
        let kinv_ij = if i >= j { self.kinv[(i, j)] } else { self.kinv[(j, i)] };
        let w_ij = kinv_ij - alpha_i * alpha_j;
        grad_theta += w_ij * dk_dtheta[i * N + j];
    }
}
gradient[idx] += -0.5 * grad_theta;
```

## Common Kernels

**ExpQuad (RBF):**
```
K_ij = eta^2 * exp(-0.5 * |x_i - x_j|^2 / ls^2)
dK/d(log_ls) = K_ij * |x_i - x_j|^2 / ls^2  (chain rule through log)
dK/d(log_eta) = 2 * K_ij  (chain rule through log)
```

**White noise (diagonal):**
```
K_ij += sigma^2 * delta_ij
dK/d(log_sigma) = 2 * sigma^2 * delta_ij
```

## JITTER

Always add a small jitter (1e-6) to the diagonal for numerical stability:
```rust
const JITTER: f64 = 1e-6;
// When building K:
if i == j { self.k_mat[(i,j)] = kernel_ij + sigma_sq + JITTER; }
```

## Building the Kernel Matrix (complete pattern)

Use `self.k_mat` and `self.dk_dls` — never allocate new matrices:
```rust
let eta_sq = eta * eta;
let inv_ls_sq = 1.0 / (ls * ls);
let sigma_sq = sigma * sigma;

for i in 0..N {
    for j in 0..=i {
        let d = X_DATA[i] - X_DATA[j];
        let d_sq = d * d;
        let r_sq_scaled = d_sq * inv_ls_sq;
        let exp_term = (-0.5 * r_sq_scaled).exp();
        let k_ij = eta_sq * exp_term;

        // Store dK/d(log_ls) for gradient computation
        self.dk_dls[i * N + j] = k_ij * r_sq_scaled;
        self.dk_dls[j * N + i] = self.dk_dls[i * N + j];

        if i == j {
            self.k_mat[(i, j)] = k_ij + sigma_sq + JITTER;
        } else {
            self.k_mat[(i, j)] = k_ij;
            self.k_mat[(j, i)] = k_ij;
        }
    }
}
```
