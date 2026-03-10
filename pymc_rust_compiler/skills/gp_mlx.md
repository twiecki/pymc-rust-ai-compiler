# Skill: GPU-Accelerated Gaussian Process Models (Apple Silicon / MLX)

This model contains a Gaussian Process and Apple Silicon with Metal is available.
Use GPU-accelerated linear algebra via mlx-rs (Rust bindings to Apple's MLX framework)
for the heavy matrix operations (Cholesky, solve, inverse). MLX uses Apple's unified
memory — CPU and GPU share the same memory pool.

## Dependencies

`mlx-rs = "0.25"` is added to Cargo.toml automatically. Metal support is built-in
on macOS (no feature flags needed).

## CRITICAL: Use `_device` variants with `Stream::cpu()`

MLX linalg operations (cholesky, solve_triangular, inv) must use the `_device`
variants with `Stream::cpu()`. The default stream may fail for linalg ops.

```rust
use mlx_rs::{Array, Stream};
use mlx_rs::linalg;

let cpu = Stream::cpu();
let l = linalg::cholesky_device(&k, None, &cpu)?;
```

## CRITICAL: f32 only — no f64 support for Array

`Array::from_slice` does NOT support f64. All MLX arrays must use f32.
Convert f64 data to f32 before creating arrays, and convert results back to f64.

## MANDATORY: Pre-allocated Struct

GP models MUST pre-allocate all buffers as struct fields.
DO NOT allocate inside `logp()` — it is called thousands of times.

```rust
use mlx_rs::{Array, Stream};
use mlx_rs::linalg;

const LN_2PI: f64 = 1.8378770664093453;
const JITTER: f64 = 1e-6;
const N: usize = /* set from model */;

pub struct GeneratedLogp {
    // CPU buffers (f64 precision)
    k_host: Vec<f64>,         // N*N kernel matrix (row-major)
    dk_dls: Vec<f64>,         // N*N dK/d(log_ls)
    alpha_host: Vec<f64>,     // N for K^{-1}y
    kinv_host: Vec<f64>,      // N*N K^{-1}
    // f32 staging buffers for MLX
    k_f32: Vec<f32>,          // N*N for GPU upload
    y_f32: Vec<f32>,          // N for GPU upload
}

impl Default for GeneratedLogp {
    fn default() -> Self {
        Self {
            k_host: vec![0.0; N * N],
            dk_dls: vec![0.0; N * N],
            alpha_host: vec![0.0; N],
            kinv_host: vec![0.0; N * N],
            k_f32: vec![0.0f32; N * N],
            y_f32: vec![0.0f32; N],
        }
    }
}
```

## Complete MLX GP logp Implementation

Here is the COMPLETE, VERIFIED logp implementation. Copy this pattern exactly.

```rust
fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, SampleError> {
    let cpu = Stream::cpu();

    // === 1. Extract log-transformed parameters ===
    let log_ls = position[0];
    let log_eta = position[1];
    let log_sigma = position[2];

    let ls = log_ls.exp();
    let eta = log_eta.exp();
    let sigma = log_sigma.exp();

    let eta_sq = eta * eta;
    let sigma_sq = sigma * sigma;
    let inv_ls_sq = 1.0 / (ls * ls);

    // === 2. Build kernel matrix K on CPU in f64 ===
    for i in 0..N {
        for j in 0..=i {
            let d = X_DATA[i] - X_DATA[j];
            let d_sq = d * d;
            let r_sq_scaled = d_sq * inv_ls_sq;
            let exp_term = (-0.5 * r_sq_scaled).exp();
            let k_ij = eta_sq * exp_term;

            self.dk_dls[i * N + j] = k_ij * r_sq_scaled;
            self.dk_dls[j * N + i] = self.dk_dls[i * N + j];

            if i == j {
                self.k_host[i * N + j] = k_ij + sigma_sq + JITTER;
            } else {
                self.k_host[i * N + j] = k_ij;
                self.k_host[j * N + i] = k_ij;
            }
        }
    }

    // === 3. Convert to f32 for MLX ===
    for i in 0..N * N {
        self.k_f32[i] = self.k_host[i] as f32;
    }
    for i in 0..N {
        self.y_f32[i] = Y_DATA[i] as f32;
    }

    // === 4. Create MLX arrays ===
    let k_mlx = Array::from_slice(&self.k_f32, &[N as i32, N as i32]);
    let y_mlx = Array::from_slice(&self.y_f32, &[N as i32, 1]);

    // === 5. Cholesky decomposition: K = L L^T ===
    // cholesky_device(a, upper, stream) — upper=None means lower triangular
    let l_mlx = linalg::cholesky_device(&k_mlx, None, &cpu)
        .map_err(|e| SampleError::Recoverable(format!("Cholesky failed: {}", e)))?;

    // === 6. Triangular solve: L * tmp = y, then L^T * alpha = tmp ===
    // solve_triangular_device(a, b, upper, stream)
    // upper=false means a is lower triangular
    let tmp_mlx = linalg::solve_triangular_device(&l_mlx, &y_mlx, false, &cpu)
        .map_err(|e| SampleError::Recoverable(format!("Solve failed: {}", e)))?;
    // Now solve L^T * alpha = tmp (upper=true for L^T)
    let lt_mlx = l_mlx.t();
    let alpha_mlx = linalg::solve_triangular_device(&lt_mlx, &tmp_mlx, true, &cpu)
        .map_err(|e| SampleError::Recoverable(format!("Solve failed: {}", e)))?;

    // === 7. Compute K^{-1} via MLX inv ===
    let kinv_mlx = linalg::inv_device(&k_mlx, &cpu)
        .map_err(|e| SampleError::Recoverable(format!("Inverse failed: {}", e)))?;

    // === 8. Evaluate all lazy computations ===
    l_mlx.eval().map_err(|e| SampleError::Recoverable(format!("Eval failed: {}", e)))?;
    alpha_mlx.eval().map_err(|e| SampleError::Recoverable(format!("Eval failed: {}", e)))?;
    kinv_mlx.eval().map_err(|e| SampleError::Recoverable(format!("Eval failed: {}", e)))?;

    // === 9. Read results back to f64 ===
    // Log-determinant from Cholesky diagonal
    let l_data: &[f32] = l_mlx.as_slice();
    let mut log_det = 0.0f64;
    for i in 0..N {
        log_det += (l_data[i * N + i] as f64).ln();
    }
    log_det *= 2.0;

    // Alpha (K^{-1} y)
    let alpha_f32: &[f32] = alpha_mlx.as_slice();
    for i in 0..N {
        self.alpha_host[i] = alpha_f32[i] as f64;
    }

    // K^{-1}
    let kinv_f32: &[f32] = kinv_mlx.as_slice();
    for i in 0..N * N {
        self.kinv_host[i] = kinv_f32[i] as f64;
    }

    // === 10. Compute y^T K^{-1} y ===
    let mut y_t_alpha = 0.0;
    for i in 0..N {
        y_t_alpha += Y_DATA[i] * self.alpha_host[i];
    }

    // === 11. GP log-likelihood ===
    let gp_logp = -0.5 * (N as f64 * LN_2PI + log_det + y_t_alpha);

    // === 12. Prior log-probabilities (HalfNormal with sigma=5) ===
    // HalfNormal(sigma=s): logp = log(2) - 0.5*ln(2*pi) - ln(s) - 0.5*(x/s)^2
    // For log-transformed: add log_jacobian = log_x
    let prior_const = -1.83522929514961; // log(2) - 0.5*ln(2*pi) - ln(5)
    let ls_prior = prior_const - 0.5 * (ls / 5.0).powi(2) + log_ls;
    let eta_prior = prior_const - 0.5 * (eta / 5.0).powi(2) + log_eta;
    let sigma_prior = prior_const - 0.5 * (sigma / 5.0).powi(2) + log_sigma;

    let total_logp = gp_logp + ls_prior + eta_prior + sigma_prior;

    // === 13. Gradients ===
    gradient.fill(0.0);

    let mut grad_log_ls = 0.0;
    let mut grad_log_eta = 0.0;
    let mut grad_log_sigma = 0.0;

    for i in 0..N {
        let alpha_i = self.alpha_host[i];
        for j in 0..N {
            let alpha_j = self.alpha_host[j];
            let kinv_ij = self.kinv_host[i * N + j];
            let w_ij = kinv_ij - alpha_i * alpha_j;

            grad_log_ls += w_ij * self.dk_dls[i * N + j];
            // dK/d(log_eta) = 2 * K_ij (only the kernel part, not noise)
            let k_ij_no_noise = if i == j {
                self.k_host[i * N + j] - sigma_sq - JITTER
            } else {
                self.k_host[i * N + j]
            };
            grad_log_eta += w_ij * 2.0 * k_ij_no_noise;
            if i == j {
                grad_log_sigma += w_ij * 2.0 * sigma_sq;
            }
        }
    }

    gradient[0] = -0.5 * grad_log_ls;
    gradient[1] = -0.5 * grad_log_eta;
    gradient[2] = -0.5 * grad_log_sigma;

    // Prior gradients: d/d(log_x) [-0.5*(x/s)^2 + log_x] = -x^2/s^2 + 1
    gradient[0] += 1.0 - ls * ls / 25.0;
    gradient[1] += 1.0 - eta * eta / 25.0;
    gradient[2] += 1.0 - sigma * sigma / 25.0;

    Ok(total_logp)
}
```

## MLX API Reference (verified, mlx-rs 0.25)

### Creating Arrays
```rust
// f32 ONLY — f64 is NOT supported
let a = Array::from_slice(&data_f32, &[rows as i32, cols as i32]);
```

### Cholesky
```rust
// cholesky_device(a, upper, stream)
// upper: None or Some(false) = lower triangular L where K = L L^T
// upper: Some(true) = upper triangular U where K = U^T U
let l = linalg::cholesky_device(&a, None, &cpu)?;
```

### Triangular Solve
```rust
// solve_triangular_device(a, b, upper, stream)
// Solves a*x = b where a is triangular
// upper=false: a is lower triangular
// upper=true: a is upper triangular
let x = linalg::solve_triangular_device(&l, &b, false, &cpu)?;
```

### Matrix Inverse
```rust
let a_inv = linalg::inv_device(&a, &cpu)?;
```

### Transpose
```rust
let at = a.t();  // convenience method
```

### Reading Data Back
```rust
array.eval()?;  // MUST eval before reading
let data: &[f32] = array.as_slice();  // returns &[f32], NOT Vec
```

### Stream
```rust
let cpu = Stream::cpu();  // ALWAYS use this for linalg ops
```

## Common Kernels

**ExpQuad (RBF):**
```
K_ij = eta^2 * exp(-0.5 * |x_i - x_j|^2 / ls^2)
dK/d(log_ls) = K_ij * |x_i - x_j|^2 / ls^2
dK/d(log_eta) = 2 * K_ij
```

**White noise (diagonal):**
```
K_ij += sigma^2 * delta_ij
dK/d(log_sigma) = 2 * sigma^2 * delta_ij
```

## JITTER

Always add jitter (1e-6) to the diagonal for numerical stability:
```rust
if i == j { self.k_host[i * N + j] = k_ij + sigma_sq + JITTER; }
```

## Error Handling

All MLX operations return `Result`. Map errors to `SampleError::Recoverable`:
```rust
.map_err(|e| SampleError::Recoverable(format!("MLX error: {}", e)))?;
```
