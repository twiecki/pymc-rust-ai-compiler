# Skill: GPU-Accelerated Gaussian Process Models (Apple Silicon / MLX)

This model contains a Gaussian Process and Apple Silicon with Metal is available.
Use GPU-accelerated linear algebra via mlx-rs (Rust bindings to Apple's MLX framework)
for the heavy matrix operations. MLX leverages Apple's unified memory architecture —
CPU and GPU share the same memory pool, so there is zero copy overhead.

For N>100, Metal GPU gives significant speedup over CPU. For N<50, CPU (faer) may
still be faster due to kernel launch overhead.

## Dependencies

`mlx-rs = { version = "0.25", features = ["metal"] }` is added to Cargo.toml
automatically. mlx-rs provides safe Rust bindings to Apple's MLX C++ framework.

## IMPORTANT: float32 on GPU, float64 on CPU

MLX's Metal backend only supports **float32** for linalg operations (Cholesky, solve, etc.).
float64 is available only on CPU via the Accelerate framework.

**Strategy for GP models:**
- Build kernel matrix K in f64 on CPU (element-wise, fast)
- Convert to f32 for GPU Cholesky + solve (sufficient precision for MCMC)
- Convert results back to f64 for logp + gradient computation
- For N<50, stay entirely in f64 on CPU (use faer instead)

## Architecture

MLX-accelerated GP logp has this flow:
1. Build kernel matrix K on CPU in f64 (element-wise, fast)
2. Convert K to f32 MLX array (zero-copy on unified memory)
3. MLX: Cholesky factorization on Metal GPU
4. MLX: triangular solve for L \ y
5. Read results back (zero-copy) and convert to f64
6. Compute logp + gradients on CPU using alpha, K^{-1}

## MANDATORY: Pre-allocated Struct with MLX Context

```rust
use mlx_rs::Array;
use mlx_rs::ops;
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
    // f32 staging buffer for MLX
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

## Building Kernel Matrix (CPU, f64)

Build the kernel matrix in row-major on CPU (will be reshaped for MLX):

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
            self.k_host[i * N + j] = k_ij + sigma_sq + JITTER;
        } else {
            self.k_host[i * N + j] = k_ij;
            self.k_host[j * N + i] = k_ij;
        }
    }
}
```

## Cholesky + Solve on Metal GPU (via MLX)

```rust
// Convert f64 -> f32 for GPU
for i in 0..N * N {
    self.k_f32[i] = self.k_host[i] as f32;
}
for i in 0..N {
    self.y_f32[i] = Y_DATA[i] as f32;
}

// Create MLX arrays (uses unified memory — no explicit copy needed)
let k_mlx = Array::from_slice(&self.k_f32, &[N as i32, N as i32]);
let y_mlx = Array::from_slice(&self.y_f32, &[N as i32, 1]);

// Cholesky: K = L L^T (lower triangular)
let l_mlx = linalg::cholesky(&k_mlx, false)  // upper=false → lower
    .map_err(|e| SampleError::Recoverable(format!("Cholesky failed: {}", e)))?;

// Solve L L^T alpha = y  →  alpha = K^{-1} y
let alpha_mlx = linalg::solve_triangular(&l_mlx, &y_mlx, false, false)
    .map_err(|e| SampleError::Recoverable(format!("Solve failed: {}", e)))?;
// Second solve for L^T
let alpha_mlx = linalg::solve_triangular(
    &ops::transpose(&l_mlx, &[1, 0]).unwrap(),
    &alpha_mlx, true, false,
).map_err(|e| SampleError::Recoverable(format!("Solve failed: {}", e)))?;

// Evaluate lazily computed results
alpha_mlx.eval().unwrap();
l_mlx.eval().unwrap();

// Read back to f64
let alpha_f32: Vec<f32> = alpha_mlx.as_slice().to_vec();
for i in 0..N {
    self.alpha_host[i] = alpha_f32[i] as f64;
}
```

## Log-determinant from MLX Cholesky

```rust
let l_data: Vec<f32> = l_mlx.as_slice().to_vec();
let mut log_det = 0.0f64;
for i in 0..N {
    // Row-major: L[i,i] at index i*N+i
    log_det += (l_data[i * N + i] as f64).ln();
}
log_det *= 2.0;
```

## Computing K^{-1} on GPU (for gradients)

```rust
// K^{-1} = (L L^T)^{-1} via MLX inverse
let kinv_mlx = linalg::inv(&k_mlx)
    .map_err(|e| SampleError::Recoverable(format!("Inverse failed: {}", e)))?;
kinv_mlx.eval().unwrap();

let kinv_f32: Vec<f32> = kinv_mlx.as_slice().to_vec();
for i in 0..N * N {
    self.kinv_host[i] = kinv_f32[i] as f64;
}
```

## GP Log-likelihood (same as CPU)

```
logp = -0.5 * (N * LN_2PI + log_det + y_t_alpha)
```
where `y_t_alpha = sum(y[i] * alpha[i])` is computed on CPU after reading back alpha.

## GP Gradients (CPU, using K^{-1} and alpha from GPU)

Same formula as CPU skill, using row-major indexing:

```rust
// K^{-1} in row-major: kinv[i,j] = kinv_host[i*N+j]
for i in 0..N {
    let alpha_i = self.alpha_host[i];
    for j in 0..N {
        let alpha_j = self.alpha_host[j];
        let kinv_ij = self.kinv_host[i * N + j];
        let w_ij = kinv_ij - alpha_i * alpha_j;
        grad_ls += w_ij * self.dk_dls[i * N + j];
        // For ExpQuad: dk_deta = 2*K_ij/eta
        // For diagonal noise: dk_dsigma = 2*sigma * delta_ij
        if i == j {
            grad_sigma += w_ij * 2.0 * sigma_sq;
        }
    }
}
```

## When to Use MLX GPU vs CPU

- **N < 50**: CPU (faer) is faster — Metal kernel launch overhead dominates
- **N = 50-100**: MLX and CPU are comparable; MLX wins with repeated evaluations
- **N > 100**: MLX is significantly faster (Cholesky is O(N^3))
- **N > 500**: MLX is 10-100x faster, unified memory avoids copy bottleneck

The compiler auto-detects Apple Silicon and selects MLX or CPU skill accordingly.

## Error Handling

MLX operations return `Result`. Map errors to `SampleError::Recoverable`:
```rust
.map_err(|e| SampleError::Recoverable(format!("MLX error: {}", e)))?;
```

## Key Differences from CUDA Approach

| Aspect | CUDA (cudarc) | MLX (mlx-rs) |
|--------|--------------|--------------|
| Memory | Explicit GPU↔CPU copy | Unified (zero-copy) |
| Precision | float64 native | float32 on GPU, float64 on CPU only |
| Layout | Column-major (Fortran) | Row-major (C) |
| Lazy eval | No | Yes — call `.eval()` to materialize |
| Struct | Holds CudaDevice, CudaSlice | Holds only CPU buffers, create Arrays on the fly |

## Lazy Evaluation

MLX uses lazy evaluation. Operations are recorded but not executed until `.eval()` is called.
Chain operations before evaluating for best performance:

```rust
let l = linalg::cholesky(&k, false)?;
let alpha = linalg::solve_triangular(&l, &y, false, false)?;
// Both are computed when we call eval:
alpha.eval().unwrap();
// l is also evaluated as a dependency
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

Always add a small jitter (1e-6) to the diagonal for numerical stability:
```rust
const JITTER: f64 = 1e-6;
if i == j { self.k_host[i * N + j] = kernel_ij + sigma_sq + JITTER; }
```
