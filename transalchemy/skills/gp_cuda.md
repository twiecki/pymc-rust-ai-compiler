# Skill: GPU-Accelerated Gaussian Process Models (CUDA)

This model contains a Gaussian Process and CUDA is available. Use GPU-accelerated
linear algebra via cudarc (Rust bindings to cuBLAS/cuSOLVER) for the heavy matrix
operations. For N=50, GPU gives ~3-5x speedup over CPU faer. For N>200, GPU
dominates by 10-100x.

## Dependencies

`cudarc = { version = "0.12", features = ["cublas", "cusolver"] }` is added to
Cargo.toml automatically. cudarc provides safe Rust wrappers around NVIDIA's
cuBLAS and cuSOLVER libraries via dynamic loading (no CUDA SDK needed at build time).

## Architecture

GPU-accelerated GP logp has this flow:
1. Build kernel matrix K on CPU (element-wise, fast)
2. Upload K to GPU
3. cuSOLVER: Cholesky factorization (potrf) on GPU
4. cuBLAS: triangular solve (trsv) for K^{-1}y on GPU
5. Download results to CPU
6. Compute logp + gradients on CPU using K^{-1}, alpha

For small N (<50), the upload/download overhead may not be worth it.
For N>100, GPU is significantly faster.

## MANDATORY: Pre-allocated Struct with GPU Context

```rust
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr};
use cudarc::cublas::{CudaBlas, sys::cublasOperation_t};
use cudarc::cusolver::{CudaSolver, sys::cublasFillMode_t};
use std::sync::Arc;

const LN_2PI: f64 = 1.8378770664093453;
const JITTER: f64 = 1e-6;

pub struct GeneratedLogp {
    // GPU context (initialized once, reused)
    dev: Arc<CudaDevice>,
    blas: CudaBlas,
    solver: CudaSolver,
    // Device memory (pre-allocated)
    d_k: CudaSlice<f64>,      // N*N kernel matrix
    d_y: CudaSlice<f64>,      // N observations
    d_alpha: CudaSlice<f64>,  // N for K^{-1}y result
    d_work: CudaSlice<f64>,   // cuSOLVER workspace
    d_info: CudaSlice<i32>,   // cuSOLVER info
    // CPU buffers
    k_host: Vec<f64>,         // N*N column-major kernel matrix
    dk_dls: Vec<f64>,         // N*N dK/d(log_ls)
    alpha_host: Vec<f64>,     // N for K^{-1}y
    kinv_host: Vec<f64>,      // N*N K^{-1} (computed via batched solve)
}

impl Default for GeneratedLogp {
    fn default() -> Self {
        let dev = CudaDevice::new(0).expect("CUDA device 0");
        let blas = CudaBlas::new(dev.clone()).expect("cuBLAS");
        let solver = CudaSolver::new(dev.clone()).expect("cuSOLVER");

        // Query workspace size for potrf
        let work_size = {
            let tmp = dev.alloc_zeros::<f64>(N * N).unwrap();
            solver.potrf_buffer_size(
                cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
                N as i32,
                &tmp,
                N as i32,
            ).unwrap()
        };

        Self {
            d_k: dev.alloc_zeros::<f64>(N * N).unwrap(),
            d_y: dev.alloc_zeros::<f64>(N).unwrap(),
            d_alpha: dev.alloc_zeros::<f64>(N).unwrap(),
            d_work: dev.alloc_zeros::<f64>(work_size).unwrap(),
            d_info: dev.alloc_zeros::<i32>(1).unwrap(),
            k_host: vec![0.0; N * N],
            dk_dls: vec![0.0; N * N],
            alpha_host: vec![0.0; N],
            kinv_host: vec![0.0; N * N],
            dev,
            blas,
            solver,
        }
    }
}
```

## Building Kernel Matrix (CPU, column-major for cuSOLVER)

cuSOLVER expects **column-major** layout. Build the kernel matrix in column-major order:

```rust
let eta_sq = eta * eta;
let inv_ls_sq = 1.0 / (ls * ls);
let sigma_sq = sigma * sigma;

for j in 0..N {
    for i in 0..N {
        let d = X_DATA[i] - X_DATA[j];
        let d_sq = d * d;
        let r_sq_scaled = d_sq * inv_ls_sq;
        let exp_term = (-0.5 * r_sq_scaled).exp();
        let k_ij = eta_sq * exp_term;

        // Column-major index
        let idx = j * N + i;
        self.dk_dls[idx] = k_ij * r_sq_scaled;

        self.k_host[idx] = if i == j {
            k_ij + sigma_sq + JITTER
        } else {
            k_ij
        };
    }
}
```

## Cholesky + Solve on GPU

```rust
// Upload K to GPU
self.dev.htod_copy_into(self.k_host.as_slice(), &mut self.d_k)?;

// Upload y to GPU
self.dev.htod_copy_into(Y_DATA, &mut self.d_y)?;

// Cholesky: K = L L^T (in-place, overwrites lower triangle of d_k)
self.solver.potrf(
    cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
    N as i32,
    &mut self.d_k,
    N as i32,
    &mut self.d_work,
    work_size as i32,
    &mut self.d_info,
)?;

// Check info
let info = self.dev.dtoh_sync_copy(&self.d_info)?;
if info[0] != 0 {
    return Err(SampleError::Recoverable(
        format!("Cholesky failed: info={}", info[0])
    ));
}

// Copy y -> alpha, then solve L L^T alpha = y
self.dev.dtod_copy(&self.d_y, &mut self.d_alpha)?;
self.solver.potrs(
    cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
    N as i32,
    1,  // nrhs
    &self.d_k,
    N as i32,
    &mut self.d_alpha,
    N as i32,
    &mut self.d_info,
)?;

// Download alpha = K^{-1}y
self.dev.dtoh_sync_copy_into(&self.d_alpha, &mut self.alpha_host)?;
```

## Log-determinant from GPU Cholesky

Download the diagonal of L (the Cholesky factor) to compute log|K|:

```rust
// Download L to compute log-det
self.dev.dtoh_sync_copy_into(&self.d_k, &mut self.k_host)?;

let mut log_det = 0.0;
for i in 0..N {
    // Column-major: L[i,i] at index i*N+i
    log_det += self.k_host[i * N + i].ln();
}
log_det *= 2.0;
```

## Computing K^{-1} on GPU (for gradients)

Use potri (Cholesky inverse) if available, or solve with identity:

```rust
// Option A: potri (in-place inverse from Cholesky factor)
self.solver.potri(
    cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
    N as i32,
    &mut self.d_k,  // L in, K^{-1} out (lower triangle)
    N as i32,
    &mut self.d_work,
    work_size as i32,
    &mut self.d_info,
)?;
self.dev.dtoh_sync_copy_into(&self.d_k, &mut self.kinv_host)?;

// Option B: Solve L L^T X = I column by column (if potri unavailable)
// Set up identity matrix, use potrs with nrhs=N
```

## GP Log-likelihood (same as CPU)

```
logp = -0.5 * (N * LN_2PI + log_det + y_t_alpha)
```
where `y_t_alpha = sum(y[i] * alpha[i])` is computed on CPU after downloading alpha.

## GP Gradients (CPU, using downloaded K^{-1} and alpha)

Same formula as CPU skill, but using column-major indexing for kinv_host:

```rust
// K^{-1} in column-major: kinv[i,j] = kinv_host[j*N+i]
for i in 0..N {
    let alpha_i = self.alpha_host[i];
    for j in 0..N {
        let alpha_j = self.alpha_host[j];
        let kinv_ij = if i >= j {
            self.kinv_host[j * N + i]  // lower triangle, column-major
        } else {
            self.kinv_host[i * N + j]  // symmetric
        };
        let w_ij = kinv_ij - alpha_i * alpha_j;
        // dk_dls is also column-major
        grad_ls += w_ij * self.dk_dls[j * N + i];
        grad_eta += w_ij * dk_deta;  // 2*K_ij/eta for ExpQuad
        if i == j {
            grad_sigma += w_ij * 2.0 * sigma_sq;
        }
    }
}
```

## When to Use GPU vs CPU

- **N < 50**: CPU (faer) is faster due to GPU launch overhead
- **N = 50-100**: GPU and CPU are similar; GPU wins if many evaluations
- **N > 100**: GPU is significantly faster (Cholesky is O(N^3))
- **N > 500**: GPU is 10-100x faster

The compiler auto-detects CUDA availability and selects GPU or CPU skill accordingly.

## Error Handling

cudarc operations return `Result`. Map errors to `SampleError::Recoverable`:
```rust
.map_err(|e| SampleError::Recoverable(format!("CUDA error: {}", e)))?;
```

## IMPORTANT: Column-Major Layout

cuBLAS/cuSOLVER use **column-major** (Fortran) layout, NOT row-major.
- Row-major: `A[i][j]` → `data[i * N + j]`
- Column-major: `A[i][j]` → `data[j * N + i]`

ALL matrices uploaded to GPU must be in column-major format.
