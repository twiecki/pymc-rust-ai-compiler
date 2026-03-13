# Skill: Hardware-Accelerated Gaussian Process Models (Apple Accelerate / AMX)

This model contains a Gaussian Process and Apple Silicon is available.
Use Apple's Accelerate framework for hardware-accelerated LAPACK operations
(Cholesky, solve, inverse). Accelerate uses the AMX coprocessor on Apple Silicon
for high-performance matrix operations in f64 precision — ~1.3x faster than faer.

## Dependencies

No crate dependencies needed. The Accelerate framework is linked via `build.rs`
which is generated automatically.

## CRITICAL: LAPACK extern declarations

Declare the required LAPACK functions at the top of the file. These are provided
by Apple's Accelerate framework (linked automatically).

```rust
extern "C" {
    fn dpotrf_(uplo: *const u8, n: *const i32, a: *mut f64, lda: *const i32, info: *mut i32);
    fn dpotrs_(
        uplo: *const u8, n: *const i32, nrhs: *const i32,
        a: *const f64, lda: *const i32,
        b: *mut f64, ldb: *const i32, info: *mut i32,
    );
    fn dpotri_(uplo: *const u8, n: *const i32, a: *mut f64, lda: *const i32, info: *mut i32);
}
```

## CRITICAL: Use uplo="U" for row-major flat arrays

LAPACK assumes column-major storage. Our flat `Vec<f64>` arrays use row-major
layout (`a[i * N + j]`). For symmetric matrices, the data is identical, but
LAPACK's "lower triangle" maps to our row-major "upper triangle" and vice versa.

**ALWAYS use `b"U"` as the uplo parameter.** This makes LAPACK's upper triangle
(column-major) correspond to our lower triangle (row-major), so
`kinv[i * N + j]` where `i >= j` correctly accesses the filled triangle.

## MANDATORY: Pre-allocated Struct with Default

GP models MUST pre-allocate all buffers as struct fields.
DO NOT allocate inside `logp()` — it is called thousands of times.

**NOTE**: `dpotrf` overwrites the input matrix with the Cholesky factor, so we
need `k_orig` to preserve the original K for gradient computation.

```rust
const LN_2PI: f64 = 1.8378770664093453;
const JITTER: f64 = 1e-6;
const N: usize = /* set from model */;

pub struct GeneratedLogp {
    k_mat: Vec<f64>,      // N*N — overwritten by dpotrf with Cholesky factor
    k_orig: Vec<f64>,     // N*N — original kernel matrix (preserved for gradients)
    kinv: Vec<f64>,       // N*N — K^{-1} (from dpotrf + dpotri)
    alpha: Vec<f64>,      // N for K^{-1}y
    dk_dls: Vec<f64>,     // N*N for dK/d(log_ls)
}

impl Default for GeneratedLogp {
    fn default() -> Self {
        Self {
            k_mat: vec![0.0; N * N],
            k_orig: vec![0.0; N * N],
            kinv: vec![0.0; N * N],
            alpha: vec![0.0; N],
            dk_dls: vec![0.0; N * N],
        }
    }
}
```

## Complete Accelerate GP logp Implementation

Here is the COMPLETE, VERIFIED logp implementation. Copy this pattern exactly.

```rust
fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, SampleError> {
    let n = N as i32;

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

    // === 2. Build kernel matrix K (flat, row-major, symmetric) ===
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
                self.k_mat[i * N + j] = k_ij + sigma_sq + JITTER;
            } else {
                self.k_mat[i * N + j] = k_ij;
                self.k_mat[j * N + i] = k_ij;
            }
        }
    }

    // === 3. Save original K, prepare alpha and kinv ===
    self.k_orig.copy_from_slice(&self.k_mat);
    self.kinv.copy_from_slice(&self.k_mat);
    for i in 0..N {
        self.alpha[i] = Y_DATA[i];
    }

    // === 4. Cholesky factorization ===
    // Use "U" because our row-major layout = LAPACK's transposed column-major
    let mut info = 0i32;
    unsafe {
        dpotrf_(b"U".as_ptr(), &n, self.k_mat.as_mut_ptr(), &n, &mut info);
    }
    if info != 0 {
        return Err(SampleError::Recoverable(
            format!("Cholesky failed: dpotrf info={}", info),
        ));
    }

    // === 5. Log-determinant from Cholesky diagonal ===
    let mut log_det = 0.0;
    for i in 0..N {
        log_det += self.k_mat[i * N + i].ln();
    }
    log_det *= 2.0;

    // === 6. Solve K * alpha = y ===
    let nrhs = 1i32;
    unsafe {
        dpotrs_(
            b"U".as_ptr(), &n, &nrhs,
            self.k_mat.as_ptr(), &n,
            self.alpha.as_mut_ptr(), &n, &mut info,
        );
    }
    if info != 0 {
        return Err(SampleError::Recoverable(
            format!("Solve failed: dpotrs info={}", info),
        ));
    }

    // === 7. K^{-1}: Cholesky then invert ===
    unsafe {
        dpotrf_(b"U".as_ptr(), &n, self.kinv.as_mut_ptr(), &n, &mut info);
    }
    if info != 0 {
        return Err(SampleError::Recoverable(
            format!("Cholesky for inverse failed: info={}", info),
        ));
    }
    unsafe {
        dpotri_(b"U".as_ptr(), &n, self.kinv.as_mut_ptr(), &n, &mut info);
    }
    if info != 0 {
        return Err(SampleError::Recoverable(
            format!("Inverse failed: dpotri info={}", info),
        ));
    }

    // === 8. y^T K^{-1} y ===
    let mut y_t_alpha = 0.0;
    for i in 0..N {
        y_t_alpha += Y_DATA[i] * self.alpha[i];
    }

    // === 9. GP log-likelihood ===
    let gp_logp = -0.5 * (N as f64 * LN_2PI + log_det + y_t_alpha);

    // === 10. Priors (HalfNormal sigma=5) ===
    let prior_const = -1.83522929514961; // log(2) - 0.5*ln(2*pi) - ln(5)
    let ls_prior = prior_const - 0.5 * (ls / 5.0).powi(2) + log_ls;
    let eta_prior = prior_const - 0.5 * (eta / 5.0).powi(2) + log_eta;
    let sigma_prior = prior_const - 0.5 * (sigma / 5.0).powi(2) + log_sigma;

    let total_logp = gp_logp + ls_prior + eta_prior + sigma_prior;

    // === 11. Gradients ===
    gradient.fill(0.0);

    let mut grad_log_ls = 0.0;
    let mut grad_log_eta = 0.0;
    let mut grad_log_sigma = 0.0;

    // With uplo="U", dpotri fills our row-major lower triangle
    // Access kinv symmetrically: lower triangle at [i*N+j] where i >= j
    for i in 0..N {
        let alpha_i = self.alpha[i];
        for j in 0..N {
            let alpha_j = self.alpha[j];
            let kinv_ij = if i >= j { self.kinv[i * N + j] } else { self.kinv[j * N + i] };
            let w_ij = kinv_ij - alpha_i * alpha_j;

            grad_log_ls += w_ij * self.dk_dls[i * N + j];

            // dK/d(log_eta) = 2 * K_ij (kernel part only, no noise)
            let k_ij_no_noise = if i == j {
                self.k_orig[i * N + j] - sigma_sq - JITTER
            } else {
                self.k_orig[i * N + j]
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

    // Prior gradients
    gradient[0] += 1.0 - ls * ls / 25.0;
    gradient[1] += 1.0 - eta * eta / 25.0;
    gradient[2] += 1.0 - sigma * sigma / 25.0;

    Ok(total_logp)
}
```

## LAPACK API Reference

### dpotrf — Cholesky Factorization
```rust
// Factors symmetric positive-definite matrix A
// uplo: b"U" (MUST use "U" for row-major flat arrays)
// Overwrites triangle of A with Cholesky factor
unsafe { dpotrf_(b"U".as_ptr(), &n, a.as_mut_ptr(), &n, &mut info); }
```

### dpotrs — Solve Using Cholesky Factor
```rust
// Solves A*X = B where A was factored by dpotrf
// a: Cholesky factor from dpotrf (NOT modified)
// b: right-hand side, overwritten with solution X
unsafe { dpotrs_(b"U".as_ptr(), &n, &nrhs, a.as_ptr(), &n, b.as_mut_ptr(), &n, &mut info); }
```

### dpotri — Inverse From Cholesky Factor
```rust
// Computes A^{-1} from Cholesky factor (from dpotrf)
// With uplo="U", fills our row-major lower triangle
// Access: if i >= j { kinv[i*N+j] } else { kinv[j*N+i] }
unsafe { dpotri_(b"U".as_ptr(), &n, a.as_mut_ptr(), &n, &mut info); }
```

## Key Differences from faer Skill

1. **No crate dependencies** — uses Apple Accelerate via build.rs linking
2. **Flat Vec<f64>** instead of `faer::Mat<f64>` — index as `[i * N + j]`
3. **dpotrf overwrites input** — must save original K in `k_orig` for gradients
4. **uplo="U" required** — compensates for row-major vs column-major mismatch
5. **Full f64 precision** — no conversion needed
6. **unsafe blocks** required for LAPACK FFI calls

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
if i == j { self.k_mat[i * N + j] = k_ij + sigma_sq + JITTER; }
```

## Error Handling

Check `info` after every LAPACK call. Map errors to `SampleError::Recoverable`:
```rust
if info != 0 {
    return Err(SampleError::Recoverable(format!("LAPACK error: info={}", info)));
}
```
