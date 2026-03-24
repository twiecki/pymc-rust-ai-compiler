use std::collections::HashMap;
use nuts_rs::{CpuLogpFunc, CpuMathError, LogpError, Storable};
use nuts_storable::HasDims;
use thiserror::Error;
use crate::data::*;
use mlx_rs::{Array, Stream};
use mlx_rs::linalg;

#[derive(Debug, Error)]
pub enum SampleError {
    #[error("Recoverable: {0}")]
    Recoverable(String),
}

impl LogpError for SampleError {
    fn is_recoverable(&self) -> bool { true }
}

pub const N_PARAMS: usize = 3;
const N: usize = 200; // Y_N
const LN_2PI: f64 = 1.8378770664093453;
const JITTER: f64 = 1e-6;

#[derive(Storable, Clone)]
pub struct Draw {
    #[storable(dims("param"))]
    pub parameters: Vec<f64>,
}

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

impl HasDims for GeneratedLogp {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::from([("param".to_string(), N_PARAMS as u64)])
    }
}

impl CpuLogpFunc for GeneratedLogp {
    type LogpError = SampleError;
    type FlowParameters = ();
    type ExpandedVector = Draw;

    fn dim(&self) -> usize { N_PARAMS }

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
                let d = X_1_DATA[i] - X_1_DATA[j];
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
        let l_mlx = linalg::cholesky_device(&k_mlx, None, &cpu)
            .map_err(|e| SampleError::Recoverable(format!("Cholesky failed: {}", e)))?;

        // === 6. Triangular solve: L * tmp = y, then L^T * alpha = tmp ===
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

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}