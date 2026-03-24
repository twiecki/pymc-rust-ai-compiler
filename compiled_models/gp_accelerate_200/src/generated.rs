use std::collections::HashMap;
use nuts_rs::{CpuLogpFunc, CpuMathError, LogpError, Storable};
use nuts_storable::HasDims;
use thiserror::Error;
use crate::data::*;

extern "C" {
    fn dpotrf_(uplo: *const u8, n: *const i32, a: *mut f64, lda: *const i32, info: *mut i32);
    fn dpotrs_(
        uplo: *const u8, n: *const i32, nrhs: *const i32,
        a: *const f64, lda: *const i32,
        b: *mut f64, ldb: *const i32, info: *mut i32,
    );
    fn dpotri_(uplo: *const u8, n: *const i32, a: *mut f64, lda: *const i32, info: *mut i32);
}

#[derive(Debug, Error)]
pub enum SampleError {
    #[error("Recoverable: {0}")]
    Recoverable(String),
}

impl LogpError for SampleError {
    fn is_recoverable(&self) -> bool { true }
}

pub const N_PARAMS: usize = 3;
const N: usize = 200;
const LN_2PI: f64 = 1.8378770664093453;
const JITTER: f64 = 1e-6;

#[derive(Storable, Clone)]
pub struct Draw {
    #[storable(dims("param"))]
    pub parameters: Vec<f64>,
}

pub struct GeneratedLogp {
    k_mat: Vec<f64>,      // N*N — after dpotrf, holds Cholesky factor L
    k_orig: Vec<f64>,     // N*N — original kernel matrix (preserved for gradients)
    kinv: Vec<f64>,       // N*N — K^{-1} (from dpotrf + dpotri)
    alpha: Vec<f64>,      // N for K^{-1}y
    dk_dls: Vec<f64>,     // N*N for dK/d(log_ls)
    dk_deta: Vec<f64>,    // N*N for dK/d(log_eta)
}

impl Default for GeneratedLogp {
    fn default() -> Self {
        Self {
            k_mat: vec![0.0; N * N],
            k_orig: vec![0.0; N * N],
            kinv: vec![0.0; N * N],
            alpha: vec![0.0; N],
            dk_dls: vec![0.0; N * N],
            dk_deta: vec![0.0; N * N],
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
        let n = N as i32;

        // Extract unconstrained parameters
        let log_ls = position[0];
        let log_eta = position[1];
        let log_sigma = position[2];

        // Transform to constrained space
        let ls = log_ls.exp();
        let eta = log_eta.exp();
        let sigma = log_sigma.exp();

        // Initialize gradient
        gradient[0] = 0.0;
        gradient[1] = 0.0;
        gradient[2] = 0.0;

        let mut logp = 0.0;

        // Prior terms
        let inv_scale = 0.2; // 1/5
        let log_const = -1.83522929514961;

        let ls_scaled_sq = (inv_scale * ls).powi(2);
        logp += log_const - 0.5 * ls_scaled_sq + log_ls;
        gradient[0] += -ls_scaled_sq + 1.0;

        let eta_scaled_sq = (inv_scale * eta).powi(2);
        logp += log_const - 0.5 * eta_scaled_sq + log_eta;
        gradient[1] += -eta_scaled_sq + 1.0;

        let sigma_scaled_sq = (inv_scale * sigma).powi(2);
        logp += log_const - 0.5 * sigma_scaled_sq + log_sigma;
        gradient[2] += -sigma_scaled_sq + 1.0;

        // Build GP kernel matrix
        let eta_sq = eta * eta;
        let inv_ls_sq = 1.0 / (ls * ls);
        let sigma_sq = sigma * sigma;

        for i in 0..N {
            for j in 0..=i {
                let xi = X_1_DATA[i];
                let xj = X_1_DATA[j];
                let d = xi - xj;
                let d_sq = d * d;
                let r_sq_scaled = d_sq * inv_ls_sq;
                let exp_term = (-0.5 * r_sq_scaled).exp();
                let k_ij = eta_sq * exp_term;

                self.dk_dls[i * N + j] = k_ij * r_sq_scaled;
                self.dk_dls[j * N + i] = self.dk_dls[i * N + j];

                self.dk_deta[i * N + j] = 2.0 * k_ij;
                self.dk_deta[j * N + i] = self.dk_deta[i * N + j];

                if i == j {
                    self.k_mat[i * N + j] = k_ij + sigma_sq + JITTER;
                } else {
                    self.k_mat[i * N + j] = k_ij;
                    self.k_mat[j * N + i] = k_ij;
                }
            }
        }

        // Save original K for gradients, prepare kinv and alpha
        self.k_orig.copy_from_slice(&self.k_mat);
        self.kinv.copy_from_slice(&self.k_mat);
        for i in 0..N {
            self.alpha[i] = Y_DATA[i];
        }

        // Cholesky: k_mat = L where K = L L^T
        let mut info = 0i32;
        unsafe {
            dpotrf_(b"U".as_ptr(), &n, self.k_mat.as_mut_ptr(), &n, &mut info);
        }
        if info != 0 {
            return Err(SampleError::Recoverable(
                format!("Cholesky failed: dpotrf info={}", info),
            ));
        }

        // Log-determinant from Cholesky diagonal
        let mut log_det = 0.0;
        for i in 0..N {
            log_det += self.k_mat[i * N + i].ln();
        }
        log_det *= 2.0;

        // Solve K * alpha = y using Cholesky factor
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

        // y^T K^{-1} y
        let mut quad_form = 0.0;
        for i in 0..N {
            quad_form += Y_DATA[i] * self.alpha[i];
        }

        // GP log-likelihood
        let gp_const = -183.78770664093454; // -N * LN_2PI / 2
        logp += gp_const - 0.5 * log_det - 0.5 * quad_form;

        // K^{-1}: Cholesky then invert
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

        // Gradient w.r.t. log_ls
        let mut grad_ls = 0.0;
        for i in 0..N {
            let alpha_i = self.alpha[i];
            for j in 0..N {
                let alpha_j = self.alpha[j];
                let kinv_ij = if i >= j { self.kinv[i * N + j] } else { self.kinv[j * N + i] };
                let w_ij = kinv_ij - alpha_i * alpha_j;
                grad_ls += w_ij * self.dk_dls[i * N + j];
            }
        }
        gradient[0] += -0.5 * grad_ls;

        // Gradient w.r.t. log_eta
        let mut grad_eta = 0.0;
        for i in 0..N {
            let alpha_i = self.alpha[i];
            for j in 0..N {
                let alpha_j = self.alpha[j];
                let kinv_ij = if i >= j { self.kinv[i * N + j] } else { self.kinv[j * N + i] };
                let w_ij = kinv_ij - alpha_i * alpha_j;
                grad_eta += w_ij * self.dk_deta[i * N + j];
            }
        }
        gradient[1] += -0.5 * grad_eta;

        // Gradient w.r.t. log_sigma
        let mut grad_sigma = 0.0;
        for i in 0..N {
            let alpha_i = self.alpha[i];
            let kinv_ii = if i >= i { self.kinv[i * N + i] } else { self.kinv[i * N + i] };
            let w_ii = kinv_ii - alpha_i * alpha_i;
            grad_sigma += w_ii * 2.0 * sigma_sq;
        }
        gradient[2] += -0.5 * grad_sigma;

        Ok(logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}
