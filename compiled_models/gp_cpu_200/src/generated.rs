use std::collections::HashMap;
use nuts_rs::{CpuLogpFunc, CpuMathError, LogpError, Storable};
use nuts_storable::HasDims;
use thiserror::Error;
use faer::{Mat, Par, Spec};
use faer::linalg::cholesky::llt::{factor, solve, inverse};
use faer::linalg::cholesky::llt::factor::LltParams;
use faer::dyn_stack::{MemBuffer, MemStack};
use crate::data::*;

#[derive(Debug, Error)]
pub enum SampleError {
    #[error("Recoverable: {0}")]
    Recoverable(String),
}

impl LogpError for SampleError {
    fn is_recoverable(&self) -> bool { true }
}

pub const N_PARAMS: usize = 3;
const N: usize = 200; // Number of observations
const LN_2PI: f64 = 1.8378770664093453; // ln(2π) — use this EXACT value
const JITTER: f64 = 1e-6;

#[derive(Storable, Clone)]
pub struct Draw {
    #[storable(dims("param"))]
    pub parameters: Vec<f64>,
}

pub struct GeneratedLogp {
    k_mat: Mat<f64>,      // N×N kernel matrix (overwritten each call)
    alpha: Mat<f64>,      // N×1 for K^{-1} y
    kinv: Mat<f64>,       // N×N for inverse
    dk_dls: Vec<f64>,     // N*N flat storage for dK/d(log_ls)
    dk_deta: Vec<f64>,    // N*N flat storage for dK/d(log_eta)
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
            dk_deta: vec![0.0; N * N],
            chol_buf: MemBuffer::new(chol_scratch),
            inv_buf: MemBuffer::new(inv_scratch),
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
        // Extract unconstrained parameters
        let log_ls = position[0];   // ls_log__
        let log_eta = position[1];  // eta_log__  
        let log_sigma = position[2]; // sigma_log__
        
        // Transform to constrained space
        let ls = log_ls.exp();
        let eta = log_eta.exp(); 
        let sigma = log_sigma.exp();
        
        // Initialize gradient
        gradient[0] = 0.0;
        gradient[1] = 0.0;
        gradient[2] = 0.0;
        
        let mut logp = 0.0;
        
        // Prior terms - using exact PyMC constants from PyTensor graph
        let inv_scale = 0.2; // 1/5
        let log_const = -1.83522929514961; // From PyTensor: log(2) - 0.5*log(2π) - log(5)
        
        // ls prior
        let ls_scaled_sq = (inv_scale * ls).powi(2);
        let ls_prior = log_const - 0.5 * ls_scaled_sq + log_ls;
        logp += ls_prior;
        gradient[0] += -ls_scaled_sq + 1.0;
        
        // eta prior  
        let eta_scaled_sq = (inv_scale * eta).powi(2);
        let eta_prior = log_const - 0.5 * eta_scaled_sq + log_eta;
        logp += eta_prior;
        gradient[1] += -eta_scaled_sq + 1.0;
        
        // sigma prior
        let sigma_scaled_sq = (inv_scale * sigma).powi(2);
        let sigma_prior = log_const - 0.5 * sigma_scaled_sq + log_sigma;
        logp += sigma_prior;
        gradient[2] += -sigma_scaled_sq + 1.0;
        
        // Build GP kernel matrix K with ExpQuad kernel
        // K_ij = eta^2 * exp(-0.5 * |x_i - x_j|^2 / ls^2) + sigma^2 * δ_ij + jitter * δ_ij
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
                
                // Store derivatives for gradient computation
                // dK/d(log_ls): chain rule gives k_ij * r_sq_scaled
                self.dk_dls[i * N + j] = k_ij * r_sq_scaled;
                self.dk_dls[j * N + i] = self.dk_dls[i * N + j];
                
                // dK/d(log_eta): chain rule gives 2 * k_ij  
                self.dk_deta[i * N + j] = 2.0 * k_ij;
                self.dk_deta[j * N + i] = self.dk_deta[i * N + j];
                
                if i == j {
                    // Diagonal: signal kernel + noise + jitter
                    self.k_mat[(i, j)] = k_ij + sigma_sq + JITTER;
                } else {
                    // Off-diagonal: only signal kernel
                    self.k_mat[(i, j)] = k_ij;
                    self.k_mat[(j, i)] = k_ij;
                }
            }
        }
        
        // Cholesky decomposition K = L L^T
        factor::cholesky_in_place(
            self.k_mat.as_mut(),
            Default::default(),
            Par::Seq,
            MemStack::new(&mut self.chol_buf),
            Spec::<LltParams, f64>::default(),
        ).map_err(|_| {
            SampleError::Recoverable("Cholesky failed: not positive definite".to_string())
        })?;
        
        // Compute log determinant: log|K| = 2 * sum(log(L_ii))
        let mut log_det = 0.0;
        for i in 0..N {
            log_det += self.k_mat[(i, i)].ln();
        }
        log_det *= 2.0;
        
        // Solve K * alpha = y to get alpha = K^{-1} * y
        for i in 0..N {
            self.alpha[(i, 0)] = Y_DATA[i];
        }
        solve::solve_in_place(
            self.k_mat.as_ref(),  // L from Cholesky
            self.alpha.as_mut(),
            Par::Seq,
            MemStack::new(&mut []),
        );
        
        // Compute y^T K^{-1} y = alpha^T alpha (since alpha = K^{-1} y)
        let mut quad_form = 0.0;
        for i in 0..N {
            let alpha_i = self.alpha[(i, 0)];
            quad_form += alpha_i * alpha_i;
        }
        
        // GP likelihood: logp(y | K) = -0.5 * (N * log(2π) + log|K| + y^T K^{-1} y)
        // Using the exact constants from PyMC's PyTensor graph
        let gp_const = -183.78770664093454; // -N * log(2π) / 2 exactly
        let gp_logp = gp_const - 0.5 * log_det - 0.5 * quad_form;
        logp += gp_logp;
        
        // Compute K^{-1} for gradients
        inverse::inverse(
            self.kinv.as_mut(),
            self.k_mat.as_ref(),  // L from Cholesky
            Par::Seq,
            MemStack::new(&mut self.inv_buf),
        );
        
        // Gradient computation for GP hyperparameters
        // For parameter θ: d(logp)/dθ = -0.5 * tr((K^{-1} - α α^T) dK/dθ)
        
        // Gradient w.r.t. log_ls
        let mut grad_ls = 0.0;
        for i in 0..N {
            let alpha_i = self.alpha[(i, 0)];
            for j in 0..N {
                let alpha_j = self.alpha[(j, 0)];
                let kinv_ij = if i >= j { self.kinv[(i, j)] } else { self.kinv[(j, i)] };
                let w_ij = kinv_ij - alpha_i * alpha_j;
                grad_ls += w_ij * self.dk_dls[i * N + j];
            }
        }
        gradient[0] += -0.5 * grad_ls;
        
        // Gradient w.r.t. log_eta
        let mut grad_eta = 0.0;
        for i in 0..N {
            let alpha_i = self.alpha[(i, 0)];
            for j in 0..N {
                let alpha_j = self.alpha[(j, 0)];
                let kinv_ij = if i >= j { self.kinv[(i, j)] } else { self.kinv[(j, i)] };
                let w_ij = kinv_ij - alpha_i * alpha_j;
                grad_eta += w_ij * self.dk_deta[i * N + j];
            }
        }
        gradient[1] += -0.5 * grad_eta;
        
        // Gradient w.r.t. log_sigma (noise parameter)
        // dK/d(log_sigma) = 2 * sigma^2 * I (only diagonal elements)
        let mut grad_sigma = 0.0;
        for i in 0..N {
            let alpha_i = self.alpha[(i, 0)];
            let kinv_ii = self.kinv[(i, i)];
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