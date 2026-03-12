use std::collections::HashMap;
use nuts_rs::{CpuLogpFunc, CpuMathError, LogpError, Storable};
use nuts_storable::HasDims;
use thiserror::Error;
use crate::data::*;

#[derive(Debug, Error)]
pub enum SampleError {
    #[error("Recoverable: {0}")]
    Recoverable(String),
}

impl LogpError for SampleError {
    fn is_recoverable(&self) -> bool { true }
}

pub const N_PARAMS: usize = 2;

const LN_2PI: f64 = 1.8378770664093453; // ln(2π)

// Precomputed constants for priors
const MU_PRIOR_LOGP_CONST: f64 = -0.5 * LN_2PI - 2.3025850929940457; // -ln(10)
const MU_PRIOR_VAR_INV: f64 = 0.01; // 1/100

const SIGMA_PRIOR_LOGP_CONST: f64 = 0.6931471805599453 - 0.5 * LN_2PI - 1.6094379124341003; // ln(2) - ln(5)
const SIGMA_PRIOR_SCALE_SQ: f64 = 0.04; // 1/25

// Likelihood constants
const LIKELIHOOD_LOGP_CONST: f64 = -0.5 * LN_2PI;

#[derive(Storable, Clone)]
pub struct Draw {
    #[storable(dims("param"))]
    pub parameters: Vec<f64>,
}

#[derive(Clone, Default)]
pub struct GeneratedLogp;

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
        // Extract parameters
        let mu = position[0];
        let log_sigma = position[1];
        let sigma = log_sigma.exp();
        
        // Precompute values used in likelihood
        let inv_sigma = 1.0 / sigma;
        let inv_sigma_sq = inv_sigma * inv_sigma;
        
        // Prior: mu ~ Normal(0, 10)
        let mu_sq = mu * mu;
        let mu_logp = MU_PRIOR_LOGP_CONST - 0.5 * mu_sq * MU_PRIOR_VAR_INV;
        let mu_grad = -mu * MU_PRIOR_VAR_INV;
        
        // Prior: sigma ~ HalfNormal(5) with LogTransform + Jacobian
        let sigma_sq = sigma * sigma;
        let sigma_logp = SIGMA_PRIOR_LOGP_CONST - 0.5 * sigma_sq * SIGMA_PRIOR_SCALE_SQ + log_sigma;
        let sigma_grad = -sigma_sq * SIGMA_PRIOR_SCALE_SQ + 1.0;
        
        // Likelihood: y ~ Normal(mu, sigma) - highly optimized loop
        let likelihood_log_norm = LIKELIHOOD_LOGP_CONST - log_sigma;
        
        // Use separate accumulator variables for better SIMD potential
        let mut sum_logp_terms = 0.0;
        let mut sum_residuals = 0.0;
        let mut sum_residual_squares = 0.0;
        
        // Optimized loop with explicit vectorization hints
        unsafe {
            let y_ptr = Y_DATA.as_ptr();
            
            // Process in chunks of 8 for better SIMD utilization
            let chunks = Y_N / 8;
            let remainder = Y_N % 8;
            
            for chunk in 0..chunks {
                let base_idx = chunk * 8;
                
                // Load 8 values and compute residuals
                let y0 = *y_ptr.add(base_idx);
                let y1 = *y_ptr.add(base_idx + 1);
                let y2 = *y_ptr.add(base_idx + 2);
                let y3 = *y_ptr.add(base_idx + 3);
                let y4 = *y_ptr.add(base_idx + 4);
                let y5 = *y_ptr.add(base_idx + 5);
                let y6 = *y_ptr.add(base_idx + 6);
                let y7 = *y_ptr.add(base_idx + 7);
                
                let r0 = y0 - mu;
                let r1 = y1 - mu;
                let r2 = y2 - mu;
                let r3 = y3 - mu;
                let r4 = y4 - mu;
                let r5 = y5 - mu;
                let r6 = y6 - mu;
                let r7 = y7 - mu;
                
                // Accumulate residuals for gradient
                sum_residuals += r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
                
                // Compute and accumulate squared residuals
                let r0_sq = r0 * r0;
                let r1_sq = r1 * r1;
                let r2_sq = r2 * r2;
                let r3_sq = r3 * r3;
                let r4_sq = r4 * r4;
                let r5_sq = r5 * r5;
                let r6_sq = r6 * r6;
                let r7_sq = r7 * r7;
                
                sum_residual_squares += r0_sq + r1_sq + r2_sq + r3_sq + r4_sq + r5_sq + r6_sq + r7_sq;
            }
            
            // Handle remaining elements
            for i in (chunks * 8)..Y_N {
                let y_i = *y_ptr.add(i);
                let residual = y_i - mu;
                sum_residuals += residual;
                sum_residual_squares += residual * residual;
            }
        }
        
        // Compute final values
        let sum_residual_squares_scaled = sum_residual_squares * inv_sigma_sq;
        let likelihood_logp = (Y_N as f64) * likelihood_log_norm - 0.5 * sum_residual_squares_scaled;
        
        // Total logp and gradients
        let total_logp = mu_logp + sigma_logp + likelihood_logp;
        
        gradient[0] = mu_grad + sum_residuals * inv_sigma_sq;
        gradient[1] = sigma_grad + sum_residual_squares_scaled - (Y_N as f64);
        
        Ok(total_logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}