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

pub const N_PARAMS: usize = 12;

const LN_2PI: f64 = 1.8378770664093453;
const LN_2: f64 = 0.6931471805599453;

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
        // Parameter extraction
        let mu_a = position[0];
        let log_sigma_a = position[1];
        let a_offset = &position[2..10];  // 8 elements
        let b = position[10];
        let log_sigma_y = position[11];

        // Transform log parameters to constrained space
        let sigma_a = log_sigma_a.exp();
        let sigma_y = log_sigma_y.exp();

        // Precompute all expensive operations outside the loop
        let sigma_a_scaled = sigma_a * 0.2;  // sigma_a / 5
        let sigma_y_scaled = sigma_y * 0.2;  // sigma_y / 5
        let inv_sigma_y = 1.0 / sigma_y;
        let inv_sigma_y_sq = inv_sigma_y * inv_sigma_y;
        let log_norm = -0.5 * LN_2PI - log_sigma_y;
        let mu_a_scaled = mu_a * 0.1;  // mu_a / 10
        let b_scaled = b * 0.1;  // b / 10
        
        // Precompute group intercepts in a single operation
        let mut a = [0.0f64; 8];
        for i in 0..8 {
            a[i] = mu_a + sigma_a * a_offset[i];
        }

        let mut logp = 0.0;
        gradient.fill(0.0);

        // Prior: mu_a ~ Normal(0, 10)
        logp += -0.5 * LN_2PI - (10.0_f64).ln() - 0.5 * mu_a_scaled * mu_a_scaled;
        gradient[0] = -mu_a_scaled * 0.1;

        // Prior: sigma_a ~ HalfNormal(5) with LogTransform + Jacobian
        logp += LN_2 - 0.5 * LN_2PI - (5.0_f64).ln() - 0.5 * sigma_a_scaled * sigma_a_scaled + log_sigma_a;
        gradient[1] = 1.0 - sigma_a_scaled * sigma_a_scaled;

        // Prior: a_offset ~ Normal(0, 1, shape=8)
        let mut a_offset_logp = 0.0;
        for i in 0..8 {
            let offset_sq = a_offset[i] * a_offset[i];
            a_offset_logp += -0.5 * LN_2PI - 0.5 * offset_sq;
            gradient[2 + i] = -a_offset[i];
        }
        logp += a_offset_logp;

        // Prior: b ~ Normal(0, 10)
        logp += -0.5 * LN_2PI - (10.0_f64).ln() - 0.5 * b_scaled * b_scaled;
        gradient[10] = -b_scaled * 0.1;

        // Prior: sigma_y ~ HalfNormal(5) with LogTransform + Jacobian
        logp += LN_2 - 0.5 * LN_2PI - (5.0_f64).ln() - 0.5 * sigma_y_scaled * sigma_y_scaled + log_sigma_y;
        gradient[11] = 1.0 - sigma_y_scaled * sigma_y_scaled;

        // Likelihood computation with fused gradient accumulation
        let mut grad_a = [0.0f64; 8];
        let mut grad_b_acc = 0.0;
        let mut sum_residual_sq = 0.0;

        // Manual loop unrolling for better performance
        let mut i = 0;
        while i + 3 < Y_N {
            // Process 4 observations at once
            for j in 0..4 {
                let idx = i + j;
                let group = unsafe { *X_1_DATA.get_unchecked(idx) } as usize;
                let x = unsafe { *X_0_DATA.get_unchecked(idx) };
                let y = unsafe { *Y_DATA.get_unchecked(idx) };
                
                let mu_i = unsafe { *a.get_unchecked(group) } + b * x;
                let residual = y - mu_i;
                let residual_sq = residual * residual;
                
                logp += log_norm - 0.5 * residual_sq * inv_sigma_y_sq;
                
                let residual_scaled = residual * inv_sigma_y_sq;
                unsafe { *grad_a.get_unchecked_mut(group) += residual_scaled; }
                grad_b_acc += residual_scaled * x;
                sum_residual_sq += residual_sq;
            }
            i += 4;
        }
        
        // Handle remaining observations
        while i < Y_N {
            let group = unsafe { *X_1_DATA.get_unchecked(i) } as usize;
            let x = unsafe { *X_0_DATA.get_unchecked(i) };
            let y = unsafe { *Y_DATA.get_unchecked(i) };
            
            let mu_i = unsafe { *a.get_unchecked(group) } + b * x;
            let residual = y - mu_i;
            let residual_sq = residual * residual;
            
            logp += log_norm - 0.5 * residual_sq * inv_sigma_y_sq;
            
            let residual_scaled = residual * inv_sigma_y_sq;
            unsafe { *grad_a.get_unchecked_mut(group) += residual_scaled; }
            grad_b_acc += residual_scaled * x;
            sum_residual_sq += residual_sq;
            i += 1;
        }

        // Chain rule for unconstrained parameters
        
        // mu_a gradient: sum of all group gradients
        let grad_mu_a_likelihood = grad_a[0] + grad_a[1] + grad_a[2] + grad_a[3] + 
                                  grad_a[4] + grad_a[5] + grad_a[6] + grad_a[7];
        gradient[0] += grad_mu_a_likelihood;

        // log_sigma_a gradient: chain rule through group intercepts
        let mut grad_sigma_a = 0.0;
        for i in 0..8 {
            grad_sigma_a += grad_a[i] * a_offset[i];
            gradient[2 + i] += grad_a[i] * sigma_a;  // a_offset gradient
        }
        gradient[1] += grad_sigma_a * sigma_a;

        // b and log_sigma_y gradients
        gradient[10] += grad_b_acc;
        gradient[11] += -(Y_N as f64) + sum_residual_sq * inv_sigma_y_sq;

        Ok(logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}