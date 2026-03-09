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

const LN_2PI: f64 = 1.8378770664093454835606594728112; // ln(2π)

#[derive(Storable, Clone)]
pub struct Draw {
    #[storable(dims("param"))]
    pub parameters: Vec<f64>,
}

#[derive(Clone)]
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
        // Clear gradient
        gradient.fill(0.0);
        
        // Extract parameters
        let mu_a = position[0];
        let log_sigma_a = position[1];
        let a_offset = &position[2..10];  // 8 elements
        let b = position[10];
        let log_sigma_y = position[11];
        
        // Transform to constrained space
        let sigma_a = log_sigma_a.exp();
        let sigma_y = log_sigma_y.exp();
        
        // Compute hierarchical parameters a = mu_a + sigma_a * a_offset
        let mut a = [0.0; 8];
        for i in 0..8 {
            a[i] = mu_a + sigma_a * a_offset[i];
        }
        
        let mut total_logp = 0.0;
        
        // 1. mu_a ~ Normal(0, 10)
        let mu_a_logp = -0.5 * LN_2PI - (10.0_f64).ln() - 0.5 * (mu_a / 10.0).powi(2);
        total_logp += mu_a_logp;
        gradient[0] = -mu_a / 100.0;
        
        // 2. sigma_a ~ HalfNormal(5) with LogTransform (includes Jacobian)
        let sigma_a_logp = (2.0_f64).ln() - 0.5 * LN_2PI - (5.0_f64).ln() 
            - 0.5 * (sigma_a / 5.0).powi(2) + log_sigma_a;
        total_logp += sigma_a_logp;
        // For HalfNormal(sigma_a | scale) with LogTransform:
        // d/d(log_sigma_a) = -sigma_a^2/scale^2 + 1 (Jacobian)
        gradient[1] = -sigma_a * sigma_a / 25.0 + 1.0;
        
        // 3. a_offset ~ Normal(0, 1) for each of 8 groups
        for i in 0..8 {
            let offset_logp = -0.5 * LN_2PI - 0.5 * a_offset[i].powi(2);
            total_logp += offset_logp;
            gradient[2 + i] = -a_offset[i];
        }
        
        // 4. b ~ Normal(0, 10)
        let b_logp = -0.5 * LN_2PI - (10.0_f64).ln() - 0.5 * (b / 10.0).powi(2);
        total_logp += b_logp;
        gradient[10] = -b / 100.0;
        
        // 5. sigma_y ~ HalfNormal(5) with LogTransform (includes Jacobian)
        let sigma_y_logp = (2.0_f64).ln() - 0.5 * LN_2PI - (5.0_f64).ln() 
            - 0.5 * (sigma_y / 5.0).powi(2) + log_sigma_y;
        total_logp += sigma_y_logp;
        gradient[11] = -sigma_y * sigma_y / 25.0 + 1.0;
        
        // 6. Observed likelihood: y ~ Normal(a[group_idx] + b * x, sigma_y)
        for i in 0..Y_N {
            let group_idx = X_1_DATA[i] as usize;
            let x = X_0_DATA[i];
            let y = Y_DATA[i];
            
            let mu_i = a[group_idx] + b * x;
            let residual = y - mu_i;
            
            let obs_logp = -0.5 * LN_2PI - log_sigma_y - 0.5 * (residual / sigma_y).powi(2);
            total_logp += obs_logp;
            
            // Gradients for observed likelihood
            let d_residual = residual / (sigma_y * sigma_y);
            
            // d/d_mu_a (affects mu_i through a[group_idx])
            gradient[0] += d_residual;
            
            // d/d_log_sigma_a (affects mu_i through a[group_idx])
            // mu_i = a[group_idx] = mu_a + sigma_a * a_offset[group_idx]
            // d(mu_i)/d(log_sigma_a) = d(mu_i)/d(sigma_a) * d(sigma_a)/d(log_sigma_a)
            //                        = a_offset[group_idx] * sigma_a
            gradient[1] += d_residual * a_offset[group_idx] * sigma_a;
            
            // d/d_a_offset[group_idx] (affects mu_i through a[group_idx])
            gradient[2 + group_idx] += d_residual * sigma_a;
            
            // d/d_b (affects mu_i directly)
            gradient[10] += d_residual * x;
            
            // d/d_log_sigma_y
            gradient[11] += -1.0 + (residual / sigma_y).powi(2);
        }
        
        Ok(total_logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}