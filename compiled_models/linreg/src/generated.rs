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

pub const N_PARAMS: usize = 3;

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
        // Extract parameters (unconstrained space)
        let alpha = position[0];
        let beta = position[1];
        let sigma_log = position[2];
        
        // Transform sigma from log space
        let sigma = sigma_log.exp();
        
        // Initialize gradient
        gradient.fill(0.0);
        let mut total_logp = 0.0;
        
        // Prior for alpha ~ Normal(0, 10)
        let alpha_diff = alpha - 0.0;
        let alpha_std = 10.0f64;
        let alpha_logp = -0.5 * std::f64::consts::TAU.ln() - alpha_std.ln() - 0.5 * (alpha_diff / alpha_std).powi(2);
        total_logp += alpha_logp;
        
        // Gradient for alpha
        gradient[0] += -alpha_diff / (alpha_std * alpha_std);
        
        // Prior for beta ~ Normal(0, 10)
        let beta_diff = beta - 0.0;
        let beta_std = 10.0f64;
        let beta_logp = -0.5 * std::f64::consts::TAU.ln() - beta_std.ln() - 0.5 * (beta_diff / beta_std).powi(2);
        total_logp += beta_logp;
        
        // Gradient for beta
        gradient[1] += -beta_diff / (beta_std * beta_std);
        
        // Prior for sigma ~ HalfNormal(5) with LogTransform
        // HalfNormal(x | sigma_scale) = 2 * Normal(x | 0, sigma_scale) for x >= 0
        // The factor of 2 comes from restricting to positive domain
        let sigma_scale = 5.0f64;
        let sigma_logp = 2.0f64.ln() - 0.5 * std::f64::consts::TAU.ln() - sigma_scale.ln() 
                        - 0.5 * (sigma / sigma_scale).powi(2) + sigma_log; // Jacobian adjustment
        total_logp += sigma_logp;
        
        // Gradient for sigma_log (chain rule)
        gradient[2] += -sigma * sigma / (sigma_scale * sigma_scale) + 1.0;
        
        // Likelihood for y ~ Normal(alpha + beta * x, sigma)
        let mut y_logp = 0.0;
        let mut alpha_grad_contrib = 0.0;
        let mut beta_grad_contrib = 0.0;
        let mut sigma_grad_contrib = 0.0;
        
        for i in 0..Y_N {
            let y_i = Y_DATA[i];
            let x_i = X_0_DATA[i];
            let mu_i = alpha + beta * x_i;
            let residual = y_i - mu_i;
            
            // Log-likelihood for this observation
            let obs_logp = -0.5 * std::f64::consts::TAU.ln() - sigma.ln() - 0.5 * (residual / sigma).powi(2);
            y_logp += obs_logp;
            
            // Gradient contributions
            alpha_grad_contrib += residual / (sigma * sigma);
            beta_grad_contrib += residual * x_i / (sigma * sigma);
            sigma_grad_contrib += -1.0 / sigma + residual * residual / (sigma * sigma * sigma);
        }
        
        total_logp += y_logp;
        
        // Add likelihood gradients
        gradient[0] += alpha_grad_contrib;
        gradient[1] += beta_grad_contrib;
        gradient[2] += sigma_grad_contrib * sigma; // Chain rule for log-transform
        
        Ok(total_logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}