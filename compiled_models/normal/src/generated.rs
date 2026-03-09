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
        // Extract parameters from unconstrained space
        let mu = position[0];
        let sigma_log = position[1];
        let sigma = sigma_log.exp();
        
        // Initialize gradients
        gradient[0] = 0.0; // d/d_mu
        gradient[1] = 0.0; // d/d_sigma_log
        
        let mut total_logp = 0.0;
        
        // Prior for mu ~ Normal(0, 10)
        // logp = -0.5*log(2*pi) - log(10) - 0.5*((mu-0)/10)^2
        let mu_centered = mu - 0.0;
        let mu_scaled = mu_centered / 10.0;
        let mu_logp = -0.5 * (2.0 * std::f64::consts::PI).ln() - 10.0_f64.ln() - 0.5 * mu_scaled * mu_scaled;
        total_logp += mu_logp;
        
        // Gradient for mu prior: d/d_mu = -(mu-0)/10^2 = -mu/100
        gradient[0] += -mu / 100.0;
        
        // Prior for sigma ~ HalfNormal(5) with LogTransform
        // HalfNormal(x|scale) = 2 * Normal(x|0,scale) for x >= 0
        // logp = log(2) - 0.5*log(2*pi) - log(5) - 0.5*(sigma/5)^2 + sigma_log (Jacobian)
        let sigma_scaled = sigma / 5.0;
        let sigma_logp = 2.0_f64.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln() - 5.0_f64.ln() 
                        - 0.5 * sigma_scaled * sigma_scaled + sigma_log;
        total_logp += sigma_logp;
        
        // Gradient for sigma prior w.r.t. sigma_log:
        // d/d_sigma_log = d/d_sigma * d_sigma/d_sigma_log = d/d_sigma * sigma
        // d/d_sigma = -sigma/25, so d/d_sigma_log = -sigma^2/25 + 1
        gradient[1] += -sigma * sigma / 25.0 + 1.0;
        
        // Likelihood for y ~ Normal(mu, sigma)
        // For each observation: logp = -0.5*log(2*pi) - log(sigma) - 0.5*((y_i - mu)/sigma)^2
        let log_2pi = (2.0 * std::f64::consts::PI).ln();
        let log_sigma = sigma.ln();
        let sigma_inv = 1.0 / sigma;
        let sigma_inv_sq = sigma_inv * sigma_inv;
        
        let mut sum_residual_sq = 0.0;
        let mut sum_residual = 0.0;
        
        for &y_i in Y_DATA {
            let residual = y_i - mu;
            let scaled_residual = residual * sigma_inv;
            let y_logp_i = -0.5 * log_2pi - log_sigma - 0.5 * scaled_residual * scaled_residual;
            total_logp += y_logp_i;
            
            sum_residual += residual;
            sum_residual_sq += residual * residual;
        }
        
        // Gradient contributions from likelihood
        // d/d_mu = sum_i (y_i - mu) / sigma^2 = sum_residual / sigma^2
        gradient[0] += sum_residual * sigma_inv_sq;
        
        // d/d_sigma_log = d/d_sigma * sigma
        // d/d_sigma = -N/sigma + sum_i (y_i - mu)^2 / sigma^3
        // d/d_sigma_log = -N + sum_residual_sq / sigma^2
        gradient[1] += -(Y_N as f64) + sum_residual_sq * sigma_inv_sq;
        
        Ok(total_logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}