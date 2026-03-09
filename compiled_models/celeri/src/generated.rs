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

pub const N_PARAMS: usize = 17;

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
        // Extract parameters
        let rotation_raw = &position[0..9];
        let slip_rate = &position[9..17];
        
        let mut total_logp = 0.0;
        gradient.fill(0.0);
        
        // Constants
        let log_2pi = (2.0 * std::f64::consts::PI).ln();
        
        // rotation_raw ~ Normal(0, 1), shape=9
        for i in 0..9 {
            let x = rotation_raw[i];
            total_logp += -0.5 * log_2pi - 0.5 * x * x;
            gradient[i] = -x;
        }
        
        // slip_rate ~ Normal(0, 5.0), shape=8
        let sigma_slip = 5.0_f64;
        for i in 0..8 {
            let x = slip_rate[i];
            total_logp += -0.5 * log_2pi - sigma_slip.ln() - 0.5 * (x / sigma_slip).powi(2);
            gradient[9 + i] = -x / (sigma_slip * sigma_slip);
        }
        
        // rotation = rotation_raw * rotation_scale
        // X_6_DATA is 9 elements, use it directly as rotation_scale
        let mut rotation = [0.0; 9];
        for i in 0..9 {
            rotation[i] = rotation_raw[i] * X_6_DATA[i];
        }
        
        // predicted = G_rotation @ rotation + G_slip @ slip_rate
        let mut predicted = [0.0; 50];
        
        // G_rotation @ rotation: X_7_DATA as 50×9 matrix (450 elements)
        for i in 0..50 {
            for j in 0..9 {
                predicted[i] += X_7_DATA[i * 9 + j] * rotation[j];
            }
        }
        
        // G_slip @ slip_rate: X_5_DATA as 50×8 matrix (400 elements)
        for i in 0..50 {
            for j in 0..8 {
                predicted[i] += X_5_DATA[i * 8 + j] * slip_rate[j];
            }
        }
        
        // station_velocity ~ StudentT(nu=6, mu=predicted, sigma=sigma_obs)
        let nu = 6.0_f64;
        
        for i in 0..50 {
            let y = STATION_VELOCITY_DATA[i];
            let mu = predicted[i];
            let sigma = X_4_DATA[i]; // sigma_obs
            
            let z = (y - mu) / sigma;
            let term = 1.0 + z * z / nu;
            
            // StudentT logp for nu=6
            let lgamma_3_5 = 1.2009736023470743; // log(gamma(3.5))
            let lgamma_3 = (2.0_f64).ln(); // log(gamma(3))
            
            let logp_t = lgamma_3_5 - lgamma_3 - 0.5 * (nu * std::f64::consts::PI).ln() - sigma.ln() - 0.5 * (nu + 1.0) * term.ln();
            total_logp += logp_t;
            
            // Gradients
            let dlogp_dz = -(nu + 1.0) * z / (nu * term);
            let dlogp_dmu = dlogp_dz * (-1.0 / sigma);
            
            // Gradients w.r.t. rotation_raw via rotation
            for j in 0..9 {
                gradient[j] += dlogp_dmu * X_7_DATA[i * 9 + j] * X_6_DATA[j];
            }
            
            // Gradients w.r.t. slip_rate
            for j in 0..8 {
                gradient[9 + j] += dlogp_dmu * X_5_DATA[i * 8 + j];
            }
        }
        
        // Slip regularization: StudentT(nu=5, mu=0, sigma=2.0)
        let reg_nu = 5.0_f64;
        let reg_sigma = 2.0_f64;
        
        for i in 0..8 {
            let x = slip_rate[i];
            let z = x / reg_sigma;
            let term = 1.0 + z * z / reg_nu;
            
            // StudentT logp for nu=5
            let lgamma_3 = (2.0_f64).ln(); // log(gamma(3))
            let lgamma_2_5 = 0.2846828704729192; // log(gamma(2.5))
            
            let logp_reg = lgamma_3 - lgamma_2_5 - 0.5 * (reg_nu * std::f64::consts::PI).ln() - reg_sigma.ln() - 0.5 * (reg_nu + 1.0) * term.ln();
            total_logp += logp_reg;
            
            // Gradient of StudentT w.r.t. x
            let dlogp_dx = -(reg_nu + 1.0) * z / (reg_nu * reg_sigma * term);
            gradient[9 + i] += dlogp_dx;
        }
        
        // Geologic bounds: Censored Normal
        // The PyTensor graph shows Switch logic that handles three cases
        let bounded_idx = [0, 2, 4];
        
        for i in 0..3 {
            let slip_idx = bounded_idx[i];
            let mu = slip_rate[slip_idx];
            let sigma = X_2_DATA[i]; // bound_sigma
            let lower = X_1_DATA[i];  // lower_bounds
            let upper = X_3_DATA[i];  // upper_bounds
            let observed = GEOLOGIC_BOUNDS_DATA[i];
            
            // PyTensor uses exact equality checks for switching
            if observed == lower {
                // At lower bound: use log CDF
                let z = (lower - mu) / sigma;
                total_logp += Self::log_normal_cdf(z);
                
                // Gradient: d/dmu log(Phi(z)) = phi(z)/Phi(z) * dz/dmu = -phi(z)/(Phi(z)*sigma)  
                let phi_z = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
                let cdf_z = Self::normal_cdf(z);
                if cdf_z > 1e-15 {
                    gradient[9 + slip_idx] += -phi_z / (cdf_z * sigma);
                }
            } else if observed == upper {
                // At upper bound: use log CCDF  
                let z = (upper - mu) / sigma;
                total_logp += Self::log_normal_ccdf(z);
                
                // Gradient: d/dmu log(1-Phi(z)) = -phi(z)/(1-Phi(z)) * dz/dmu = phi(z)/((1-Phi(z))*sigma)
                let phi_z = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt(); 
                let ccdf_z = Self::normal_ccdf(z);
                if ccdf_z > 1e-15 {
                    gradient[9 + slip_idx] += phi_z / (ccdf_z * sigma);
                }
            } else {
                // Interior point: use normal PDF
                let z = (observed - mu) / sigma;
                total_logp += -0.5 * log_2pi - sigma.ln() - 0.5 * z * z;
                
                // Gradient: d/dmu [-0.5*((observed-mu)/sigma)^2] = (observed-mu)/sigma^2 = z/sigma  
                gradient[9 + slip_idx] += z / sigma;
            }
        }
        
        Ok(total_logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}

impl GeneratedLogp {
    fn normal_cdf(x: f64) -> f64 {
        0.5 * (1.0 + Self::erf(x / std::f64::consts::SQRT_2))
    }
    
    fn normal_ccdf(x: f64) -> f64 {
        0.5 * (1.0 - Self::erf(x / std::f64::consts::SQRT_2))
    }
    
    fn log_normal_cdf(x: f64) -> f64 {
        if x < -5.0 {
            let z = -x / std::f64::consts::SQRT_2;
            (Self::erfcx(z) / 2.0).ln() - z * z
        } else {
            let cdf_val = Self::normal_cdf(x);
            if cdf_val > 1e-200 {
                cdf_val.ln()
            } else {
                // Use erfcx for very small values
                let z = -x / std::f64::consts::SQRT_2;
                (Self::erfcx(z) / 2.0).ln() - z * z
            }
        }
    }
    
    fn log_normal_ccdf(x: f64) -> f64 {
        if x > 5.0 {
            let z = x / std::f64::consts::SQRT_2;
            (Self::erfcx(z) / 2.0).ln() - z * z
        } else {
            let ccdf_val = Self::normal_ccdf(x);
            if ccdf_val > 1e-200 {
                ccdf_val.ln()
            } else {
                let z = x / std::f64::consts::SQRT_2;
                (Self::erfcx(z) / 2.0).ln() - z * z
            }
        }
    }
    
    fn erf(x: f64) -> f64 {
        if x.abs() > 4.0 {
            return x.signum();
        }
        
        // Abramowitz and Stegun 7.1.26 (higher precision version)
        let a = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429];
        let p = 0.3275911;
        
        let x_abs = x.abs();
        let t = 1.0 / (1.0 + p * x_abs);
        
        let poly = a.iter().enumerate().fold(0.0, |acc, (i, &coef)| {
            acc + coef * t.powi(i as i32 + 1)
        });
        
        let erf_val = 1.0 - poly * (-x_abs * x_abs).exp();
        if x >= 0.0 { erf_val } else { -erf_val }
    }
    
    fn erfcx(x: f64) -> f64 {
        if x >= 0.0 {
            if x < 26.0 {
                // Use continued fraction expansion
                let mut cf = 0.0;
                for n in (1..=20).rev() {
                    cf = (n as f64) / (2.0 * x + cf);
                }
                1.0 / (std::f64::consts::PI.sqrt() * (x + cf))
            } else {
                // For very large x, use asymptotic expansion
                1.0 / (x * std::f64::consts::PI.sqrt())
            }
        } else {
            // For negative x
            2.0 * (x * x).exp() - Self::erfcx(-x)
        }
    }
}