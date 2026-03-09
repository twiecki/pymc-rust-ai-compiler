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

pub const N_PARAMS: usize = 124;

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
        // Initialize gradient to zero
        for g in gradient.iter_mut() {
            *g = 0.0;
        }

        let mut total_logp = 0.0;
        
        // Extract unconstrained parameters
        let grand_mean = position[0];
        let sigma_store_log = position[1];
        let sigma_day_log = position[2]; 
        let sigma_cat_log = position[3];
        let store_effect_unc = &position[4..9];  // 5 elements
        let day_effect_unc = &position[9..15];   // 6 elements
        let interaction_unc = &position[15..123]; // 108 elements (6*6*3)
        let sigma_y_log = position[123];

        // Transform to constrained space
        let sigma_store = sigma_store_log.exp();
        let sigma_day = sigma_day_log.exp();
        let sigma_cat = sigma_cat_log.exp();
        let sigma_y = sigma_y_log.exp();

        // ============= grand_mean: Normal(0, 10) =============
        let diff = grand_mean - 0.0;
        let sigma_gm = 10.0_f64;
        let term = diff / sigma_gm;
        let gm_logp = -0.5 * (2.0 * std::f64::consts::PI).ln() - sigma_gm.ln() - 0.5 * term * term;
        total_logp += gm_logp;
        gradient[0] += -diff / (sigma_gm * sigma_gm);

        // ============= sigma_store: HalfNormal(2) with LogTransform =============
        let x = sigma_store;
        let sigma_hn = 2.0_f64;
        let hn_logp = (2.0_f64).ln() - 0.5 * (2.0 * std::f64::consts::PI).ln() - sigma_hn.ln() - 0.5 * (x / sigma_hn).powi(2);
        let jacobian = sigma_store_log; // log jacobian for exp transform
        let store_logp = hn_logp + jacobian;
        total_logp += store_logp;
        
        // For HalfNormal(x|sigma) with LogTransform: gradient = (-x²/sigma² + 1) * x  
        gradient[1] += (-x * x / (sigma_hn * sigma_hn) + 1.0) * sigma_store;

        // ============= sigma_day: HalfNormal(2) with LogTransform =============
        let x = sigma_day;
        let hn_logp = (2.0_f64).ln() - 0.5 * (2.0 * std::f64::consts::PI).ln() - sigma_hn.ln() - 0.5 * (x / sigma_hn).powi(2);
        let jacobian = sigma_day_log;
        let day_logp = hn_logp + jacobian;
        total_logp += day_logp;
        
        gradient[2] += (-x * x / (sigma_hn * sigma_hn) + 1.0) * sigma_day;

        // ============= sigma_cat: HalfNormal(2) with LogTransform =============
        let x = sigma_cat;
        let hn_logp = (2.0_f64).ln() - 0.5 * (2.0 * std::f64::consts::PI).ln() - sigma_hn.ln() - 0.5 * (x / sigma_hn).powi(2);
        let jacobian = sigma_cat_log;
        let cat_logp = hn_logp + jacobian;
        total_logp += cat_logp;
        
        gradient[3] += (-x * x / (sigma_hn * sigma_hn) + 1.0) * sigma_cat;

        // ============= store_effect: ZeroSumNormal(sigma_store, shape=6) =============
        // ZeroSumNormal logp uses only the 5 unconstrained elements
        let n_unc_store = 5;
        let mut sum_sq_store = 0.0;
        for &x in store_effect_unc {
            sum_sq_store += x * x;
        }
        let store_eff_logp = (n_unc_store as f64) * (-0.5 * (2.0 * std::f64::consts::PI).ln() - sigma_store.ln()) - 0.5 * sum_sq_store / (sigma_store * sigma_store);
        total_logp += store_eff_logp;
        
        // Gradient w.r.t unconstrained elements
        for (i, &x) in store_effect_unc.iter().enumerate() {
            gradient[4 + i] += -x / (sigma_store * sigma_store);
        }
        // Gradient w.r.t log(sigma_store) from ZeroSumNormal - exactly as computed
        gradient[1] += (-n_unc_store as f64 + sum_sq_store / (sigma_store * sigma_store)) * sigma_store;

        // ============= day_effect: ZeroSumNormal(sigma_day, shape=7) =============
        let n_unc_day = 6;
        let mut sum_sq_day = 0.0;
        for &x in day_effect_unc {
            sum_sq_day += x * x;
        }
        let day_eff_logp = (n_unc_day as f64) * (-0.5 * (2.0 * std::f64::consts::PI).ln() - sigma_day.ln()) - 0.5 * sum_sq_day / (sigma_day * sigma_day);
        total_logp += day_eff_logp;
        
        // Gradient w.r.t unconstrained elements
        for (i, &x) in day_effect_unc.iter().enumerate() {
            gradient[9 + i] += -x / (sigma_day * sigma_day);
        }
        // Gradient w.r.t log(sigma_day) from ZeroSumNormal - exactly as computed
        gradient[2] += (-n_unc_day as f64 + sum_sq_day / (sigma_day * sigma_day)) * sigma_day;

        // ============= interaction: ZeroSumNormal(sigma_cat, shape=(6,7,4), n_zerosum_axes=2) =============
        let n_unc_int = 108; // 6*6*3
        let mut sum_sq_int = 0.0;
        for &x in interaction_unc {
            sum_sq_int += x * x;
        }
        let int_logp = (n_unc_int as f64) * (-0.5 * (2.0 * std::f64::consts::PI).ln() - sigma_cat.ln()) - 0.5 * sum_sq_int / (sigma_cat * sigma_cat);
        total_logp += int_logp;
        
        // Gradient w.r.t unconstrained elements
        for (i, &x) in interaction_unc.iter().enumerate() {
            gradient[15 + i] += -x / (sigma_cat * sigma_cat);
        }
        // Gradient w.r.t log(sigma_cat) from ZeroSumNormal - exactly as computed
        gradient[3] += (-n_unc_int as f64 + sum_sq_int / (sigma_cat * sigma_cat)) * sigma_cat;

        // ============= sigma_y: HalfNormal(5) with LogTransform =============
        let x = sigma_y;
        let sigma_hn = 5.0_f64;
        let hn_logp = (2.0_f64).ln() - 0.5 * (2.0 * std::f64::consts::PI).ln() - sigma_hn.ln() - 0.5 * (x / sigma_hn).powi(2);
        let jacobian = sigma_y_log;
        let y_sigma_logp = hn_logp + jacobian;
        total_logp += y_sigma_logp;
        
        gradient[123] += (-x * x / (sigma_hn * sigma_hn) + 1.0) * sigma_y;

        // ============= Transform unconstrained effects to constrained for likelihood =============
        
        // Transform store_effect (5 -> 6)
        let mut store_effect_full = vec![0.0; 6];
        let n = 6.0_f64;
        let sum_x: f64 = store_effect_unc.iter().sum();
        let norm = sum_x / (n.sqrt() + n);
        let fill = norm - sum_x / n.sqrt();
        for i in 0..5 {
            store_effect_full[i] = store_effect_unc[i] - norm;
        }
        store_effect_full[5] = fill - norm;

        // Transform day_effect (6 -> 7)
        let mut day_effect_full = vec![0.0; 7];
        let n = 7.0_f64;
        let sum_x: f64 = day_effect_unc.iter().sum();
        let norm = sum_x / (n.sqrt() + n);
        let fill = norm - sum_x / n.sqrt();
        for i in 0..6 {
            day_effect_full[i] = day_effect_unc[i] - norm;
        }
        day_effect_full[6] = fill - norm;

        // Transform interaction: (6,6,3) -> (6,7,3) -> (6,7,4)
        // Step 1: extend axis -2 (middle axis): (6,6,3) -> (6,7,3)
        let mut interaction_temp = vec![vec![vec![0.0; 3]; 7]; 6];
        for i in 0..6 {
            let n = 7.0_f64;
            // Sum over middle axis
            let mut sums = vec![0.0; 3];
            for k in 0..3 {
                for j in 0..6 {
                    let idx = i * 6 * 3 + j * 3 + k;
                    sums[k] += interaction_unc[idx];
                }
            }
            
            for k in 0..3 {
                let sum_x = sums[k];
                let norm = sum_x / (n.sqrt() + n);
                let fill = norm - sum_x / n.sqrt();
                
                // Copy original elements
                for j in 0..6 {
                    let idx = i * 6 * 3 + j * 3 + k;
                    interaction_temp[i][j][k] = interaction_unc[idx] - norm;
                }
                // Fill element
                interaction_temp[i][6][k] = fill - norm;
            }
        }

        // Step 2: extend axis -1 (last axis): (6,7,3) -> (6,7,4)
        let mut interaction_full = vec![vec![vec![0.0; 4]; 7]; 6];
        for i in 0..6 {
            for j in 0..7 {
                let n = 4.0_f64;
                let sum_x: f64 = interaction_temp[i][j].iter().sum();
                let norm = sum_x / (n.sqrt() + n);
                let fill = norm - sum_x / n.sqrt();
                
                for k in 0..3 {
                    interaction_full[i][j][k] = interaction_temp[i][j][k] - norm;
                }
                interaction_full[i][j][3] = fill - norm;
            }
        }

        // ============= y likelihood: Normal(mu, sigma_y) =============
        let mut y_logp = 0.0;
        let mut grad_grand_mean = 0.0;
        let mut grad_sigma_y_log = 0.0;
        let mut grad_store_effect = vec![0.0; 6];
        let mut grad_day_effect = vec![0.0; 7]; 
        let mut grad_interaction = vec![vec![vec![0.0; 4]; 7]; 6];

        for obs in 0..Y_N {
            let y_obs = Y_DATA[obs];
            // Based on data description:
            // X_0_DATA → cat_idx (values 0..3, 4 groups)  
            // X_1_DATA → day_idx (values 0..6, 7 groups)
            // X_2_DATA → store_idx (values 0..5, 6 groups)
            let cat_idx = X_0_DATA[obs] as usize;
            let day_idx = X_1_DATA[obs] as usize;   
            let store_idx = X_2_DATA[obs] as usize; 
            
            let mu = grand_mean + store_effect_full[store_idx] + day_effect_full[day_idx] + interaction_full[store_idx][day_idx][cat_idx];
            let residual = y_obs - mu;
            let term = residual / sigma_y;
            
            let obs_logp = -0.5 * (2.0 * std::f64::consts::PI).ln() - sigma_y.ln() - 0.5 * term * term;
            y_logp += obs_logp;

            // Gradients for observed likelihood
            let grad_mu = residual / (sigma_y * sigma_y);
            grad_grand_mean += grad_mu;
            grad_store_effect[store_idx] += grad_mu;
            grad_day_effect[day_idx] += grad_mu;
            grad_interaction[store_idx][day_idx][cat_idx] += grad_mu;
            
            grad_sigma_y_log += (-1.0 / sigma_y + residual * residual / (sigma_y * sigma_y * sigma_y)) * sigma_y;
        }

        total_logp += y_logp;
        gradient[0] += grad_grand_mean;
        gradient[123] += grad_sigma_y_log;

        // ============= Backpropagate gradients through ZeroSum transforms =============
        
        // Store effect gradient
        let n = 6.0_f64;
        let sum_grad: f64 = grad_store_effect[0..5].iter().sum();
        let grad_fill = grad_store_effect[5];
        for j in 0..5 {
            gradient[4 + j] += grad_store_effect[j] - sum_grad / (n.sqrt() + n) - grad_fill / n.sqrt();
        }

        // Day effect gradient  
        let n = 7.0_f64;
        let sum_grad: f64 = grad_day_effect[0..6].iter().sum();
        let grad_fill = grad_day_effect[6];
        for j in 0..6 {
            gradient[9 + j] += grad_day_effect[j] - sum_grad / (n.sqrt() + n) - grad_fill / n.sqrt();
        }

        // Interaction gradient - reverse order of transforms
        // First: backprop through axis -1 transform: (6,7,4) -> (6,7,3)
        let mut grad_interaction_temp = vec![vec![vec![0.0; 3]; 7]; 6];
        for i in 0..6 {
            for j in 0..7 {
                let n = 4.0_f64;
                let sum_grad: f64 = grad_interaction[i][j][0..3].iter().sum();
                let grad_fill = grad_interaction[i][j][3];
                for k in 0..3 {
                    grad_interaction_temp[i][j][k] = grad_interaction[i][j][k] - sum_grad / (n.sqrt() + n) - grad_fill / n.sqrt();
                }
            }
        }

        // Second: backprop through axis -2 transform: (6,7,3) -> (6,6,3)
        for i in 0..6 {
            for k in 0..3 {
                let n = 7.0_f64;
                let sum_grad: f64 = grad_interaction_temp[i][0..6].iter().map(|row| row[k]).sum();
                let grad_fill = grad_interaction_temp[i][6][k];
                for j in 0..6 {
                    let idx = i * 6 * 3 + j * 3 + k;
                    gradient[15 + idx] += grad_interaction_temp[i][j][k] - sum_grad / (n.sqrt() + n) - grad_fill / n.sqrt();
                }
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