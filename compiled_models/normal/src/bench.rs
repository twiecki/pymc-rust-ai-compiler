
use pymc_compiled_model::generated::GeneratedLogp;
use nuts_rs::CpuLogpFunc;
use std::io::{self, BufRead};
use std::time::Instant;

fn main() {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    // First line: number of iterations
    let n_iters: usize = lines.next().unwrap().unwrap().trim().parse().unwrap();

    // Second line: parameter vector
    let param_line = lines.next().unwrap().unwrap();
    let position: Vec<f64> = param_line.split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();

    let mut logp_fn = GeneratedLogp::default();
    let n = logp_fn.dim();
    let mut gradient = vec![0.0f64; n];
    let mut logp_val = 0.0f64;

    // Warmup
    for _ in 0..100 {
        logp_val = logp_fn.logp(&position, &mut gradient).unwrap();
    }

    // Timed loop
    let start = Instant::now();
    for _ in 0..n_iters {
        logp_val = logp_fn.logp(&position, &mut gradient).unwrap();
    }
    let elapsed = start.elapsed();

    let nanos = elapsed.as_nanos() as f64;
    let us_per_eval = nanos / (n_iters as f64) / 1000.0;
    print!("{:.6},{:.17e}", us_per_eval, logp_val);
    for g in &gradient {
        print!(",{:.17e}", g);
    }
    println!();
}
