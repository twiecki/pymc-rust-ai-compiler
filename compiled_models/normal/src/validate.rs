
use pymc_compiled_model::generated::GeneratedLogp;
use nuts_rs::CpuLogpFunc;
use std::io::{self, BufRead, Write};

fn main() {
    let mut logp_fn = GeneratedLogp;
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines() {
        let line = line.unwrap();
        if line.is_empty() { continue; }

        let position: Vec<f64> = line.split(',')
            .map(|s| s.trim().parse().unwrap())
            .collect();

        let n = logp_fn.dim();
        let mut gradient = vec![0.0f64; n];
        match logp_fn.logp(&position, &mut gradient) {
            Ok(logp) => {
                write!(out, "{:.17e}", logp).unwrap();
                for g in &gradient {
                    write!(out, ",{:.17e}", g).unwrap();
                }
                writeln!(out).unwrap();
            }
            Err(e) => {
                writeln!(out, "ERROR:{}", e).unwrap();
            }
        }
    }
}
