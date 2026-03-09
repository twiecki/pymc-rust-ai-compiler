# PyMC Rust AI Compiler

Compile PyMC models to optimized Rust via LLM. The AI doesn't just translate ops mechanically — it reasons about the full computational graph and applies optimizations a human expert would: loop fusion, memory pre-allocation, cache-friendly access patterns.

## How it works

```
PyMC Model → Extract logp graph + validation points → Claude API → Rust code → Verify → Compile
```

1. **Extract**: Read `pm.Model()` to get parameters, transforms, logp graph, and reference values
2. **Generate**: Claude (as an agent with tools) generates a complete Rust `CpuLogpFunc` implementation
3. **Verify**: Build and validate logp + gradients against PyMC's reference values
4. **Iterate**: If validation fails, the agent reads errors, inspects data, and fixes code autonomously

The agent has four tools: `write_rust_code`, `cargo_build`, `validate_logp`, and `read_file`. It loops until the model compiles and validates correctly (up to 5 attempts).

## Benchmarks

### Compilation (Claude Sonnet 4)

| Model | Params | Builds | Tokens | Result |
|---|---|---|---|---|
| Normal | 2 | 1 | 39K | First try |
| Linear Regression | 3 | 1 | 52K | First try |
| Hierarchical | 12 | 4 | 288K | Fixed gradients in 3 retries |
| ZeroSumNormal | 124 | 1 | 213K | First try |
| GP (ExpQuad) | 3 | 10 | 1.6M | Failed — Cholesky gradients |

### Runtime: logp+dlogp evaluation speed

Rust vs nutpie's Numba backend (500K evaluations, lower is better):

| Model | Numba (us/eval) | Rust (us/eval) | Speedup |
|---|---|---|---|
| Normal (2 params) | 0.99 | 0.15 | **6.6x** |
| LinReg (3 params) | 1.62 | 0.37 | **4.3x** |
| Hierarchical (12 params) | 2.82 | 0.62 | **4.6x** |
| GP regression (3 params) | 126.70 | 52.84 | **2.4x** |

Numba column = `numba.cfunc` called from Rust in a tight loop (how nutpie actually works). The AI-compiled Rust is 2-7x faster across all models.

## Quick start

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv sync  # or: pip install -e .
```

```python
import pymc as pm
from pymc_rust_compiler import compile_model

with pm.Model() as model:
    mu = pm.Normal("mu", 0, 10)
    sigma = pm.HalfNormal("sigma", 5)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)

result = compile_model(model)

if result.success:
    print(f"Compiled in {result.n_attempts} build(s), {result.token_usage['total_tokens']} tokens")
```

## Examples

```bash
# Single models
python examples/01_normal.py
python examples/02_linear_regression.py
python examples/03_hierarchical.py

# Full benchmark suite
python examples/run_benchmark.py

# logp+dlogp evaluation benchmark (Rust vs nutpie/Numba)
python examples/bench_logp.py
```

## Architecture

```
pymc_rust_compiler/
├── exporter.py     # Extract parameters, transforms, logp graph from pm.Model()
├── compiler.py     # Agentic loop: Claude API → Rust code → build → validate
└── benchmark.py    # logp eval benchmarks: Rust vs Numba (jit + cfunc)

rust_template/      # Template Rust project (Cargo.toml, data loading, validation)
bench_runner/       # Rust lib for calling Numba cfunc from Rust (like nutpie)
compiled_models/    # Pre-compiled models (normal, linreg, hierarchical, GP, ...)
```

The key insight: `pm.Model()` already contains everything needed — parameters, transforms, shapes, logp functions. We extract it all and let the AI generate optimized Rust that matches PyMC's exact numerical output.
