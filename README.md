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

The agent has four tools: `write_rust_code`, `cargo_build`, `validate_logp`, and `read_file`. It loops until the model compiles and validates correctly. Model-specific "skills" are detected automatically and loaded to provide specialized knowledge:

- **GP**: CPU linear algebra via faer (Cholesky, solves, inverses)
- **GP CUDA**: GPU-accelerated via cudarc + cuSOLVER for NVIDIA GPUs
- **GP MLX**: GPU-accelerated via mlx-rs + Metal for Apple Silicon (M1-M5)
- **ZeroSumNormal**: ZeroSum transform formulas and constraint handling

Hardware is auto-detected: CUDA → MLX → CPU fallback.

## Benchmarks

### Compilation (Claude Sonnet 4)

| Model | Params | Tool Calls | Tokens | Result |
|---|---|---|---|---|
| Normal | 2 | 4 | 40K | First try |
| Linear Regression | 3 | 4 | 54K | First try |
| Hierarchical | 12 | 8 | 153K | Fixed gradients in 1 retry |
| GP (ExpQuad) | 3 | 11 | 467K | Passed (GP skill) |
| ZeroSumNormal | 142 | 9 | 484K | Passed (ZeroSumNormal skill) |

### Runtime: logp+dlogp evaluation speed

Rust vs nutpie's Numba backend (500K evaluations, lower is better):

| Model | Numba (us/eval) | Rust (us/eval) | Speedup |
|---|---|---|---|
| Normal (2 params) | 0.96 | 0.14 | **6.8x** |
| LinReg (3 params) | 1.60 | 0.33 | **4.9x** |
| Hierarchical (12 params) | 2.63 | 0.76 | **3.5x** |
| GP regression (3 params) | 116.57 | 35.31 | **3.3x** |

Numba column = `numba.cfunc` called from Rust in a tight loop (how nutpie actually works). The AI-compiled Rust is 3-7x faster across all models.

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
├── exporter.py       # Extract parameters, transforms, logp graph from pm.Model()
├── compiler.py       # Agentic loop: Claude API → Rust code → build → validate
├── nutpie_bridge.py  # nutpie integration: compiled Rust → nutpie.sample()
├── benchmark.py      # logp eval benchmarks: Rust vs Numba (jit + cfunc)
└── skills/           # Model-specific knowledge for the AI agent
    ├── gp.md         # CPU GP (faer Cholesky)
    ├── gp_cuda.md    # NVIDIA GPU GP (cudarc + cuSOLVER)
    ├── gp_mlx.md     # Apple Silicon GP (mlx-rs + Metal)
    └── zerosumnormal.md

rust_template/      # Template Rust project (Cargo.toml, data loading, validation)
bench_runner/       # Rust lib for calling Numba cfunc from Rust (like nutpie)
compiled_models/    # Pre-compiled models (normal, linreg, hierarchical, GP, ...)
```

The key insight: `pm.Model()` already contains everything needed — parameters, transforms, shapes, logp functions. We extract it all and let the AI generate optimized Rust that matches PyMC's exact numerical output.
