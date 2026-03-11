# Bayes AI Compiler

An AI agent that acts as a compiler for computational models. It transpiles between probabilistic programming languages (PyMC, Stan), deep learning frameworks (JAX, PyTorch), and compiles to optimized Rust — with numerical validation at every step.

**[Read the blog post →](https://twiecki.io/blog/2026/03/10/pymc-rust-ai-compiler/)**

## How it works

The agent doesn't translate ops mechanically. It reasons about the full computational graph and applies the optimizations a domain expert would: loop fusion, memory pre-allocation, cache-friendly access patterns.

```
Source (PyMC/Stan/JAX/PyTorch) → Extract outputs + validation points → Claude agent loop → Target (Rust/PyMC/PyTorch/JAX) → Verify
```

1. **Extract**: Read model to get parameters, transforms, logp graph, and reference values
2. **Generate**: Claude (as an agent with tools) generates the target implementation
3. **Verify**: Validate logp + gradients against reference values
4. **Iterate**: If validation fails, the agent reads errors, inspects data, and fixes code autonomously

The agent has four tools: `write_rust_code`, `cargo_build`, `validate_logp`, and `read_file`. It loops until the output compiles and validates correctly. Model-specific "skills" are detected automatically:

- **GP (CPU)**: Linear algebra via faer (Cholesky, solves, inverses)
- **GP (Accelerate)**: Apple Accelerate framework (AMX coprocessor) for Apple Silicon
- **GP (CUDA)**: GPU-accelerated via cudarc + cuSOLVER for NVIDIA GPUs
- **ZeroSumNormal**: ZeroSum transform formulas and constraint handling
- **Stan → Rust**: Stan model extraction via BridgeStan
- **Stan → PyMC**: Distribution mappings, idiom translation, constraint handling
- **JAX → PyTorch**: Functional-to-stateful translation, op mapping, weight transposition
- **PyTorch → JAX**: Stateful-to-functional translation, pure function extraction
- **PyTorch → Rust**: Neural net to zero-dependency Rust binary, forward + gradient validation

Hardware is auto-detected: CUDA → Accelerate (Apple Silicon) → CPU fallback.

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

### Apple Accelerate (AMX coprocessor) acceleration

For GP models on Apple Silicon, the compiler auto-detects the platform and uses Apple's Accelerate framework via direct LAPACK FFI (`dpotrf`, `dpotrs`, `dpotri`). This leverages the AMX coprocessor for hardware-accelerated matrix operations in full f64 precision. Benchmarks on M4 Max, N=200 GP:

| Backend | µs/eval | Speedup vs faer |
|---|---|---|
| Rust + faer (pure Rust) | 487 | 1.0x |
| Rust + Accelerate (AMX) | 366 | **1.33x** |

No extra crate dependencies needed — Accelerate is linked via `build.rs` to the system framework.

### Neural network inference: PyTorch → Rust

minGPT-nano (3 layers, 3 heads, 48-dim embeddings) transpiled to zero-dependency Rust:

| Backend | µs/call | Speedup |
|---|---|---|
| PyTorch (eager) | 660 | 1.0x |
| Rust (transpiled) | 284 | **2.3x** |
| Rust + Enzyme (forward+backward) | 899 | **3.1x** vs PyTorch fwd+bwd (2777 µs) |

Forward pass numerical accuracy: max_diff = 5.36e-07. Enzyme gradients match PyTorch autograd to ~5e-7 (f32 precision).

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

# Stan → PyMC transpilation
python examples/stan_pymc_01_normal.py
python examples/stan_pymc_02_hierarchical.py

# JAX ↔ PyTorch transpilation
python examples/jax_to_pytorch_mlp.py
python examples/pytorch_to_jax_mlp.py

# PyTorch → Rust (zero-dependency inference binary)
python examples/pytorch_to_rust_mlp.py

# minGPT → Rust (transformer inference)
python examples/mingpt_to_rust.py
```

## Architecture

```
pymc_rust_compiler/
├── exporter.py       # Extract parameters, transforms, logp graph from pm.Model()
├── compiler.py       # Agentic loop: Claude API → Rust code → build → validate
├── stan_exporter.py  # Extract Stan model context via BridgeStan
├── stan_compiler.py  # Stan → Rust agentic compiler
├── stan_to_pymc.py           # Stan → PyMC agentic transpiler
├── jax_exporter.py           # Extract model info from JAX functions
├── pytorch_exporter.py       # Extract model info from PyTorch modules
├── jax_pytorch_transpiler.py  # JAX ↔ PyTorch agentic transpiler
├── pytorch_rust_transpiler.py # PyTorch → Rust agentic transpiler (zero-dep inference)
├── nutpie_bridge.py          # nutpie integration: compiled Rust → nutpie.sample()
├── benchmark.py              # logp eval benchmarks: Rust vs Numba (jit + cfunc)
└── skills/                   # Model-specific knowledge for the AI agent
    ├── gp.md              # CPU GP (faer Cholesky)
    ├── gp_accelerate.md   # Apple Silicon GP (Accelerate LAPACK / AMX)
    ├── gp_cuda.md         # NVIDIA GPU GP (cudarc + cuSOLVER)
    ├── zerosumnormal.md
    ├── stan.md            # Stan → Rust translation knowledge
    ├── stan_to_pymc.md    # Stan → PyMC translation knowledge
    ├── jax_to_pytorch.md  # JAX → PyTorch op mapping + idioms
    ├── pytorch_to_jax.md  # PyTorch → JAX op mapping + idioms
    └── pytorch_to_rust.md # PyTorch → Rust: matmul, activations, backprop

rust_template/      # Template Rust project (Cargo.toml, data loading, validation)
bench_runner/       # Rust lib for calling Numba cfunc from Rust (like nutpie)
compiled_models/    # Pre-compiled models (normal, linreg, hierarchical, GP, ...)
```

## Stan → PyMC Transpiler

The same agentic architecture works for language-to-language translation. Claude generates PyMC code, validates logp against BridgeStan reference values, and iterates until the models match numerically. Used to translate all 120 models from [posteriordb](https://github.com/stan-dev/posteriordb) from Stan to PyMC.

```python
from pymc_rust_compiler import transpile_stan_to_pymc

stan_code = """
data { int<lower=0> N; array[N] real y; }
parameters { real mu; real<lower=0> sigma; }
model { mu ~ normal(0, 10); sigma ~ normal(0, 5); y ~ normal(mu, sigma); }
"""

result = transpile_stan_to_pymc(stan_code, data={"N": 100, "y": [...]})
if result.success:
    model = result.get_model(data)  # returns a pm.Model
    print(result.pymc_code)         # generated Python code
```

## JAX ↔ PyTorch Transpiler

The same agentic architecture generalizes to deep learning frameworks. Claude translates between JAX's functional style and PyTorch's stateful modules, validating forward pass outputs and gradients at multiple test points.

### JAX → PyTorch

```python
import jax.numpy as jnp
from pymc_rust_compiler import transpile_jax_to_pytorch

def forward(params, x):
    x = jax.nn.relu(x @ params["w1"] + params["b1"])
    return x @ params["w2"] + params["b2"]

params = {"w1": jnp.ones((4, 8)), "b1": jnp.zeros(8),
          "w2": jnp.ones((8, 2)), "b2": jnp.zeros(2)}

result = transpile_jax_to_pytorch(forward, params, sample_input=jnp.ones((1, 4)))
if result.success:
    model = result.get_model(params)  # returns a torch.nn.Module
```

### PyTorch → JAX

```python
import torch.nn as nn
from pymc_rust_compiler import transpile_pytorch_to_jax

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

result = transpile_pytorch_to_jax(MLP(), sample_input=torch.randn(1, 4))
if result.success:
    jax_params, forward_fn = result.get_model(param_data)
    output = forward_fn(jax_params, jnp.ones((1, 4)))
```

## PyTorch → Rust Transpiler

The killer feature for inference deployment. Takes a PyTorch `nn.Module` and generates a **zero-dependency Rust binary** — no ML framework, no Python runtime, just raw f32 math compiled to native code. Parameters are baked in as `const` arrays.

The agent uses the same agentic architecture with tools: `write_code` → `cargo_build` → `validate_model` (forward pass + gradient matching). The generated Rust code includes both `forward()` and manual backpropagation for gradient validation.

```python
import torch.nn as nn
from pymc_rust_compiler import transpile_pytorch_to_rust

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

result = transpile_pytorch_to_rust(MLP(), sample_input=torch.randn(1, 4))
if result.success:
    print(f"Binary at: {result.binary_path}")  # Zero-dependency native binary
    result.save("model.rs")                     # Save the generated Rust code
```
