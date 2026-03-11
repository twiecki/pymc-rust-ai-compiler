"""Bayes AI Compiler: compile and transpile models across frameworks via LLM.

Supports: PyMC/Stan → Rust, Stan → PyMC, JAX ↔ PyTorch.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# JAX ↔ PyTorch (no heavy deps at import time)
from pymc_rust_compiler.jax_pytorch_transpiler import (
    transpile_jax_to_pytorch,
    transpile_pytorch_to_jax,
    TranspileResult,
)
from pymc_rust_compiler.jax_exporter import (
    JaxModelExporter,
    export_jax_model,
)
from pymc_rust_compiler.pytorch_exporter import (
    PytorchModelExporter,
    export_pytorch_model,
)
from pymc_rust_compiler.pytorch_rust_transpiler import (
    transpile_pytorch_to_rust,
    RustTranspileResult,
)

# PyMC/Stan imports are lazy — they pull in heavy deps (pymc, bridgestan)
if TYPE_CHECKING:
    from pymc_rust_compiler.exporter import ModelContext, RustModelExporter, export_model
    from pymc_rust_compiler.compiler import compile_model, optimize_model
    from pymc_rust_compiler.stan_exporter import (
        StanModelContext,
        StanModelExporter,
        export_stan_model,
    )
    from pymc_rust_compiler.stan_compiler import compile_stan_model, StanCompilationResult
    from pymc_rust_compiler.stan_to_pymc import transpile_stan_to_pymc, StanToPyMCResult


def __getattr__(name: str):
    """Lazy import for PyMC/Stan components."""
    _lazy_imports = {
        "ModelContext": ("pymc_rust_compiler.exporter", "ModelContext"),
        "RustModelExporter": ("pymc_rust_compiler.exporter", "RustModelExporter"),
        "export_model": ("pymc_rust_compiler.exporter", "export_model"),
        "compile_model": ("pymc_rust_compiler.compiler", "compile_model"),
        "optimize_model": ("pymc_rust_compiler.compiler", "optimize_model"),
        "StanModelContext": ("pymc_rust_compiler.stan_exporter", "StanModelContext"),
        "StanModelExporter": ("pymc_rust_compiler.stan_exporter", "StanModelExporter"),
        "export_stan_model": ("pymc_rust_compiler.stan_exporter", "export_stan_model"),
        "compile_stan_model": ("pymc_rust_compiler.stan_compiler", "compile_stan_model"),
        "StanCompilationResult": ("pymc_rust_compiler.stan_compiler", "StanCompilationResult"),
        "transpile_stan_to_pymc": ("pymc_rust_compiler.stan_to_pymc", "transpile_stan_to_pymc"),
        "StanToPyMCResult": ("pymc_rust_compiler.stan_to_pymc", "StanToPyMCResult"),
    }
    if name in _lazy_imports:
        module_path, attr = _lazy_imports[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module 'pymc_rust_compiler' has no attribute {name!r}")


__all__ = [
    # PyMC → Rust (lazy)
    "compile_model",
    "optimize_model",
    "export_model",
    "ModelContext",
    "RustModelExporter",
    "to_nutpie",
    # Stan → Rust (lazy)
    "compile_stan_model",
    "export_stan_model",
    "StanModelContext",
    "StanModelExporter",
    "StanCompilationResult",
    # Stan → PyMC (lazy)
    "transpile_stan_to_pymc",
    "StanToPyMCResult",
    # JAX ↔ PyTorch
    "transpile_jax_to_pytorch",
    "transpile_pytorch_to_jax",
    "TranspileResult",
    "JaxModelExporter",
    "export_jax_model",
    "PytorchModelExporter",
    "export_pytorch_model",
    # PyTorch → Rust
    "transpile_pytorch_to_rust",
    "RustTranspileResult",
]


def to_nutpie(compile_result, model):
    """Convert a CompilationResult to a nutpie-compatible model. Lazy import."""
    from pymc_rust_compiler.nutpie_bridge import to_nutpie as _to_nutpie
    return _to_nutpie(compile_result, model)
