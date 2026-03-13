"""Transpailer: compile and transpile models across frameworks via LLM.

Supports: PyMC/Stan → Rust, Stan → PyMC, JAX ↔ PyTorch.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# JAX ↔ PyTorch (no heavy deps at import time)
from transpailer.jax_pytorch_transpiler import (
    transpile_jax_to_pytorch,
    transpile_pytorch_to_jax,
    TranspileResult,
)
from transpailer.jax_exporter import (
    JaxModelExporter,
    export_jax_model,
)
from transpailer.pytorch_exporter import (
    PytorchModelExporter,
    export_pytorch_model,
)
from transpailer.pytorch_rust_transpiler import (
    transpile_pytorch_to_rust,
    RustTranspileResult,
)

# PyMC/Stan imports are lazy — they pull in heavy deps (pymc, bridgestan)
if TYPE_CHECKING:
    from transpailer.exporter import (
        ModelContext,
        RustModelExporter,
        export_model,
    )
    from transpailer.compiler import (
        compile_model,
        optimize_model,
        OptimizationEvent,
    )
    from transpailer.analysis import (
        plot_optimization_progress,
        plot_waterfall,
        plot_timeline,
        print_summary,
    )
    from transpailer.stan_exporter import (
        StanModelContext,
        StanModelExporter,
        export_stan_model,
    )
    from transpailer.stan_compiler import (
        compile_stan_model,
        StanCompilationResult,
    )
    from transpailer.stan_to_pymc import transpile_stan_to_pymc, StanToPyMCResult


def __getattr__(name: str):
    """Lazy import for PyMC/Stan components."""
    _lazy_imports = {
        "ModelContext": ("transpailer.exporter", "ModelContext"),
        "RustModelExporter": ("transpailer.exporter", "RustModelExporter"),
        "export_model": ("transpailer.exporter", "export_model"),
        "compile_model": ("transpailer.compiler", "compile_model"),
        "optimize_model": ("transpailer.compiler", "optimize_model"),
        "OptimizationEvent": ("transpailer.compiler", "OptimizationEvent"),
        "plot_optimization_progress": (
            "transpailer.analysis",
            "plot_optimization_progress",
        ),
        "plot_waterfall": ("transpailer.analysis", "plot_waterfall"),
        "plot_timeline": ("transpailer.analysis", "plot_timeline"),
        "print_summary": ("transpailer.analysis", "print_summary"),
        "StanModelContext": ("transpailer.stan_exporter", "StanModelContext"),
        "StanModelExporter": ("transpailer.stan_exporter", "StanModelExporter"),
        "export_stan_model": ("transpailer.stan_exporter", "export_stan_model"),
        "compile_stan_model": (
            "transpailer.stan_compiler",
            "compile_stan_model",
        ),
        "StanCompilationResult": (
            "transpailer.stan_compiler",
            "StanCompilationResult",
        ),
        "transpile_stan_to_pymc": (
            "transpailer.stan_to_pymc",
            "transpile_stan_to_pymc",
        ),
        "StanToPyMCResult": ("transpailer.stan_to_pymc", "StanToPyMCResult"),
    }
    if name in _lazy_imports:
        module_path, attr = _lazy_imports[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module 'transpailer' has no attribute {name!r}")


__all__ = [
    # PyMC → Rust (lazy)
    "compile_model",
    "optimize_model",
    "OptimizationEvent",
    # Analysis / plotting
    "plot_optimization_progress",
    "plot_waterfall",
    "plot_timeline",
    "print_summary",
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
    from transpailer.nutpie_bridge import to_nutpie as _to_nutpie

    return _to_nutpie(compile_result, model)
