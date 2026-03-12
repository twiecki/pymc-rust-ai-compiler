"""Tests for JAX↔PyTorch transpilation components.

Tests the exporters (which don't need an API key) and the transpiler's
tool execution logic (mocked Claude API).
"""

from __future__ import annotations

import numpy as np
import pytest

_has_jax = pytest.importorskip is not None  # helper below
try:
    import jax  # noqa: F401

    _has_jax = True
except ImportError:
    _has_jax = False

jax_required = pytest.mark.skipif(not _has_jax, reason="JAX not installed")


# ── JAX Exporter Tests ──────────────────────────────────────────────────────


@jax_required
class TestJaxExporter:
    """Test JaxModelExporter extracts correct model context."""

    @pytest.fixture
    def simple_model(self):
        import jax.numpy as jnp

        params = {
            "w": jnp.array(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),
            "b": jnp.array(np.array([0.1, 0.2], dtype=np.float32)),
        }
        x = jnp.array(np.array([[1.0, 0.5]], dtype=np.float32))

        def forward(params, x):
            return x @ params["w"] + params["b"]

        return forward, params, x

    def test_extract_params(self, simple_model):
        from pymc_rust_compiler.jax_exporter import JaxModelExporter

        fn, params, x = simple_model
        exporter = JaxModelExporter(fn, params, x)
        ctx = exporter.context

        assert ctx.source_framework == "jax"
        assert len(ctx.params) == 2

        w_info = next(p for p in ctx.params if p.name == "w")
        assert w_info.shape == [2, 2]
        assert w_info.size == 4

        b_info = next(p for p in ctx.params if p.name == "b")
        assert b_info.shape == [2]
        assert b_info.size == 2

    def test_extract_validation_points(self, simple_model):
        from pymc_rust_compiler.jax_exporter import JaxModelExporter

        fn, params, x = simple_model
        exporter = JaxModelExporter(fn, params, x, n_extra_points=2)
        ctx = exporter.context

        assert len(ctx.validation_points) == 3  # 1 original + 2 extra

        vp0 = ctx.validation_points[0]
        assert "w" in vp0.params
        assert "b" in vp0.params
        assert "x" in vp0.inputs
        assert vp0.output is not None
        assert "w" in vp0.grad_params
        assert "b" in vp0.grad_params

    def test_forward_output_matches(self, simple_model):
        from pymc_rust_compiler.jax_exporter import JaxModelExporter

        fn, params, x = simple_model
        exporter = JaxModelExporter(fn, params, x)
        ctx = exporter.context

        expected = np.asarray(fn(params, x))
        got = np.array(ctx.validation_points[0].output)
        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_to_dict(self, simple_model):
        from pymc_rust_compiler.jax_exporter import JaxModelExporter

        fn, params, x = simple_model
        exporter = JaxModelExporter(fn, params, x)
        d = exporter.context.to_dict()

        assert d["source_framework"] == "jax"
        assert len(d["parameters"]) == 2
        assert len(d["validation_points"]) >= 1


# ── PyTorch Exporter Tests ──────────────────────────────────────────────────


class TestPytorchExporter:
    """Test PytorchModelExporter extracts correct model context."""

    @pytest.fixture
    def simple_model(self):
        import torch
        import torch.nn as nn

        class Linear(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(2, 3)
                with torch.no_grad():
                    self.fc.weight.copy_(
                        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                    )
                    self.fc.bias.copy_(torch.tensor([0.1, 0.2, 0.3]))

            def forward(self, x):
                return self.fc(x)

        model = Linear()
        x = np.array([[1.0, 0.5]], dtype=np.float32)
        return model, x

    def test_extract_params(self, simple_model):
        from pymc_rust_compiler.pytorch_exporter import PytorchModelExporter

        model, x = simple_model
        exporter = PytorchModelExporter(model, x)
        ctx = exporter.context

        assert ctx.source_framework == "pytorch"
        assert len(ctx.params) == 2  # weight + bias

        weight_info = next(p for p in ctx.params if p.name == "fc.weight")
        assert weight_info.shape == [3, 2]
        assert weight_info.size == 6

    def test_extract_validation_points(self, simple_model):
        from pymc_rust_compiler.pytorch_exporter import PytorchModelExporter

        model, x = simple_model
        exporter = PytorchModelExporter(model, x, n_extra_points=2)
        ctx = exporter.context

        assert len(ctx.validation_points) == 3

        vp0 = ctx.validation_points[0]
        assert "fc.weight" in vp0.params
        assert "fc.bias" in vp0.params
        assert vp0.output is not None
        assert "fc.weight" in vp0.grad_params

    def test_forward_output_matches(self, simple_model):
        import torch
        from pymc_rust_compiler.pytorch_exporter import PytorchModelExporter

        model, x = simple_model
        exporter = PytorchModelExporter(model, x)
        ctx = exporter.context

        with torch.no_grad():
            expected = model(torch.tensor(x)).numpy()
        got = np.array(ctx.validation_points[0].output)
        np.testing.assert_allclose(got, expected, atol=1e-6)


# ── Transpiler Tool Tests ───────────────────────────────────────────────────


class TestTranspilerTools:
    """Test the transpiler's tool execution logic without API calls."""

    def test_write_code_syntax_check(self):
        from pymc_rust_compiler.jax_pytorch_transpiler import (
            _tool_write_code,
            _AgentState,
        )
        from pymc_rust_compiler.jax_exporter import ModelContext

        state = _AgentState(
            direction="jax_to_pytorch",
            source_context=ModelContext(
                source_framework="jax",
                source_code=None,
                params=[],
                inputs=[],
                outputs=[],
                validation_points=[],
            ),
            generated_code="",
        )

        # Valid code
        result = _tool_write_code({"code": "x = 1 + 2"}, state, verbose=False)
        assert "Written" in result
        assert state.generated_code == "x = 1 + 2"

        # Invalid code
        result = _tool_write_code({"code": "def f(:"}, state, verbose=False)
        assert "Syntax error" in result

    def test_validate_no_code(self):
        from pymc_rust_compiler.jax_pytorch_transpiler import (
            _tool_validate,
            _AgentState,
        )
        from pymc_rust_compiler.jax_exporter import ModelContext

        state = _AgentState(
            direction="jax_to_pytorch",
            source_context=ModelContext(
                source_framework="jax",
                source_code=None,
                params=[],
                inputs=[],
                outputs=[],
                validation_points=[],
            ),
            generated_code="",
        )

        result = _tool_validate(state, verbose=False)
        assert "no code" in result.lower()

    @jax_required
    def test_validate_pytorch_correct_model(self):
        """Test that validation passes for a correctly transpiled model."""
        import jax.numpy as jnp
        from pymc_rust_compiler.jax_exporter import JaxModelExporter
        from pymc_rust_compiler.jax_pytorch_transpiler import (
            _tool_write_code,
            _tool_validate,
            _AgentState,
        )

        # Create a simple JAX model
        params = {
            "w": jnp.array(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),
            "b": jnp.array(np.array([0.1, 0.2], dtype=np.float32)),
        }
        x = jnp.array(np.array([[1.0, 0.5]], dtype=np.float32))

        def forward(params, x):
            return x @ params["w"] + params["b"]

        exporter = JaxModelExporter(forward, params, x)
        ctx = exporter.context

        state = _AgentState(
            direction="jax_to_pytorch",
            source_context=ctx,
            generated_code="",
        )

        # Write correct PyTorch code
        pytorch_code = """
import torch
import torch.nn as nn
import numpy as np

def make_model(params):
    class Model(nn.Module):
        def __init__(self, params):
            super().__init__()
            self.w = nn.Parameter(torch.tensor(np.array(params["w"]), dtype=torch.float32))
            self.b = nn.Parameter(torch.tensor(np.array(params["b"]), dtype=torch.float32))

        def forward(self, x):
            return x @ self.w + self.b

    return Model(params)
"""
        _tool_write_code({"code": pytorch_code}, state, verbose=False)
        result = _tool_validate(state, verbose=False)

        assert "PASSED" in result
        assert state.validated is True

    @jax_required
    def test_validate_jax_correct_model(self):
        """Test that validation passes for a correctly transpiled JAX model."""
        import torch
        import torch.nn as nn
        from pymc_rust_compiler.pytorch_exporter import PytorchModelExporter
        from pymc_rust_compiler.jax_pytorch_transpiler import (
            _tool_write_code,
            _tool_validate,
            _AgentState,
        )

        # Create a simple PyTorch model
        class Linear(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
                self.b = nn.Parameter(torch.tensor([0.1, 0.2]))

            def forward(self, x):
                return x @ self.w + self.b

        model = Linear()
        x = np.array([[1.0, 0.5]], dtype=np.float32)

        exporter = PytorchModelExporter(model, x)
        ctx = exporter.context

        state = _AgentState(
            direction="pytorch_to_jax",
            source_context=ctx,
            generated_code="",
        )

        # Write correct JAX code
        jax_code = """
import jax
import jax.numpy as jnp
import numpy as np

def init_params(param_data):
    return {k: jnp.array(np.array(v), dtype=jnp.float32) for k, v in param_data.items()}

def forward(params, x):
    return x @ params["w"] + params["b"]
"""
        _tool_write_code({"code": jax_code}, state, verbose=False)
        result = _tool_validate(state, verbose=False)

        assert "PASSED" in result
        assert state.validated is True


# ── Skill Loading Tests ─────────────────────────────────────────────────────


class TestSkills:
    """Test that skill files load correctly."""

    def test_jax_to_pytorch_skill_exists(self):
        from pymc_rust_compiler.jax_pytorch_transpiler import _load_skill

        skill = _load_skill("jax_to_pytorch")
        assert len(skill) > 0
        assert "PyTorch" in skill

    def test_pytorch_to_jax_skill_exists(self):
        from pymc_rust_compiler.jax_pytorch_transpiler import _load_skill

        skill = _load_skill("pytorch_to_jax")
        assert len(skill) > 0
        assert "JAX" in skill
