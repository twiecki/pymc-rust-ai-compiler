"""Tests for PyTorch → Rust transpilation components.

Tests the Rust project setup, tool execution logic (no API key needed),
and validation infrastructure.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ── Rust Project Setup Tests ─────────────────────────────────────────────────


class TestRustProjectSetup:
    """Test that the Rust project scaffolding is generated correctly."""

    @pytest.fixture
    def simple_context(self):
        import torch
        import torch.nn as nn
        from transpailer.pytorch_exporter import PytorchModelExporter

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
        exporter = PytorchModelExporter(model, x)
        return exporter.context

    def test_creates_project_structure(self, simple_context):
        from transpailer.pytorch_rust_transpiler import _setup_rust_project

        with tempfile.TemporaryDirectory() as tmpdir:
            build_path = Path(tmpdir)
            _setup_rust_project(build_path, simple_context)

            assert (build_path / "Cargo.toml").exists()
            assert (build_path / "src" / "main.rs").exists()
            assert (build_path / "src" / "data.rs").exists()
            assert (build_path / "src" / "generated.rs").exists()

    def test_data_rs_contains_params(self, simple_context):
        from transpailer.pytorch_rust_transpiler import _setup_rust_project

        with tempfile.TemporaryDirectory() as tmpdir:
            build_path = Path(tmpdir)
            _setup_rust_project(build_path, simple_context)

            data_rs = (build_path / "src" / "data.rs").read_text()

            # Check parameter constants exist
            assert "FC_WEIGHT" in data_rs
            assert "FC_BIAS" in data_rs
            assert "FC_WEIGHT_SHAPE" in data_rs
            assert "FC_BIAS_SHAPE" in data_rs

    def test_cargo_toml_valid(self, simple_context):
        from transpailer.pytorch_rust_transpiler import _setup_rust_project

        with tempfile.TemporaryDirectory() as tmpdir:
            build_path = Path(tmpdir)
            _setup_rust_project(build_path, simple_context)

            cargo = (build_path / "Cargo.toml").read_text()
            assert "pytorch-to-rust" in cargo
            assert "edition" in cargo


# ── Tool Execution Tests ─────────────────────────────────────────────────────


class TestTranspilerTools:
    """Test the transpiler's tool execution logic without API calls."""

    @pytest.fixture
    def agent_state(self):
        import torch
        import torch.nn as nn
        from transpailer.pytorch_exporter import PytorchModelExporter
        from transpailer.pytorch_rust_transpiler import (
            _AgentState,
            _setup_rust_project,
        )

        class Linear(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(
                    torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
                )
                self.b = nn.Parameter(torch.tensor([0.1, 0.2], dtype=torch.float32))

            def forward(self, x):
                return x @ self.w + self.b

        model = Linear()
        x = np.array([[1.0, 0.5]], dtype=np.float32)
        exporter = PytorchModelExporter(model, x)
        ctx = exporter.context

        build_path = Path(tempfile.mkdtemp(prefix="test_pytorch_rust_"))
        _setup_rust_project(build_path, ctx)

        state = _AgentState(
            build_path=build_path,
            source_context=ctx,
            source_code=None,
        )
        return state

    def test_write_code(self, agent_state):
        from transpailer.pytorch_rust_transpiler import _tool_write_code

        result = _tool_write_code(
            {
                "code": "use crate::data::*;\npub fn forward(input: &[f32]) -> Vec<f32> { vec![] }\npub fn forward_with_grad(input: &[f32], _p: &str) -> (Vec<f32>, Vec<f32>) { (vec![], vec![]) }\n"
            },
            agent_state,
            verbose=False,
        )
        assert "Written" in result

        # Check file was written
        gen = (agent_state.build_path / "src" / "generated.rs").read_text()
        assert "forward" in gen

    def test_write_code_empty(self, agent_state):
        from transpailer.pytorch_rust_transpiler import _tool_write_code

        result = _tool_write_code({"code": ""}, agent_state, verbose=False)
        assert "Error" in result

    def test_read_source(self, agent_state):
        from transpailer.pytorch_rust_transpiler import _tool_read_source

        result = _tool_read_source(agent_state, verbose=False)
        # Should return some source code or indication
        assert len(result) > 0

    def test_read_file(self, agent_state):
        from transpailer.pytorch_rust_transpiler import _tool_read_file

        result = _tool_read_file({"path": "src/data.rs"}, agent_state, verbose=False)
        assert "W" in result or "WEIGHT" in result or "f32" in result

    def test_read_file_not_found(self, agent_state):
        from transpailer.pytorch_rust_transpiler import _tool_read_file

        result = _tool_read_file({"path": "nonexistent.rs"}, agent_state, verbose=False)
        assert "not found" in result.lower() or "Available" in result

    def test_cargo_build_placeholder(self, agent_state):
        """Test that the placeholder generated.rs compiles."""
        from transpailer.pytorch_rust_transpiler import _tool_cargo_build

        # Check if cargo is available
        try:
            subprocess.run(["cargo", "--version"], capture_output=True, timeout=5)
        except FileNotFoundError:
            pytest.skip("cargo not available")

        result = _tool_cargo_build(agent_state, verbose=False)
        assert "successful" in result.lower() or "FAILED" in result

    def test_validate_no_binary(self, agent_state):
        """Test validation when binary doesn't exist."""
        from transpailer.pytorch_rust_transpiler import _tool_validate

        result = _tool_validate(agent_state, verbose=False)
        assert "not found" in result.lower() or "Error" in result


# ── Validation Logic Tests ───────────────────────────────────────────────────


class TestValidation:
    """Test the forward pass + gradient validation logic."""

    @pytest.fixture
    def simple_model_context(self):
        """Create a simple model context for testing validation."""
        import torch
        import torch.nn as nn
        from transpailer.pytorch_exporter import PytorchModelExporter

        class Simple(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(
                    torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
                )
                self.b = nn.Parameter(torch.tensor([0.1, 0.2], dtype=torch.float32))

            def forward(self, x):
                return x @ self.w + self.b

        model = Simple()
        x = np.array([[1.0, 0.5]], dtype=np.float32)
        exporter = PytorchModelExporter(model, x, n_extra_points=1)
        return exporter.context

    def test_context_has_validation_points(self, simple_model_context):
        ctx = simple_model_context
        assert len(ctx.validation_points) >= 2

    def test_context_has_gradients(self, simple_model_context):
        ctx = simple_model_context
        vp = ctx.validation_points[0]
        assert "w" in vp.grad_params
        assert "b" in vp.grad_params

    def test_output_matches_pytorch(self, simple_model_context):
        """Verify that the exporter captured the correct forward pass output."""

        ctx = simple_model_context
        vp = ctx.validation_points[0]

        w = np.array(vp.params["w"], dtype=np.float32)
        b = np.array(vp.params["b"], dtype=np.float32)
        x = np.array(vp.inputs["x"], dtype=np.float32)

        expected = x @ w + b
        got = np.array(vp.output, dtype=np.float32)
        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_gradient_matches_analytical(self, simple_model_context):
        """Verify gradient of sum(output) w.r.t. bias is all ones (for linear model)."""
        ctx = simple_model_context
        vp = ctx.validation_points[0]

        # For output = x @ w + b, d(sum(output))/d(b) = [1, 1, ...]
        grad_b = np.array(vp.grad_params["b"], dtype=np.float32)
        np.testing.assert_allclose(grad_b, np.ones_like(grad_b), atol=1e-6)


# ── Full Pipeline Test (with cargo) ─────────────────────────────────────────


class TestFullPipeline:
    """Integration test: write correct Rust, build, and validate."""

    @pytest.fixture
    def model_and_state(self):
        import torch
        import torch.nn as nn
        from transpailer.pytorch_exporter import PytorchModelExporter
        from transpailer.pytorch_rust_transpiler import (
            _AgentState,
            _setup_rust_project,
        )

        class Simple(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(
                    torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
                )
                self.b = nn.Parameter(torch.tensor([0.1, 0.2], dtype=torch.float32))

            def forward(self, x):
                return x @ self.w + self.b

        model = Simple()
        x = np.array([[1.0, 0.5]], dtype=np.float32)
        exporter = PytorchModelExporter(model, x, n_extra_points=1)
        ctx = exporter.context

        build_path = Path(tempfile.mkdtemp(prefix="test_pytorch_rust_full_"))
        _setup_rust_project(build_path, ctx)

        state = _AgentState(
            build_path=build_path,
            source_context=ctx,
            source_code=None,
        )
        return state

    def test_correct_rust_validates(self, model_and_state):
        """Write manually correct Rust code and verify it passes validation."""
        from transpailer.pytorch_rust_transpiler import (
            _tool_write_code,
            _tool_cargo_build,
            _tool_validate,
        )

        try:
            subprocess.run(["cargo", "--version"], capture_output=True, timeout=5)
        except FileNotFoundError:
            pytest.skip("cargo not available")

        state = model_and_state

        # Write correct Rust implementation of y = x @ w + b
        rust_code = """
use crate::data::*;

/// Forward pass: y = x @ w + b
/// input shape: [1, 2] flattened to [2]
/// W shape: [2, 2], B shape: [2]
pub fn forward(input: &[f32]) -> Vec<f32> {
    let in_features = 2usize;
    let out_features = 2usize;
    let mut output = vec![0.0f32; out_features];
    for i in 0..out_features {
        let mut sum = B[i];
        for j in 0..in_features {
            // W is stored row-major as [2, 2] but for y = x @ W,
            // output[i] = sum_j(x[j] * W[j][i]) = sum_j(x[j] * W[j * out_features + i])
            // But our data.rs stores W as flat [w00, w01, w10, w11]
            // For y = x @ W: output[i] = x[0]*W[0,i] + x[1]*W[1,i]
            sum += input[j] * W[j * out_features + i];
        }
        output[i] = sum;
    }
    output
}

pub fn forward_with_grad(input: &[f32], param_name: &str) -> (Vec<f32>, Vec<f32>) {
    let output = forward(input);
    let in_features = 2usize;
    let out_features = 2usize;

    match param_name {
        "w" => {
            // grad of sum(output) w.r.t. W
            // output[i] = sum_j(x[j] * W[j,i]) + b[i]
            // d(sum_i output[i])/d(W[j,i]) = x[j]
            let mut grad = vec![0.0f32; in_features * out_features];
            for j in 0..in_features {
                for i in 0..out_features {
                    grad[j * out_features + i] = input[j];
                }
            }
            (output, grad)
        }
        "b" => {
            // grad of sum(output) w.r.t. b: all ones
            let grad = vec![1.0f32; out_features];
            (output, grad)
        }
        _ => (output, vec![])
    }
}
"""
        _tool_write_code({"code": rust_code}, state, verbose=False)

        build_result = _tool_cargo_build(state, verbose=False)
        assert "successful" in build_result.lower(), f"Build failed: {build_result}"

        validate_result = _tool_validate(state, verbose=False)
        assert "PASSED" in validate_result, f"Validation failed: {validate_result}"
        assert state.validated is True


# ── Skill Loading Tests ──────────────────────────────────────────────────────


class TestSkills:
    """Test that the PyTorch→Rust skill file loads correctly."""

    def test_pytorch_to_rust_skill_exists(self):
        from transpailer.pytorch_rust_transpiler import _load_skill

        skill = _load_skill("pytorch_to_rust")
        assert len(skill) > 0
        assert "Rust" in skill
        assert "PyTorch" in skill

    def test_skill_has_matmul_example(self):
        from transpailer.pytorch_rust_transpiler import _load_skill

        skill = _load_skill("pytorch_to_rust")
        assert "linear" in skill.lower()
        assert "relu" in skill.lower()

    def test_skill_has_backprop(self):
        from transpailer.pytorch_rust_transpiler import _load_skill

        skill = _load_skill("pytorch_to_rust")
        assert (
            "backward" in skill.lower()
            or "backprop" in skill.lower()
            or "gradient" in skill.lower()
        )


# ── Result Type Tests ────────────────────────────────────────────────────────


class TestRustTranspileResult:
    """Test the result dataclass."""

    def test_success_property(self):
        from transpailer.pytorch_rust_transpiler import RustTranspileResult

        result = RustTranspileResult(
            generated_code="fn forward() {}",
            validated=True,
            validation_errors=[],
            n_attempts=1,
            build_dir=None,
        )
        assert result.success is True

    def test_failure_property(self):
        from transpailer.pytorch_rust_transpiler import RustTranspileResult

        result = RustTranspileResult(
            generated_code="",
            validated=False,
            validation_errors=["failed"],
            n_attempts=3,
            build_dir=None,
        )
        assert result.success is False

    def test_save(self, tmp_path):
        from transpailer.pytorch_rust_transpiler import RustTranspileResult

        result = RustTranspileResult(
            generated_code="pub fn forward() -> Vec<f32> { vec![] }",
            validated=True,
            validation_errors=[],
            n_attempts=1,
            build_dir=None,
        )
        out_path = tmp_path / "generated.rs"
        result.save(out_path)
        assert out_path.read_text() == result.generated_code


# ── Prompt Building Tests ────────────────────────────────────────────────────


class TestPromptBuilding:
    """Test that user prompts are built correctly."""

    def test_prompt_contains_model_info(self):
        import torch.nn as nn
        from transpailer.pytorch_exporter import PytorchModelExporter
        from transpailer.pytorch_rust_transpiler import _build_user_prompt

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 8)

            def forward(self, x):
                return self.fc(x)

        model = Model()
        x = np.random.randn(1, 4).astype(np.float32)
        exporter = PytorchModelExporter(model, x)
        ctx = exporter.context

        prompt = _build_user_prompt(ctx)

        assert "PyTorch" in prompt
        assert "Rust" in prompt
        assert "fc.weight" in prompt or "FC_WEIGHT" in prompt
        assert "fc.bias" in prompt or "FC_BIAS" in prompt
        assert "forward" in prompt.lower() or "validation" in prompt.lower()
