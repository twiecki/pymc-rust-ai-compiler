"""Tests for pymc_rust_compiler.compiler."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock


from pymc_rust_compiler.compiler import (
    CompilationResult,
    _AgentState,
    _execute_tool,
    _generate_data_rs,
    _setup_rust_project,
    _tool_read_file,
    _tool_write_rust_code,
)
from pymc_rust_compiler.exporter import (
    RustModelExporter,
)


# ---------------------------------------------------------------------------
# CompilationResult
# ---------------------------------------------------------------------------


class TestCompilationResult:
    def test_success_when_validated(self):
        r = CompilationResult(
            rust_code="fn logp() {}",
            logp_validated=True,
            validation_errors=[],
            n_attempts=1,
            build_dir=None,
            timings={},
        )
        assert r.success is True

    def test_failure_when_not_validated(self):
        r = CompilationResult(
            rust_code="",
            logp_validated=False,
            validation_errors=["validation failed"],
            n_attempts=5,
            build_dir=None,
            timings={},
        )
        assert r.success is False

    def test_default_token_usage(self):
        r = CompilationResult(
            rust_code="",
            logp_validated=False,
            validation_errors=[],
            n_attempts=0,
            build_dir=None,
            timings={},
        )
        assert r.token_usage["input_tokens"] == 0
        assert r.token_usage["output_tokens"] == 0
        assert r.token_usage["total_tokens"] == 0


# ---------------------------------------------------------------------------
# _AgentState
# ---------------------------------------------------------------------------


class TestAgentState:
    def test_initial_state(self):
        state = _AgentState(
            build_path=Path("/tmp/test"),
            ctx=None,
            messages=[],
        )
        assert state.tool_calls == 0
        assert state.builds == 0
        assert state.validations == 0
        assert state.validated is False


# ---------------------------------------------------------------------------
# _generate_data_rs
# ---------------------------------------------------------------------------


class TestGenerateDataRs:
    def _make_ctx(self, observed_data, covariate_data=None):
        """Create a minimal mock context with observed/covariate data."""
        ctx = MagicMock()
        ctx.observed_data = observed_data
        ctx.covariate_data = covariate_data or {}
        return ctx

    def test_basic_observed_data(self):
        ctx = self._make_ctx(
            {
                "y": {
                    "shape": [3],
                    "dtype": "float64",
                    "n": 3,
                    "values": [1.0, 2.0, 3.0],
                }
            }
        )
        rs = _generate_data_rs(ctx)

        assert "Y_N: usize = 3" in rs
        assert "Y_DATA: &[f64]" in rs

    def test_multidimensional_data_flattened(self):
        ctx = self._make_ctx(
            {
                "y": {
                    "shape": [2, 3],
                    "dtype": "float64",
                    "n": 6,
                    "values": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                }
            }
        )
        rs = _generate_data_rs(ctx)

        assert "Y_N: usize = 6" in rs

    def test_covariate_data(self):
        ctx = self._make_ctx(
            observed_data={},
            covariate_data={
                "x_0": {
                    "shape": [10],
                    "dtype": "float64",
                    "n": 10,
                    "values": list(range(10)),
                    "is_index_array": False,
                    "n_groups": 0,
                }
            },
        )
        rs = _generate_data_rs(ctx)

        assert "X_0_N: usize = 10" in rs
        assert "X_0_DATA: &[f64]" in rs

    def test_empty_data(self):
        ctx = self._make_ctx({})
        rs = _generate_data_rs(ctx)

        assert "Auto-generated" in rs

    def test_no_values_key(self):
        ctx = self._make_ctx({"y": {"shape": "unknown"}})
        rs = _generate_data_rs(ctx)

        # Should not crash, just skip the entry
        assert "Y_DATA" not in rs

    def test_full_precision(self):
        ctx = self._make_ctx(
            {
                "y": {
                    "shape": [1],
                    "n": 1,
                    "values": [3.141592653589793],
                }
            }
        )
        rs = _generate_data_rs(ctx)

        # Should contain full precision representation
        assert "3.14159265358979" in rs


# ---------------------------------------------------------------------------
# _setup_rust_project
# ---------------------------------------------------------------------------


class TestSetupRustProject:
    def test_creates_project_structure(self, normal_model):
        exporter = RustModelExporter(normal_model)
        ctx = exporter.context

        with tempfile.TemporaryDirectory() as tmpdir:
            build_path = Path(tmpdir)
            _setup_rust_project(build_path, ctx)

            assert (build_path / "Cargo.toml").exists()
            assert (build_path / "src" / "lib.rs").exists()
            assert (build_path / "src" / "data.rs").exists()
            assert (build_path / "src" / "generated.rs").exists()
            assert (build_path / "src" / "validate.rs").exists()
            assert (build_path / "src" / "bench.rs").exists()

    def test_cargo_toml_content(self, normal_model):
        exporter = RustModelExporter(normal_model)
        ctx = exporter.context

        with tempfile.TemporaryDirectory() as tmpdir:
            build_path = Path(tmpdir)
            _setup_rust_project(build_path, ctx)

            cargo = (build_path / "Cargo.toml").read_text()
            assert "nuts-rs" in cargo
            assert "nuts-storable" in cargo
            assert 'name = "pymc-compiled-model"' in cargo

    def test_lib_rs_content(self, normal_model):
        exporter = RustModelExporter(normal_model)
        ctx = exporter.context

        with tempfile.TemporaryDirectory() as tmpdir:
            build_path = Path(tmpdir)
            _setup_rust_project(build_path, ctx)

            lib_rs = (build_path / "src" / "lib.rs").read_text()
            assert "pub mod data;" in lib_rs
            assert "pub mod generated;" in lib_rs

    def test_data_rs_contains_observed(self, normal_model):
        exporter = RustModelExporter(normal_model)
        ctx = exporter.context

        with tempfile.TemporaryDirectory() as tmpdir:
            build_path = Path(tmpdir)
            _setup_rust_project(build_path, ctx)

            data_rs = (build_path / "src" / "data.rs").read_text()
            assert "Y_DATA" in data_rs
            assert "Y_N" in data_rs

    def test_npy_files_saved(self, normal_model):
        exporter = RustModelExporter(normal_model)
        ctx = exporter.context

        with tempfile.TemporaryDirectory() as tmpdir:
            build_path = Path(tmpdir)
            _setup_rust_project(build_path, ctx)

            npy_files = list(build_path.glob("*.npy"))
            assert len(npy_files) >= 1

    def test_validate_rs_content(self, normal_model):
        exporter = RustModelExporter(normal_model)
        ctx = exporter.context

        with tempfile.TemporaryDirectory() as tmpdir:
            build_path = Path(tmpdir)
            _setup_rust_project(build_path, ctx)

            validate_rs = (build_path / "src" / "validate.rs").read_text()
            assert "GeneratedLogp" in validate_rs
            assert "CpuLogpFunc" in validate_rs


# ---------------------------------------------------------------------------
# Tool execution: _tool_write_rust_code
# ---------------------------------------------------------------------------


class TestToolWriteRustCode:
    def test_write_code(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            build_path = Path(tmpdir)
            (build_path / "src").mkdir()

            state = _AgentState(
                build_path=build_path,
                ctx=None,
                messages=[],
            )

            code = "pub fn hello() {}"
            result = _tool_write_rust_code({"code": code}, state, verbose=False)

            assert "Written" in result
            assert (build_path / "src" / "generated.rs").read_text() == code

    def test_write_empty_code(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            build_path = Path(tmpdir)
            (build_path / "src").mkdir()

            state = _AgentState(
                build_path=build_path,
                ctx=None,
                messages=[],
            )

            result = _tool_write_rust_code({"code": ""}, state, verbose=False)
            assert "Error" in result


# ---------------------------------------------------------------------------
# Tool execution: _tool_read_file
# ---------------------------------------------------------------------------


class TestToolReadFile:
    def test_read_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            build_path = Path(tmpdir)
            (build_path / "src").mkdir()
            (build_path / "src" / "data.rs").write_text("// data")

            state = _AgentState(
                build_path=build_path,
                ctx=None,
                messages=[],
            )

            result = _tool_read_file({"path": "src/data.rs"}, state, verbose=False)
            assert "// data" in result

    def test_read_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            build_path = Path(tmpdir)

            state = _AgentState(
                build_path=build_path,
                ctx=None,
                messages=[],
            )

            result = _tool_read_file({"path": "nonexistent.rs"}, state, verbose=False)
            assert "File not found" in result

    def test_read_empty_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _AgentState(
                build_path=Path(tmpdir),
                ctx=None,
                messages=[],
            )
            result = _tool_read_file({"path": ""}, state, verbose=False)
            assert "Error" in result

    def test_read_truncation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            build_path = Path(tmpdir)
            (build_path / "big.txt").write_text("x" * 10000)

            state = _AgentState(
                build_path=build_path,
                ctx=None,
                messages=[],
            )

            result = _tool_read_file({"path": "big.txt"}, state, verbose=False)
            assert "truncated" in result
            assert len(result) < 10000


# ---------------------------------------------------------------------------
# _execute_tool dispatch
# ---------------------------------------------------------------------------


class TestExecuteTool:
    def test_unknown_tool(self):
        state = _AgentState(
            build_path=Path("/tmp"),
            ctx=None,
            messages=[],
        )
        result = _execute_tool("nonexistent_tool", {}, state, verbose=False)
        assert "Unknown tool" in result

    def test_dispatches_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            build_path = Path(tmpdir)
            (build_path / "src").mkdir()

            state = _AgentState(
                build_path=build_path,
                ctx=None,
                messages=[],
            )

            result = _execute_tool(
                "write_rust_code", {"code": "// test"}, state, verbose=False
            )
            assert "Written" in result

    def test_dispatches_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            build_path = Path(tmpdir)
            (build_path / "test.rs").write_text("// hello")

            state = _AgentState(
                build_path=build_path,
                ctx=None,
                messages=[],
            )

            result = _execute_tool(
                "read_file", {"path": "test.rs"}, state, verbose=False
            )
            assert "// hello" in result


# ---------------------------------------------------------------------------
# Tool/constant definitions
# ---------------------------------------------------------------------------


class TestToolDefinitions:
    def test_tools_list_has_five_tools(self):
        from pymc_rust_compiler.compiler import TOOLS

        assert len(TOOLS) == 5
        tool_names = {t["name"] for t in TOOLS}
        assert tool_names == {
            "write_rust_code",
            "cargo_build",
            "validate_logp",
            "read_file",
            "add_cargo_dependency",
        }

    def test_system_prompt_exists(self):
        from pymc_rust_compiler.compiler import SYSTEM_PROMPT

        assert len(SYSTEM_PROMPT) > 100
        assert "CpuLogpFunc" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Integration: extract + setup project
# ---------------------------------------------------------------------------


class TestIntegrationExtractAndSetup:
    def test_extract_and_setup_normal(self, normal_model):
        """Full pipeline: extract context → set up Rust project."""
        exporter = RustModelExporter(normal_model)
        ctx = exporter.context

        with tempfile.TemporaryDirectory() as tmpdir:
            build_path = Path(tmpdir)
            _setup_rust_project(build_path, ctx)

            # Verify data.rs has the right number of observed values
            data_rs = (build_path / "src" / "data.rs").read_text()
            assert "Y_N: usize = 50" in data_rs

    def test_extract_and_setup_linreg(self, linreg_model):
        """Full pipeline for linear regression model."""
        exporter = RustModelExporter(linreg_model)
        ctx = exporter.context

        with tempfile.TemporaryDirectory() as tmpdir:
            build_path = Path(tmpdir)
            _setup_rust_project(build_path, ctx)

            data_rs = (build_path / "src" / "data.rs").read_text()
            # Should have observed data
            assert "Y_N" in data_rs
            # Should have covariate data
            assert "_DATA" in data_rs

    def test_extract_and_setup_hierarchical(self, hierarchical_model):
        """Full pipeline for hierarchical model."""
        exporter = RustModelExporter(hierarchical_model)
        ctx = exporter.context

        with tempfile.TemporaryDirectory() as tmpdir:
            build_path = Path(tmpdir)
            _setup_rust_project(build_path, ctx)

            data_rs = (build_path / "src" / "data.rs").read_text()
            assert "Y_N" in data_rs

            # Should have npy files
            npy_files = list(build_path.glob("*.npy"))
            assert len(npy_files) >= 1
