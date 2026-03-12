"""Tests for pymc_rust_compiler.exporter."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from pymc_rust_compiler.exporter import (
    ParamInfo,
    RustModelExporter,
    ValidationPoint,
    export_model,
)


# ---------------------------------------------------------------------------
# ParamInfo dataclass
# ---------------------------------------------------------------------------


class TestParamInfo:
    def test_scalar_param(self):
        p = ParamInfo(
            name="mu",
            value_var="mu",
            transform=None,
            shape=[],
            unc_shape=[],
            size=1,
        )
        assert p.is_scalar is True

    def test_vector_param(self):
        p = ParamInfo(
            name="offset",
            value_var="offset",
            transform=None,
            shape=[4],
            unc_shape=[4],
            size=4,
        )
        assert p.is_scalar is False

    def test_log_transformed_param(self):
        p = ParamInfo(
            name="sigma",
            value_var="sigma_log__",
            transform="LogTransform",
            shape=[],
            unc_shape=[],
            size=1,
        )
        assert p.transform == "LogTransform"
        assert p.is_scalar is True

    def test_zerosum_param(self):
        p = ParamInfo(
            name="effect",
            value_var="effect_zerosum__",
            transform="ZeroSumTransform",
            shape=[6],
            unc_shape=[5],
            size=5,
            zerosum_axes=[0],
        )
        assert p.zerosum_axes == [0]
        assert p.size == 5
        assert p.is_scalar is False


# ---------------------------------------------------------------------------
# ValidationPoint dataclass
# ---------------------------------------------------------------------------


class TestValidationPoint:
    def test_basic(self):
        vp = ValidationPoint(
            point={"mu": 0.0, "sigma_log__": 0.5},
            logp=-123.45,
            dlogp=[1.0, -2.0],
        )
        assert vp.logp == -123.45
        assert vp.per_rv_logp is None

    def test_with_per_rv(self):
        vp = ValidationPoint(
            point={"mu": 0.0},
            logp=-10.0,
            dlogp=[1.0],
            per_rv_logp={"mu": -3.0, "y": -7.0},
        )
        assert vp.per_rv_logp["mu"] == -3.0


# ---------------------------------------------------------------------------
# RustModelExporter with real PyMC models
# ---------------------------------------------------------------------------


class TestExporterNormalModel:
    def test_extract_params(self, normal_model):
        exporter = RustModelExporter(normal_model)
        ctx = exporter.context

        assert len(ctx.params) == 2
        param_names = {p.name for p in ctx.params}
        assert "mu" in param_names
        assert "sigma" in param_names

    def test_param_transforms(self, normal_model):
        exporter = RustModelExporter(normal_model)
        ctx = exporter.context

        params_by_name = {p.name: p for p in ctx.params}
        assert params_by_name["mu"].transform is None
        assert params_by_name["sigma"].transform is not None

    def test_n_params(self, normal_model):
        exporter = RustModelExporter(normal_model)
        ctx = exporter.context

        assert ctx.n_params == 2  # mu (1) + sigma (1)

    def test_observed_data(self, normal_model):
        exporter = RustModelExporter(normal_model)
        ctx = exporter.context

        assert "y" in ctx.observed_data
        y_info = ctx.observed_data["y"]
        assert y_info["n"] == 50
        assert y_info["shape"] == [50]
        assert "values" in y_info

    def test_initial_point(self, normal_model):
        exporter = RustModelExporter(normal_model)
        ctx = exporter.context

        assert isinstance(ctx.initial_point, ValidationPoint)
        assert np.isfinite(ctx.initial_point.logp)
        assert len(ctx.initial_point.dlogp) == 2

    def test_extra_points(self, normal_model):
        exporter = RustModelExporter(normal_model, n_extra_points=3)
        ctx = exporter.context

        assert len(ctx.extra_points) == 3
        for pt in ctx.extra_points:
            assert np.isfinite(pt.logp)
            assert len(pt.dlogp) == 2

    def test_extra_points_count(self, normal_model):
        exporter = RustModelExporter(normal_model, n_extra_points=5)
        ctx = exporter.context
        assert len(ctx.extra_points) == 5

    def test_logp_graph_not_empty(self, normal_model):
        exporter = RustModelExporter(normal_model)
        ctx = exporter.context

        assert len(ctx.logp_graph) > 0
        assert len(ctx.dlogp_graph) > 0

    def test_logp_terms(self, normal_model):
        exporter = RustModelExporter(normal_model)
        ctx = exporter.context

        assert len(ctx.logp_terms) > 0
        assert "y" in ctx.logp_terms

    def test_per_rv_logp_at_initial_point(self, normal_model):
        exporter = RustModelExporter(normal_model)
        ctx = exporter.context

        assert ctx.initial_point.per_rv_logp is not None
        per_rv = ctx.initial_point.per_rv_logp
        assert "mu" in per_rv
        assert "sigma" in per_rv
        assert "y" in per_rv

        # Sum of per-RV logp should approximately equal total logp
        total_per_rv = sum(per_rv.values())
        assert abs(total_per_rv - ctx.initial_point.logp) < 1e-6

    def test_seed_reproducibility(self, normal_model):
        ctx1 = RustModelExporter(normal_model, seed=42).context
        ctx2 = RustModelExporter(normal_model, seed=42).context

        for p1, p2 in zip(ctx1.extra_points, ctx2.extra_points):
            assert p1.point == p2.point
            assert p1.logp == p2.logp

    def test_different_seeds_differ(self, normal_model):
        ctx1 = RustModelExporter(normal_model, seed=1).context
        ctx2 = RustModelExporter(normal_model, seed=2).context

        # At least one extra point should differ
        assert any(
            p1.point != p2.point for p1, p2 in zip(ctx1.extra_points, ctx2.extra_points)
        )


class TestExporterLinregModel:
    def test_covariate_extraction(self, linreg_model):
        exporter = RustModelExporter(linreg_model)
        ctx = exporter.context

        # The x data should be extracted as a covariate
        assert len(ctx.covariate_data) >= 1

    def test_param_count(self, linreg_model):
        exporter = RustModelExporter(linreg_model)
        ctx = exporter.context

        assert ctx.n_params == 3  # alpha, beta, sigma

    def test_param_order_matches_value_vars(self, linreg_model):
        exporter = RustModelExporter(linreg_model)
        ctx = exporter.context

        assert len(ctx.param_order) == len(ctx.params)


class TestExporterHierarchicalModel:
    def test_group_index_detection(self, hierarchical_model):
        exporter = RustModelExporter(hierarchical_model)
        ctx = exporter.context

        # Should detect the group index array
        index_covariates = {
            name: info
            for name, info in ctx.covariate_data.items()
            if info.get("is_index_array")
        }
        assert len(index_covariates) >= 1

        # Check the index array properties
        for _name, info in index_covariates.items():
            assert info["n_groups"] == 4

    def test_vector_param_shape(self, hierarchical_model):
        exporter = RustModelExporter(hierarchical_model)
        ctx = exporter.context

        params_by_name = {p.name: p for p in ctx.params}
        assert params_by_name["offset"].size == 4
        assert params_by_name["offset"].shape == [4]

    def test_n_params(self, hierarchical_model):
        exporter = RustModelExporter(hierarchical_model)
        ctx = exporter.context

        # mu(1) + sigma_group(1) + offset(4) + sigma_y(1) = 7
        assert ctx.n_params == 7


# ---------------------------------------------------------------------------
# ModelContext serialization
# ---------------------------------------------------------------------------


class TestModelContextSerialization:
    def test_to_dict(self, normal_model):
        exporter = RustModelExporter(normal_model)
        ctx = exporter.context
        d = ctx.to_dict()

        assert "parameters" in d
        assert "param_order" in d
        assert "n_params" in d
        assert "logp_graph" in d
        assert "dlogp_graph" in d
        assert "observed_data" in d
        assert "covariate_data" in d
        assert "validation" in d

    def test_to_dict_roundtrip_json(self, normal_model):
        exporter = RustModelExporter(normal_model)
        ctx = exporter.context
        d = ctx.to_dict()

        # Should be JSON-serializable
        json_str = json.dumps(d, default=str)
        parsed = json.loads(json_str)
        assert parsed["n_params"] == ctx.n_params

    def test_save_to_file(self, normal_model):
        exporter = RustModelExporter(normal_model)
        ctx = exporter.context

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            ctx.save(f.name)
            saved = json.loads(Path(f.name).read_text())
            assert saved["n_params"] == 2


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------


class TestPromptGeneration:
    def test_prompt_contains_validation(self, normal_model):
        exporter = RustModelExporter(normal_model)
        prompt = exporter.to_prompt()

        assert "Validation" in prompt
        assert "logp" in prompt
        assert "gradient" in prompt

    def test_prompt_contains_param_layout(self, normal_model):
        exporter = RustModelExporter(normal_model)
        prompt = exporter.to_prompt()

        assert "Parameter Layout" in prompt
        assert "position[" in prompt

    def test_prompt_contains_data_section(self, normal_model):
        exporter = RustModelExporter(normal_model)
        prompt = exporter.to_prompt()

        assert "Data" in prompt
        assert "Y_DATA" in prompt.upper() or "data.rs" in prompt

    def test_prompt_contains_logp_graph(self, normal_model):
        exporter = RustModelExporter(normal_model)
        prompt = exporter.to_prompt()

        assert "logp" in prompt.lower()
        assert "Graph" in prompt


# ---------------------------------------------------------------------------
# Rust test generation
# ---------------------------------------------------------------------------


class TestRustTestGeneration:
    def test_generates_test_functions(self, normal_model):
        exporter = RustModelExporter(normal_model)
        rust_tests = exporter.to_rust_tests()

        assert "#[cfg(test)]" in rust_tests
        assert "#[test]" in rust_tests
        assert "test_logp_at_initial_point" in rust_tests
        assert "test_logp_at_point_1" in rust_tests

    def test_test_count_matches_points(self, normal_model):
        exporter = RustModelExporter(normal_model, n_extra_points=3)
        rust_tests = exporter.to_rust_tests()

        assert rust_tests.count("#[test]") == 4  # 1 initial + 3 extra


# ---------------------------------------------------------------------------
# save_all
# ---------------------------------------------------------------------------


class TestSaveAll:
    def test_save_all_creates_files(self, normal_model):
        exporter = RustModelExporter(normal_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter.save_all(tmpdir)
            assert (Path(tmpdir) / "codegen_prompt.txt").exists()
            assert (Path(tmpdir) / "codegen_context.json").exists()
            assert (Path(tmpdir) / "validation_test.rs").exists()


# ---------------------------------------------------------------------------
# export_model convenience function
# ---------------------------------------------------------------------------


class TestExportModel:
    def test_export_returns_exporter(self, normal_model):
        exporter = export_model(normal_model)
        assert isinstance(exporter, RustModelExporter)
        assert exporter.context.n_params == 2

    def test_export_with_output_dir(self, normal_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            export_model(normal_model, output_dir=tmpdir)
            assert (Path(tmpdir) / "codegen_prompt.txt").exists()

    def test_export_with_source_code(self, normal_model):
        source = "mu = pm.Normal('mu', 0, 10)"
        exporter = export_model(normal_model, source_code=source)
        assert exporter.context.source_code == source
