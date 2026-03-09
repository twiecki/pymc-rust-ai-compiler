"""Extract everything from a pm.Model() needed to generate Rust logp code."""

from __future__ import annotations

import inspect
import io
import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymc as pm
try:
    from pytensor.graph.traversal import graph_inputs
except ImportError:
    from pytensor.graph.basic import graph_inputs
from pytensor.printing import debugprint
from pytensor.tensor import TensorConstant


@dataclass
class ParamInfo:
    name: str
    value_var: str
    transform: str | None
    shape: list[int]
    size: int

    @property
    def is_scalar(self) -> bool:
        return self.size == 1


@dataclass
class ValidationPoint:
    point: dict[str, list | float]
    logp: float
    dlogp: list[float]


@dataclass
class ModelContext:
    """All information extracted from a PyMC model."""

    source_code: str | None
    params: list[ParamInfo]
    param_order: list[str]
    n_params: int
    logp_graph: str
    logp_terms: dict[str, str]
    observed_data: dict[str, dict]
    covariate_data: dict[str, dict]  # predictor/input data (e.g. x in regression)
    initial_point: ValidationPoint
    extra_points: list[ValidationPoint]

    def to_dict(self) -> dict:
        return {
            "source_code": self.source_code,
            "parameters": [
                {
                    "name": p.name,
                    "value_var": p.value_var,
                    "transform": p.transform,
                    "shape": p.shape,
                    "size": p.size,
                }
                for p in self.params
            ],
            "param_order": self.param_order,
            "n_params": self.n_params,
            "logp_graph": self.logp_graph,
            "logp_terms": self.logp_terms,
            "observed_data": self.observed_data,
            "covariate_data": self.covariate_data,
            "validation": {
                "initial_point": {
                    "point": self.initial_point.point,
                    "logp": self.initial_point.logp,
                    "dlogp": self.initial_point.dlogp,
                },
                "extra_points": [
                    {"point": p.point, "logp": p.logp, "dlogp": p.dlogp}
                    for p in self.extra_points
                ],
            },
        }

    def save(self, path: str | Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class RustModelExporter:
    """Extract everything from a pm.Model() needed for Rust code generation."""

    def __init__(
        self,
        model: pm.Model,
        source_code: str | None = None,
        n_extra_points: int = 3,
        seed: int = 123,
    ):
        self.model = model
        self._source_code = source_code
        self._n_extra_points = n_extra_points
        self._seed = seed
        self._context: ModelContext | None = None

    @property
    def context(self) -> ModelContext:
        if self._context is None:
            self._context = self._extract()
        return self._context

    def _extract(self) -> ModelContext:
        model = self.model

        params = []
        for rv in model.free_RVs:
            value_var = model.rvs_to_values[rv]
            transform = model.rvs_to_transforms.get(rv, None)
            shape = list(rv.type.shape) if hasattr(rv.type, "shape") else []
            size = int(np.prod(shape)) if shape else 1
            params.append(
                ParamInfo(
                    name=rv.name,
                    value_var=value_var.name,
                    transform=type(transform).__name__ if transform else None,
                    shape=shape,
                    size=size,
                )
            )

        param_order = [v.name for v in model.value_vars]
        n_params = sum(p.size for p in params)

        logp_graph = _capture_debugprint(model.logp())

        logp_terms = {}
        for rv in model.free_RVs:
            term = model.logp(vars=[rv], sum=False)
            logp_terms[rv.name] = _capture_debugprint(term)
        for rv in model.observed_RVs:
            term = model.logp(vars=[rv], sum=False)
            logp_terms[rv.name] = _capture_debugprint(term)

        observed_data = {}
        for rv in model.observed_RVs:
            obs = model.rvs_to_values[rv]
            # PyMC 5: TensorConstant has .data, SharedVariable has .get_value()
            if hasattr(obs, "data"):
                data = np.asarray(obs.data)
            elif hasattr(obs, "get_value"):
                data = obs.get_value()
            else:
                data = None
            if data is not None:
                observed_data[rv.name] = {
                    "shape": list(data.shape),
                    "dtype": str(data.dtype),
                    "n": int(np.prod(data.shape)),
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "mean": float(np.mean(data)),
                    "values": data.tolist(),
                }
            else:
                observed_data[rv.name] = {"shape": "unknown"}

        # Extract covariate/predictor data from the computational graph
        # These are non-scalar TensorConstants that aren't observed data
        covariate_data = self._extract_covariates(model, observed_data)

        logp_fn = model.compile_logp()
        dlogp_fn = model.compile_dlogp()
        test_point = model.initial_point()

        initial = ValidationPoint(
            point={
                k: v.tolist() if hasattr(v, "tolist") else v
                for k, v in test_point.items()
            },
            logp=float(logp_fn(test_point)),
            dlogp=dlogp_fn(test_point).tolist(),
        )

        rng = np.random.default_rng(self._seed)
        extra_points = []
        for _ in range(self._n_extra_points):
            point = {}
            for k, v in test_point.items():
                if hasattr(v, "shape") and v.shape:
                    point[k] = (rng.standard_normal(v.shape) * 0.5).tolist()
                else:
                    point[k] = float(rng.standard_normal() * 0.5)
            extra_points.append(
                ValidationPoint(
                    point=point,
                    logp=float(logp_fn(point)),
                    dlogp=dlogp_fn(point).tolist(),
                )
            )

        source = self._source_code or self._try_extract_source()

        return ModelContext(
            source_code=source,
            params=params,
            param_order=param_order,
            n_params=n_params,
            logp_graph=logp_graph,
            logp_terms=logp_terms,
            observed_data=observed_data,
            covariate_data=covariate_data,
            initial_point=initial,
            extra_points=extra_points,
        )

    @staticmethod
    def _extract_covariates(
        model: pm.Model, observed_data: dict[str, dict]
    ) -> dict[str, dict]:
        """Find non-scalar constant arrays in the logp graph (predictors/covariates).

        These are numpy arrays passed into the model (e.g. x in regression)
        that aren't observed RVs but are needed for computation.
        """
        logp = model.logp()
        inputs = graph_inputs([logp])

        # Collect observed data values for deduplication
        obs_arrays = []
        for info in observed_data.values():
            vals = info.get("values")
            if vals is not None:
                obs_arrays.append(np.asarray(vals))

        covariates = {}
        unnamed_idx = 0
        for inp in inputs:
            if not isinstance(inp, TensorConstant):
                continue
            data = np.asarray(inp.data)
            if data.ndim == 0 or data.size <= 1:
                continue

            # Skip if this is already in observed data
            is_observed = False
            for obs in obs_arrays:
                if data.shape == obs.shape and np.allclose(data, obs):
                    is_observed = True
                    break
            if is_observed:
                continue

            # Use the tensor's name if available, otherwise auto-generate
            if inp.name:
                name = inp.name
            else:
                name = f"x_{unnamed_idx}"
                unnamed_idx += 1

            # Detect integer index arrays (e.g., group_idx with values 0..n-1)
            # Require at least 3 distinct values to avoid classifying binary covariates
            is_index = False
            n_groups = 0
            flat = data.ravel()
            if np.all(flat == np.floor(flat)) and np.min(flat) == 0:
                unique_vals = np.unique(flat)
                max_val = int(np.max(flat))
                if max_val >= 2 and max_val < 200 and np.array_equal(unique_vals, np.arange(max_val + 1)):
                    is_index = True
                    n_groups = max_val + 1

            covariates[name] = {
                "shape": list(data.shape),
                "dtype": str(data.dtype),
                "n": int(np.prod(data.shape)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "mean": float(np.mean(data)),
                "values": data.tolist(),
                "is_index_array": is_index,
                "n_groups": n_groups,
            }

        return covariates

    def _infer_data_mapping(self, ctx) -> list[str]:
        """Try to match source code variable names to extracted data constants.

        Looks for patterns like var[index_var] and matches based on data characteristics.
        """
        source = ctx.source_code
        if not source:
            return []

        import re as _re

        hints = []

        # Find variable names used in indexing: a[group_idx] → group_idx
        index_vars = set(_re.findall(r'\w+\[(\w+)\]', source))
        # Find other variable names that are likely data arrays (not parameters)
        # Parameters are defined with ~ or = expressions
        param_names = set(_re.findall(r'(\w+)\s*~', source))
        param_names.update(_re.findall(r'(\w+)\s*=\s*\w+', source))

        # Covariates: match index arrays to index variables, others to remaining vars
        covariate_items = list(ctx.covariate_data.items())
        index_covariates = [(n, i) for n, i in covariate_items if i.get("is_index_array")]
        non_index_covariates = [(n, i) for n, i in covariate_items if not i.get("is_index_array")]

        # Match index covariates to index variables from source
        # If only one index covariate and one index variable, match them directly
        if len(index_covariates) == 1 and len(index_vars) >= 1:
            cov_name, cov_info = index_covariates[0]
            n_groups = cov_info.get("n_groups", 0)
            src_var = sorted(index_vars)[0]
            hints.append(
                f"`{src_var}` in source → `{cov_name.upper()}_DATA` "
                f"(integer indices, {n_groups} groups, cast to `usize` for indexing)"
            )
        else:
            for (cov_name, cov_info), src_var in zip(index_covariates, sorted(index_vars)):
                n_groups = cov_info.get("n_groups", 0)
                hints.append(
                    f"`{src_var}` in source → `{cov_name.upper()}_DATA` "
                    f"(integer indices, {n_groups} groups, cast to `usize` for indexing)"
                )

        # Match remaining covariates to non-index, non-parameter variables
        # Look for variables used in arithmetic but not defined as params
        arith_vars = set(_re.findall(r'[\+\-\*]\s*(\w+)', source))
        arith_vars -= param_names
        arith_vars -= index_vars
        arith_vars -= {'observed', 'shape'}
        remaining_src_vars = sorted(arith_vars)

        for (cov_name, cov_info), src_var in zip(non_index_covariates, remaining_src_vars):
            hints.append(
                f"`{src_var}` in source → `{cov_name.upper()}_DATA`"
            )

        # Observed data mapping
        for obs_name in ctx.observed_data:
            # Find the observed variable in source (pattern: var ~ ..., observed)
            obs_match = _re.search(r'(\w+)\s*~.*observed', source)
            if obs_match:
                src_obs = obs_match.group(1)
                hints.append(
                    f"`{src_obs}` (observed) in source → `{obs_name.upper()}_DATA`"
                )

        return hints

    def _try_extract_source(self) -> str | None:
        try:
            for frame_info in inspect.stack():
                source = inspect.getsource(frame_info.frame)
                if "pm.Model" in source or "pymc.Model" in source:
                    return textwrap.dedent(source)
        except (OSError, TypeError):
            pass
        return None

    def to_prompt(self) -> str:
        """Generate a complete LLM prompt for Rust code generation."""
        ctx = self.context
        parts = []

        parts.append("Generate a Rust logp+gradient function for this PyMC model.\n")

        if ctx.source_code:
            parts.append(f"## PyMC Model Code\n```python\n{ctx.source_code}\n```\n")

        parts.append("## Parameter Layout (unconstrained space)")
        parts.append("The `position` slice contains these parameters in order:\n")
        offset = 0
        for p in ctx.params:
            if p.is_scalar:
                line = f"- position[{offset}] = {p.value_var}"
            else:
                line = f"- position[{offset}..{offset + p.size}] = {p.value_var} (shape {p.shape})"
            if p.transform:
                line += f"  [{p.transform} of {p.name}]"
            parts.append(line)
            offset += p.size
        parts.append(f"\nTotal unconstrained parameters: {ctx.n_params}\n")

        # List all data available in data.rs
        all_data = {}
        all_data.update(ctx.observed_data)
        all_data.update(ctx.covariate_data)
        if all_data:
            parts.append("## Data (available via `use crate::data::*;`)\n")
            parts.append(
                "All data arrays are pre-generated in `data.rs` with full f64 precision.\n"
                "Do NOT embed data in your code — use the constants from `data.rs`.\n"
            )
            parts.append("Available constants:")
            for name, info in all_data.items():
                rust_name = name.upper()
                n = info.get("n", "?")
                mn = info.get("min")
                mx = info.get("max")
                mean = info.get("mean")
                range_str = f"[{mn:.3f}, {mx:.3f}]" if mn is not None else "unknown"
                mean_str = f"{mean:.3f}" if mean is not None else "unknown"
                is_obs = name in ctx.observed_data
                is_idx = info.get("is_index_array", False)
                n_groups = info.get("n_groups", 0)
                if is_obs:
                    label = "observed"
                elif is_idx:
                    label = f"INTEGER INDEX ARRAY (values 0..{n_groups-1}, {n_groups} groups) — cast to usize for array indexing"
                else:
                    label = "covariate/predictor"
                parts.append(
                    f"- `{rust_name}_DATA: &[f64]` — {label}, n={n}, "
                    f"range={range_str}, mean={mean_str}"
                )
                parts.append(f"  `{rust_name}_N: usize = {n}`")

            # Try to add source-to-data mapping hints
            if ctx.source_code:
                mapping_hints = self._infer_data_mapping(ctx)
                if mapping_hints:
                    parts.append("")
                    parts.append("**Data mapping** (source variable → Rust constant):")
                    for hint in mapping_hints:
                        parts.append(f"- {hint}")
            parts.append("")

        parts.append(f"## PyTensor Computational Graph (logp)\n```\n{ctx.logp_graph}\n```\n")

        parts.append("## Individual logp terms\n")
        for name, term in ctx.logp_terms.items():
            display = term[:2000] + ("..." if len(term) > 2000 else "")
            parts.append(f"### {name}\n```\n{display}\n```\n")

        parts.append("## Validation")
        parts.append("Your generated code MUST produce these exact values (within float64 precision):\n")

        parts.append(f"At initial point: {json.dumps(ctx.initial_point.point)}")
        parts.append(f"- logp = {ctx.initial_point.logp:.10f}")
        parts.append(f"- gradient = {ctx.initial_point.dlogp}\n")

        for i, pt in enumerate(ctx.extra_points):
            parts.append(f"At test point {i + 1}: {json.dumps(pt.point)}")
            parts.append(f"- logp = {pt.logp:.10f}")
            parts.append(f"- gradient = {pt.dlogp}\n")

        parts.append("""## Output
Generate ONLY the Rust function body. Include comments mapping each section
to the corresponding PyMC variable. The function signature is:
```rust
fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, SampleError>
```""")

        return "\n".join(parts)

    def to_context(self) -> dict:
        return self.context.to_dict()

    def to_rust_tests(self, struct_name: str = "GeneratedLogp") -> str:
        """Generate Rust test code that validates logp+gradient at known points."""
        ctx = self.context
        tests = []

        tests.append("""
#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64, tol: f64, name: &str) {
        let diff = (a - b).abs();
        assert!(diff < tol, "{}: expected {}, got {}, diff={}", name, b, a, diff);
    }
""")

        all_points = [("initial_point", ctx.initial_point)] + [
            (f"point_{i + 1}", p) for i, p in enumerate(ctx.extra_points)
        ]

        for test_name, vp in all_points:
            position = []
            for name in ctx.param_order:
                val = vp.point[name]
                if isinstance(val, list):
                    position.extend(val)
                else:
                    position.append(val)

            tests.append(f"""
    #[test]
    fn test_logp_at_{test_name}() {{
        let mut logp_fn = {struct_name};
        let position = vec!{position};
        let mut gradient = vec![0.0f64; {ctx.n_params}];
        let logp = logp_fn.logp(&position, &mut gradient).unwrap();

        assert_close(logp, {vp.logp:.10f}, 1e-6, "logp");""")

            for i, g in enumerate(vp.dlogp):
                tests.append(
                    f'        assert_close(gradient[{i}], {g:.10e}, 1e-4, "grad[{i}]");'
                )
            tests.append("    }\n")

        tests.append("}\n")
        return "\n".join(tests)

    def save_all(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "codegen_prompt.txt").write_text(self.to_prompt())
        self.context.save(output_dir / "codegen_context.json")
        (output_dir / "validation_test.rs").write_text(self.to_rust_tests())


def _capture_debugprint(var) -> str:
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    debugprint(var, print_type=True)
    sys.stdout = old_stdout
    return buffer.getvalue()


def export_model(
    model: pm.Model,
    source_code: str | None = None,
    output_dir: str | Path | None = None,
    n_extra_points: int = 3,
) -> RustModelExporter:
    """One-liner to export a PyMC model for Rust compilation.

    Usage:
        exporter = export_model(model)
        prompt = exporter.to_prompt()
    """
    exporter = RustModelExporter(
        model, source_code=source_code, n_extra_points=n_extra_points
    )
    if output_dir:
        exporter.save_all(output_dir)
    return exporter
