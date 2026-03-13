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
from pytensor.tensor import TensorConstant


@dataclass
class ParamInfo:
    name: str
    value_var: str
    transform: str | None
    shape: list[int]  # original RV shape
    unc_shape: list[int]  # unconstrained (value_var) shape
    size: int  # unconstrained size
    zerosum_axes: list[int] | None = None  # axes for ZeroSumTransform

    @property
    def is_scalar(self) -> bool:
        return self.size == 1


@dataclass
class ValidationPoint:
    point: dict[str, list | float]
    logp: float
    dlogp: list[float]
    per_rv_logp: dict[str, float] | None = None  # logp contribution per RV


@dataclass
class ModelContext:
    """All information extracted from a PyMC model."""

    source_code: str | None
    params: list[ParamInfo]
    param_order: list[str]
    n_params: int
    logp_graph: str
    dlogp_graph: str
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
                    "unc_shape": p.unc_shape,
                    "size": p.size,
                }
                for p in self.params
            ],
            "param_order": self.param_order,
            "n_params": self.n_params,
            "logp_graph": self.logp_graph,
            "dlogp_graph": self.dlogp_graph,
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
            # Use value_var shape for unconstrained size (transforms may change dims)
            rv_shape = list(rv.type.shape) if hasattr(rv.type, "shape") else []
            unc_shape = (
                list(value_var.type.shape) if hasattr(value_var.type, "shape") else []
            )
            size = int(np.prod(unc_shape)) if unc_shape else 1
            zs_axes = None
            if transform and type(transform).__name__ == "ZeroSumTransform":
                zs_axes = list(transform.zerosum_axes)

            params.append(
                ParamInfo(
                    name=rv.name,
                    value_var=value_var.name,
                    transform=type(transform).__name__ if transform else None,
                    shape=rv_shape,
                    unc_shape=unc_shape,
                    size=size,
                    zerosum_axes=zs_axes,
                )
            )

        param_order = [v.name for v in model.value_vars]
        n_params = sum(p.size for p in params)

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

        # Compile and extract optimized graphs (PyTensor optimization removes cruft)
        logp_fn = model.compile_logp()
        dlogp_fn = model.compile_dlogp()
        logp_graph = _capture_fgraph(logp_fn.f.maker.fgraph)
        dlogp_graph = _capture_fgraph(dlogp_fn.f.maker.fgraph)

        # Per-RV logp terms (also optimized)
        logp_terms = {}
        all_rvs = list(model.free_RVs) + list(model.observed_RVs)
        for rv in all_rvs:
            rv_fn = model.compile_logp(vars=[rv], sum=False)
            logp_terms[rv.name] = _capture_fgraph(rv_fn.f.maker.fgraph)

        test_point = model.initial_point()

        # Compute per-RV logp for debugging (reuse compiled per-RV functions)
        per_rv_logp_fns = {}
        for rv in all_rvs:
            per_rv_logp_fns[rv.name] = model.compile_logp(vars=[rv])

        def _per_rv_logp(point):
            return {name: float(fn(point)) for name, fn in per_rv_logp_fns.items()}

        initial = ValidationPoint(
            point={
                k: v.tolist() if hasattr(v, "tolist") else v
                for k, v in test_point.items()
            },
            logp=float(logp_fn(test_point)),
            dlogp=dlogp_fn(test_point).tolist(),
            per_rv_logp=_per_rv_logp(test_point),
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
            dlogp_graph=dlogp_graph,
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
                if (
                    max_val >= 2
                    and max_val < 200
                    and np.array_equal(unique_vals, np.arange(max_val + 1))
                ):
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
        index_vars = set(_re.findall(r"\w+\[(\w+)\]", source))
        # Find other variable names that are likely data arrays (not parameters)
        # Parameters are defined with ~ or = expressions
        param_names = set(_re.findall(r"(\w+)\s*~", source))
        param_names.update(_re.findall(r"(\w+)\s*=\s*\w+", source))

        # Covariates: match index arrays to index variables, others to remaining vars
        covariate_items = list(ctx.covariate_data.items())
        index_covariates = [
            (n, i) for n, i in covariate_items if i.get("is_index_array")
        ]
        non_index_covariates = [
            (n, i) for n, i in covariate_items if not i.get("is_index_array")
        ]

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
            for (cov_name, cov_info), src_var in zip(
                index_covariates, sorted(index_vars)
            ):
                n_groups = cov_info.get("n_groups", 0)
                hints.append(
                    f"`{src_var}` in source → `{cov_name.upper()}_DATA` "
                    f"(integer indices, {n_groups} groups, cast to `usize` for indexing)"
                )

        # Match remaining covariates to non-index, non-parameter variables
        # Look for variables used in arithmetic but not defined as params
        arith_vars = set(_re.findall(r"[\+\-\*]\s*(\w+)", source))
        arith_vars -= param_names
        arith_vars -= index_vars
        arith_vars -= {"observed", "shape"}
        remaining_src_vars = sorted(arith_vars)

        for (cov_name, cov_info), src_var in zip(
            non_index_covariates, remaining_src_vars
        ):
            hints.append(f"`{src_var}` in source → `{cov_name.upper()}_DATA`")

        # Observed data mapping
        for obs_name in ctx.observed_data:
            # Find the observed variable in source (pattern: var ~ ..., observed)
            obs_match = _re.search(r"(\w+)\s*~.*observed", source)
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
        has_zerosum = False
        for p in ctx.params:
            if p.is_scalar:
                line = f"- position[{offset}] = {p.value_var}"
            else:
                line = f"- position[{offset}..{offset + p.size}] = {p.value_var} (unconstrained shape {p.unc_shape})"
            if p.transform:
                line += f"  [{p.transform} of {p.name}, original shape {p.shape}]"
                if p.transform == "ZeroSumTransform":
                    has_zerosum = True
            parts.append(line)
            offset += p.size
        parts.append(f"\nTotal unconstrained parameters: {ctx.n_params}\n")

        if has_zerosum:
            parts.append("## ZeroSumTransform Details")
            parts.append(
                "ZeroSumTransform maps n-1 unconstrained params to n constrained params that sum to zero.\n"
                "The log_jac_det of ZeroSumTransform is 0 (no Jacobian correction needed).\n\n"
                "**Exact backward formula** (`extend_axis`) for a single axis with constrained size n:\n"
                "Given unconstrained `x[0..n-2]` (length n-1), produce constrained `y[0..n-1]` (length n):\n"
                "```rust\n"
                "let n: f64 = constrained_size as f64;  // e.g. 6.0_f64 for shape=[6]\n"
                "let sum_x: f64 = x.iter().sum();\n"
                "let norm: f64 = sum_x / (n.sqrt() + n);\n"
                "let fill: f64 = norm - sum_x / n.sqrt();\n"
                "y[i] = x[i] - norm    // for i in 0..n-2\n"
                "y[n-1] = fill - norm  // the last element\n"
                "// Guarantee: y.iter().sum() ≈ 0.0\n"
                "// IMPORTANT: Always use f64 type annotations or _f64 suffix for n!\n"
                "```\n\n"
                "**Multi-axis ZeroSumTransform** (`n_zerosum_axes > 1`):\n"
                "When applied to multiple axes (e.g., axes [-2, -1] for a 3D array),\n"
                "the transform is applied sequentially: first extend axis -2, then extend axis -1.\n"
                "Each `extend_axis` call independently adds one element along that axis.\n"
                "For example, shape (6,7,4) with zerosum_axes=[-2,-1]:\n"
                "  unconstrained: (6, 6, 3) → after extend axis -2: (6, 7, 3) → after extend axis -1: (6, 7, 4)\n\n"
                "**CRITICAL: ZeroSumNormal logp uses UNCONSTRAINED element count, not constrained!**\n"
                "ZeroSumNormal evaluates Normal(0, sigma) logp for only n_unconstrained elements\n"
                "(not the full constrained array). The last element per sum-to-zero axis is deterministic\n"
                "and does NOT contribute a logp term.\n\n"
                "The logp is computed on the UNCONSTRAINED values directly:\n"
                "```\n"
                "logp = sum over all unconstrained elements x_i of:\n"
                "    -0.5*log(2*pi) - log(sigma) - 0.5*(x_i/sigma)^2\n"
                "```\n"
                "This means the gradient w.r.t. log(sigma) uses N_UNCONSTRAINED, not N_CONSTRAINED:\n"
                "```\n"
                "d(logp)/d(log_sigma) = (-N_unc/sigma + sum(x_i^2)/sigma^3) * sigma\n"
                "                     = -N_unc + sum(x_i^2)/sigma^2\n"
                "```\n"
                "And the gradient w.r.t. unconstrained x_i is simply:\n"
                "```\n"
                "d(logp)/d(x_i) = -x_i/sigma^2\n"
                "```\n"
                "NO ZeroSumTransform backward/gradient needed! The logp is evaluated directly\n"
                "on the unconstrained parameters. The only place you need the backward transform\n"
                "is for computing the OBSERVED LIKELIHOOD (mu = ... + effect[idx] where effect\n"
                "is the constrained version).\n\n"
            )

            # Show specific ZeroSum params
            for p in ctx.params:
                if p.transform == "ZeroSumTransform":
                    n_unc = int(np.prod(p.unc_shape))
                    n_con = int(np.prod(p.shape))
                    parts.append(
                        f"- `{p.name}`: constrained shape {p.shape} ({n_con} elements), "
                        f"unconstrained shape {p.unc_shape} ({n_unc} elements), "
                        f"zerosum_axes={p.zerosum_axes}\n"
                        f"  → logp uses {n_unc} Normal terms (NOT {n_con}!)\n"
                        f"  → gradient w.r.t. sigma uses -({n_unc}/sigma + ...) (NOT -{n_con}/sigma)"
                    )
            parts.append("")

            parts.append(
                "**Computing constrained values for the likelihood:**\n"
                "You still need the constrained (full) values for the observed likelihood.\n"
                "Apply `extend_axis` sequentially for each zerosum axis.\n\n"
                "**Gradient of likelihood through ZeroSumTransform (per axis):**\n"
                "If `dlogp_dy[i]` is the gradient w.r.t. constrained y along one axis (constrained size n),\n"
                "the gradient w.r.t. unconstrained x (size n-1) is:\n"
                "```rust\n"
                "let n: f64 = constrained_size as f64;  // use _f64 suffix or type annotation!\n"
                "let sum_grad: f64 = dlogp_dy[0..n-1].iter().sum();\n"
                "let grad_fill: f64 = dlogp_dy[n-1];\n"
                "dlogp_dx[j] = dlogp_dy[j] - sum_grad / (n.sqrt() + n) - grad_fill / n.sqrt();\n"
                "// for j in 0..n-2\n"
                "```\n"
                "For multi-axis, apply the gradient transform in reverse order of axes.\n"
            )

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
                    label = f"INTEGER INDEX ARRAY (values 0..{n_groups - 1}, {n_groups} groups) — cast to usize for array indexing"
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

        parts.append(
            f"## Optimized PyTensor Graph (logp)\n```\n{ctx.logp_graph}\n```\n"
        )

        parts.append(
            f"## Optimized PyTensor Graph (dlogp/gradient)\n```\n{ctx.dlogp_graph}\n```\n"
        )

        parts.append("## Individual logp terms (optimized, per RV)\n")
        for name, term in ctx.logp_terms.items():
            display = term[:2000] + ("..." if len(term) > 2000 else "")
            parts.append(f"### {name}\n```\n{display}\n```\n")

        parts.append("## Validation")
        parts.append(
            "Your generated code MUST produce these exact values (within float64 precision):\n"
        )

        parts.append(f"At initial point: {json.dumps(ctx.initial_point.point)}")
        parts.append(f"- logp = {ctx.initial_point.logp:.10f}")
        if ctx.initial_point.per_rv_logp:
            parts.append("- logp decomposition per RV (use this to debug each term):")
            for rv_name, rv_logp in ctx.initial_point.per_rv_logp.items():
                parts.append(f"    {rv_name}: {rv_logp:.10f}")
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

        def _flatten(v):
            if isinstance(v, (list, tuple)):
                for item in v:
                    yield from _flatten(item)
            else:
                yield float(v)

        for test_name, vp in all_points:
            position = []
            for name in ctx.param_order:
                val = vp.point[name]
                position.extend(_flatten(val))

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


def _capture_fgraph(fgraph) -> str:
    """Capture the dprint output of an optimized FunctionGraph."""
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    fgraph.dprint()
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
