"""Extract everything from a Stan model needed to generate Rust logp code.

Uses BridgeStan to compile the Stan model and compute reference
logp + gradient values for validation.
"""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class StanParamInfo:
    name: str
    unc_name: str
    constrained_size: int
    unconstrained_size: int


@dataclass
class StanValidationPoint:
    point: list[float]
    logp: float
    dlogp: list[float]


@dataclass
class StanModelContext:
    """All information extracted from a Stan model."""

    stan_code: str
    params: list[StanParamInfo]
    param_names: list[str]  # constrained names
    unc_param_names: list[str]  # unconstrained names
    n_params: int  # unconstrained count
    n_params_constrained: int
    data_json: str | None
    data_summary: dict[str, dict]  # name → {shape, dtype, min, max, mean}
    initial_point: StanValidationPoint
    extra_points: list[StanValidationPoint]

    def to_dict(self) -> dict:
        return {
            "stan_code": self.stan_code,
            "parameters": [
                {
                    "name": p.name,
                    "unc_name": p.unc_name,
                    "constrained_size": p.constrained_size,
                    "unconstrained_size": p.unconstrained_size,
                }
                for p in self.params
            ],
            "param_names": self.param_names,
            "unc_param_names": self.unc_param_names,
            "n_params": self.n_params,
            "n_params_constrained": self.n_params_constrained,
            "data_summary": self.data_summary,
            "validation": {
                "initial_point": {
                    "point": self.initial_point.point,
                    "logp": self.initial_point.logp,
                    "dlogp": self.initial_point.dlogp,
                },
                "extra_points": [{"point": p.point, "logp": p.logp, "dlogp": p.dlogp} for p in self.extra_points],
            },
        }

    def save(self, path: str | Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class StanModelExporter:
    """Extract everything from a Stan model needed for Rust code generation."""

    def __init__(
        self,
        stan_code: str,
        data: dict | str | None = None,
        n_extra_points: int = 3,
        seed: int = 123,
    ):
        self.stan_code = stan_code
        self._data = data
        self._n_extra_points = n_extra_points
        self._seed = seed
        self._context: StanModelContext | None = None
        self._bs_model = None

    def _get_bridgestan_model(self):
        """Compile the Stan model and return a BridgeStan StanModel instance."""
        if self._bs_model is not None:
            return self._bs_model

        import bridgestan as bs

        # Write Stan code to a temp file
        self._tmp_dir = tempfile.mkdtemp(prefix="stan_export_")
        stan_file = Path(self._tmp_dir) / "model.stan"
        stan_file.write_text(self.stan_code)

        # Prepare data
        if isinstance(self._data, dict):
            data_arg = self._data
        elif isinstance(self._data, str):
            data_arg = self._data
        else:
            data_arg = None

        self._bs_model = bs.StanModel(
            str(stan_file),
            data=data_arg,
            seed=self._seed,
        )
        return self._bs_model

    @property
    def context(self) -> StanModelContext:
        if self._context is None:
            self._context = self._extract()
        return self._context

    def _extract(self) -> StanModelContext:
        model = self._get_bridgestan_model()

        # Parameter info
        param_names = model.param_names()
        unc_param_names = model.param_unc_names()
        n_constrained = model.param_num()
        n_unconstrained = model.param_unc_num()

        # Build param info list by grouping related parameters
        params = _build_param_info(param_names, unc_param_names)

        # Parse data from Stan code and provided data
        data_summary = self._extract_data_summary()

        # Compute validation points
        rng = np.random.default_rng(self._seed)

        # Initial point at zeros
        x0 = np.zeros(n_unconstrained)
        logp0, grad0 = model.log_density_gradient(x0)
        initial = StanValidationPoint(
            point=x0.tolist(),
            logp=float(logp0),
            dlogp=grad0.tolist(),
        )

        # Extra random points
        extra_points = []
        for _ in range(self._n_extra_points):
            x = (rng.standard_normal(n_unconstrained) * 0.5).tolist()
            logp, grad = model.log_density_gradient(np.array(x))
            extra_points.append(
                StanValidationPoint(
                    point=x,
                    logp=float(logp),
                    dlogp=grad.tolist(),
                )
            )

        return StanModelContext(
            stan_code=self.stan_code,
            params=params,
            param_names=list(param_names),
            unc_param_names=list(unc_param_names),
            n_params=n_unconstrained,
            n_params_constrained=n_constrained,
            data_json=json.dumps(self._data) if isinstance(self._data, dict) else self._data,
            data_summary=data_summary,
            initial_point=initial,
            extra_points=extra_points,
        )

    def _extract_data_summary(self) -> dict[str, dict]:
        """Extract summary of data variables from the Stan data block."""
        summary = {}
        if not isinstance(self._data, dict):
            return summary

        for name, value in self._data.items():
            arr = np.asarray(value)
            info = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "n": int(np.prod(arr.shape)) if arr.shape else 1,
            }
            if np.issubdtype(arr.dtype, np.number) and arr.size > 0:
                info["min"] = float(np.min(arr))
                info["max"] = float(np.max(arr))
                info["mean"] = float(np.mean(arr))
            summary[name] = info

        return summary

    def to_prompt(self) -> str:
        """Generate a complete LLM prompt for Rust code generation from Stan."""
        ctx = self.context
        parts = []

        parts.append("Generate a Rust logp+gradient function for this Stan model.\n")
        parts.append(f"## Stan Model Code\n```stan\n{ctx.stan_code}\n```\n")

        # Parameter layout
        parts.append("## Parameter Layout (unconstrained space)")
        parts.append(f"Total unconstrained parameters: {ctx.n_params}")
        parts.append(f"Total constrained parameters: {ctx.n_params_constrained}\n")

        parts.append("Unconstrained parameter names (in order):")
        for i, name in enumerate(ctx.unc_param_names):
            parts.append(f"- position[{i}] = {name}")

        parts.append("\nConstrained parameter names:")
        for name in ctx.param_names:
            parts.append(f"- {name}")

        # Data
        if ctx.data_summary:
            parts.append("\n## Data (available via `use crate::data::*;`)\n")
            parts.append(
                "All data arrays are pre-generated in `data.rs` with full f64 precision.\n"
                "Do NOT embed data in your code — use the constants from `data.rs`.\n"
            )
            for name, info in ctx.data_summary.items():
                n = info.get("n", "?")
                mn = info.get("min")
                mx = info.get("max")
                mean = info.get("mean")
                shape = info.get("shape", [])
                range_str = f"[{mn:.3f}, {mx:.3f}]" if mn is not None else "unknown"
                mean_str = f"{mean:.3f}" if mean is not None else "unknown"
                parts.append(
                    f"- `{name.upper()}_DATA: &[f64]` — shape={shape}, n={n}, range={range_str}, mean={mean_str}"
                )
                parts.append(f"  `{name.upper()}_N: usize = {n}`")

            # Add scalar int data as usize constants
            if isinstance(self._data, dict):
                for name, value in self._data.items():
                    arr = np.asarray(value)
                    if arr.ndim == 0 and np.issubdtype(arr.dtype, np.integer):
                        parts.append(f"- `{name.upper()}: usize = {int(arr)}`  (scalar integer)")

        parts.append("")

        # Validation
        parts.append("## Validation")
        parts.append("Your generated code MUST produce these exact values (within float64 precision):\n")
        parts.append(
            "NOTE: BridgeStan computes log_density with jacobian=True, propto=True.\n"
            "This means the log density INCLUDES Jacobian adjustments for constrained parameters\n"
            "but may DROP constant terms that don't depend on parameters.\n"
            "Your Rust implementation must match these values exactly.\n"
        )

        parts.append(f"At initial point (all zeros): {ctx.initial_point.point}")
        parts.append(f"- logp = {ctx.initial_point.logp:.10f}")
        parts.append(f"- gradient = {ctx.initial_point.dlogp}\n")

        for i, pt in enumerate(ctx.extra_points):
            parts.append(f"At test point {i + 1}: {pt.point}")
            parts.append(f"- logp = {pt.logp:.10f}")
            parts.append(f"- gradient = {pt.dlogp}\n")

        parts.append("""## Output
Generate ONLY the Rust function body. Include comments mapping each section
to the corresponding Stan block/variable. The function signature is:
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
            tests.append(f"""
    #[test]
    fn test_logp_at_{test_name}() {{
        let mut logp_fn = {struct_name}::default();
        let position = vec!{vp.point!r};
        let mut gradient = vec![0.0f64; {ctx.n_params}];
        let logp = logp_fn.logp(&position, &mut gradient).unwrap();

        assert_close(logp, {vp.logp:.10f}, 1e-6, "logp");""")

            for i, g in enumerate(vp.dlogp):
                tests.append(f'        assert_close(gradient[{i}], {g:.10e}, 1e-4, "grad[{i}]");')
            tests.append("    }\n")

        tests.append("}\n")
        return "\n".join(tests)

    def save_all(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "codegen_prompt.txt").write_text(self.to_prompt())
        self.context.save(output_dir / "codegen_context.json")
        (output_dir / "validation_test.rs").write_text(self.to_rust_tests())


def _build_param_info(
    param_names: list[str],
    unc_param_names: list[str],
) -> list[StanParamInfo]:
    """Build parameter info by grouping related constrained/unconstrained names.

    Stan parameter names follow patterns like:
    - Scalar: "mu" (constrained), "mu" (unconstrained)
    - Vector: "theta.1", "theta.2" (constrained), "theta.1", "theta.2" (unc)
    - Constrained: "sigma" (constrained), "sigma" (unconstrained, log-transformed)
    """

    # Group by base name (strip .N suffixes)
    def base_name(name: str) -> str:
        return re.sub(r"\.\d+$", "", name)

    # Count constrained params per base name
    constrained_counts: dict[str, int] = {}
    for name in param_names:
        bn = base_name(name)
        constrained_counts[bn] = constrained_counts.get(bn, 0) + 1

    # Count unconstrained params per base name
    unconstrained_counts: dict[str, int] = {}
    for name in unc_param_names:
        bn = base_name(name)
        unconstrained_counts[bn] = unconstrained_counts.get(bn, 0) + 1

    # Build param list preserving order
    seen = set()
    params = []
    for name in param_names:
        bn = base_name(name)
        if bn in seen:
            continue
        seen.add(bn)
        params.append(
            StanParamInfo(
                name=bn,
                unc_name=bn,
                constrained_size=constrained_counts.get(bn, 1),
                unconstrained_size=unconstrained_counts.get(bn, 1),
            )
        )

    return params


def export_stan_model(
    stan_code: str,
    data: dict | str | None = None,
    output_dir: str | Path | None = None,
    n_extra_points: int = 3,
) -> StanModelExporter:
    """One-liner to export a Stan model for Rust compilation.

    Usage:
        exporter = export_stan_model(stan_code, data={"N": 10, "y": [...]})
        prompt = exporter.to_prompt()
    """
    exporter = StanModelExporter(stan_code, data=data, n_extra_points=n_extra_points)
    if output_dir:
        exporter.save_all(output_dir)
    return exporter
