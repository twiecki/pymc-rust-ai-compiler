"""Extract model information from a JAX function for cross-framework transpilation.

Given a JAX function (pure function + params), extracts:
- Parameter shapes and dtypes
- Forward pass reference outputs at test points
- Gradient reference values (via jax.grad)
- Source code if available
"""

from __future__ import annotations

import inspect
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class TensorInfo:
    """Metadata about a single parameter tensor or output."""

    name: str
    shape: list[int]
    dtype: str
    size: int

    @property
    def is_scalar(self) -> bool:
        return self.size == 1


@dataclass
class ValidationPoint:
    """A set of inputs/params and expected outputs + gradients."""

    params: dict[str, list | float]
    inputs: dict[str, list | float]
    output: list | float
    grad_params: dict[str, list | float]


@dataclass
class ModelContext:
    """All information extracted from a JAX model."""

    source_framework: str  # "jax" or "pytorch"
    source_code: str | None
    params: list[TensorInfo]
    inputs: list[TensorInfo]
    outputs: list[TensorInfo]
    validation_points: list[ValidationPoint]
    extra_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_framework": self.source_framework,
            "source_code": self.source_code,
            "parameters": [{"name": p.name, "shape": p.shape, "dtype": p.dtype, "size": p.size} for p in self.params],
            "inputs": [{"name": i.name, "shape": i.shape, "dtype": i.dtype, "size": i.size} for i in self.inputs],
            "outputs": [{"name": o.name, "shape": o.shape, "dtype": o.dtype, "size": o.size} for o in self.outputs],
            "validation_points": [
                {
                    "params": vp.params,
                    "inputs": vp.inputs,
                    "output": vp.output,
                    "grad_params": vp.grad_params,
                }
                for vp in self.validation_points
            ],
            "extra_info": self.extra_info,
        }


class JaxModelExporter:
    """Extract model info from a JAX pure function.

    The function should have signature: f(params, x) -> output
    where params is a dict of {name: jnp.array} and x is the input.
    """

    def __init__(
        self,
        fn: Callable,
        params: dict[str, Any],
        sample_input: Any,
        input_names: list[str] | None = None,
        source_code: str | None = None,
        n_extra_points: int = 3,
        seed: int = 42,
        loss_fn: Callable | None = None,
    ):
        self.fn = fn
        self.params = params
        self.sample_input = sample_input
        self.input_names = input_names
        self._source_code = source_code
        self._n_extra_points = n_extra_points
        self._seed = seed
        self._loss_fn = loss_fn
        self._context: ModelContext | None = None

    @property
    def context(self) -> ModelContext:
        if self._context is None:
            self._context = self._extract()
        return self._context

    def _extract(self) -> ModelContext:
        import jax
        import jax.numpy as jnp

        # Extract parameter info
        param_infos = []
        for name, val in self.params.items():
            arr = np.asarray(val)
            param_infos.append(
                TensorInfo(
                    name=name,
                    shape=list(arr.shape),
                    dtype=str(arr.dtype),
                    size=int(np.prod(arr.shape)) if arr.shape else 1,
                )
            )

        # Extract input info
        input_infos = []
        if isinstance(self.sample_input, dict):
            for name, val in self.sample_input.items():
                arr = np.asarray(val)
                input_infos.append(
                    TensorInfo(
                        name=name,
                        shape=list(arr.shape),
                        dtype=str(arr.dtype),
                        size=int(np.prod(arr.shape)) if arr.shape else 1,
                    )
                )
        else:
            arr = np.asarray(self.sample_input)
            name = self.input_names[0] if self.input_names else "x"
            input_infos.append(
                TensorInfo(
                    name=name,
                    shape=list(arr.shape),
                    dtype=str(arr.dtype),
                    size=int(np.prod(arr.shape)) if arr.shape else 1,
                )
            )

        # Determine the function to differentiate
        if self._loss_fn is not None:
            # User provided a scalar loss: grad w.r.t. params of loss_fn(fn(params, x))
            def scalar_fn(params, x):
                return self._loss_fn(self.fn(params, x))
        else:
            # If output is scalar, use it directly; otherwise sum for gradient
            test_out = self.fn(self.params, self.sample_input)
            test_arr = np.asarray(test_out)
            if test_arr.ndim == 0 or test_arr.size == 1:

                def scalar_fn(params, x):
                    return jnp.sum(self.fn(params, x))
            else:

                def scalar_fn(params, x):
                    return jnp.sum(self.fn(params, x))

        grad_fn = jax.grad(scalar_fn, argnums=0)

        # Extract output info from a forward pass
        output = self.fn(self.params, self.sample_input)
        out_arr = np.asarray(output)
        output_infos = [
            TensorInfo(
                name="output",
                shape=list(out_arr.shape),
                dtype=str(out_arr.dtype),
                size=int(np.prod(out_arr.shape)) if out_arr.shape else 1,
            )
        ]

        # Generate validation points
        rng = np.random.default_rng(self._seed)
        validation_points = []

        # Point 0: original params
        grads = grad_fn(self.params, self.sample_input)
        validation_points.append(
            ValidationPoint(
                params={k: np.asarray(v).tolist() for k, v in self.params.items()},
                inputs=self._input_to_dict(self.sample_input),
                output=out_arr.tolist(),
                grad_params={k: np.asarray(v).tolist() for k, v in grads.items()},
            )
        )

        # Extra points: perturbed params
        for _ in range(self._n_extra_points):
            perturbed = {}
            for name, val in self.params.items():
                arr = np.asarray(val)
                perturbed[name] = jnp.array(arr + rng.standard_normal(arr.shape) * 0.1)

            out = self.fn(perturbed, self.sample_input)
            grads = grad_fn(perturbed, self.sample_input)
            validation_points.append(
                ValidationPoint(
                    params={k: np.asarray(v).tolist() for k, v in perturbed.items()},
                    inputs=self._input_to_dict(self.sample_input),
                    output=np.asarray(out).tolist(),
                    grad_params={k: np.asarray(v).tolist() for k, v in grads.items()},
                )
            )

        source = self._source_code or self._try_extract_source()

        return ModelContext(
            source_framework="jax",
            source_code=source,
            params=param_infos,
            inputs=input_infos,
            outputs=output_infos,
            validation_points=validation_points,
        )

    def _input_to_dict(self, inp: Any) -> dict:
        if isinstance(inp, dict):
            return {k: np.asarray(v).tolist() for k, v in inp.items()}
        return {"x": np.asarray(inp).tolist()}

    def _try_extract_source(self) -> str | None:
        try:
            return textwrap.dedent(inspect.getsource(self.fn))
        except (OSError, TypeError):
            return None


def export_jax_model(
    fn: Callable,
    params: dict[str, Any],
    sample_input: Any,
    **kwargs,
) -> JaxModelExporter:
    """One-liner to export a JAX model for transpilation."""
    return JaxModelExporter(fn, params, sample_input, **kwargs)
