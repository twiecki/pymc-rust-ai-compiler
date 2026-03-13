"""Extract model information from a PyTorch nn.Module for cross-framework transpilation.

Given a PyTorch module, extracts:
- Parameter names, shapes, and dtypes
- Forward pass reference outputs at test points
- Gradient reference values (via autograd)
- Source code if available
"""

from __future__ import annotations

import inspect
import textwrap
from typing import Any, Callable

import numpy as np

from transpailer.jax_exporter import ModelContext, TensorInfo, ValidationPoint


class PytorchModelExporter:
    """Extract model info from a PyTorch nn.Module.

    The module's forward method should accept input tensors and return output tensors.
    """

    def __init__(
        self,
        module: Any,  # torch.nn.Module
        sample_input: Any,
        input_names: list[str] | None = None,
        source_code: str | None = None,
        n_extra_points: int = 3,
        seed: int = 42,
        loss_fn: Callable | None = None,
    ):
        self.module = module
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
        import torch

        module = self.module

        # Extract parameter info
        param_infos = []
        for name, param in module.named_parameters():
            param_infos.append(
                TensorInfo(
                    name=name,
                    shape=list(param.shape),
                    dtype=str(param.dtype).replace("torch.", ""),
                    size=int(param.numel()),
                )
            )

        # Extract input info
        input_infos = []
        if isinstance(self.sample_input, dict):
            for name, val in self.sample_input.items():
                t = torch.as_tensor(val)
                input_infos.append(
                    TensorInfo(
                        name=name,
                        shape=list(t.shape),
                        dtype=str(t.dtype).replace("torch.", ""),
                        size=int(t.numel()),
                    )
                )
        elif isinstance(self.sample_input, (tuple, list)):
            for i, val in enumerate(self.sample_input):
                t = torch.as_tensor(val)
                name = (
                    self.input_names[i]
                    if self.input_names and i < len(self.input_names)
                    else f"x_{i}"
                )
                input_infos.append(
                    TensorInfo(
                        name=name,
                        shape=list(t.shape),
                        dtype=str(t.dtype).replace("torch.", ""),
                        size=int(t.numel()),
                    )
                )
        else:
            t = torch.as_tensor(self.sample_input)
            name = self.input_names[0] if self.input_names else "x"
            input_infos.append(
                TensorInfo(
                    name=name,
                    shape=list(t.shape),
                    dtype=str(t.dtype).replace("torch.", ""),
                    size=int(t.numel()),
                )
            )

        # Forward pass to get output info
        module.eval()
        with torch.no_grad():
            output = self._forward(module, self.sample_input)
        out_np = output.detach().cpu().numpy()

        output_infos = [
            TensorInfo(
                name="output",
                shape=list(out_np.shape),
                dtype=str(output.dtype).replace("torch.", ""),
                size=int(np.prod(out_np.shape)) if out_np.shape else 1,
            )
        ]

        # Generate validation points
        rng = np.random.default_rng(self._seed)
        validation_points = []

        # Point 0: original params
        vp = self._compute_validation_point(module, self.sample_input)
        validation_points.append(vp)

        # Extra points: perturbed params
        original_state = {k: v.clone() for k, v in module.state_dict().items()}
        for _ in range(self._n_extra_points):
            with torch.no_grad():
                for name, param in module.named_parameters():
                    noise = torch.tensor(
                        rng.standard_normal(param.shape) * 0.1,
                        dtype=param.dtype,
                    )
                    param.copy_(original_state[name] + noise)

            vp = self._compute_validation_point(module, self.sample_input)
            validation_points.append(vp)

        # Restore original params
        module.load_state_dict(original_state)

        source = self._source_code or self._try_extract_source()

        return ModelContext(
            source_framework="pytorch",
            source_code=source,
            params=param_infos,
            inputs=input_infos,
            outputs=output_infos,
            validation_points=validation_points,
        )

    def _forward(self, module, inp):
        import torch

        if isinstance(inp, dict):
            return module(
                **{k: torch.as_tensor(v, dtype=torch.float32) for k, v in inp.items()}
            )
        elif isinstance(inp, (tuple, list)):
            return module(*[torch.as_tensor(v, dtype=torch.float32) for v in inp])
        else:
            return module(torch.as_tensor(inp, dtype=torch.float32))

    def _compute_validation_point(self, module, inp) -> ValidationPoint:

        module.zero_grad()
        module.train()
        output = self._forward(module, inp)

        # Compute loss for gradient
        if self._loss_fn is not None:
            loss = self._loss_fn(output)
        else:
            loss = output.sum()

        loss.backward()

        params_dict = {}
        grad_dict = {}
        for name, param in module.named_parameters():
            params_dict[name] = param.detach().cpu().numpy().tolist()
            if param.grad is not None:
                grad_dict[name] = param.grad.detach().cpu().numpy().tolist()
            else:
                grad_dict[name] = np.zeros_like(param.detach().cpu().numpy()).tolist()

        return ValidationPoint(
            params=params_dict,
            inputs=self._input_to_dict(inp),
            output=output.detach().cpu().numpy().tolist(),
            grad_params=grad_dict,
        )

    def _input_to_dict(self, inp) -> dict:
        import torch

        if isinstance(inp, dict):
            return {
                k: np.asarray(v).tolist()
                if not isinstance(v, torch.Tensor)
                else v.detach().cpu().numpy().tolist()
                for k, v in inp.items()
            }
        elif isinstance(inp, (tuple, list)):
            return {
                f"x_{i}": np.asarray(v).tolist()
                if not isinstance(v, torch.Tensor)
                else v.detach().cpu().numpy().tolist()
                for i, v in enumerate(inp)
            }
        else:
            if isinstance(inp, torch.Tensor):
                return {"x": inp.detach().cpu().numpy().tolist()}
            return {"x": np.asarray(inp).tolist()}

    def _try_extract_source(self) -> str | None:
        try:
            return textwrap.dedent(inspect.getsource(type(self.module)))
        except (OSError, TypeError):
            return None


def export_pytorch_model(
    module: Any,
    sample_input: Any,
    **kwargs,
) -> PytorchModelExporter:
    """One-liner to export a PyTorch model for transpilation."""
    return PytorchModelExporter(module, sample_input, **kwargs)
