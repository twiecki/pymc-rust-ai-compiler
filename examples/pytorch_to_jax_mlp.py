"""Example: Transpile a PyTorch MLP to JAX.

This example defines a simple 2-layer MLP in PyTorch and transpiles it to
an equivalent JAX pure function using the AI compiler.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transpailer import transpile_pytorch_to_jax


class SimpleMLP(nn.Module):
    """Simple 2-layer MLP in PyTorch."""

    def __init__(self, in_dim=4, hidden_dim=8, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    # Create model
    torch.manual_seed(42)
    model = SimpleMLP()

    # Sample input
    x = torch.randn(3, 4)

    # Test the original
    with torch.no_grad():
        out = model(x)
    print(f"PyTorch output shape: {out.shape}")
    print(f"PyTorch output:\n{out.numpy()}\n")

    # Transpile to JAX
    result = transpile_pytorch_to_jax(
        module=model,
        sample_input=x.numpy(),
        verbose=True,
    )

    if result.success:
        print("\nTranspilation successful!")
        print(f"  Tool calls: {result.n_tool_calls}")
        print(f"  Tokens: {result.token_usage['total_tokens']}")
        print("\nGenerated JAX code:")
        print(result.generated_code)

        # Test the generated model
        import jax.numpy as jnp

        param_data = {name: param.detach().numpy() for name, param in model.named_parameters()}
        jax_params, forward_fn = result.get_model(param_data)
        jax_out = forward_fn(jax_params, jnp.array(x.numpy()))
        print(f"\nJAX output:\n{np.asarray(jax_out)}")
        print(f"Max diff: {np.max(np.abs(np.asarray(jax_out) - out.numpy())):.2e}")
    else:
        print(f"\nTranspilation failed: {result.validation_errors}")


if __name__ == "__main__":
    main()
