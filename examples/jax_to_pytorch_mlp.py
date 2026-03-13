"""Example: Transpile a JAX MLP to PyTorch.

This example defines a simple 2-layer MLP in JAX and transpiles it to
an equivalent PyTorch nn.Module using the AI compiler.
"""

import jax
import jax.numpy as jnp
import numpy as np

from transpailer import transpile_jax_to_pytorch


def mlp_forward(params, x):
    """Simple 2-layer MLP in JAX."""
    x = jax.nn.relu(x @ params["w1"] + params["b1"])
    x = x @ params["w2"] + params["b2"]
    return x


def main():
    # Initialize parameters
    rng = np.random.default_rng(42)
    params = {
        "w1": jnp.array(rng.standard_normal((4, 8)).astype(np.float32)),
        "b1": jnp.array(rng.standard_normal(8).astype(np.float32)),
        "w2": jnp.array(rng.standard_normal((8, 2)).astype(np.float32)),
        "b2": jnp.array(rng.standard_normal(2).astype(np.float32)),
    }

    # Sample input
    x = jnp.array(rng.standard_normal((3, 4)).astype(np.float32))

    # Test the original
    out = mlp_forward(params, x)
    print(f"JAX output shape: {out.shape}")
    print(f"JAX output:\n{out}\n")

    # Transpile to PyTorch
    result = transpile_jax_to_pytorch(
        fn=mlp_forward,
        params=params,
        sample_input=x,
        verbose=True,
    )

    if result.success:
        print("\nTranspilation successful!")
        print(f"  Tool calls: {result.n_tool_calls}")
        print(f"  Tokens: {result.token_usage['total_tokens']}")
        print("\nGenerated PyTorch code:")
        print(result.generated_code)

        # Test the generated model
        import torch

        model = result.get_model({k: np.asarray(v) for k, v in params.items()})
        pt_out = model(torch.tensor(np.asarray(x)))
        print(f"\nPyTorch output:\n{pt_out.detach().numpy()}")
        print(
            f"Max diff: {np.max(np.abs(pt_out.detach().numpy() - np.asarray(out))):.2e}"
        )
    else:
        print(f"\nTranspilation failed: {result.validation_errors}")


if __name__ == "__main__":
    main()
