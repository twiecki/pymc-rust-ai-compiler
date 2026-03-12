"""Example: Transpile a PyTorch MLP to pure Rust.

This demonstrates the PyTorch → Rust transpiler on a simple 2-layer MLP.
The generated Rust code has zero dependencies — just raw f32 math.
"""

import torch
import torch.nn as nn

from pymc_rust_compiler import transpile_pytorch_to_rust


# Define a simple MLP
class MLP(nn.Module):
    def __init__(self, in_dim=4, hidden=8, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def main():
    # Create model with fixed weights for reproducibility
    torch.manual_seed(42)
    model = MLP()

    # Sample input
    x = torch.randn(1, 4)

    # PyTorch forward pass
    with torch.no_grad():
        pytorch_output = model(x).numpy()
    print(f"PyTorch output: {pytorch_output}")

    # Source code for context
    source = """
class MLP(nn.Module):
    def __init__(self, in_dim=4, hidden=8, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
"""

    # Transpile to Rust
    result = transpile_pytorch_to_rust(
        model,
        sample_input=x.numpy(),
        source_code=source,
        verbose=True,
    )

    if result.success:
        print("\nTranspilation successful!")
        print(f"  Build dir: {result.build_dir}")
        print(f"  Tool calls: {result.n_tool_calls}")
        print(f"  Builds: {result.n_attempts}")
        print(f"  Tokens: {result.token_usage['total_tokens']}")
        print(f"\nGenerated Rust code ({len(result.generated_code)} chars):")
        print(result.generated_code[:500])
        if len(result.generated_code) > 500:
            print("...")
    else:
        print("\nTranspilation failed:")
        for err in result.validation_errors:
            print(f"  - {err}")


if __name__ == "__main__":
    main()
