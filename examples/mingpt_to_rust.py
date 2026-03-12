"""Transpile Karpathy's minGPT to Rust using bayes-ai-compiler.

This script:
1. Creates a small GPT model (gpt-nano: 3 layers, 3 heads, 48 embd)
2. Wraps the transformer core (skipping embedding lookup) for float input
3. Transpiles to pure Rust via the agentic Claude loop
4. Benchmarks PyTorch vs Rust inference speed
"""

import sys
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

# ── Inline minGPT model (from karpathy/minGPT, simplified for inference) ──


class NewGELU(nn.Module):
    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    (2.0 / 3.141592653589793) ** 0.5 * (x + 0.044715 * x.pow(3.0))
                )
            )
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.act = NewGELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.c_proj(self.act(self.c_fc(self.ln_2(x))))
        return x


class MinGPTCore(nn.Module):
    """The transformer core of minGPT — takes pre-embedded float input.

    Skips embedding lookup (which just indexes a table) and focuses on
    the compute-heavy transformer blocks + final projection.

    Input:  flat f32 array of shape [seq_len * n_embd]
    Output: flat f32 array of shape [seq_len * vocab_size]
    """

    def __init__(self, n_layer, n_head, n_embd, block_size, vocab_size):
        super().__init__()
        self.n_embd = n_embd
        self.seq_len = block_size  # fixed sequence length for transpilation
        self.blocks = nn.ModuleList(
            [TransformerBlock(n_embd, n_head, block_size) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x):
        # x is flat: [seq_len * n_embd] → reshape to [1, seq_len, n_embd]
        x = x.view(1, self.seq_len, self.n_embd)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [1, seq_len, vocab_size]
        return logits.view(-1)  # flatten to [seq_len * vocab_size]


def create_mingpt_nano():
    """Create a tiny GPT model for transpilation testing.

    gpt-nano config: 3 layers, 3 heads, 48 embedding dim
    With vocab_size=32 and block_size=8 → very small but structurally complete.
    """
    n_layer = 3
    n_head = 3
    n_embd = 48
    block_size = 8  # sequence length
    vocab_size = 32

    model = MinGPTCore(n_layer, n_head, n_embd, block_size, vocab_size)
    model.eval()

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MinGPT-nano: {n_layer} layers, {n_head} heads, {n_embd} embd")
    print(f"  vocab_size={vocab_size}, block_size={block_size}")
    print(f"  Parameters: {n_params:,} ({n_params / 1e3:.1f}K)")

    sample_input = torch.randn(block_size * n_embd)
    return model, sample_input


def benchmark_pytorch(model, sample_input, n_warmup=100, n_runs=10000):
    """Benchmark PyTorch inference."""
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            model(sample_input)

        # Timed runs
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(sample_input)
        elapsed = time.perf_counter() - t0

    us_per_call = (elapsed / n_runs) * 1e6
    return us_per_call


def benchmark_rust(binary_path, sample_input, n_warmup=100, n_runs=10000):
    """Benchmark Rust inference via subprocess (measures binary execution)."""
    import subprocess

    flat_input = sample_input.detach().numpy().ravel()
    input_str = ",".join(f"{v:.9e}" for v in flat_input)

    # Build a batch of commands for efficiency
    # Single invocation with many forward calls
    commands = []
    for _ in range(n_warmup + n_runs):
        commands.append(f"forward {input_str}")
    batch = "\n".join(commands) + "\n"

    t0 = time.perf_counter()
    result = subprocess.run(
        [str(binary_path)],
        input=batch,
        capture_output=True,
        text=True,
        timeout=300,
    )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"Rust benchmark error: {result.stderr[:500]}")
        return None

    lines = [line for line in result.stdout.strip().split("\n") if line.strip()]
    # Skip warmup lines
    lines[n_warmup:]

    # The total elapsed includes warmup, so estimate per-call from total runs
    us_per_call = (elapsed / (n_warmup + n_runs)) * 1e6
    return us_per_call


def main():
    print("=" * 60)
    print("minGPT → Rust Transpilation Experiment")
    print("=" * 60)
    print()

    # Step 1: Create model
    print("Step 1: Creating MinGPT-nano model...")
    model, sample_input = create_mingpt_nano()

    # Quick PyTorch forward check
    with torch.no_grad():
        out = model(sample_input)
    print(f"  PyTorch output shape: {out.shape}")
    print(f"  PyTorch output sample: mean={out.mean():.4f}, std={out.std():.4f}")
    print()

    # Step 2: Benchmark PyTorch
    print("Step 2: Benchmarking PyTorch inference...")
    pytorch_us = benchmark_pytorch(model, sample_input, n_warmup=200, n_runs=5000)
    print(f"  PyTorch: {pytorch_us:.1f} µs/call")
    print()

    # Step 3: Transpile to Rust
    print("Step 3: Transpiling to Rust via bayes-ai-compiler...")
    print("  (This will run an agentic Claude loop — may take a few minutes)")
    print()

    # Read source code for better context
    source_code = """
class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            (2.0 / 3.141592653589793) ** 0.5 * (x + 0.044715 * x.pow(3.0))
        ))

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.act = NewGELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.c_proj(self.act(self.c_fc(self.ln_2(x))))
        return x

class MinGPTCore(nn.Module):
    '''Transformer core: input flat [seq_len * n_embd] → output flat [seq_len * vocab_size].
    n_layer=3, n_head=3, n_embd=48, seq_len=8, vocab_size=32.

    The input is pre-embedded (token + position embeddings already applied).
    The model reshapes input to [1, 8, 48], runs 3 transformer blocks with
    causal self-attention, then layer norm + linear projection to vocab logits.

    IMPORTANT for Rust implementation:
    - The causal attention mask is a lower-triangular 8x8 matrix
    - Softmax with -inf masking for future positions
    - Multi-head attention: 3 heads, head_dim=16
    - Layer norm with learned weight and bias
    - GELU activation in the MLP
    '''

    def __init__(self, n_layer, n_head, n_embd, block_size, vocab_size):
        super().__init__()
        self.n_embd = n_embd
        self.seq_len = block_size
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, block_size) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x):
        x = x.view(1, self.seq_len, self.n_embd)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits.view(-1)
"""

    from pymc_rust_compiler import transpile_pytorch_to_rust

    # Use "burn" backend for optimized tensors + autodiff, or "pure" for zero-dep
    backend = sys.argv[1] if len(sys.argv) > 1 else "burn"
    print(f"  Using backend: {backend}")

    result = transpile_pytorch_to_rust(
        model,
        sample_input,
        max_turns=40,  # transformers are complex, give more turns
        model_name="claude-sonnet-4-20250514",
        verbose=True,
        source_code=source_code,
        backend=backend,
    )

    print()
    print(f"  Success: {result.success}")
    print(f"  Tool calls: {result.n_tool_calls}")
    print(f"  Build attempts: {result.n_attempts}")
    print(f"  Tokens: {result.token_usage}")

    if result.validation_errors:
        print(f"  Errors: {result.validation_errors}")

    if result.success:
        # Save the generated Rust code
        result.save("mingpt_generated.rs")
        print("\n  Generated Rust code saved to mingpt_generated.rs")
        print(f"  Build dir: {result.build_dir}")

        # Step 4: Benchmark Rust
        print()
        print("Step 4: Benchmarking Rust inference...")
        binary = result.binary_path
        if binary and binary.exists():
            rust_us = benchmark_rust(binary, sample_input, n_warmup=200, n_runs=5000)
            if rust_us:
                print(f"  Rust: {rust_us:.1f} µs/call")
                print()
                print("=" * 60)
                print("RESULTS")
                print("=" * 60)
                print(f"  PyTorch:  {pytorch_us:.1f} µs/call")
                print(f"  Rust:     {rust_us:.1f} µs/call")
                speedup = pytorch_us / rust_us
                print(
                    f"  Speedup:  {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}"
                )
            else:
                print("  Rust benchmark failed!")
        else:
            print("  No binary found for benchmarking")
    else:
        print("\n  Transpilation failed — see errors above.")
        print("  The generated code (possibly incomplete) is at:")
        print(f"  {result.build_dir}/src/generated.rs")


if __name__ == "__main__":
    main()
