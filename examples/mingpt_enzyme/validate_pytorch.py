"""Validate Enzyme gradients against PyTorch autograd.
Loads weights from data.rs (the same weights the Rust code uses)."""

import re
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def parse_data_rs(path):
    """Parse Rust data.rs to extract weight constants."""
    with open(path) as f:
        content = f.read()

    weights = {}
    # Match: pub const NAME: &[f32] = &[ ... ];
    pattern = r"pub const (\w+): &\[f32\] = &\[([\s\S]*?)\];"
    for m in re.finditer(pattern, content):
        name = m.group(1)
        if name.endswith("_SHAPE"):
            continue
        vals = [float(x.strip().rstrip(",")) for x in m.group(2).split(",") if x.strip()]
        weights[name] = np.array(vals, dtype=np.float32)
    return weights


class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh((2.0 / 3.141592653589793) ** 0.5 * (x + 0.044715 * x.pow(3.0))))


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
    def __init__(self, n_layer=3, n_head=3, n_embd=48, block_size=8, vocab_size=32):
        super().__init__()
        self.n_embd = n_embd
        self.seq_len = block_size
        self.blocks = nn.ModuleList([TransformerBlock(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x):
        x = x.view(1, self.seq_len, self.n_embd)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits.view(-1)


def load_weights_from_data_rs(model, weights):
    """Load parsed data.rs weights into PyTorch model."""
    with torch.no_grad():
        for i in range(3):
            b = model.blocks[i]
            prefix = f"BLOCKS_{i}_"
            b.ln_1.weight.copy_(torch.tensor(weights[f"{prefix}LN_1_WEIGHT"]))
            b.ln_1.bias.copy_(torch.tensor(weights[f"{prefix}LN_1_BIAS"]))
            b.attn.c_attn.weight.copy_(torch.tensor(weights[f"{prefix}ATTN_C_ATTN_WEIGHT"]).reshape(144, 48))
            b.attn.c_attn.bias.copy_(torch.tensor(weights[f"{prefix}ATTN_C_ATTN_BIAS"]))
            b.attn.c_proj.weight.copy_(torch.tensor(weights[f"{prefix}ATTN_C_PROJ_WEIGHT"]).reshape(48, 48))
            b.attn.c_proj.bias.copy_(torch.tensor(weights[f"{prefix}ATTN_C_PROJ_BIAS"]))
            b.ln_2.weight.copy_(torch.tensor(weights[f"{prefix}LN_2_WEIGHT"]))
            b.ln_2.bias.copy_(torch.tensor(weights[f"{prefix}LN_2_BIAS"]))
            b.c_fc.weight.copy_(torch.tensor(weights[f"{prefix}C_FC_WEIGHT"]).reshape(192, 48))
            b.c_fc.bias.copy_(torch.tensor(weights[f"{prefix}C_FC_BIAS"]))
            b.c_proj.weight.copy_(torch.tensor(weights[f"{prefix}C_PROJ_WEIGHT"]).reshape(48, 192))
            b.c_proj.bias.copy_(torch.tensor(weights[f"{prefix}C_PROJ_BIAS"]))

        model.ln_f.weight.copy_(torch.tensor(weights["LN_F_WEIGHT"]))
        model.ln_f.bias.copy_(torch.tensor(weights["LN_F_BIAS"]))
        model.lm_head.weight.copy_(torch.tensor(weights["LM_HEAD_WEIGHT"]).reshape(32, 48))


def main():
    print("Parsing data.rs weights...")
    weights = parse_data_rs("/tmp/mingpt_enzyme/src/data.rs")
    print(f"  Found {len(weights)} weight arrays")

    model = MinGPTCore()
    load_weights_from_data_rs(model, weights)
    model.eval()

    # Same input as Rust: sin(i * 0.01) for i in 0..384
    input_flat = np.array([np.sin(i * 0.01) for i in range(384)], dtype=np.float32)
    x = torch.tensor(input_flat, dtype=torch.float32, requires_grad=True)

    # Forward
    logits = model(x)  # [256] = 8*32

    # Loss on last position, target=5 (same as Rust)
    last_logits = logits[-32:]  # last position's 32 logits
    target = torch.tensor([5], dtype=torch.long)
    loss = F.cross_entropy(last_logits.unsqueeze(0), target)

    print(f"\nPyTorch Loss = {loss.item():.6f}")

    # Backward
    loss.backward()
    grad = x.grad.numpy()

    print(f"Gradient shape: {grad.shape}")
    print("Gradient stats:")
    print(f"  sum     = {grad.sum():.6f}")
    print(f"  abs_sum = {np.abs(grad).sum():.6f}")
    print(f"  max     = {grad.max():.6f}")
    print(f"  min     = {grad.min():.6f}")
    print("First 10 gradients:")
    for i in range(10):
        print(f"  grad[{i}] = {grad[i]:.8f}")

    # Compare with Enzyme results
    enzyme_grads = [
        -0.06038946,
        0.19937083,
        -0.17638184,
        -0.23096114,
        0.14284474,
        0.30951929,
        -0.02494755,
        -0.27090901,
        -0.05807663,
        0.07012614,
    ]

    print("\n--- Enzyme vs PyTorch comparison (first 10) ---")
    print(f"{'idx':>4} {'Enzyme':>14} {'PyTorch':>14} {'abs_diff':>12} {'rel_diff':>12}")
    for i in range(10):
        e = enzyme_grads[i]
        p = grad[i]
        abs_diff = abs(e - p)
        rel_diff = abs_diff / (abs(p) + 1e-10)
        print(f"{i:4d} {e:14.8f} {p:14.8f} {abs_diff:12.2e} {rel_diff:12.2e}")

    # Overall comparison
    all_abs_diff = np.abs(np.array(enzyme_grads[:10]) - grad[:10])
    print(f"\nMax abs diff (first 10): {all_abs_diff.max():.2e}")
    print(f"Mean abs diff (first 10): {all_abs_diff.mean():.2e}")

    # Benchmark PyTorch forward+backward
    target_t = torch.tensor([5], dtype=torch.long)
    # Warmup
    for _ in range(200):
        x2 = torch.tensor(input_flat, dtype=torch.float32, requires_grad=True)
        logits2 = model(x2)
        loss2 = F.cross_entropy(logits2[-32:].unsqueeze(0), target_t)
        loss2.backward()

    n_runs = 10000
    start = time.perf_counter()
    for _ in range(n_runs):
        x2 = torch.tensor(input_flat, dtype=torch.float32, requires_grad=True)
        logits2 = model(x2)
        loss2 = F.cross_entropy(logits2[-32:].unsqueeze(0), target_t)
        loss2.backward()
    elapsed = time.perf_counter() - start
    us_per_call = elapsed / n_runs * 1e6
    print(f"\nPyTorch forward+backward: {us_per_call:.1f} µs/call ({n_runs} runs)")


if __name__ == "__main__":
    main()
