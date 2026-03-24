# Skill: PyTorch → Rust (Burn Framework) Model Translation

You are translating a **PyTorch** `nn.Module` to **Rust using the Burn deep learning framework**. Burn provides optimized tensor operations, automatic differentiation, and multiple backends (CPU via NdArray, GPU via WGPU/CUDA). This gives us both high performance AND gradients for free.

## Why Burn over Pure Rust

- **Autodiff**: Burn's `Autodiff<B>` backend decorator gives us reverse-mode AD automatically
- **Optimized matmul**: BLAS-accelerated on CPU, GPU-accelerated with WGPU/CUDA
- **Type-safe tensors**: `Tensor<B, D>` with compile-time dimension checking
- **Still pure Rust**: Compiles to native binary, no Python runtime

## Core Paradigm Shift

| PyTorch | Burn (Rust) |
|---|---|
| `torch.Tensor` | `Tensor<B, D>` (generic over backend + dims) |
| `nn.Module` | `#[derive(Module)] struct` |
| `nn.Linear(in, out)` | `LinearConfig::new(in, out).init(&device)` |
| `nn.LayerNorm(dim)` | `LayerNormConfig::new(dim).init(&device)` |
| `F.softmax(x, dim=-1)` | `activation::softmax(x, dim)` |
| `F.gelu(x)` | `activation::gelu(x)` |
| `x @ y` | `x.matmul(y)` |
| `x.view(B, T, C)` | `x.reshape([b, t, c])` |
| `x.transpose(1, 2)` | `x.swap_dims(1, 2)` |
| `model.parameters()` | Automatic via `#[derive(Module)]` |
| `loss.backward()` | `loss.backward()` → `Gradients` |
| `param.grad` | `grads.get(&tensor)` |

## Architecture

### File Structure

```
src/
├── main.rs          # Validation binary (reads stdin commands)
├── model.rs         # Model definition using Burn modules
├── data.rs          # Parameter constants as flat f32 arrays
└── generated.rs     # Forward function + weight loading + inference
```

### Cargo.toml

```toml
[package]
name = "pytorch-to-rust"
version = "0.1.0"
edition = "2021"

[dependencies]
burn = { version = "0.20", features = ["ndarray", "autodiff"] }

[[bin]]
name = "validate"
path = "src/main.rs"
```

## Tensor API Quick Reference

### Creation

```rust
use burn::prelude::*;
use burn::backend::NdArray;

type B = NdArray<f32>;

// From flat slice (most common for loading weights)
let data = TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [2, 2]);
let tensor: Tensor<B, 2> = Tensor::from_data(data, &device);

// From small constant array
let tensor: Tensor<B, 1> = Tensor::from_floats([1.0, 2.0, 3.0], &device);

// Zeros/ones
let zeros: Tensor<B, 2> = Tensor::zeros([8, 48], &device);
```

### Operations

```rust
// Matmul: [B, T, C] x [C, D] → [B, T, D]
let output = input.matmul(weight.transpose());

// Reshape
let x = x.reshape([batch, seq_len, n_embd]);

// Transpose / swap dims
let k = k.swap_dims(2, 3);  // [B, nh, T, hs] → [B, nh, hs, T]

// Slice
let q = qkv.slice([0..batch, 0..seq, 0..n_embd]);

// Narrow (select range along a dim)
let q = qkv.narrow(2, 0, n_embd);       // first n_embd along dim 2
let k = qkv.narrow(2, n_embd, n_embd);  // next n_embd
let v = qkv.narrow(2, 2*n_embd, n_embd); // last n_embd

// Chunk (split into equal parts)
let parts = qkv.chunk(3, 2);  // split dim 2 into 3 chunks

// Cat (concatenate)
let combined = Tensor::cat(vec![a, b, c], 2);

// Element-wise ops
let scaled = att * (1.0 / (head_dim as f32).sqrt());

// Mask fill
let mask: Tensor<B, 4, Bool> = causal_mask.bool_not(); // True where to mask
let att = att.mask_fill(mask, f32::NEG_INFINITY);

// Softmax
use burn::tensor::activation;
let att = activation::softmax(att, 3); // softmax along last dim

// GELU
let x = activation::gelu(x);

// Sum for loss
let loss = output.sum();
```

### Autodiff (Gradients)

```rust
use burn::backend::{Autodiff, NdArray};

type MyBackend = Autodiff<NdArray<f32>>;

// Forward pass with gradient tracking
let input = Tensor::<MyBackend, 1>::from_data(data, &device);
let output = forward(&input, &weights);
let loss = output.sum();

// Backward pass
let grads = loss.backward();

// Extract gradient for a specific tensor
let grad_w = weights.grad(&grads).unwrap();
let grad_values: Vec<f32> = grad_w.into_data().to_vec().unwrap();
```

## Operation Mapping

### Linear Layer

```python
# PyTorch: nn.Linear(in_features, out_features)
# weight: (out_features, in_features), bias: (out_features,)
# output = input @ weight.T + bias
```

```rust
// Using Burn's Linear module (preferred for modules):
use burn::nn::{Linear, LinearConfig};
let linear = LinearConfig::new(in_features, out_features).init(&device);
let output = linear.forward(input);

// Manual with tensors (for generated code):
fn linear_forward<B: Backend>(
    input: Tensor<B, 2>,     // [batch, in_features]
    weight: Tensor<B, 2>,    // [out_features, in_features]
    bias: Tensor<B, 1>,      // [out_features]
) -> Tensor<B, 2> {
    // PyTorch Linear: output = input @ weight.T + bias
    input.matmul(weight.transpose()).add(bias.unsqueeze())
}
```

### Layer Normalization

```rust
// Manual implementation (when loading custom weights):
fn layer_norm<B: Backend>(
    input: Tensor<B, 3>,     // [batch, seq, dim]
    weight: Tensor<B, 1>,    // [dim]
    bias: Tensor<B, 1>,      // [dim]
    eps: f64,
) -> Tensor<B, 3> {
    let mean = input.clone().mean_dim(2);  // [batch, seq, 1]
    let diff = input - mean;
    let var = diff.clone().powf_scalar(2.0).mean_dim(2);
    let inv_std = (var + eps).sqrt().recip();
    let normalized = diff * inv_std;
    normalized * weight.unsqueeze() + bias.unsqueeze()
}
```

### Causal Self-Attention

```rust
fn causal_self_attention<B: Backend>(
    x: Tensor<B, 3>,               // [1, seq_len, n_embd]
    c_attn_weight: Tensor<B, 2>,    // [3*n_embd, n_embd]
    c_attn_bias: Tensor<B, 1>,      // [3*n_embd]
    c_proj_weight: Tensor<B, 2>,    // [n_embd, n_embd]
    c_proj_bias: Tensor<B, 1>,      // [n_embd]
    n_head: usize,
    n_embd: usize,
) -> Tensor<B, 3> {
    let [b, t, c] = x.dims();
    let head_dim = c / n_head;
    let device = x.device();

    // QKV projection
    let qkv = x.matmul(c_attn_weight.transpose())
        + c_attn_bias.clone().unsqueeze::<2>().unsqueeze();  // [1, T, 3C]

    // Split into Q, K, V
    let q = qkv.clone().narrow(2, 0, n_embd);
    let k = qkv.clone().narrow(2, n_embd, n_embd);
    let v = qkv.narrow(2, 2 * n_embd, n_embd);

    // Reshape to [B, n_head, T, head_dim]
    let q = q.reshape([b, t, n_head, head_dim]).swap_dims(1, 2);
    let k = k.reshape([b, t, n_head, head_dim]).swap_dims(1, 2);
    let v = v.reshape([b, t, n_head, head_dim]).swap_dims(1, 2);

    // Attention scores: [B, nh, T, T]
    let scale = 1.0 / (head_dim as f64).sqrt();
    let att = q.matmul(k.swap_dims(2, 3)) * scale;

    // Causal mask: lower triangular
    let causal_mask = Tensor::<B, 2>::ones([t, t], &device)
        .triu(1)           // upper triangular with diagonal=1
        .bool();           // True above diagonal
    let att = att.mask_fill(causal_mask.unsqueeze::<4>(), f32::NEG_INFINITY);

    // Softmax
    let att = activation::softmax(att, 3);

    // Apply attention to values
    let y = att.matmul(v);  // [B, nh, T, hs]

    // Reassemble heads: [B, T, C]
    let y = y.swap_dims(1, 2).reshape([b, t, c]);

    // Output projection
    y.matmul(c_proj_weight.transpose())
        + c_proj_bias.clone().unsqueeze::<2>().unsqueeze()
}
```

### Transformer Block

```rust
fn transformer_block<B: Backend>(
    x: Tensor<B, 3>,
    // LayerNorm 1
    ln1_weight: Tensor<B, 1>, ln1_bias: Tensor<B, 1>,
    // Attention
    attn_c_attn_w: Tensor<B, 2>, attn_c_attn_b: Tensor<B, 1>,
    attn_c_proj_w: Tensor<B, 2>, attn_c_proj_b: Tensor<B, 1>,
    // LayerNorm 2
    ln2_weight: Tensor<B, 1>, ln2_bias: Tensor<B, 1>,
    // MLP
    mlp_fc_w: Tensor<B, 2>, mlp_fc_b: Tensor<B, 1>,
    mlp_proj_w: Tensor<B, 2>, mlp_proj_b: Tensor<B, 1>,
    n_head: usize, n_embd: usize,
) -> Tensor<B, 3> {
    // Pre-norm attention with residual
    let x_norm = layer_norm(x.clone(), ln1_weight, ln1_bias, 1e-5);
    let attn_out = causal_self_attention(
        x_norm, attn_c_attn_w, attn_c_attn_b,
        attn_c_proj_w, attn_c_proj_b, n_head, n_embd,
    );
    let x = x + attn_out;

    // Pre-norm MLP with residual
    let x_norm = layer_norm(x.clone(), ln2_weight, ln2_bias, 1e-5);
    let hidden = x_norm.matmul(mlp_fc_w.transpose())
        + mlp_fc_b.unsqueeze::<2>().unsqueeze();
    let hidden = activation::gelu(hidden);
    let mlp_out = hidden.matmul(mlp_proj_w.transpose())
        + mlp_proj_b.unsqueeze::<2>().unsqueeze();
    x + mlp_out
}
```

## Loading Weights from data.rs

Parameters are baked into `data.rs` as flat `&[f32]` constants (same as pure-Rust mode).
Load them into Burn tensors at the start of `forward()`:

```rust
use crate::data::*;

fn load_tensor_2d<B: Backend>(
    data: &[f32], rows: usize, cols: usize, device: &B::Device,
) -> Tensor<B, 2> {
    let td = TensorData::new(data.to_vec(), [rows, cols]);
    Tensor::from_data(td, device)
}

fn load_tensor_1d<B: Backend>(
    data: &[f32], len: usize, device: &B::Device,
) -> Tensor<B, 1> {
    let td = TensorData::new(data.to_vec(), [len]);
    Tensor::from_data(td, device)
}
```

## Complete Forward Function Pattern

```rust
use burn::prelude::*;
use burn::tensor::activation;
use burn::backend::NdArray;
use crate::data::*;

type B = NdArray<f32>;

pub fn forward(input: &[f32]) -> Vec<f32> {
    let device = Default::default();

    // Load input
    let x: Tensor<B, 1> = Tensor::from_data(
        TensorData::new(input.to_vec(), [input.len()]),
        &device,
    );
    let x = x.reshape([1, SEQ_LEN, N_EMBD]);

    // Load weights (example for one block)
    let ln1_w = load_tensor_1d::<B>(BLOCKS_0_LN_1_WEIGHT, N_EMBD, &device);
    let ln1_b = load_tensor_1d::<B>(BLOCKS_0_LN_1_BIAS, N_EMBD, &device);
    // ... load all weights ...

    // Run transformer blocks
    let x = transformer_block(x, ln1_w, ln1_b, /* ... */);

    // Final layer norm + lm_head projection
    let x = layer_norm(x, ln_f_w, ln_f_b, 1e-5);
    let logits = x.matmul(lm_head_w.transpose());

    // Flatten output to Vec<f32>
    logits.reshape([SEQ_LEN * VOCAB_SIZE])
        .into_data().to_vec::<f32>().unwrap()
}
```

## Gradient Function Pattern

```rust
use burn::backend::{Autodiff, NdArray};

type AB = Autodiff<NdArray<f32>>;

pub fn forward_with_grad(input: &[f32], param_name: &str) -> (Vec<f32>, Vec<f32>) {
    let device = Default::default();

    // Load input and weights as Autodiff tensors
    let x: Tensor<AB, 1> = Tensor::from_data(
        TensorData::new(input.to_vec(), [input.len()]),
        &device,
    );

    // Load the target parameter with require_grad
    let target_param: Tensor<AB, 2> = load_tensor_2d::<AB>(
        get_param_data(param_name), rows, cols, &device,
    ).require_grad();

    // ... build forward pass using target_param ...
    let output = /* forward computation */;

    // Compute loss = sum(output) for gradient
    let loss = output.clone().sum();

    // Backward pass
    let grads = loss.backward();

    // Extract gradient
    let grad = target_param.grad(&grads).unwrap();

    let output_vec: Vec<f32> = output.into_data().to_vec().unwrap();
    let grad_vec: Vec<f32> = grad.into_data().to_vec().unwrap();

    (output_vec, grad_vec)
}
```

## Important Conventions

1. **Parameters are in `data.rs`** as `pub const NAME: &[f32]` — same as pure-Rust mode
2. **Parameter naming**: PyTorch `fc1.weight` → `FC1_WEIGHT` constant + load into `Tensor<B, 2>`
3. **Weight layout**: PyTorch `nn.Linear` weight is `(out, in)` — use `.transpose()` in matmul
4. **f32 everywhere**: Use `NdArray<f32>` backend
5. **Forward function**: `pub fn forward(input: &[f32]) -> Vec<f32>` — same interface
6. **Gradient function**: `pub fn forward_with_grad(input: &[f32], param_name: &str) -> (Vec<f32>, Vec<f32>)`
7. **Device**: Use `Default::default()` for NdArray (CPU)
8. **No `nn::Module`**: Don't use Burn's module system — load weights manually from data.rs constants

## Performance Tips

1. **Minimize tensor creation**: Load weights once, not per-call if possible
2. **Use `clone()` sparingly**: Burn tensors are reference-counted, but clone when needed for autodiff
3. **Build with `--release`**: Enables BLAS optimizations
4. **Batch operations**: Use tensor ops instead of loops where possible
5. **`.narrow()` over `.slice()`**: More efficient for contiguous sub-tensors

## Common Pitfalls

1. **Unsqueeze for broadcasting**: Burn requires explicit unsqueeze for broadcasting (no auto-broadcast like PyTorch in all cases)
2. **Dimension ordering**: Burn dimension indices are 0-based, same as PyTorch
3. **Causal mask**: Use `triu(1)` to create upper triangular mask, then mask_fill with -inf
4. **Softmax dim**: Burn's `activation::softmax(tensor, dim)` — dim is the dimension index, not negative indexing
5. **Transpose**: Use `.transpose()` for 2D (swaps last two dims) or `.swap_dims(a, b)` for ND
