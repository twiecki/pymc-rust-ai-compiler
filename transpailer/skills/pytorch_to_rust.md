# Skill: PyTorch → Rust Model Translation

You are translating a **PyTorch** `nn.Module` to **pure Rust** — no ML framework, no external crates, just `std` and raw f32 math. The goal is zero-overhead inference deployment.

## Core Paradigm Shift

| PyTorch | Rust |
|---|---|
| `nn.Module` with `self.param` | `const` arrays in `data.rs` |
| `torch.Tensor` (dynamic) | `&[f32]` slices (stack) or `Vec<f32>` (heap) |
| Autograd (automatic differentiation) | Manual backpropagation |
| `F.linear(x, w, b)` | Explicit loop-based matmul |
| `model.eval()` / `model.train()` | No concept — always inference |
| GPU/CPU transparent | CPU only (SIMD auto-vectorized) |
| Dynamic shapes | Fixed shapes via constants |

## Operation Mapping

### Linear Layers

```python
# PyTorch: nn.Linear(in_features, out_features)
# weight shape: (out_features, in_features)
# bias shape: (out_features,)
# Computes: output = input @ weight.T + bias
```

```rust
// Rust: manual matmul
// W is stored as (out_features * in_features) row-major in data.rs
// The PyTorch weight is already (out, in), so row i of W gives output[i]
fn linear(input: &[f32], weight: &[f32], bias: &[f32],
          in_features: usize, out_features: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; out_features];
    for i in 0..out_features {
        let mut sum = bias[i];
        for j in 0..in_features {
            sum += input[j] * weight[i * in_features + j];
        }
        output[i] = sum;
    }
    output
}
```

**IMPORTANT**: PyTorch `nn.Linear` weight is `(out_features, in_features)` and computes
`x @ W.T + b`. When flattened to data.rs, row `i` (elements `i*in_features..(i+1)*in_features`)
corresponds to output neuron `i`. So: `output[i] = sum_j(input[j] * weight[i * in + j]) + bias[i]`.

### Activation Functions

```rust
#[inline]
fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn tanh_f32(x: f32) -> f32 {
    x.tanh()
}

#[inline]
fn gelu(x: f32) -> f32 {
    // Approximate GELU (matches PyTorch's default)
    0.5 * x * (1.0 + (0.7978845608_f32 * (x + 0.044715 * x * x * x)).tanh())
}

#[inline]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

#[inline]
fn leaky_relu(x: f32, negative_slope: f32) -> f32 {
    if x > 0.0 { x } else { negative_slope * x }
}

#[inline]
fn elu(x: f32, alpha: f32) -> f32 {
    if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }
}

#[inline]
fn softmax(input: &[f32]) -> Vec<f32> {
    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}
```

### Gradient (Backpropagation) Rules

For `forward_with_grad`, you need to implement manual backprop. Key derivatives:

```rust
// ReLU backward: grad_input = if x > 0 { grad_output } else { 0 }
fn relu_backward(grad_output: &[f32], input: &[f32]) -> Vec<f32> {
    grad_output.iter().zip(input.iter())
        .map(|(&g, &x)| if x > 0.0 { g } else { 0.0 })
        .collect()
}

// Sigmoid backward: grad_input = grad_output * sigmoid(x) * (1 - sigmoid(x))
fn sigmoid_backward(grad_output: &[f32], output: &[f32]) -> Vec<f32> {
    grad_output.iter().zip(output.iter())
        .map(|(&g, &o)| g * o * (1.0 - o))
        .collect()
}

// Tanh backward: grad_input = grad_output * (1 - tanh(x)^2)
fn tanh_backward(grad_output: &[f32], output: &[f32]) -> Vec<f32> {
    grad_output.iter().zip(output.iter())
        .map(|(&g, &o)| g * (1.0 - o * o))
        .collect()
}

// Linear backward:
// grad_input = grad_output @ weight (for propagating to previous layer)
// grad_weight[i][j] = grad_output[i] * input[j]  (outer product)
// grad_bias[i] = grad_output[i]
fn linear_backward(
    grad_output: &[f32],  // shape: (out_features,)
    input: &[f32],        // shape: (in_features,)
    weight: &[f32],       // shape: (out_features * in_features,)
    in_features: usize,
    out_features: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // grad_input: (in_features,)
    let mut grad_input = vec![0.0f32; in_features];
    for j in 0..in_features {
        for i in 0..out_features {
            grad_input[j] += grad_output[i] * weight[i * in_features + j];
        }
    }

    // grad_weight: (out_features * in_features,) — same layout as weight
    let mut grad_weight = vec![0.0f32; out_features * in_features];
    for i in 0..out_features {
        for j in 0..in_features {
            grad_weight[i * in_features + j] = grad_output[i] * input[j];
        }
    }

    // grad_bias: (out_features,)
    let grad_bias = grad_output.to_vec();

    (grad_input, grad_weight, grad_bias)
}
```

### Batch Dimension Handling

PyTorch models often process batched inputs. For validation, we typically use batch_size=1:

```rust
// Input is flattened: for batch_size=1, shape [1, in_features] → &[f32] of len in_features
// Output is flattened: for batch_size=1, shape [1, out_features] → Vec<f32> of len out_features
```

If the model expects a batch dimension, handle it:
```rust
pub fn forward(input: &[f32]) -> Vec<f32> {
    let batch_size = input.len() / IN_FEATURES;
    let mut output = Vec::with_capacity(batch_size * OUT_FEATURES);
    for b in 0..batch_size {
        let x = &input[b * IN_FEATURES..(b + 1) * IN_FEATURES];
        // ... process single sample
        output.extend_from_slice(&sample_output);
    }
    output
}
```

### Common Patterns

#### MLP (Multi-Layer Perceptron)

```rust
use crate::data::*;

#[inline]
fn relu(x: f32) -> f32 { if x > 0.0 { x } else { 0.0 } }

fn linear(input: &[f32], weight: &[f32], bias: &[f32],
          in_f: usize, out_f: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_f];
    for i in 0..out_f {
        let mut sum = bias[i];
        for j in 0..in_f {
            sum += input[j] * weight[i * in_f + j];
        }
        out[i] = sum;
    }
    out
}

pub fn forward(input: &[f32]) -> Vec<f32> {
    // Layer 1: linear + relu
    let h1 = linear(input, FC1_WEIGHT, FC1_BIAS, 4, 8);
    let h1: Vec<f32> = h1.iter().map(|&x| relu(x)).collect();

    // Layer 2: linear (output)
    linear(&h1, FC2_WEIGHT, FC2_BIAS, 8, 2)
}
```

#### Conv2d (2D Convolution)

```rust
// PyTorch: nn.Conv2d(in_channels, out_channels, kernel_size)
// weight shape: (out_channels, in_channels, kH, kW)
fn conv2d(input: &[f32], weight: &[f32], bias: &[f32],
          in_c: usize, out_c: usize, h: usize, w: usize,
          kh: usize, kw: usize, padding: usize) -> Vec<f32> {
    let out_h = h + 2 * padding - kh + 1;
    let out_w = w + 2 * padding - kw + 1;
    let mut output = vec![0.0f32; out_c * out_h * out_w];

    for oc in 0..out_c {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut sum = bias[oc];
                for ic in 0..in_c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let ih = oh + ki;
                            let iw = ow + kj;
                            if ih >= padding && ih < h + padding &&
                               iw >= padding && iw < w + padding {
                                let in_idx = ic * h * w + (ih - padding) * w + (iw - padding);
                                let w_idx = oc * in_c * kh * kw + ic * kh * kw + ki * kw + kj;
                                sum += input[in_idx] * weight[w_idx];
                            }
                        }
                    }
                }
                output[oc * out_h * out_w + oh * out_w + ow] = sum;
            }
        }
    }
    output
}
```

### LayerNorm

```rust
fn layer_norm(input: &[f32], weight: &[f32], bias: &[f32],
              normalized_shape: usize, eps: f32) -> Vec<f32> {
    let mean: f32 = input[..normalized_shape].iter().sum::<f32>() / normalized_shape as f32;
    let var: f32 = input[..normalized_shape].iter()
        .map(|&x| (x - mean) * (x - mean))
        .sum::<f32>() / normalized_shape as f32;
    let inv_std = 1.0 / (var + eps).sqrt();

    input.iter().enumerate().map(|(i, &x)| {
        weight[i % normalized_shape] * (x - mean) * inv_std + bias[i % normalized_shape]
    }).collect()
}
```

## Performance Tips

1. **Use `#[inline]` on activation functions** — they're called millions of times
2. **Pre-allocate vectors**: `vec![0.0f32; size]` instead of pushing
3. **Use iterators**: `iter().zip()` patterns auto-vectorize well
4. **Minimize allocations**: reuse buffers across layers when possible
5. **Loop order for matmul**: iterate output rows first, then columns for cache locality
6. **Build with `--release`**: enables -O3 and auto-vectorization (SIMD)
7. **Avoid branching in hot loops**: use `f32::max(0.0, x)` for ReLU in some cases

## Important Conventions

1. **All parameters are in `data.rs`** as `pub const NAME: &[f32]` — do NOT embed arrays in generated.rs
2. **Parameter names**: PyTorch's dot notation (`fc1.weight`) becomes `FC1_WEIGHT` in Rust constants
3. **Weight layout**: PyTorch `nn.Linear` weight is `(out, in)`, stored row-major when flattened
4. **f32 everywhere**: match PyTorch's default dtype
5. **Shape constants**: `NAME_SHAPE: &[usize]` available for each parameter
6. **Forward function**: `pub fn forward(input: &[f32]) -> Vec<f32>` — flat input, flat output
7. **Gradient function**: `pub fn forward_with_grad(input: &[f32], param_name: &str) -> (Vec<f32>, Vec<f32>)` — returns (output, grad_of_param)
