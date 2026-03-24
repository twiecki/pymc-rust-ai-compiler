# Skill: ZeroSumNormal Models

This model uses ZeroSumNormal distributions with ZeroSumTransform. The key challenge
is implementing the forward transform (unconstrained → constrained) and backpropagating
gradients through it correctly.

## CRITICAL: Unconstrained Dimension Sizes

With `n_zerosum_axes`, EACH zero-sum axis reduces its dimension by 1 in unconstrained space.

Examples:
- shape=(6,), n_zerosum_axes=1 → unconstrained size = **5**
- shape=(7,), n_zerosum_axes=1 → unconstrained size = **6**
- shape=(6,7,4), n_zerosum_axes=2 → unconstrained = (6, **6**, **3**) = **108** elements
  (axis -2: 7→6, axis -1: 4→3, axis 0 is NOT a zero-sum axis)
- shape=(6,7,4), n_zerosum_axes=3 → unconstrained = (**5**, **6**, **3**) = **90** elements

The parameter layout in `position[]` uses the UNCONSTRAINED sizes. Always check the
`unc_shape` field in the parameter info to get the correct unconstrained dimensions.
N_PARAMS is the sum of all unconstrained sizes.

## How ZeroSumTransform Works

ZeroSumNormal(sigma, shape=(K,)) has K-1 unconstrained parameters that map to K
constrained values summing to zero.

**Forward transform** (K-1 unconstrained → K constrained):
```
n = K  (full/constrained dimension size)
sum_x = sum of unconstrained elements
norm = sum_x / (sqrt(n) + n)
fill = norm - sum_x / sqrt(n)    // this is the "extra" element

constrained[i] = unconstrained[i] - norm    for i = 0..K-2
constrained[K-1] = fill - norm              // last element
```

The result sums to zero: sum(constrained) = 0.

## Multi-axis ZeroSumTransform

For tensors with `n_zerosum_axes > 1`, the transform is applied sequentially,
starting from the SECOND-TO-LAST zero-sum axis and ending with the LAST.

For shape=(6,7,4) and n_zerosum_axes=2 (axes -2 and -1 are zero-sum):
- Unconstrained shape: (6, 6, 3)

Forward transform steps:
1. First extend axis -2: (6, **6**, 3) → (6, **7**, 3) — apply transform along middle dim
2. Then extend axis -1: (6, 7, **3**) → (6, 7, **4**) — apply transform along last dim

**Concrete example for axis -2 extension** (6→7 for each i,k slice):
```rust
for i in 0..6 {
    for k in 0..3 {
        // Gather the 6 unconstrained values along axis -2
        let mut sum_x = 0.0;
        for j in 0..6 {
            sum_x += unc[i * 6 * 3 + j * 3 + k];
        }
        let n = 7.0;
        let norm = sum_x / (n.sqrt() + n);
        let fill = norm - sum_x / n.sqrt();
        for j in 0..6 {
            temp[i * 7 * 3 + j * 3 + k] = unc[i * 6 * 3 + j * 3 + k] - norm;
        }
        temp[i * 7 * 3 + 6 * 3 + k] = fill - norm;  // j=6 is the new element
    }
}
```

**Then axis -1 extension** (3→4 for each i,j slice):
```rust
for i in 0..6 {
    for j in 0..7 {
        let mut sum_x = 0.0;
        for k in 0..3 {
            sum_x += temp[i * 7 * 3 + j * 3 + k];
        }
        let n = 4.0;
        let norm = sum_x / (n.sqrt() + n);
        let fill = norm - sum_x / n.sqrt();
        for k in 0..3 {
            full[i * 7 * 4 + j * 4 + k] = temp[i * 7 * 3 + j * 3 + k] - norm;
        }
        full[i * 7 * 4 + j * 4 + 3] = fill - norm;
    }
}
```

## Gradient Backpropagation

Reverse the transform order. If forward was axis -2 then axis -1:
- First backprop through axis -1 transform
- Then backprop through axis -2 transform

**Backprop formula** (K constrained gradients → K-1 unconstrained gradients):
```
n = K  (full/constrained dimension size)
sum_grad = sum of constrained gradients for elements 0..K-2
grad_fill = constrained gradient for element K-1

unconstrained_grad[i] = constrained_grad[i] - sum_grad / (sqrt(n) + n) - grad_fill / sqrt(n)
```

**Concrete backprop for axis -1** (4→3 for each i,j):
```rust
let mut grad_temp = [0.0; 6 * 7 * 3];
for i in 0..6 {
    for j in 0..7 {
        let n = 4.0;
        let mut sum_grad = 0.0;
        for k in 0..3 { sum_grad += grad_full[i*7*4 + j*4 + k]; }
        let grad_fill = grad_full[i*7*4 + j*4 + 3];
        for k in 0..3 {
            grad_temp[i*7*3 + j*3 + k] = grad_full[i*7*4 + j*4 + k]
                - sum_grad / (n.sqrt() + n) - grad_fill / n.sqrt();
        }
    }
}
```

**Then backprop for axis -2** (7→6 for each i,k):
```rust
for i in 0..6 {
    for k in 0..3 {
        let n = 7.0;
        let mut sum_grad = 0.0;
        for j in 0..6 { sum_grad += grad_temp[i*7*3 + j*3 + k]; }
        let grad_fill = grad_temp[i*7*3 + 6*3 + k];
        for j in 0..6 {
            gradient[offset + i*6*3 + j*3 + k] += grad_temp[i*7*3 + j*3 + k]
                - sum_grad / (n.sqrt() + n) - grad_fill / n.sqrt();
        }
    }
}
```

## ZeroSumNormal Log-density

The logp uses ONLY the unconstrained elements (no Jacobian — it's baked in):
```
n_unc = number of unconstrained elements (product of unconstrained shape)
logp = n_unc * (-0.5*log(2*PI) - log(sigma)) - 0.5 * sum(x_unc²) / sigma²
```

IMPORTANT: Use `log(2*PI) = 1.8378770664093453` as a constant. Do NOT compute it as
`LN_2 * PI` (that's `ln(2) * π` which is completely wrong!). Define:
```rust
const LN_2PI: f64 = 1.8378770664093453;  // ln(2π)
```

Gradient w.r.t. unconstrained element x_i:
```
d(logp)/d(x_i) = -x_i / sigma²
```

Gradient w.r.t. log(sigma) from ZeroSumNormal term:
```
d(logp)/d(log_sigma) = (-n_unc + sum(x_unc²) / sigma²) * sigma
```

## Pre-allocation for Effect Arrays

For small dimensions (< 64 total elements), use stack-allocated fixed arrays:
```rust
let mut store_effect_full = [0.0; 6];  // NOT vec![0.0; 6]
let mut grad_store_effect = [0.0; 6];
```

For larger tensors, pre-allocate in the struct:
```rust
pub struct GeneratedLogp {
    interaction_temp: Vec<f64>,  // flattened 3D array, reused
    interaction_full: Vec<f64>,
    grad_interaction: Vec<f64>,
}
```

Use flat indexing for multi-dimensional arrays: `arr[i * D2 * D3 + j * D3 + k]`
instead of `Vec<Vec<Vec<f64>>>` (which allocates per row).
