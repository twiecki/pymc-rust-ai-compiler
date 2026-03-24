# Skill: JAX → PyTorch Model Translation

You are translating a **JAX** model (pure function + params dict) to a **PyTorch** `nn.Module`.

## Core Paradigm Shift

| JAX | PyTorch |
|---|---|
| Pure functions, explicit params | Stateful `nn.Module` with `self.param` |
| `params` dict passed explicitly | Parameters registered via `nn.Parameter` |
| `jax.grad(f)` for gradients | `loss.backward()` + `param.grad` |
| `jax.jit(f)` for compilation | `torch.compile(model)` (optional) |
| Immutable arrays | Mutable tensors |
| `jax.vmap` for batching | Manual batching or `torch.vmap` |

## Operation Mapping

### Array Creation & Manipulation

| JAX | PyTorch |
|---|---|
| `jnp.array(x)` | `torch.tensor(x)` |
| `jnp.zeros(shape)` | `torch.zeros(shape)` |
| `jnp.ones(shape)` | `torch.ones(shape)` |
| `jnp.arange(n)` | `torch.arange(n)` |
| `jnp.linspace(a, b, n)` | `torch.linspace(a, b, n)` |
| `jnp.concatenate([a, b])` | `torch.cat([a, b])` |
| `jnp.stack([a, b])` | `torch.stack([a, b])` |
| `jnp.reshape(x, shape)` | `x.reshape(shape)` or `x.view(shape)` |
| `jnp.expand_dims(x, axis)` | `x.unsqueeze(axis)` |
| `jnp.squeeze(x)` | `x.squeeze()` |
| `x.at[i].set(v)` | `x[i] = v` (in-place) or `x.clone(); x[i] = v` |

### Math Operations

| JAX | PyTorch |
|---|---|
| `jnp.dot(a, b)` | `torch.matmul(a, b)` or `a @ b` |
| `jnp.einsum('ij,jk->ik', a, b)` | `torch.einsum('ij,jk->ik', a, b)` |
| `jnp.sum(x, axis=0)` | `x.sum(dim=0)` |
| `jnp.mean(x, axis=0)` | `x.mean(dim=0)` |
| `jnp.max(x, axis=0)` | `x.max(dim=0).values` |
| `jnp.clip(x, a, b)` | `torch.clamp(x, a, b)` |
| `jnp.where(cond, a, b)` | `torch.where(cond, a, b)` |
| `jnp.log(x)` | `torch.log(x)` |
| `jnp.exp(x)` | `torch.exp(x)` |
| `jnp.sqrt(x)` | `torch.sqrt(x)` |
| `jnp.abs(x)` | `torch.abs(x)` |
| `jnp.power(x, n)` | `torch.pow(x, n)` or `x ** n` |

### Activation Functions

| JAX | PyTorch |
|---|---|
| `jax.nn.relu(x)` | `F.relu(x)` or `nn.ReLU()` |
| `jax.nn.gelu(x)` | `F.gelu(x)` or `nn.GELU()` |
| `jax.nn.silu(x)` / `jax.nn.swish(x)` | `F.silu(x)` or `nn.SiLU()` |
| `jax.nn.sigmoid(x)` | `torch.sigmoid(x)` or `nn.Sigmoid()` |
| `jax.nn.softmax(x, axis=-1)` | `F.softmax(x, dim=-1)` |
| `jax.nn.log_softmax(x, axis=-1)` | `F.log_softmax(x, dim=-1)` |
| `jax.nn.tanh(x)` | `torch.tanh(x)` |
| `jax.nn.elu(x)` | `F.elu(x)` |
| `jax.nn.leaky_relu(x)` | `F.leaky_relu(x)` |

### Linear Algebra

| JAX | PyTorch |
|---|---|
| `jnp.linalg.cholesky(A)` | `torch.linalg.cholesky(A)` |
| `jnp.linalg.solve(A, b)` | `torch.linalg.solve(A, b)` |
| `jnp.linalg.inv(A)` | `torch.linalg.inv(A)` |
| `jnp.linalg.det(A)` | `torch.linalg.det(A)` |
| `jnp.linalg.eigh(A)` | `torch.linalg.eigh(A)` |
| `jnp.linalg.svd(A)` | `torch.linalg.svd(A)` |
| `jnp.linalg.norm(x)` | `torch.linalg.norm(x)` |

### Layer Equivalents

| JAX (manual) | PyTorch |
|---|---|
| `x @ w + b` | `nn.Linear(in, out)` or `F.linear(x, w, b)` |
| `jax.lax.conv(...)` | `nn.Conv2d(...)` or `F.conv2d(...)` |
| Manual batch norm | `nn.BatchNorm1d/2d(...)` |
| Manual layer norm | `nn.LayerNorm(...)` |
| Manual dropout (with key) | `nn.Dropout(p)` |
| `jax.nn.one_hot(x, n)` | `F.one_hot(x, n).float()` |

## Common Patterns

### JAX params dict → PyTorch Module

```python
# JAX
params = {
    'linear1': {'weight': w1, 'bias': b1},
    'linear2': {'weight': w2, 'bias': b2},
}

def forward(params, x):
    x = jax.nn.relu(x @ params['linear1']['weight'] + params['linear1']['bias'])
    x = x @ params['linear2']['weight'] + params['linear2']['bias']
    return x
```

```python
# PyTorch
class Model(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.linear1 = nn.Linear(in1, out1)
        self.linear2 = nn.Linear(in2, out2)
        # Initialize from params
        with torch.no_grad():
            self.linear1.weight.copy_(torch.tensor(params['linear1']['weight']).T)
            self.linear1.bias.copy_(torch.tensor(params['linear1']['bias']))
            self.linear2.weight.copy_(torch.tensor(params['linear2']['weight']).T)
            self.linear2.bias.copy_(torch.tensor(params['linear2']['bias']))

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
```

**IMPORTANT**: `nn.Linear` stores weights transposed! If JAX does `x @ W` where W is (in, out),
then PyTorch's `nn.Linear` weight is (out, in). Either:
- Use `nn.Linear` and transpose when loading: `linear.weight = W.T`
- Or use raw `nn.Parameter` and do `x @ self.W + self.b` manually

### Flat params dict → PyTorch

```python
# JAX flat params
params = {'w': array(shape=(3, 4)), 'b': array(shape=(4,))}

# PyTorch: use nn.Parameter directly
class Model(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(np.array(params['w']), dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(np.array(params['b']), dtype=torch.float32))

    def forward(self, x):
        return x @ self.w + self.b
```

### JAX scan → PyTorch loop

```python
# JAX
def step(carry, x):
    h = jax.nn.tanh(carry @ w_h + x @ w_x + b)
    return h, h

final, outputs = jax.lax.scan(step, init_carry, xs)
```

```python
# PyTorch
outputs = []
h = init_carry
for t in range(seq_len):
    h = torch.tanh(h @ self.w_h + xs[t] @ self.w_x + self.b)
    outputs.append(h)
outputs = torch.stack(outputs)
```

## Important Conventions

1. **Dtype**: JAX defaults to float32 on GPU, float64 on CPU. PyTorch defaults to float32.
   Always use `dtype=torch.float32` unless the source uses float64.
2. **Parameter naming**: Preserve JAX param names. Use nested `nn.Module` for nested param dicts.
3. **Weight transposition**: `nn.Linear` weight is `(out_features, in_features)`. If JAX
   does `x @ W` with W shape `(in, out)`, you need to transpose when loading into `nn.Linear`.
4. **Random keys**: JAX uses explicit PRNG keys; PyTorch uses global state. For deterministic
   behavior, use `torch.manual_seed()`.
5. **In-place ops**: JAX arrays are immutable (`x.at[i].set(v)`). PyTorch allows in-place
   ops but be careful with autograd.
