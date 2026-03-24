# Skill: PyTorch → JAX Model Translation

You are translating a **PyTorch** `nn.Module` to a **JAX** pure function + params dict.

## Core Paradigm Shift

| PyTorch | JAX |
|---|---|
| Stateful `nn.Module` | Pure functions, explicit params |
| `self.weight` as `nn.Parameter` | `params['weight']` passed explicitly |
| `loss.backward()` | `jax.grad(loss_fn)(params, x)` |
| `torch.compile()` | `jax.jit(f)` |
| In-place mutation OK | Immutable arrays (`x.at[i].set(v)`) |
| `DataParallel` / `DistributedDataParallel` | `jax.pmap` / `jax.sharding` |

## Operation Mapping

### Array Creation & Manipulation

| PyTorch | JAX |
|---|---|
| `torch.tensor(x)` | `jnp.array(x)` |
| `torch.zeros(shape)` | `jnp.zeros(shape)` |
| `torch.ones(shape)` | `jnp.ones(shape)` |
| `torch.arange(n)` | `jnp.arange(n)` |
| `torch.cat([a, b])` | `jnp.concatenate([a, b])` |
| `torch.stack([a, b])` | `jnp.stack([a, b])` |
| `x.view(shape)` / `x.reshape(shape)` | `jnp.reshape(x, shape)` or `x.reshape(shape)` |
| `x.unsqueeze(dim)` | `jnp.expand_dims(x, axis=dim)` |
| `x.squeeze()` | `jnp.squeeze(x)` |
| `x.permute(dims)` | `jnp.transpose(x, axes=dims)` |
| `x.contiguous()` | Not needed (JAX handles memory layout) |
| `x.clone()` | `jnp.array(x)` (arrays are immutable) |
| `x[i] = v` (in-place) | `x = x.at[i].set(v)` |

### Math Operations

| PyTorch | JAX |
|---|---|
| `torch.matmul(a, b)` / `a @ b` | `jnp.dot(a, b)` or `a @ b` |
| `torch.einsum('ij,jk->ik', a, b)` | `jnp.einsum('ij,jk->ik', a, b)` |
| `x.sum(dim=0)` | `jnp.sum(x, axis=0)` |
| `x.mean(dim=0)` | `jnp.mean(x, axis=0)` |
| `x.max(dim=0).values` | `jnp.max(x, axis=0)` |
| `torch.clamp(x, a, b)` | `jnp.clip(x, a, b)` |
| `torch.where(cond, a, b)` | `jnp.where(cond, a, b)` |
| `torch.log(x)` | `jnp.log(x)` |
| `torch.exp(x)` | `jnp.exp(x)` |

### Activation Functions

| PyTorch | JAX |
|---|---|
| `F.relu(x)` / `nn.ReLU()` | `jax.nn.relu(x)` |
| `F.gelu(x)` | `jax.nn.gelu(x)` |
| `F.silu(x)` | `jax.nn.silu(x)` |
| `torch.sigmoid(x)` | `jax.nn.sigmoid(x)` |
| `F.softmax(x, dim=-1)` | `jax.nn.softmax(x, axis=-1)` |
| `F.log_softmax(x, dim=-1)` | `jax.nn.log_softmax(x, axis=-1)` |
| `torch.tanh(x)` | `jnp.tanh(x)` |
| `F.elu(x)` | `jax.nn.elu(x)` |
| `F.leaky_relu(x)` | `jax.nn.leaky_relu(x)` |
| `F.dropout(x, p, training)` | Manual with `jax.random.bernoulli` |

### Linear Algebra

| PyTorch | JAX |
|---|---|
| `torch.linalg.cholesky(A)` | `jnp.linalg.cholesky(A)` |
| `torch.linalg.solve(A, b)` | `jnp.linalg.solve(A, b)` |
| `torch.linalg.inv(A)` | `jnp.linalg.inv(A)` |
| `torch.linalg.det(A)` | `jnp.linalg.det(A)` |
| `torch.linalg.eigh(A)` | `jnp.linalg.eigh(A)` |
| `torch.linalg.norm(x)` | `jnp.linalg.norm(x)` |

## Common Patterns

### nn.Module → Pure function

```python
# PyTorch
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

```python
# JAX
def init_params(param_data):
    # param_data comes from PyTorch's named_parameters()
    # nn.Linear weight is (out, in), but JAX convention is (in, out)
    return {
        'fc1.weight': jnp.array(param_data['fc1.weight']),
        'fc1.bias': jnp.array(param_data['fc1.bias']),
        'fc2.weight': jnp.array(param_data['fc2.weight']),
        'fc2.bias': jnp.array(param_data['fc2.bias']),
    }

def forward(params, x):
    # nn.Linear computes x @ W.T + b, so we transpose here
    x = jax.nn.relu(x @ params['fc1.weight'].T + params['fc1.bias'])
    x = x @ params['fc2.weight'].T + params['fc2.bias']
    return x
```

**IMPORTANT**: PyTorch's `nn.Linear` stores weight as `(out_features, in_features)`.
The computation is `F.linear(x, w, b) = x @ w.T + b`. In JAX, replicate this exactly:
`x @ params['weight'].T + params['bias']`.

### nn.Sequential → function chain

```python
# PyTorch
model = nn.Sequential(
    nn.Linear(10, 32), nn.ReLU(),
    nn.Linear(32, 16), nn.ReLU(),
    nn.Linear(16, 1),
)
```

```python
# JAX: explicit function chain
def forward(params, x):
    x = jax.nn.relu(x @ params['0.weight'].T + params['0.bias'])
    x = jax.nn.relu(x @ params['2.weight'].T + params['2.bias'])
    x = x @ params['4.weight'].T + params['4.bias']
    return x
```

### BatchNorm → manual stats

```python
# PyTorch
self.bn = nn.BatchNorm1d(features)
```

```python
# JAX (inference mode — use running stats)
def batch_norm(x, weight, bias, running_mean, running_var, eps=1e-5):
    x_norm = (x - running_mean) / jnp.sqrt(running_var + eps)
    return weight * x_norm + bias
```

### RNN/LSTM patterns

```python
# PyTorch
self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
```

```python
# JAX: manual LSTM step
def lstm_step(carry, x_t, params):
    h, c = carry
    gates = x_t @ params['weight_ih'].T + h @ params['weight_hh'].T + params['bias']
    i, f, g, o = jnp.split(gates, 4, axis=-1)
    i, f, o = jax.nn.sigmoid(i), jax.nn.sigmoid(f), jax.nn.sigmoid(o)
    g = jnp.tanh(g)
    c = f * c + i * g
    h = o * jnp.tanh(c)
    return (h, c), h

# Use jax.lax.scan for the sequence
(final_h, final_c), outputs = jax.lax.scan(
    lambda carry, x: lstm_step(carry, x, params), init_carry, xs
)
```

## Important Conventions

1. **Weight transposition**: PyTorch `nn.Linear` weight is `(out, in)`. In JAX, do
   `x @ W.T + b` to match `F.linear(x, W, b)`.
2. **Parameter names**: PyTorch uses dot notation: `layer1.weight`, `layer1.bias`.
   Preserve these names in the params dict.
3. **Batch dimension**: PyTorch usually expects batch-first. JAX is flexible.
4. **Dropout**: JAX needs explicit PRNG keys. For validation (no dropout), this doesn't matter.
5. **Eval vs train**: For translation, focus on eval/inference mode. Skip dropout, use
   running stats for batch norm.
6. **Dtype**: Use `jnp.float32` to match PyTorch's default.
7. **Device**: JAX auto-places on available accelerator. No `.to(device)` needed.
