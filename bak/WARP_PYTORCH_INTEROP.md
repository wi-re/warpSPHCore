# Smooth Warp-PyTorch Interoperability

## The Problem

When using Warp with PyTorch, the default interop doesn't connect the autograd graphs. This means:
- You must manually create and manage `wp.Tape()` objects
- You must manually call `tape.backward()` 
- Gradients don't flow seamlessly through PyTorch's `.backward()`

## The Solution

Create a custom `torch.autograd.Function` that bridges the two autograd systems. This allows you to:
- Use Warp operations like any PyTorch operation
- Call `.backward()` once and have gradients flow through both systems
- Integrate Warp kernels seamlessly into PyTorch training loops

## Comparison

### ❌ Old (Clunky) Way

```python
# Create tensors
positions = torch.randn(..., requires_grad=True)
areas = torch.ones(..., requires_grad=True)

# Convert to Warp
positions_warp = castTorchToWarp(positions)
areas_warp = castTorchToWarp(areas)
areas_warp.requires_grad = True
densities = wp.zeros(..., requires_grad=True)

# Manual tape management
tape = wp.Tape()
with tape:
    wp.launch(sphDensity_warp, ...)
    deviation = (densities - 1.0)**2.0
    loss = wp.zeros(1, requires_grad=True)
    wp.launch(loss_kernel, ...)

# Manual backward pass
tape.backward(loss)

# Convert back and do PyTorch backward separately
densities_torch = wp.to_torch(densities)
error = torch.sum((1-densities_torch)**2)
error.backward()  # This doesn't see the Warp operations!

# Gradients are disconnected - doesn't work as expected
```

### ✅ New (Smooth) Way

```python
from wp_autograd import warp_op, warp_loss_op

# Create tensors with requires_grad
positions = torch.randn(..., requires_grad=True)
areas = torch.ones(..., requires_grad=True)

# Compute loss using the wrapper
loss = compute_sph_loss_smooth(
    positions, areas, ...,
    sph_kernel, loss_kernel
)

# Standard PyTorch backward - everything flows automatically!
loss.backward()

# Gradients are automatically in positions.grad and areas.grad
optimizer.step()
```

## How It Works

The `torch.autograd.Function` wrapper:

1. **Forward Pass:**
   - Converts PyTorch tensors to Warp arrays
   - Creates a `wp.Tape()` and records operations
   - Saves the tape for backward pass
   - Returns PyTorch tensors

2. **Backward Pass:**
   - Receives gradients from PyTorch (from any downstream operations!)
   - Transfers them to Warp arrays
   - Calls `tape.backward()` to compute Warp gradients
   - Transfers gradients back to PyTorch tensors

**Key insight:** The backward pass receives gradients from whatever PyTorch operations come after. This means you can:
- Compute loss in PyTorch: `loss = torch.sum((densities - 1.0)**2); loss.backward()`
- Use PyTorch loss functions: `loss = F.mse_loss(densities, target); loss.backward()`
- Chain with other PyTorch models: `pred = model(densities); loss = criterion(pred, label); loss.backward()`

All gradients flow backward through the Warp operations automatically!

## Implementation Files

- `wp_autograd.py`: Core autograd function implementations
  - `WarpFunction`: General-purpose wrapper for any Warp operation
  - `WarpLossFunction`: Specialized wrapper for loss computations
  - `warp_op()` and `warp_loss_op()`: Convenience wrappers

- `warp_sph_example.py`: Examples showing usage patterns
  - Drop-in replacement for your current code
  - Training loop integration
  - Multiple usage patterns

## Usage Patterns

### Pattern 1: Warp Physics + PyTorch Loss (RECOMMENDED ✨)
```python
# Wrap only the Warp kernel
densities = warp_op(
    kernel=sph_kernel,
    dim=n_particles,
    device=device_warp,
    output_shape=(n_particles,),
    output_dtype=wp.float32,
    # All inputs (tensors will auto-track gradients)
    positions, supports, areas, ...
)

# Compute loss in PyTorch (simpler and more flexible!)
loss = torch.sum((densities - 1.0) ** 2)
# Or use ANY PyTorch loss: F.mse_loss(densities, target)

loss.backward()  # Gradients flow through Warp AND PyTorch!
```

**Why this is best:**
- ✅ Simpler: wrap only the physics kernel
- ✅ Flexible: use any PyTorch loss function
- ✅ Easy to debug: loss stays in PyTorch
- ✅ Can combine with other PyTorch operations/models

### Pattern 2: Full Loss Computation in Warp (Advanced)
```python
def forward_fn(*inputs):
    # Your Warp computation under tape
    tape = wp.Tape()
    with tape:
        # ... Warp operations ...
    return loss, tape, warp_arrays, input_indices

loss = warp_loss_op(forward_fn, *pytorch_tensors)
loss.backward()  # Gradients flow automatically!
```

**Use this only if:** you need to compute the loss in Warp for performance reasons.

## Benefits

1. **Cleaner Code**: No manual tape management
2. **Seamless Integration**: Works with PyTorch optimizers, training loops, etc.
3. **Automatic Gradients**: Single `.backward()` call handles everything
4. **Type Safety**: PyTorch's autograd checks ensure correct gradient flow
5. **Debuggable**: Can use PyTorch's gradient checking tools

## Migration Guide

Replace this pattern:
```python
tape = wp.Tape()
with tape:
    # ... operations ...
tape.backward(loss)
# Manual gradient handling
```

With this:
```python
loss = warp_loss_op(forward_fn, *inputs)
loss.backward()  # Done!
```
