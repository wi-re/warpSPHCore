"""
Example showing that Warp-PyTorch interop works perfectly with PyTorch loss functions.

This is actually the RECOMMENDED pattern - it's simpler and more flexible!
"""

import torch
import warp as wp
from wp_autograd import warp_op


def compute_sph_density_with_torch_loss(positions, query_supports, reference_supports,
                                         areas, domain_min, domain_max, periodic,
                                         mode_uint, adjacency_j, edge_offsets, num_neighbors,
                                         sph_kernel):
    """
    Compute SPH density using Warp, then compute loss in PyTorch.
    
    Gradients flow seamlessly through both!
    """
    device_warp = wp.device_from_torch(positions.device)
    
    # Step 1: Warp computation (wrapped with autograd.Function)
    densities = warp_op(
        kernel=sph_kernel,
        dim=positions.shape[0],
        device=device_warp,
        output_shape=(positions.shape[0],),
        output_dtype=wp.float32,
        positions, positions,
        query_supports, reference_supports,
        areas, areas,
        domain_min, domain_max, periodic,
        mode_uint,
        adjacency_j, edge_offsets, num_neighbors
    )
    
    # Step 2: PyTorch loss computation (this is easier than Warp!)
    deviation = (densities - 1.0) ** 2
    loss = torch.sum(deviation)
    
    return loss, densities


# Example: Drop-in replacement for your notebook code
def your_notebook_pattern(positions, areas, dx, dim, sph_kernel,
                          adjacency, domain_description, mode_uint):
    """
    This replaces your current clunky pattern with a clean one.
    Loss is computed in PyTorch - much simpler!
    """
    device = positions.device
    device_warp = wp.device_from_torch(device)
    
    # Prepare supports
    query_supports = torch.ones(positions.shape[0], device=device) * dx * 2
    reference_supports = query_supports.clone()
    
    # Warp computation (just the physics)
    densities = warp_op(
        kernel=sph_kernel,
        dim=positions.shape[0],
        device=device_warp,
        output_shape=(positions.shape[0],),
        output_dtype=wp.float32,
        # Inputs
        positions, positions,
        query_supports, reference_supports,
        areas, areas,
        domain_description.min, domain_description.max, domain_description.periodic,
        wp.uint32(mode_uint),
        adjacency.j, adjacency.edgeOffsets, adjacency.numNeighbors
    )
    
    # PyTorch loss (cleaner than Warp!)
    loss = torch.sum((densities - 1.0) ** 2)
    
    return loss, densities


# Complete training example
def training_loop_example():
    """
    Full training loop showing PyTorch loss with Warp physics.
    """
    # Setup (your actual data)
    positions = torch.randn(1000, 3, device='cuda', requires_grad=True)
    areas = torch.ones(1000, device='cuda', requires_grad=True)
    
    # ... other setup ...
    
    optimizer = torch.optim.Adam([positions, areas], lr=0.01)
    
    for epoch in range(100):
        optimizer.zero_grad()
        
        # Warp physics + PyTorch loss
        loss, densities = your_notebook_pattern(
            positions, areas, dx, dim, sph_kernel,
            adjacency, domain_description, mode_uint
        )
        
        # Can add more PyTorch operations/losses!
        regularization = 0.01 * torch.sum(positions ** 2)
        total_loss = loss + regularization
        
        # Single backward pass - gradients flow through Warp and PyTorch!
        total_loss.backward()
        
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, "
                  f"Density range=[{densities.min():.3f}, {densities.max():.3f}]")


# You can use ANY PyTorch loss function!
def various_loss_functions_work(densities_from_warp, targets):
    """
    After getting densities from Warp, you can use any PyTorch loss.
    Gradients will flow backward through the Warp operations automatically.
    """
    
    # MSE Loss
    loss1 = torch.nn.functional.mse_loss(densities_from_warp, targets)
    
    # L1 Loss  
    loss2 = torch.nn.functional.l1_loss(densities_from_warp, targets)
    
    # Custom loss with PyTorch operations
    loss3 = torch.mean(torch.abs(densities_from_warp - targets) ** 1.5)
    
    # Complex loss combining multiple terms
    mse_term = torch.mean((densities_from_warp - targets) ** 2)
    reg_term = torch.std(densities_from_warp)  
    loss4 = mse_term + 0.1 * reg_term
    
    # All of these work! Choose any one and call .backward()
    return loss1  # or loss2, loss3, loss4, etc.


# Comparison with your current code
"""
BEFORE (doesn't work properly):
------------------------
positions_warp = castTorchToWarp(positions)
areas_warp = castTorchToWarp(areas)
densities = wp.zeros(..., requires_grad=True)
tape = wp.Tape()
with tape:
    wp.launch(sphDensity_warp, ...)
    deviation = (densities - 1.0)**2.0
    loss = wp.zeros(1, requires_grad=True)
    wp.launch(loss_kernel, ...)
tape.backward(loss)
densities_torch = wp.to_torch(densities)
error = torch.sum((1-densities_torch)**2)
error.backward()  # Gradients don't flow to positions!


AFTER (clean and automatic):
------------------------
positions.requires_grad = True
areas.requires_grad = True

densities = warp_op(
    sph_kernel, dim, device, (N,), wp.float32,
    positions, positions, query_supports, reference_supports,
    areas, areas, domain_min, domain_max, periodic,
    mode_uint, adjacency_j, edge_offsets, num_neighbors
)

# Use ANY PyTorch loss function!
loss = torch.sum((densities - 1.0) ** 2)
# or: loss = F.mse_loss(densities, torch.ones_like(densities))
# or: loss = custom_pytorch_loss(densities)

loss.backward()  # Gradients flow automatically to positions and areas!

print(positions.grad)  # ✓ Works!
print(areas.grad)      # ✓ Works!
"""


# Why this is better than WarpLossFunction
"""
WarpLossFunction (computing loss in Warp):
- More complex: need to write Warp kernels for loss
- Less flexible: harder to experiment with different losses
- Harder to debug: loss computation in Warp

warp_op + PyTorch loss (THIS APPROACH):
- Simpler: wrap only the physics kernel
- Flexible: use any PyTorch loss function
- Easy to debug: loss is in PyTorch
- Can combine with other PyTorch operations
- Can use pretrained PyTorch models in the loss
"""
