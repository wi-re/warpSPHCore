"""
Example demonstrating smooth PyTorch-Warp interop for SPH density computation.

This shows how to use the WarpFunction autograd wrapper to seamlessly integrate
Warp kernels into PyTorch's computation graph without manually managing tapes.
"""

import torch
import warp as wp
from wp_autograd import warp_op, warp_loss_op
from wp_util import castTorchToWarp


# Example 1: Simple wrapper using warp_op
def compute_sph_density_smooth(positions, query_supports, reference_supports, 
                                 areas, domain_min, domain_max, periodic,
                                 mode_uint, adjacency_j, edge_offsets, num_neighbors,
                                 sph_kernel):
    """
    Compute SPH densities with smooth gradient flow through PyTorch and Warp.
    
    Args:
        positions: torch.Tensor of positions (requires_grad=True if needed)
        ... (other parameters)
        sph_kernel: Warp kernel function for SPH density computation
    
    Returns:
        densities: torch.Tensor with gradients connected to inputs
    """
    device_warp = wp.device_from_torch(positions.device)
    
    # Use the warp_op wrapper - gradients will flow automatically!
    densities = warp_op(
        kernel=sph_kernel,
        dim=positions.shape[0],
        device=device_warp,
        output_shape=(positions.shape[0],),
        output_dtype=wp.float32,
        # Inputs - can mix tensors with requires_grad=True and scalars
        positions, positions,
        query_supports, reference_supports,
        areas, areas,
        domain_min, domain_max, periodic,
        mode_uint,
        adjacency_j, edge_offsets, num_neighbors
    )
    
    return densities


# Example 2: End-to-end loss computation with warp_loss_op
def compute_sph_loss_smooth(positions, query_supports, reference_supports,
                             areas, domain_min, domain_max, periodic,
                             mode_uint, adjacency_j, edge_offsets, num_neighbors,
                             sph_kernel, loss_kernel):
    """
    Compute SPH density loss with fully automatic gradient flow.
    
    This is even cleaner for end-to-end loss computation.
    """
    
    def warp_forward_fn(positions, query_supports, reference_supports, areas,
                        domain_min, domain_max, periodic):
        """
        Inner function that wraps the entire Warp computation.
        This runs under a tape and returns everything needed for backward.
        """
        device_warp = wp.device_from_torch(positions.device)
        
        # Convert inputs to Warp
        positions_warp = wp.from_torch(positions.contiguous(), requires_grad=True)
        query_supports_warp = wp.from_torch(query_supports.contiguous(), requires_grad=True)
        reference_supports_warp = wp.from_torch(reference_supports.contiguous(), requires_grad=True)
        areas_warp = wp.from_torch(areas.contiguous(), requires_grad=True)
        domain_min_warp = wp.from_torch(domain_min.contiguous(), requires_grad=False)
        domain_max_warp = wp.from_torch(domain_max.contiguous(), requires_grad=False)
        periodic_warp = wp.from_torch(periodic.contiguous(), requires_grad=False)
        
        # Create intermediate and output arrays
        densities = wp.zeros(positions.shape[0], dtype=wp.float32, device=device_warp, requires_grad=True)
        deviation = wp.zeros(positions.shape[0], dtype=wp.float32, device=device_warp, requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float32, device=device_warp, requires_grad=True)
        
        # Run computation under tape
        tape = wp.Tape()
        with tape:
            # Density computation
            wp.launch(
                sph_kernel,
                dim=positions.shape[0],
                inputs=[
                    positions_warp, positions_warp,
                    query_supports_warp, reference_supports_warp,
                    areas_warp, areas_warp,
                    domain_min_warp, domain_max_warp, periodic_warp,
                    mode_uint,
                    adjacency_j, edge_offsets, num_neighbors,
                    densities
                ],
                device=device_warp
            )
            
            # Compute deviation
            wp.launch(
                lambda tid, dens, dev: dev[tid] = (dens[tid] - 1.0) ** 2.0,
                dim=densities.shape[0],
                inputs=[densities, deviation],
                device=device_warp
            )
            
            # Compute loss
            wp.launch(
                loss_kernel,
                dim=deviation.shape[0],
                inputs=[deviation, loss],
                device=device_warp
            )
        
        # Return: loss, tape, list of warp arrays, indices of arrays corresponding to inputs
        warp_arrays = [
            positions_warp,           # 0
            query_supports_warp,      # 1
            reference_supports_warp,  # 2
            areas_warp                # 3
        ]
        input_indices = [0, 1, 2, 3]  # Which arrays correspond to inputs
        
        return loss, tape, warp_arrays, input_indices
    
    # Call the loss function - gradients will flow automatically!
    loss = warp_loss_op(
        warp_forward_fn,
        positions, query_supports, reference_supports, areas,
        domain_min, domain_max, periodic
    )
    
    return loss


# Example 3: Usage in training loop
def training_example():
    """
    Example showing how to use the smooth interop in a training loop.
    """
    # Setup data
    positions = torch.randn(1000, 3, device='cuda', requires_grad=True)
    areas = torch.ones(1000, device='cuda', requires_grad=True)
    # ... other parameters ...
    
    optimizer = torch.optim.Adam([positions, areas], lr=0.01)
    
    for epoch in range(100):
        optimizer.zero_grad()
        
        # Compute loss - this is now just like any PyTorch operation!
        loss = compute_sph_loss_smooth(
            positions, query_supports, reference_supports,
            areas, domain_min, domain_max, periodic,
            mode_uint, adjacency_j, edge_offsets, num_neighbors,
            sph_kernel, loss_kernel
        )
        
        # Standard PyTorch backward - Warp gradients flow automatically!
        loss.backward()
        
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")


# Example 4: Simplified pattern for your existing code
def simplified_density_loss(positions, areas, dx, dim, sph_kernel, loss_kernel, 
                            adjacency, domain_description, mode_uint):
    """
    Drop-in replacement for your current clunky implementation.
    
    Now you can just call this and use .backward() normally!
    """
    device = positions.device
    
    # Prepare inputs
    query_supports = torch.ones(positions.shape[0], device=device) * dx * 2
    reference_supports = query_supports.clone()
    
    def forward_fn(pos, areas):
        device_warp = wp.device_from_torch(pos.device)
        
        # Convert to Warp
        pos_warp = wp.from_torch(pos.contiguous(), requires_grad=True)
        areas_warp = wp.from_torch(areas.contiguous(), requires_grad=True)
        qs_warp = wp.from_torch(query_supports.contiguous(), requires_grad=False)
        rs_warp = wp.from_torch(reference_supports.contiguous(), requires_grad=False)
        
        # Create outputs
        densities = wp.zeros(pos.shape[0], dtype=wp.float32, device=device_warp, requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float32, device=device_warp, requires_grad=True)
        
        # Compute under tape
        tape = wp.Tape()
        with tape:
            wp.launch(
                sph_kernel,
                dim=pos.shape[0],
                inputs=[
                    pos_warp, pos_warp,
                    qs_warp, rs_warp,
                    areas_warp, areas_warp,
                    castTorchToWarp(domain_description.min),
                    castTorchToWarp(domain_description.max),
                    castTorchToWarp(domain_description.periodic),
                    wp.uint32(mode_uint),
                    castTorchToWarp(adjacency.j),
                    castTorchToWarp(adjacency.edgeOffsets),
                    castTorchToWarp(adjacency.numNeighbors),
                    densities
                ],
                device=device_warp
            )
            
            # Compute loss
            deviation = (densities - 1.0) ** 2.0
            wp.launch(loss_kernel, dim=deviation.shape[0], inputs=[deviation, loss], device=device_warp)
        
        return loss, tape, [pos_warp, areas_warp], [0, 1]
    
    # This returns a PyTorch tensor with full autograd support
    return warp_loss_op(forward_fn, positions, areas)


# Usage in your notebook:
"""
# Instead of your current clunky code:
# tape = wp.Tape()
# with tape:
#     wp.launch(...)
#     deviation = ...
#     wp.launch(loss_kernel, ...)
# tape.backward(loss)
# densities_torch = wp.to_torch(densities)
# error.backward()

# You can now just do:
positionsTemp = torch.randn(..., requires_grad=True)
areas = torch.ones(..., requires_grad=True)

loss = simplified_density_loss(
    positionsTemp, areas, dx, dim, 
    sphDensity_warp, loss_kernel,
    adjacency, domainDescription, mode_uint
)

# Standard PyTorch backward - everything just works!
loss.backward()

# Gradients are now in positionsTemp.grad and areas.grad automatically
print(positionsTemp.grad)
print(areas.grad)
"""
