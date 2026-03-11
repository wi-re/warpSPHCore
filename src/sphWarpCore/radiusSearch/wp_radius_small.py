import warp as wp
from ..mathutil.wp_math import *
from ..utils.wp_util import *

# Define kernels that work with flattened float32 arrays
@wp.kernel
def warp_radius_search_kernel_direct_2(
    x: wp.array2d(dtype=wp.float32),         # Query points (N*2,) as 1D float32 array
    y: wp.array2d(dtype=wp.float32),         # Reference points (M*2,) as 1D float32 array
    hx: wp.array(dtype=wp.float32),        # Query radii (N,)
    hy: wp.array(dtype=wp.float32),        # Reference radii (M,)
    min_domain: wp.array(dtype=wp.float32),  # Domain minimum
    max_domain: wp.array(dtype=wp.float32),  # Domain maximum
    periodic: wp.array(dtype=wp.bool),     # Periodicity per dimension
    mode: wp.uint32,                       # 0=gather, 1=scatter, 2=symmetric, 3=superSymmetric
    edge_count: wp.array(dtype=wp.int32),  # Output: number of edges (N,)
):
    """
    Warp kernel for computing radius search neighbors (GPU-direct version).
    Works with 1D flattened float32 arrays.
    """
    M = y.shape[0]
    D = x.shape[1]
    i = wp.tid()
    
    if i < wp.len(hx):  # Loop over query points
        count = int(0)
        
        for j in range(M):
            dist = computeCartesianDistance(
                x[i], 
                y[j], 
                min_domain, max_domain, periodic
            )
            
            # Determine threshold based on mode
            threshold = 0.0
            if mode == 0:  # gather
                threshold = hx[i]
            elif mode == 1:  # scatter
                threshold = hy[j]
            elif mode == 2:  # symmetric
                threshold = (hx[i] + hy[j]) / 2.0
            elif mode == 3:  # superSymmetric
                threshold = wp.max(hx[i], hy[j])
            
            # Count valid neighbors
            if dist <= threshold:
                count += 1
        
        edge_count[i] = count


@wp.kernel
def warp_radius_search_collect_kernel_direct_2(
    x: wp.array2d(dtype=wp.float32),         # Query points (N*2,)
    y: wp.array2d(dtype=wp.float32),         # Reference points (M*2,)
    hx: wp.array(dtype=wp.float32),        # Query radii (N,)
    hy: wp.array(dtype=wp.float32),        # Reference radii (M,)
    min_domain: wp.array(dtype=wp.float32),  # Domain minimum
    max_domain: wp.array(dtype=wp.float32),  # Domain maximum
    periodic: wp.array(dtype=wp.bool),     # Periodicity per dimension
    mode: wp.uint32,                       # 0=gather, 1=scatter, 2=symmetric, 3=superSymmetric
    edge_offsets: wp.array(dtype=wp.int32), # Cumulative edge counts (N,)
    edge_i: wp.array(dtype=wp.int32),      # Output: query point indices
    edge_j: wp.array(dtype=wp.int32),      # Output: reference point indices
):
    """
    Warp kernel to collect the actual neighbor pairs (GPU-direct version).
    """
    i = wp.tid()
    M = y.shape[0]
    
    if i < wp.len(hx):  # Loop over query points
        offset = edge_offsets[i]
        edge_idx = offset
        
        for j in range(M):
            dist = computeCartesianDistance(
                x[i], 
                y[j], 
                min_domain, max_domain, periodic
            )
            
            # Determine threshold based on mode
            threshold = 0.0
            if mode == 0:  # gather
                threshold = hx[i]
            elif mode == 1:  # scatter
                threshold = hy[j]
            elif mode == 2:  # symmetric
                threshold = (hx[i] + hy[j]) / 2.0
            elif mode == 3:  # superSymmetric
                threshold = wp.max(hx[i], hy[j])
            
            # Store valid neighbors
            if dist <= threshold:
                edge_i[edge_idx] = i
                edge_j[edge_idx] = j
                edge_idx += 1


import numpy as np
from .radius_util import AdjacencyList

def warp_radius_search_small(queryPositions, referencePositions, supportX, supportsY, periodicity, domainDescription, mode:str = 'gather'):
    minD = domainDescription.min.cpu()
    maxD = domainDescription.max.cpu()
    
    x = torch.vstack([component if not periodic else torch.remainder(component - minD[i], maxD[i] - minD[i]) + minD[i] for i, (component, periodic) in enumerate(zip(referencePositions.mT, periodicity))]).mT
    y = torch.vstack([component if not periodic else torch.remainder(component - minD[i], maxD[i] - minD[i]) + minD[i] for i, (component, periodic) in enumerate(zip(queryPositions.mT, periodicity))]).mT
    
    x_warp = castTorchToWarp(x)
    y_warp = castTorchToWarp(y)
    hx_warp = castTorchToWarp(supportX)
    hy_warp = castTorchToWarp(supportsY)
    periodic_warp = castTorchToWarp(periodicity)
    min_domain_warp = castTorchToWarp(domainDescription.min)
    max_domain_warp = castTorchToWarp(domainDescription.max)

    # print("\nTesting castTorchToWarp function...")
    # print('x_warp shape:', x_warp.shape, 'dtype:', x_warp.dtype, 'device:', x_warp.device)
    # print('y_warp shape:', y_warp.shape, 'dtype:', y_warp.dtype, 'device:', y_warp.device)
    # print('hx_warp shape:', hx_warp.shape, 'dtype:', hx_warp.dtype, 'device:', hx_warp.device)
    # print('hy_warp shape:', hy_warp.shape, 'dtype:', hy_warp.dtype, 'device:', hy_warp.device)
    # print('periodic_warp shape:', periodic_warp.shape, 'dtype:', periodic_warp.dtype, 'device:', periodic_warp.device)
    # print('min_domain_warp shape:', min_domain_warp.shape, 'dtype:', min_domain_warp.dtype, 'device:', min_domain_warp.device)  


    N = x.shape[0]
    M = y.shape[0]
    D = x.shape[1]

    # mode = 'gather'  # Change as needed: 'gather', 'scatter', 'symmetric', 'superSymmetric'

    mode_map = {'gather': 0, 'scatter': 1, 'symmetric': 2, 'superSymmetric': 3}
    mode_uint = mode_map.get(mode, 0)
        
    edge_count = wp.zeros(N, dtype=wp.int32, device=x_warp.device)  # Allocate on same device as input data

    # import time
    # wp.synchronize()  # Ensure all previous GPU work is done before starting timer
    # startTime = time.time()

    wp.launch(warp_radius_search_kernel_direct_2, dim=N, inputs=[
        x_warp, y_warp, hx_warp, hy_warp, min_domain_warp, max_domain_warp, periodic_warp, wp.uint32(mode_uint), edge_count
    ], device=x_warp.device)  # Ensure kernel runs on the same device as input data
    # wp.synchronize()  # Ensure kernel has finished before measuring time

    # endTime = time.time()
    # print(f"Kernel execution time: {endTime - startTime:.4f} seconds")



    # Convert counts to host (only the counts, not the main data)
    edge_count_np = edge_count.numpy()
    total_edges = int(np.sum(edge_count_np))

    # Compute cumulative offsets
    edge_offsets = np.zeros(N, dtype=np.int32)
    edge_offsets[1:] = np.cumsum(edge_count_np[:-1])
    edge_offsets_warp = wp.from_numpy(edge_offsets, device=x_warp.device)

    # Allocate output arrays on GPU
    edge_i = wp.zeros(total_edges, dtype=wp.int32, device=x_warp.device)
    edge_j = wp.zeros(total_edges, dtype=wp.int32, device=x_warp.device)

    # Second pass: collect edges
    wp.launch(warp_radius_search_collect_kernel_direct_2, dim=N, inputs=[
        x_warp, y_warp, hx_warp, hy_warp, min_domain_warp, max_domain_warp,
        periodic_warp, wp.uint32(mode_uint), edge_offsets_warp, edge_i, edge_j
    ], device=x_warp.device)  # Ensure kernel runs on the same device as input data

    # Convert Warp arrays back to PyTorch tensors using wp.to_torch() for direct GPU access
    i_torch = wp.to_torch(edge_i)
    j_torch = wp.to_torch(edge_j)
    
    return AdjacencyList(
        i=i_torch.to(dtype=torch.int64),  # Ensure dtype is long for indexing
        j=j_torch.to(dtype=torch.int64),
        numNeighbors=wp.to_torch(edge_count).to(dtype=torch.int64),
        edgeOffsets=wp.to_torch(edge_offsets_warp).to(dtype=torch.int64),
        numRows=N,
        numCols=M
    )
    
    