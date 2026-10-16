import torch
import warp as wp
from ...math import *
from ...util import *
from ...dataTypes import domainData
from ...type_config import scalar_t, index_t, offset_t, neighbor_count_t

# Define kernels that work with flattened float32 arrays
@wp.kernel
def warp_radius_search_kernel_direct_2(
    x: wp.array2d(dtype=scalar_t),         # Query points (N*2,) as 1D float32 array
    y: wp.array2d(dtype=scalar_t),         # Reference points (M*2,) as 1D float32 array
    hx: wp.array(dtype=scalar_t),        # Query radii (N,)
    hy: wp.array(dtype=scalar_t),        # Reference radii (M,)
    min_domain: wp.array(dtype=scalar_t),  # Domain minimum
    max_domain: wp.array(dtype=scalar_t),  # Domain maximum
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

    domainState = domainData()
    domainState.domainMin = min_domain
    domainState.domainMax = max_domain
    domainState.periodicity = periodic
    domainState.dim = D

    if i < wp.len(hx):  # Loop over query points
        count = int(0)

        for j in range(M):
            dist = computeCartesianDistance(
                x[i],
                y[j],
                domainState
            )

            threshold = computePairwiseSupport(hx[i], hy[j], mode)

            # Count valid neighbors
            if dist <= threshold:
                count += 1

        edge_count[i] = count


@wp.kernel
def warp_radius_search_collect_kernel_direct_2(
    x: wp.array2d(dtype=scalar_t),         # Query points (N*2,)
    y: wp.array2d(dtype=scalar_t),         # Reference points (M*2,)
    hx: wp.array(dtype=scalar_t),        # Query radii (N,)
    hy: wp.array(dtype=scalar_t),        # Reference radii (M,)
    min_domain: wp.array(dtype=scalar_t),  # Domain minimum
    max_domain: wp.array(dtype=scalar_t),  # Domain maximum
    periodic: wp.array(dtype=wp.bool),     # Periodicity per dimension
    mode: wp.uint32,                       # 0=gather, 1=scatter, 2=symmetric, 3=superSymmetric
    edge_offsets: wp.array(dtype=offset_t), # Cumulative edge counts (N,)
    edge_i: wp.array(dtype=index_t),      # Output: query point indices
    edge_j: wp.array(dtype=index_t),      # Output: reference point indices
):
    """
    Warp kernel to collect the actual neighbor pairs (GPU-direct version).
    """
    i = wp.tid()
    M = y.shape[0]

    domainState = domainData()
    domainState.domainMin = min_domain
    domainState.domainMax = max_domain
    domainState.periodicity = periodic
    domainState.dim = wp.int32(x.shape[1])

    if i < wp.len(hx):  # Loop over query points
        offset = edge_offsets[i]
        edge_idx = offset

        for j in range(M):
            dist = computeCartesianDistance(
                x[i],
                y[j],
                domainState
            )

            threshold = computePairwiseSupport(hx[i], hy[j], mode)

            # Store valid neighbors
            if dist <= threshold:
                edge_i[edge_idx] = index_t(i)
                edge_j[edge_idx] = index_t(j)
                edge_idx += offset_t(1)


import numpy as np
from ...dataTypes import AdjacencyList
from ...enumTypes import SupportScheme, supportSchemeToUint

def warp_radius_search_small(queryPositions, referencePositions, supportX, supportsY, periodicity, domainDescription, mode:SupportScheme = SupportScheme.Gather) -> AdjacencyList:
    from ..csr import buildCSROffsets, validateParticleCounts, validateEdgeCapacity

    N, M = queryPositions.shape[0], referencePositions.shape[0]
    validateParticleCounts(N, M)
    minD, maxD = domainDescription.min, domainDescription.max
    flags = periodicity.tolist()
    def wrapped(positions):
        return torch.vstack([component if not periodic else
            torch.remainder(component - minD[d], maxD[d] - minD[d]) + minD[d]
            for d, (component, periodic) in enumerate(zip(positions.mT, flags))]).mT
    # Kernel x/hx are queries; y/hy are references, including unequal populations.
    x_warp = castTorchToWarp(wrapped(queryPositions))
    y_warp = castTorchToWarp(wrapped(referencePositions))
    common = [x_warp, y_warp, castTorchToWarp(supportX), castTorchToWarp(supportsY),
              castTorchToWarp(minD), castTorchToWarp(maxD), castTorchToWarp(periodicity),
              wp.uint32(supportSchemeToUint(mode))]
    stream = wp.stream_from_torch(torch.cuda.current_stream(queryPositions.device)) if queryPositions.is_cuda else None
    edge_count_t, edge_count = allocateTorchWarp(N, neighbor_count_t, x_warp.device)
    wp.launch(warp_radius_search_kernel_direct_2, dim=N, inputs=common+[edge_count],
              device=x_warp.device, stream=stream)
    offsets, total_edges = buildCSROffsets(edge_count_t)
    validateEdgeCapacity(total_edges, total_edges)
    i_torch, edge_i = allocateTorchWarp(total_edges, index_t, x_warp.device)
    j_torch, edge_j = allocateTorchWarp(total_edges, index_t, x_warp.device)
    wp.launch(warp_radius_search_collect_kernel_direct_2, dim=N,
              inputs=common+[wp.from_torch(offsets), edge_i, edge_j], device=x_warp.device, stream=stream)
    return AdjacencyList(paddedI=i_torch, paddedJ=j_torch, numEdges=total_edges,
                         numNeighbors=edge_count_t, edgeOffsets=offsets, numRows=N, numCols=M)
