import warp as wp
import torch 
from ...type_config import *
from typing import NamedTuple, Union, Tuple, List, Optional, Any
from warp.types import vector, matrix
from ...math import *
from ...util import *

# Convert Warp arrays back to PyTorch tensors using wp.to_torch() for direct GPU access
from ...dataTypes import *
from ...enumTypes import *
from ...util import *
from .wp_countVerlet import countNeighborsVerlet
from .wp_updateVerlet import updateNeighborsVerlet
from ..csr import buildCSROffsets, validateParticleCounts, validateEdgeCapacity

# This function filters the Verlet list to produce the actual neighbor list for the current positions and supports, which may have changed since the Verlet list was built. 
def filterVerletList_(
        queryPositions: torch.Tensor, referencePositions: torch.Tensor, 
        querySupports: torch.Tensor, referenceSupports: torch.Tensor,
        domain: DomainDescription, adjacency: AdjacencyList, 
        mode : str = 'symmetric',
        capacity: Optional[CapacityPolicy] = DEFAULT_CAPACITY,
):                   
    
    validateParticleCounts(queryPositions.shape[0], referencePositions.shape[0])
    _torchStream = wp.stream_from_torch(torch.cuda.current_stream(queryPositions.device)) if queryPositions.is_cuda else None
    edge_count = torch.zeros_like(adjacency.numNeighbors)
    wp.launch(
        countNeighborsVerlet, 
        dim = queryPositions.shape[0], 
        inputs = [castTorchToWarpAsBuiltins(queryPositions), castTorchToWarpAsBuiltins(referencePositions), 
        castTorchToWarp(querySupports), castTorchToWarp(referenceSupports),
        castTorchToWarp(domain.min), castTorchToWarp(domain.max), castTorchToWarp(domain.periodic),

        supportSchemeToUint(mode),
        castTorchToWarp(adjacency.paddedJ), castTorchToWarp(adjacency.edgeOffsets), castTorchToWarp(adjacency.numNeighbors),
        castTorchToWarp(edge_count)], device=str(queryPositions.device), stream=_torchStream

    )
    warpDevice = castTorchToWarp(queryPositions).device
    # Convert counts to host (only the counts, not the main data)
    edge_count_t = edge_count
    edge_offsets, total_edges = buildCSROffsets(edge_count_t)

    N = queryPositions.shape[0]
    # Compute cumulative offsets
    edge_offsets_warp = wp.from_torch(edge_offsets)

    # Allocate output arrays on GPU
    numEdges = int(total_edges)
    capacityEdges = numEdges if capacity is None else capacity.capacityFor(numEdges)
    validateEdgeCapacity(numEdges, capacityEdges)
    i_torch, edge_i = allocateTorchWarp(capacityEdges, index_t, warpDevice)
    j_torch, edge_j = allocateTorchWarp(capacityEdges, index_t, warpDevice)
    if capacityEdges > numEdges:
        i_torch.fill_(-1)
        j_torch.fill_(-1)

    wp.launch(
        updateNeighborsVerlet, 
        dim = queryPositions.shape[0], 
        inputs = [castTorchToWarpAsBuiltins(queryPositions), castTorchToWarpAsBuiltins(referencePositions), 
        castTorchToWarp(querySupports), castTorchToWarp(referenceSupports),
        castTorchToWarp(domain.min), castTorchToWarp(domain.max), castTorchToWarp(domain.periodic),

        supportSchemeToUint(mode),
        castTorchToWarp(adjacency.paddedJ), castTorchToWarp(adjacency.edgeOffsets), castTorchToWarp(adjacency.numNeighbors),
        edge_offsets_warp, castTorchToWarp(edge_count),

        edge_i, edge_j
        ], device=str(queryPositions.device), stream=_torchStream
    )

    return AdjacencyList(
        paddedI = i_torch, paddedJ = j_torch, numEdges = numEdges,
        numNeighbors=edge_count_t, edgeOffsets=edge_offsets,
        numRows=queryPositions.shape[0], numCols=referencePositions.shape[0],
        queryPositions = queryPositions, referencePositions = referencePositions,
        querySupports = querySupports, referenceSupports = referenceSupports
    )

def filterVerletList(
        queryParticles: ParticleState, 
        domain: DomainDescription, 
        adjacency: AdjacencyList, 
        supportMode: SupportScheme = SupportScheme.Gather,
        referenceParticles: Optional[ParticleState] = None, 
        capacity: Optional[CapacityPolicy] = DEFAULT_CAPACITY,
):
    return filterVerletList_(
        queryParticles.positions, referenceParticles.positions if referenceParticles is not None else queryParticles.positions, 
        queryParticles.supports, referenceParticles.supports if referenceParticles is not None else queryParticles.supports,
        domain, adjacency, supportMode, capacity
    )
