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

# This function filters the Verlet list to produce the actual neighbor list for the current positions and supports, which may have changed since the Verlet list was built. 
def filterVerletList_(
        queryPositions: torch.Tensor, referencePositions: torch.Tensor, 
        querySupports: torch.Tensor, referenceSupports: torch.Tensor,
        domain: DomainDescription, adjacency: AdjacencyList, 
        mode : str = 'symmetric',
):                   
    
    edge_count = torch.zeros_like(adjacency.numNeighbors)
    wp.launch(
        countNeighborsVerlet, 
        dim = queryPositions.shape[0], 
        inputs = [castTorchToWarpAsBuiltins(queryPositions), castTorchToWarpAsBuiltins(referencePositions), 
        castTorchToWarp(querySupports), castTorchToWarp(referenceSupports),
        castTorchToWarp(domain.min), castTorchToWarp(domain.max), castTorchToWarp(domain.periodic),

        supportSchemeToUint(mode),
        castTorchToWarp(adjacency.j), castTorchToWarp(adjacency.edgeOffsets), castTorchToWarp(adjacency.numNeighbors),
        castTorchToWarp(edge_count)]

    )
    warpDevice = castTorchToWarp(queryPositions).device
    # Convert counts to host (only the counts, not the main data)
    edge_count_t = edge_count
    total_edges = torch.sum(edge_count_t).cpu().item()

    N = queryPositions.shape[0]
    # Compute cumulative offsets
    edge_offsets = torch.zeros(N, dtype=torch.int32, device = queryPositions.device)
    edge_offsets[1:] = torch.cumsum(edge_count_t[:-1], dim=0)
    edge_offsets_warp = wp.from_torch(edge_offsets)

    # Allocate output arrays on GPU
    i_torch, edge_i = allocateTorchWarp(total_edges, wp.int64, warpDevice)
    j_torch, edge_j = allocateTorchWarp(total_edges, wp.int64, warpDevice)

    wp.launch(
        updateNeighborsVerlet, 
        dim = queryPositions.shape[0], 
        inputs = [castTorchToWarpAsBuiltins(queryPositions), castTorchToWarpAsBuiltins(referencePositions), 
        castTorchToWarp(querySupports), castTorchToWarp(referenceSupports),
        castTorchToWarp(domain.min), castTorchToWarp(domain.max), castTorchToWarp(domain.periodic),

        supportSchemeToUint(mode),
        castTorchToWarp(adjacency.j), castTorchToWarp(adjacency.edgeOffsets), castTorchToWarp(adjacency.numNeighbors),
        edge_offsets_warp, castTorchToWarp(edge_count),

        edge_i, edge_j
        ]
    )

    return AdjacencyList(
        i = i_torch, j = j_torch,
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
):
    return filterVerletList_(
        queryParticles.positions, referenceParticles.positions if referenceParticles is not None else queryParticles.positions, 
        queryParticles.supports, referenceParticles.supports if referenceParticles is not None else queryParticles.supports,
        domain, adjacency, supportMode
    )