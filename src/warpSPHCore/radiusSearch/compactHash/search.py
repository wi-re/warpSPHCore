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

from ...profiling import record_function
from .grid import *
from .wp_countNeighbors import radiusSearchCountNeighborsCompactHashMap
from .wp_collectNeighbors import radiusSearchCollectCompactHashMap

def radiusSearchOnCompactHashMap(
    datastructure: CompactHashMap,
    queryPositions: torch.Tensor,
    referencePositions: torch.Tensor,
    querySupports: torch.Tensor,
    referenceSupports: torch.Tensor,
    periodicity: torch.Tensor,
    domainDescription: DomainDescription,
    mode: SupportScheme = SupportScheme.Gather,
    hashMapLength: int = 4096,
):                
    mode_uint = datastructure.mode_uint
    minDomain = domainDescription.min if domainDescription.min is not None else None
    maxDomain = domainDescription.max if domainDescription.max is not None else None
    hMax = computeGridSupport(querySupports, referenceSupports, mode)
    minD, maxD = getDomainExtents(referencePositions, minDomain, maxDomain)
    # periodicity = domainDescription.periodicity if domainDescription.periodicity is not None else [False] * referencePositions.shape[1]

    x = torch.vstack([component if not periodic else torch.remainder(component - minD[i], maxD[i] - minD[i]) + minD[i] for i, (component, periodic) in enumerate(zip(referencePositions.mT, periodicity))]).mT
    y = torch.vstack([component if not periodic else torch.remainder(component - minD[i], maxD[i] - minD[i]) + minD[i] for i, (component, periodic) in enumerate(zip(queryPositions.mT, periodicity))]).mT
    
    sortedPositions = datastructure.sortedPositions
    sortedSupports = datastructure.sortedSupports
    sortIndex = datastructure.sortIndex
    hashTable = datastructure.hashTable
    sortedCellTable = datastructure.sortedCellTable
    qMin = datastructure.qMin
    qMax = datastructure.qMax
    hCell = datastructure.hCell
    numCells = datastructure.numCells
    D = datastructure.D
    offsets = datastructure.cellOffsets

    warpDevice = castTorchToWarp(queryPositions).device

    with record_function("neighborSearch - count neighbors"):
        N = queryPositions.shape[0]
        M = sortedPositions.shape[0]
        edge_count_t, edge_count = allocateTorchWarp(queryPositions.shape[0], wp.int32, warpDevice)
        wp.launch(radiusSearchCountNeighborsCompactHashMap, dim=queryPositions.shape[0], inputs=[
            castTorchToWarp(y),
            castTorchToWarp(querySupports),
            castTorchToWarp(sortedPositions),
            castTorchToWarp(sortedSupports),
            castTorchToWarp(hashTable),
            castTorchToWarp(sortedCellTable),
            castTorchToWarp(qMin),
            scalar_t(hCell),
            D,
            castTorchToWarp(offsets),
            castTorchToWarp(numCells),
            castTorchToWarp(qMax),
            castTorchToWarp(qMin),
            castTorchToWarp(periodicity),
            wp.uint32(mode_uint),
            edge_count
        ], device=warpDevice)
    with record_function("neighborSearch - allocate neighbors"):


        # Synchronize so PyTorch reads the fully-written count results from Warp's stream
        wp.synchronize()
        total_edges = torch.sum(edge_count_t).cpu().item()

        # Compute cumulative offsets
        edge_offsets = torch.zeros(N, dtype=torch.int32, device = queryPositions.device)
        edge_offsets[1:] = torch.cumsum(edge_count_t[:-1], dim=0)
        # Synchronize so the Warp collect kernel reads the fully-written cumsum results from PyTorch's stream
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        edge_offsets_warp = wp.from_torch(edge_offsets)

        # Allocate output arrays on GPU
        edge_i_t, edge_i = allocateTorchWarp(total_edges, wp.int64, warpDevice)
        edge_j_t, edge_j = allocateTorchWarp(total_edges, wp.int64, warpDevice)
    with record_function("neighborSearch - collect neighbors"):
        wp.launch(radiusSearchCollectCompactHashMap, dim=queryPositions.shape[0], inputs=[
            castTorchToWarp(y),
            castTorchToWarp(querySupports),
            castTorchToWarp(sortedPositions),
            castTorchToWarp(sortedSupports),
            castTorchToWarp(hashTable),
            castTorchToWarp(sortedCellTable),
            castTorchToWarp(qMin),
            hCell,
            D,
            castTorchToWarp(offsets),
            castTorchToWarp(numCells),
            castTorchToWarp(qMax),
            castTorchToWarp(qMin),
            castTorchToWarp(periodicity),
            wp.uint32(mode_uint),   
            edge_count,
            edge_offsets_warp,
            castTorchToWarp(sortIndex),
            edge_i,
            edge_j
        ], device=warpDevice)


    with record_function("neighborSearch - build adjacency"):
        adjacencyCH = AdjacencyList(
            i=edge_i_t,  # Ensure dtype is long for indexing
            j=edge_j_t,
            numNeighbors=edge_count_t,
            edgeOffsets=edge_offsets,
            numRows=N,
            numCols=M,
            queryPositions = queryPositions,
            referencePositions = referencePositions,
            querySupports = querySupports,
            referenceSupports = referenceSupports,
            hashMap = datastructure
        )
    return adjacencyCH