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
    capacity: Optional[CapacityPolicy] = DEFAULT_CAPACITY,
    reuse: Optional[AdjacencyList] = None,
):                
    # `capacity`: spare room of the flat pair arrays (CapacityPolicy; None = exactly sized, no padding).
    # `reuse`: the list this rebuild replaces: its capacity is kept while the new count still fits (stable shapes),
    # and with `capacity.inPlace` its buffers are overwritten and shared by the result.
    mode_uint = datastructure.mode_uint
    minDomain = domainDescription.min if domainDescription.min is not None else None
    maxDomain = domainDescription.max if domainDescription.max is not None else None
    hMax = computeGridSupport(querySupports, referenceSupports, mode)
    minD, maxD = getDomainExtents(referencePositions, minDomain, maxDomain)
    # periodicity = domainDescription.periodicity if domainDescription.periodicity is not None else [False] * referencePositions.shape[1]

    # the periodic flags read to the host once (iterating the device tensor synced per axis)
    periodicHost = periodicity.tolist() if isinstance(periodicity, torch.Tensor) else list(periodicity)
    x = torch.vstack([component if not periodic else torch.remainder(component - minD[i], maxD[i] - minD[i]) + minD[i] for i, (component, periodic) in enumerate(zip(referencePositions.mT, periodicHost))]).mT
    y = torch.vstack([component if not periodic else torch.remainder(component - minD[i], maxD[i] - minD[i]) + minD[i] for i, (component, periodic) in enumerate(zip(queryPositions.mT, periodicHost))]).mT
    
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
    # Pin the warp count/collect kernels to torch's current stream (matching
    # buildHashmap.hashCells). Without this they run on warp's default stream and
    # can read the datastructure tensors (hashTable, sortedCellTable,
    # sortedPositions) before the torch build ops in buildCompactHashMap finish
    # writing them -- an unsynchronised cross-stream race that intermittently
    # yields a degenerate (~N^2) neighbour count and an OOM on edge allocation.
    _torchStream = (wp.stream_from_torch(torch.cuda.current_stream(queryPositions.device))
                    if queryPositions.device.type == 'cuda' else None)

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
        ], device=warpDevice, stream=_torchStream)
    with record_function("neighborSearch - allocate neighbors"):


        # the count kernel ran on torch's current stream, so the read below is ordered after it
        # (the device-wide wp.synchronize() this used to need predates pinning the launches)
        total_edges = torch.sum(edge_count_t).cpu().item()

        # Compute cumulative offsets
        edge_offsets = torch.zeros(N, dtype=torch.int32, device = queryPositions.device)
        edge_offsets[1:] = torch.cumsum(edge_count_t[:-1], dim=0)
        # the collect kernel runs on torch's current stream, after the cumsum (no device-wide sync needed)
        edge_offsets_warp = wp.from_torch(edge_offsets)

        # Allocate output arrays on GPU, with spare capacity beyond the real count (padded with -1, see AdjacencyList)
        numEdges = int(total_edges)
        capacityEdges = numEdges if capacity is None else capacity.capacityFor(
            numEdges, None if reuse is None else reuse.paddedI.shape[0])
        inPlace = (capacity is not None and capacity.inPlace and reuse is not None
                   and reuse.paddedI.shape[0] == capacityEdges and reuse.paddedJ.shape == reuse.paddedI.shape
                   and reuse.paddedI.device == queryPositions.device
                   and reuse.numNeighbors.shape == edge_count_t.shape and reuse.edgeOffsets.shape == edge_offsets.shape)
        if inPlace:
            edge_i_t, edge_j_t = reuse.paddedI, reuse.paddedJ
            edge_i, edge_j = castTorchToWarp(edge_i_t), castTorchToWarp(edge_j_t)
        else:
            edge_i_t, edge_i = allocateTorchWarp(capacityEdges, wp.int64, warpDevice)
            edge_j_t, edge_j = allocateTorchWarp(capacityEdges, wp.int64, warpDevice)
        if capacityEdges > numEdges or inPlace:
            edge_i_t.fill_(-1)
            edge_j_t.fill_(-1)
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
        ], device=warpDevice, stream=_torchStream)
        if inPlace:
            # the per-row arrays too: a graph reads them at fixed addresses. After the collect launch (it reads
            # the new counts and offsets from the arrays it was given); same stream, so ordered.
            reuse.numNeighbors.copy_(edge_count_t)
            reuse.edgeOffsets.copy_(edge_offsets)
            edge_count_t, edge_offsets = reuse.numNeighbors, reuse.edgeOffsets
            # the list's reference snapshot (what the validity check compares the particles with) is read by a graph
            # at fixed addresses too: the new values go into the old tensors (one tensor where query and reference
            # were one), unless a shape or dtype changed
            snapshots = {}
            for name, value in (('queryPositions', queryPositions), ('referencePositions', referencePositions),
                                ('querySupports', querySupports), ('referenceSupports', referenceSupports)):
                held = getattr(reuse, name)
                if held is not None and held.shape == value.shape and held.dtype == value.dtype and held.device == value.device:
                    if not any(held is done for done in snapshots.values()):
                        held.copy_(value)
                    snapshots[name] = held
                else:
                    snapshots[name] = value
            queryPositions, referencePositions = snapshots['queryPositions'], snapshots['referencePositions']
            querySupports, referenceSupports = snapshots['querySupports'], snapshots['referenceSupports']


    with record_function("neighborSearch - build adjacency"):
        adjacencyCH = AdjacencyList(
            paddedI=edge_i_t,
            paddedJ=edge_j_t,
            numEdges=numEdges,
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