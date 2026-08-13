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
from .grid import *
from .indexing import *
from .wp_index import indexCells

def sortReferenceParticles(referenceParticles, referenceSupport, domainMin, domainMax, periodicity: Optional[torch.Tensor] = None):
    """
    Sorts the reference particles based on their linear indices.

    Args:
        referenceParticles (torch.Tensor): The reference particles to be sorted.
        referenceSupport (scalar_t): The reference support value.
        domainMin: The minimum value of the domain.
        domainMax: The maximum value of the domain.

    Returns:
        sortedLinearIndices (torch.Tensor): The sorted linear indices of the reference particles.
        sortingIndices (torch.Tensor): The indices that sort the linear indices.
        cellCount (torch.Tensor): The number of cells in each dimension.
        domainMin: The minimum value of the domain.
        domainMax: The maximum value of the domain.
        hCell (scalar_t): The computed h value for the cells.
    """
    # with record_function("neighborSearch - sortReferenceParticles"): 
    # with record_function("neighborSearch - sortReferenceParticles[index Calculation]"): 
    hCell = compute_h(domainMin, domainMax, referenceSupport)
    qExtent = domainMax - domainMin
    # print('Domain Min:', domainMin, 'dtype:', domainMin.dtype)
    # print('Domain Max:', domainMax, 'dtype:', domainMax.dtype)
    # print('Reference Support:', referenceSupport, 'dtype:', referenceSupport.dtype)
    # print('Computed hCell:', hCell, 'dtype:', hCell.dtype)

    rawCellCount = qExtent / hCell
    # Periodic axes should not create an extra trailing cell from ceil();
    # that inserts an empty ghost cell and breaks wrap-around adjacency.
    eps = torch.tensor(1e-6, dtype=rawCellCount.dtype, device=rawCellCount.device)
    ceilCounts = torch.ceil(rawCellCount - eps)
    floorCounts = torch.floor(rawCellCount + eps)
    if periodicity is not None:
        periodicMask = periodicity.to(device=rawCellCount.device, dtype=torch.bool)
        axisCounts = torch.where(periodicMask, floorCounts, ceilCounts)
    else:
        axisCounts = ceilCounts
    cellCount = torch.clamp(axisCounts, min=1).to(torch.int32)
    indices = torch.floor((referenceParticles - domainMin) / hCell).to(torch.int32).view(-1, referenceParticles.shape[1])
    maxIndex = (cellCount - 1).view(1, -1)
    indices = torch.minimum(torch.maximum(indices, torch.zeros_like(indices)), maxIndex)
    
    
    # print('Cell count:', cellCount, 'Cell size:', hCell, 'Domain extent:', qExtent)
    # print('indices:', indices.contiguous())
    warp_indices = castTorchToWarp(indices)

    linearIndices, out = allocateTorchWarp((indices.shape[0],), wp.int64, warp_indices.device)
    warp_cell_count = castTorchToWarp(cellCount)
    wp.launch(
        indexCells,
        dim=indices.shape[0],
        inputs=[warp_indices, warp_cell_count, out],
        device=warp_indices.device,
    )
    # linearIndices = linearIndexing(indices, cellCount)
    # with record_function("neighborSearch - sortReferenceParticles[argsort]"): 
    sortingIndices = torch.argsort(linearIndices)
    # with record_function("neighborSearch - sortReferenceParticles[resort]"): 
    sortedLinearIndices = linearIndices[sortingIndices]
    return sortedLinearIndices, sortingIndices, \
            cellCount, domainMin, domainMax, scalar_t(hCell)
            