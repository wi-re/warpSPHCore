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

from .indexing import *

@wp.func
def clampCellIndex(cellIndex: wp.vec3i, numCells: wp.array(dtype=wp.int32), D: int) -> wp.vec3i:
    for d in range(D):
        if cellIndex[d] < 0:
            cellIndex[d] = 0
        elif cellIndex[d] >= numCells[d]:
            cellIndex[d] = numCells[d] - 1
    return cellIndex
    
@wp.func
def wrapCellComponentPeriodic(cell: wp.int32, numCell: wp.int32) -> wp.int32:
    # Wrap integer cell indices for periodic domains, even if they are more than one box out-of-range.
    if numCell <= 0:
        return wp.int32(0)
    return cell - wp.int32(wp.floor(scalar_t(cell) / scalar_t(numCell))) * numCell


@wp.kernel
def indexCells(
    cellIndices: wp.array2d(dtype=wp.int32),
    cellCounts: wp.array(dtype=wp.int32),
    cellIndxes: wp.array(dtype=wp.int64),
):
    i = wp.tid()
    numCells = cellIndices.shape[0]
    dim = cellIndices.shape[1]
    
    if i >= numCells:
        return
    
    cellIndex = wp.vec3i(0)
    for d in range(dim):
        cellIndex[d] = cellIndices[i, d]
        
    cellIndxes[i] = getLinearIndex64(cellIndex, cellCounts, dim)