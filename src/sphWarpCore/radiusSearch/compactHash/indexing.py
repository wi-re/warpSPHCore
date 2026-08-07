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


@wp.func
def computeZOrderIndex(index: wp.vec3i) -> wp.int32:
    # Morton encoding (Z-order curve) for 3D indices
    x = wp.uint32(index.x)
    y = wp.uint32(index.y)
    z = wp.uint32(index.z)

    # Interleave bits of x, y, and z
    def splitBy3Bits(n):
        n = (n | (n << 16)) & 0x030000FF
        n = (n | (n << 8)) & 0x0300F00F
        n = (n | (n << 4)) & 0x030C30C3
        n = (n | (n << 2)) & 0x09249249
        return n
    
    return wp.vec3i(splitBy3Bits(x) | (splitBy3Bits(y) << 1) | (splitBy3Bits(z) << 2))
@wp.func

def getLinearIndex64(
    cellIndex: wp.vec3i,
    cellCounts: wp.array(dtype=wp.int32),  # shape [D]
    D: int,
) -> wp.int64:
    # x-major (axis 0 first): x + Nx*y + Nx*Ny*z
    linearIndex = wp.int64(0)
    product = wp.int64(1)
    for d in range(D):
        linearIndex += wp.int64(cellIndex[d]) * product
        product = product * wp.int64(cellCounts[d])
    return linearIndex


# @torch.jit.script # jit script is deprecated :/
def linearIndexing(cellIndices, cellCounts):
    """
    Compute the linear index based on the given cell indices and cell counts.

    Args:
        cellIndices (torch.Tensor): Tensor containing the cell indices.
        cellCounts (torch.Tensor): Tensor containing the cell counts.

    Returns:
        torch.Tensor: Tensor containing the linear indices.
    """
    dim = cellIndices.shape[1]
    linearIndex = torch.zeros(cellIndices.shape[0], dtype=cellIndices.dtype, device=cellIndices.device)
    product = 1
    for i in range(dim):
        linearIndex += cellIndices[:, i] * product
        product = product * cellCounts[i].item()
    return linearIndex


def delinearizeIndices(linearIndices: torch.Tensor, cellCounts: torch.Tensor, D: int) -> torch.Tensor:
    """Recover D-dimensional cell indices from x-major linear indices.

    This keeps hashing and lookup consistent with getLinearIndex64/getLinearIndex.
    """
    linear = linearIndices.to(torch.int64).clone()
    grid = torch.zeros((linear.shape[0], D), dtype=torch.int32, device=linear.device)
    for d in range(D):
        cd = int(cellCounts[d].item())
        grid[:, d] = torch.remainder(linear, cd).to(torch.int32)
        linear = torch.div(linear, cd, rounding_mode="floor")
    return grid


@wp.func 
def getLinearIndex(
    cellIndex: wp.vec3i,
    cellCounts: wp.array(dtype=wp.int32),  # shape [D]
    D: int
):
    linearIndex = wp.int32(0)
    product = wp.int32(1)
    for d in range(D):
        linearIndex += cellIndex[d] * product
        product = product * cellCounts[d]
    return linearIndex