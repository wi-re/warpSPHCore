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

def computeGridSupport(supportsX, supportsY, scheme: SupportScheme):
    # The maximum of an empty role contributes zero to the radius bound.
    # Grid construction supplies a positive geometric sentinel when no pairs
    # can exist; ordinary nonempty support semantics stay unchanged.
    def maximum(supports):
        return supports.max() if supports.numel() else supports.new_zeros(())
    # print(f'SupportsX: {supportsX}')
    # print(f'SupportsY: {supportsY}')
    if scheme == SupportScheme.Gather:  # gather
        return maximum(supportsX)
    elif scheme == SupportScheme.Scatter:  # scatter
        return maximum(supportsY)
    elif scheme == SupportScheme.MeanSymmetric: # partialSymmetric
        # Query/reference populations need not match; bound every cross pair.
        return (maximum(supportsX) + maximum(supportsY)) / 2
    # torch.maximum, not Python max(): the latter compares two device tensors
    # on the host (a sync); the value is the same either way
    elif scheme == SupportScheme.KernelMeanSymmetric:  # symmetric
        return torch.maximum(maximum(supportsX), maximum(supportsY))
    elif scheme == SupportScheme.SuperSymmetric:  # superSymmetric
        return torch.maximum(maximum(supportsX), maximum(supportsY))
    elif scheme == SupportScheme.PartialSymmetric:  # partialSymmetric
        return torch.maximum(maximum(supportsX), maximum(supportsY))
    else:
        raise ValueError('Invalid scheme value. Must be a valid SupportScheme. Value is {}'.format(scheme))


def getDomainExtents(y, minDomain: Optional[torch.Tensor], maxDomain: Optional[torch.Tensor]):
    if y.shape[0] == 0 and (minDomain is None or maxDomain is None):
        raise ValueError('Empty reference populations require explicit domain min and max bounds.')
    if minDomain is not None and maxDomain is not None:
        return minDomain, maxDomain
    elif minDomain is not None:
        maxD = torch.max(y, dim=0)[0]
        return minDomain, maxD
    elif maxDomain is not None:
        minD = torch.min(y, dim=0)[0]
        return minD, maxDomain
    else:
        minD = torch.min(y, dim=0)[0]
        maxD = torch.max(y, dim=0)[0]
        return minD, maxD



# @torch.jit.script
def compute_h(qMin, qMax, referenceSupport): 
    """
    Compute the smoothing length (h) based on the given minimum and maximum coordinates (qMin and qMax)
    and the reference support value. The smoothing length is used for grid operations and is determined
    by dividing the domain into cells based on the reference support value such that h > referenceSupport.

    Args:
        qMin (torch.Tensor): The minimum coordinates.
        qMax (torch.Tensor): The maximum coordinates.
        referenceSupport (scalar_t): The reference support value.

    Returns:
        torch.Tensor: The computed smoothing length (h).
    """
    qExtent = qMax - qMin
    qCells = qExtent / referenceSupport
    qfCells = torch.floor(qCells)
    # numCells = torch.where( qCells - qfCells < 1e-4, qfCells, qfCells+1)
    # Ensure every dimension has at least one cell to avoid inf h and invalid indexing.
    numCells = torch.clamp(qfCells, min=1)
    h = qExtent / (numCells)

    # the two corrections as selects instead of host branches (same arithmetic, no device read)
    shrink = torch.any(qExtent / h - numCells > 0)
    numCellsShrunk = torch.clamp(numCells - 1, min=1)
    numCells = torch.where(shrink, numCellsShrunk, numCells)
    h = torch.where(shrink, qExtent / numCellsShrunk, h)

    h = torch.where(torch.any(torch.ceil(qExtent / h) > qExtent / h), h * (1e-4 + 1), h)

    return torch.max(h)
