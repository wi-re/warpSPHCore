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
    # print(f'SupportsX: {supportsX}')
    # print(f'SupportsY: {supportsY}')
    if scheme == SupportScheme.Gather:  # gather
        return torch.max(supportsX)
    elif scheme == SupportScheme.Scatter:  # scatter
        return torch.max(supportsY)
    elif scheme == SupportScheme.MeanSymmetric: # partialSymmetric
        return torch.max((supportsX + supportsY) / 2)
    elif scheme == SupportScheme.KernelMeanSymmetric:  # symmetric
        return max(torch.max(supportsX), torch.max(supportsY))
    elif scheme == SupportScheme.SuperSymmetric:  # superSymmetric
        return max(torch.max(supportsX), torch.max(supportsY))
    elif scheme == SupportScheme.PartialSymmetric:  # partialSymmetric
        return max(torch.max(supportsX), torch.max(supportsY))
    else:
        raise ValueError('Invalid scheme value. Must be a valid SupportScheme. Value is {}'.format(scheme))


def getDomainExtents(y, minDomain: Optional[torch.Tensor], maxDomain: Optional[torch.Tensor]):
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

    if torch.any(qExtent / h - numCells > 0):
        numCells -= 1
        numCells = torch.clamp(numCells, min=1)
        h = qExtent / numCells

    if torch.any(torch.ceil(qExtent / h) > qExtent / h):
        h = h * (1e-4 + 1)

    return torch.max(h)