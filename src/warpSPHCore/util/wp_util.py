import warp as wp
import torch
from ..type_config import *


# ---------------------------------------------------------------------------
# wp.array identity cache -- REMOVED.
# ---------------------------------------------------------------------------
# This used to key a cached wp.array wrapper by (data_ptr, shape, strides,
# dtype) so that repeated calls on the same underlying tensor storage reused
# the same wp.array object instead of rebuilding it. The problem: the reused
# wp.array's .grad buffer is also reused and is never zeroed between unrelated
# forward/backward calls, so any two calls that happen to share tensor
# storage (the same leaf tensor called into repeatedly -- true of
# torch.autograd.gradcheck, and of any training loop that updates parameters
# in place) silently accumulate gradients from prior calls into the "new"
# one, making backward non-reentrant and gradients wrong from the second call
# onward. Found while adding torch.autograd.gradcheck coverage; see
# warpier_core.md. getCachedWarpArray now always builds a fresh wrapper.


from ..profiling import record_function
from .cast import castTorchToWarpAsBuiltins, castWarpToTorch


# from wp_tensor import tensor


    
# Convention Wise kind = 0 for fluid, 1 for boundary, 2 for ghost
# class OperationDirection(Enum):
#     AllToAll = 0
#     FluidToFluid = 1
#     FluidToBoundary = 2
#     BoundaryToFluid = 3
#     BoundaryToBoundary = 4
#     FluidToGhost = 5
#     GhostToFluid = 6
#     BoundaryToGhost = 7
#     GhostToBoundary = 8


import torch
from .support import volumeToSupport
from ..dataTypes import DomainDescription

def generateNeighborTestData(nx, targetNumNeighbors, dim, periodic, device):


    minDomain = torch.tensor([-1] * dim, dtype = torch.float32, device = device)
    maxDomain = torch.tensor([ 1] * dim, dtype = torch.float32, device = device)
    periodicity = torch.tensor([periodic] * dim, device = device, dtype = torch.bool)

    extent = maxDomain - minDomain
    shortExtent = torch.min(extent, dim = 0)[0].item()
    dx = (shortExtent / nx)
    ny = int(1 // dx)
    h = volumeToSupport(dx**dim, targetNumNeighbors, dim)
    dy = dx / 1.5
    ny = int(1 // dy)

    positions = []
    for d in range(dim):
        positions.append(torch.linspace(minDomain[d] + dx / 2, maxDomain[d] - dx / 2, int((extent[d] - dx) / dx) + 1, device = device))
    grid = torch.meshgrid(*positions, indexing = 'xy')
    
    positions = torch.stack(grid, dim = -1).reshape(-1,dim).to(device)
    supports = torch.ones(positions.shape[0], device = device) * h
    
    domain = DomainDescription(minDomain, maxDomain, periodicity, dim)
    
    return positions, supports, positions.shape[0], domain, dx




from typing import Any
from warp.types import vector, matrix

