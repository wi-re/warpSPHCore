import numpy as np
from ..utils import cpow_warp, iPow, bpow_warp
from typing import Any
import warp as wp
from warp.types import vector

@torch.jit.script
def k(q, dim: int = 2):        
    return cpow(1-q, 5) - 6 * cpow(2/3 - q, 5) + 15 * cpow(1/3 - q, 5)
@torch.jit.script
def dkdq(q, dim: int = 2):     
    return -5 * cpow(1-q, 4) + 30 * cpow(2/3 - q, 4) - 75 * cpow(1/3 - q, 4)
@torch.jit.script
def d2kdq2(q, dim: int = 2):        
    return 20 * cpow(1-q,3) - 120 * cpow(2/3 - q, 3) + 300 * cpow(1/3 - q, 3)
@torch.jit.script
def d3kdq3(q, dim: int = 2):
    return -60 * cpow(1-q,2) + 360 * cpow(2/3 - q, 2) - 900 * cpow(1/3 - q, 2)

@torch.jit.script
def C_d(dim : int):
    if dim == 1: return 3**5/40
    elif dim == 2: return 3**7 * 7 / (478 * np.pi)
    else: return 3**7/ (40 * np.pi)

@torch.jit.script # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations
def kernelScale(dim: int = 2):
    if dim == 1: return 2.121321
    elif dim == 2: return 2.158131
    else: return 2.195775


@torch.jit.script
def packingRatio(): # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations Table 2
    return 1.595  * 1.1425 # Factor to be in line CRKSPH