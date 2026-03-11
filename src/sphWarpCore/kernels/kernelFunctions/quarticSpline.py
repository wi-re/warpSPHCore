import numpy as np
from ..utils import cpow_warp, iPow, bpow_warp
from typing import Any
import warp as wp
from warp.types import vector

@torch.jit.script
def k(q, dim: int = 2):   
    return cpow(1-q, 4) - 5 * cpow(3/5 - q, 4) + 10 * cpow(1/5 - q, 4)
@torch.jit.script
def dkdq(q, dim: int = 2):    
    return -4 * cpow(1-q, 3) + 20 * cpow(3/5 - q, 3) - 40 * cpow(1/5 - q, 3)
@torch.jit.script
def d2kdq2(q, dim: int = 2):        
    return 12 * cpow(1-q,2) -60 * cpow(3/5 - q,2) + 120 * cpow(1/5 - q,2)
@torch.jit.script
def d3kdq3(q, dim: int = 2):
    return -24 * (1-q) + 120 * cpow(3/5 - q, 1) - 240 * cpow(1/5 - q, 1)
    
@torch.jit.script
def C_d(dim : int):
    if dim == 1: return 5**5/768
    elif dim == 2: return 5**6 * 3 / (2398 * np.pi)
    else: return 5**6/ (512 * np.pi)

@torch.jit.script # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations
def kernelScale(dim: int = 2):
    if dim == 1: return 1.936492
    elif dim == 2: return 1.977173
    else: return 2.018932

@torch.jit.script
def packingRatio(): # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations Table 2
    return 1.203 