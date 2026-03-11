import numpy as np
from ..utils import cpow_warp, iPow, bpow_warp
from typing import Any
import warp as wp
from warp.types import vector


@torch.jit.script
def k(q, dim: int = 2):        
    eta_max = 4   
    
    result = 56 * bpow(q - 1/4, 7)  -28 * bpow(q - 2/4, 7) + 8 * bpow(q - 3/4, 7) - bpow(q - 1, 7)
    
    return result

@torch.jit.script
def dkdq(q, dim: int = 2):        
    eta_max = 4
    
    result = 56 * bpow(q - 1/4, 6)  -28 * bpow(q - 2/4, 6) + 8 * bpow(q - 3/4, 6) - bpow(q - 1, 6)
    
    return result * 7


@torch.jit.script
def d2kdq2(q, dim: int = 2):        
    eta_max = 4
    
    result = 56 * bpow(q - 1/4, 5)  -28 * bpow(q - 2/4, 5) + 8 * bpow(q - 3/4, 5) - bpow(q - 1, 5)
    
    return result * 7 * 6

@torch.jit.script
def d3kdq3(q, dim: int = 2):        
    eta_max = 4
    
    result = 56 * bpow(q - 1/4, 4)  -28 * bpow(q - 2/4, 4) + 8 * bpow(q - 3/4, 4) - bpow(q - 1, 4)
    
    return result * 7 * 6 * 5

@torch.jit.script
def C_d(dim : int):
    if dim == 1: return 4096/315
    elif dim == 2: return 589824/(7435 *np.pi)
    else: return 1024/(105 *np.pi)

@torch.jit.script # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations
def kernelScale(dim: int = 2):
    if dim == 1: return 2.121321
    elif dim == 2: return 2.158131
    else: return 2.195775


@torch.jit.script
def packingRatio(): # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations Table 2
    return 1.595  * 1.1425 # Factor to be in line CRKSPH