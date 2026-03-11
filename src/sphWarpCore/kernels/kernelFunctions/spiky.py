import numpy as np
from ..utils import cpow_warp, iPow, bpow_warp
from typing import Any
import warp as wp
from warp.types import vector

@torch.jit.script
def k(q, dim: int = 2):  
    return cpow(1-q, 3)
@torch.jit.script
def dkdq(q, dim: int = 2):    
    return -3 * cpow(1-q,2)
@torch.jit.script
def d2kdq2(q, dim: int = 2):        
    return 6 * (1-q)
@torch.jit.script
def d3kdq3(q, dim: int = 2):
    return -6 * torch.ones_like(q)
    
@torch.jit.script
def C_d(dim : int):
    if dim == 1: return 1/4
    elif dim == 2: return 2 / np.pi
    else: return 15 / np.pi

def kernelScale(dim: int = 2):
    return 1.0

def packingRatio():
    return 1.0