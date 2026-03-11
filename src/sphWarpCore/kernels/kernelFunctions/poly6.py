import numpy as np
from ..utils import cpow_warp, iPow, bpow_warp
from typing import Any
import warp as wp
from warp.types import vector

@torch.jit.script
def k(q, dim: int = 2):  
    return (1-q**2)**3
@torch.jit.script
def dkdq(q, dim: int = 2):    
    return -6 * q * (1-q**2)**2
def d2kdq2(q, dim: int = 2):        
    return -6 * (5 * q**4 - 6 * q**2 + 1)
    
def d3kdq3(q, dim: int = 2):        
    return -6 * (20 * q**3 - 12 * q)

@torch.jit.script
def C_d(dim : int):
    if dim == 1: return 35 / 16
    elif dim == 2: return 35 / (32 * np.pi)
    else: return 315 / (64 * np.pi)

def kernelScale(dim: int = 2):
    return 1.0

def packingRatio():
    return 1.0