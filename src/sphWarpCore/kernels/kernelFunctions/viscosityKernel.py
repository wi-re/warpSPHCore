import numpy as np
from ..utils import cpow_warp, iPow, bpow_warp
from typing import Any
import warp as wp
from warp.types import vector

@torch.jit.script
def k(q, dim: int = 2):  
    return -1/2 * q**3 + q**2 +1/2 / (q) - 1
def dkdq(q, dim: int = 2):    
    return -3/2 *q**2 - 1 /(2 * q**2) + 2 * q
@torch.jit.script
def d2kdq2(q, dim: int = 2):        
    return q**(-3) - 3* q + 2
@torch.jit.script
def d3kdq3(q, dim: int = 2):
    return -3 * q**(-4) - 3

@torch.jit.script
def C_d(dim : int):
    if dim == 1: return 15 / (2 * np.pi)
    elif dim == 2: return 10 / (9 * np.pi)
    else: return 15 / (2 * np.pi)


def kernelScale(dim: int = 2):
    return 1.0

def packingRatio():
    return 1.0