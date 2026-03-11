import numpy as np
from ..utils import cpow_warp, iPow, bpow_warp
from typing import Any
import warp as wp
from warp.types import vector


@torch.jit.script
def k(q, dim: int = 2):  
    return torch.where(q < 0.5, 2, 1) * (1-q)**3 * q**3 + torch.where(q < 0.5, -1/64, 0)
@torch.jit.script
def dkdq(q, dim: int = 2):    
    return torch.where(q < 0.5, 2, 1) * (-3 *q**2 * (2*q -1) * (1-q)**2)
@torch.jit.script
def d2kdq2(q, dim: int = 2):        
    return torch.where(q < 0.5, 2, 1) * (6 * q * (-5 *q **3 + 10 *q**2 - 6 *q + 1))
@torch.jit.script
def d3kdq3(q, dim: int = 2):
    return torch.where(q < 0.5, 2, 1) * (-120 * q**3 + 180 * q**2 - 72 * q + 6)

@torch.jit.script
def C_d(dim : int):
    if dim == 1: return 32 / np.pi
    elif dim == 2: return 32 / np.pi
    else: return 32 / np.pi

def kernelScale(dim: int = 2):
    return 1.0

def packingRatio():
    return 1.0