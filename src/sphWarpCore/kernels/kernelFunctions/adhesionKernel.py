import numpy as np
from ..utils import cpow_warp, iPow, bpow_warp
from typing import Any
import warp as wp
from warp.types import vector


@torch.jit.script
def k(q, dim: int = 2):  
    if q > 0.5:
        return (-4 * q**2 + 6 * q - 2)**(1/4)
    return 0.0
    # return torch.where(q > 0.5, ((-4 * q**2 + 6 * q - 2))**(1/4), 0)
@torch.jit.script
def dkdq(q, dim: int = 2):    
    if q > 0.5:
        return (6 - 8 * q) / (4 * (-4 * q**2 + 6 *q -2)** (3/4))
    return 0.0
    # return torch.where(q > 0.5, (6 - 8 * q) / (4 * (-4 * q**2 + 6 *q -2)** (3/4)),0)
@torch.jit.script
def d2kdq2(q, dim: int = 2):
    if q > 0.5:
        return (16 *q**2 - 24 *q + 11) / (8 *(-4 * q**2 + 6 *q -2)**(3/4) * (2 *q**2 - 3*q + 1))
    return 0.0        
    # return torch.where(q > 0.5, (16 *q**2 - 24 *q + 11) / (8 *(-4 * q**2 + 6 *q -2)**(3/4) * (2 *q**2 - 3*q + 1)),0)
@torch.jit.script
def d3kdq3(q, dim: int = 2):
    if q > 0.5:
        return (48 - 32 * q) / (4 * (-4 * q**2 + 6 *q -2)**(3/4) * (2 *q**2 - 3*q + 1))
    # return torch.where(q > 0.5, (48 - 32 * q) / (4 * (-4 * q**2 + 6 *q -2)**(3/4) * (2 *q**2 - 3*q + 1)) ,0)
    
@torch.jit.script
def C_d(dim : int):
    if dim == 1: return 0.007
    elif dim == 2: return 0.007
    else: return 0.007


def kernelScale(dim: int = 2):
    return 1.0

def packingRatio():
    return 1.0