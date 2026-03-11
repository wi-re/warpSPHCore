import numpy as np
from ..utils import cpow_warp, iPow, bpow_warp
from typing import Any
import warp as wp
from warp.types import vector

@wp.func
def poly6_k(q: wp.float32, dim: wp.int32 = 2):  
    return iPow(1.0 - iPow(q, 2), 3)

@wp.func
def poly6_dkdq(q: wp.float32, dim: wp.int32 = 2):    
    return -6.0 * q * iPow(1.0 - iPow(q, 2), 2)

@wp.func
def poly6_d2kdq2(q: wp.float32, dim: wp.int32 = 2):        
    return -6.0 * (5.0 * iPow(q, 4) - 6.0 * iPow(q, 2) + 1.0)

@wp.func
def poly6_d3kdq3(q: wp.float32, dim: wp.int32 = 2):        
    return -6.0 * (20.0 * iPow(q, 3) - 12.0 * q)

@wp.func
def poly6_C_d(dim: wp.int32):
    if dim == 1: return 35.0/16.0
    elif dim == 2: return 35.0 / (32.0 * np.pi)
    else: return 315.0 / (64.0 * np.pi)

@wp.func
def poly6_kernelScale(dim: wp.int32 = 2):
    return 1.0

@wp.func
def poly6_packingRatio():
    return 1.0