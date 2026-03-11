import numpy as np
from ..utils import cpow_warp, iPow, bpow_warp
from typing import Any
import warp as wp
from warp.types import vector

@wp.func
def spiky_k(q: wp.float32, dim: wp.int32 = 2):  
    return cpow_warp(1.0 - q, 3)

@wp.func
def spiky_dkdq(q: wp.float32, dim: wp.int32 = 2):    
    return -3.0 * cpow_warp(1.0 - q, 2)

@wp.func
def spiky_d2kdq2(q: wp.float32, dim: wp.int32 = 2):        
    return 6.0 * (1.0 - q)

@wp.func
def spiky_d3kdq3(q: wp.float32, dim: wp.int32 = 2):
    return -6.0

@wp.func
def spiky_C_d(dim: wp.int32):
    if dim == 1: return 0.25
    elif dim == 2: return 2.0 / np.pi
    else: return 15.0 / np.pi

@wp.func
def spiky_kernelScale(dim: wp.int32 = 2):
    return 1.0

@wp.func
def spiky_packingRatio():
    return 1.0