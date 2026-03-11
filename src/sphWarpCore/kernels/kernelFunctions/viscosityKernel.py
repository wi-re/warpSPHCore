import numpy as np
from ..utils import cpow_warp, iPow, bpow_warp
from typing import Any
import warp as wp
from warp.types import vector

@wp.func
def viscosityKernel_k(q: wp.float32, dim: wp.int32 = 2):  
    return -0.5 * iPow(q, 3) + iPow(q, 2) + 0.5 / q - 1.0

@wp.func
def viscosityKernel_dkdq(q: wp.float32, dim: wp.int32 = 2):    
    return -1.5 * iPow(q, 2) - 0.5 / iPow(q, 2) + 2.0 * q

@wp.func
def viscosityKernel_d2kdq2(q: wp.float32, dim: wp.int32 = 2):        
    return 1.0 / iPow(q, 3) - 3.0 * q + 2.0

@wp.func
def viscosityKernel_d3kdq3(q: wp.float32, dim: wp.int32 = 2):
    return -3.0 / iPow(q, 4) - 3.0

@wp.func
def viscosityKernel_C_d(dim: wp.int32):
    if dim == 1: return 15.0 / (2.0 * np.pi)
    elif dim == 2: return 10.0 / (9.0 * np.pi)
    else: return 15.0 / (2.0 * np.pi)

@wp.func
def viscosityKernel_kernelScale(dim: wp.int32 = 2):
    return 1.0

@wp.func
def viscosityKernel_packingRatio():
    return 1.0