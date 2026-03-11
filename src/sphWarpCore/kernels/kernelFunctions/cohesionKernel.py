import numpy as np
from ..utils import cpow_warp, iPow, bpow_warp
from typing import Any
import warp as wp
from warp.types import vector

@wp.func
def cohesionKernel_k(q: wp.float32, dim: wp.int32 = 2):  
    if q < 0.5:
        return 2.0 * iPow(1.0 - q, 3) * iPow(q, 3) - 1.0/64.0
    else:
        return iPow(1.0 - q, 3) * iPow(q, 3)

@wp.func
def cohesionKernel_dkdq(q: wp.float32, dim: wp.int32 = 2):    
    if q < 0.5:
        return 2.0 * (-3.0 * iPow(q, 2) * (2.0 * q - 1.0) * iPow(1.0 - q, 2))
    else:
        return -3.0 * iPow(q, 2) * (2.0 * q - 1.0) * iPow(1.0 - q, 2)

@wp.func
def cohesionKernel_d2kdq2(q: wp.float32, dim: wp.int32 = 2):
    if q < 0.5:
        return 2.0 * (6.0 * q * (-5.0 * iPow(q, 3) + 10.0 * iPow(q, 2) - 6.0 * q + 1.0))
    else:
        return 6.0 * q * (-5.0 * iPow(q, 3) + 10.0 * iPow(q, 2) - 6.0 * q + 1.0)

@wp.func
def cohesionKernel_d3kdq3(q: wp.float32, dim: wp.int32 = 2):
    if q < 0.5:
        return 2.0 * (-120.0 * iPow(q, 3) + 180.0 * iPow(q, 2) - 72.0 * q + 6.0)
    else:
        return -120.0 * iPow(q, 3) + 180.0 * iPow(q, 2) - 72.0 * q + 6.0

@wp.func
def cohesionKernel_C_d(dim: wp.int32):
    if dim == 1: return 32.0 / np.pi
    elif dim == 2: return 32.0 / np.pi
    else: return 32.0 / np.pi

@wp.func
def cohesionKernel_kernelScale(dim: wp.int32 = 2):
    return 1.0

@wp.func
def cohesionKernel_packingRatio():
    return 1.0