import numpy as np
from ..utils import cpow_warp, iPow, bpow_warp
from ...types import *
from typing import Any
import warp as wp
from warp.types import vector

@wp.func
def cohesionKernel_k(q: scalar_t, dim: wp.int32 = 2):  
    if q < scalar_t(0.5):
        return scalar_t(2.0) * iPow(scalar_t(1.0) - q, 3) * iPow(q, 3) - scalar_t(1.0)/scalar_t(64.0)
    else:
        return iPow(scalar_t(1.0) - q, 3) * iPow(q, 3)

@wp.func
def cohesionKernel_dkdq(q: scalar_t, dim: wp.int32 = 2):    
    if q < scalar_t(0.5):
        return scalar_t(2.0) * (scalar_t(-3.0) * iPow(q, 2) * (scalar_t(2.0) * q - scalar_t(1.0)) * iPow(scalar_t(1.0) - q, 2))
    else:
        return scalar_t(-3.0) * iPow(q, 2) * (scalar_t(2.0) * q - scalar_t(1.0)) * iPow(scalar_t(1.0) - q, 2)

@wp.func
def cohesionKernel_d2kdq2(q: scalar_t, dim: wp.int32 = 2):
    if q < scalar_t(0.5):
        return scalar_t(2.0) * (scalar_t(6.0) * q * (scalar_t(-5.0) * iPow(q, 3) + scalar_t(10.0) * iPow(q, 2) - scalar_t(6.0) * q + scalar_t(1.0)))
    else:
        return scalar_t(6.0) * q * (scalar_t(-5.0) * iPow(q, 3) + scalar_t(10.0) * iPow(q, 2) - scalar_t(6.0) * q + scalar_t(1.0))

@wp.func
def cohesionKernel_d3kdq3(q: scalar_t, dim: wp.int32 = 2):
    if q < scalar_t(0.5):
        return scalar_t(2.0) * (scalar_t(-120.0) * iPow(q, 3) + scalar_t(180.0) * iPow(q, 2) - scalar_t(72.0) * q + scalar_t(6.0))
    else:
        return scalar_t(-120.0) * iPow(q, 3) + scalar_t(180.0) * iPow(q, 2) - scalar_t(72.0) * q + scalar_t(6.0)

@wp.func
def cohesionKernel_C_d(dim: wp.int32):
    if dim == 1: return scalar_t(32.0) / scalar_t(np.pi)
    elif dim == 2: return scalar_t(32.0) / scalar_t(np.pi)
    else: return scalar_t(32.0) / scalar_t(np.pi)

@wp.func
def cohesionKernel_kernelScale(dim: wp.int32 = 2):
    return scalar_t(1.0)

@wp.func
def cohesionKernel_packingRatio():
    return scalar_t(1.0)