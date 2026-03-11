import numpy as np
from ..utils import cpow_warp, iPow, bpow_warp
from typing import Any
import warp as wp
from warp.types import vector

@wp.func
def adhesionKernel_k(q: wp.float32, dim: wp.int32 = 2):  
    if q > 0.5:
        return wp.pow(-4.0 * iPow(q, 2) + 6.0 * q - 2.0, wp.float32(0.25))
    return 0.0

@wp.func
def adhesionKernel_dkdq(q: wp.float32, dim: wp.int32 = 2):    
    if q > 0.5:
        return (6.0 - 8.0 * q) / (4.0 * wp.pow(-4.0 * iPow(q, 2) + 6.0 * q - 2.0, wp.float32(0.75)))
    return 0.0

@wp.func
def adhesionKernel_d2kdq2(q: wp.float32, dim: wp.int32 = 2):
    if q > 0.5:
        return (16.0 * iPow(q, 2) - 24.0 * q + 11.0) / (8.0 * wp.pow(-4.0 * iPow(q, 2) + 6.0 * q - 2.0, wp.float32(0.75)) * (2.0 * iPow(q, 2) - 3.0 * q + 1.0))
    return 0.0

@wp.func
def adhesionKernel_d3kdq3(q: wp.float32, dim: wp.int32 = 2):
    if q > 0.5:
        return (48.0 - 32.0 * q) / (4.0 * wp.pow(-4.0 * iPow(q, 2) + 6.0 * q - 2.0, wp.float32(0.75)) * (2.0 * iPow(q, 2) - 3.0 * q + 1.0))
    return 0.0

@wp.func
def adhesionKernel_C_d(dim: wp.int32):
    if dim == 1: return 0.007
    elif dim == 2: return 0.007
    else: return 0.007

@wp.func
def adhesionKernel_kernelScale(dim: wp.int32 = 2):
    return 1.0

@wp.func
def adhesionKernel_packingRatio():
    return 1.0