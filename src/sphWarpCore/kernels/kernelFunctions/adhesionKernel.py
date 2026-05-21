import numpy as np
from ..utils import cpow_warp, iPow, bpow_warp
from ...types import *
from typing import Any
import warp as wp
from warp.types import vector

@wp.func
def adhesionKernel_k(q: scalar_t, dim: wp.int32 = 2):  
    if q > scalar_t(0.5):
        return wp.pow(scalar_t(-4.0) * iPow(q, 2) + scalar_t(6.0) * q - scalar_t(2.0), scalar_t(scalar_t(0.25)))
    return scalar_t(0.0)

@wp.func
def adhesionKernel_dkdq(q: scalar_t, dim: wp.int32 = 2):    
    if q > scalar_t(0.5):
        return (scalar_t(6.0) - scalar_t(8.0) * q) / (scalar_t(4.0) * wp.pow(scalar_t(-4.0) * iPow(q, 2) + scalar_t(6.0) * q - scalar_t(2.0), scalar_t(scalar_t(0.75))))
    return scalar_t(0.0)

@wp.func
def adhesionKernel_d2kdq2(q: scalar_t, dim: wp.int32 = 2):
    if q > scalar_t(0.5):
        return (scalar_t(16.0) * iPow(q, 2) - scalar_t(24.0) * q + scalar_t(11.0)) / (scalar_t(8.0) * wp.pow(scalar_t(-4.0) * iPow(q, 2) + scalar_t(6.0) * q - scalar_t(2.0), scalar_t(scalar_t(0.75))) * (scalar_t(2.0) * iPow(q, 2) - scalar_t(3.0) * q + scalar_t(1.0)))
    return scalar_t(0.0)

@wp.func
def adhesionKernel_d3kdq3(q: scalar_t, dim: wp.int32 = 2):
    if q > scalar_t(0.5):
        return (scalar_t(48.0) - scalar_t(32.0) * q) / (scalar_t(4.0) * wp.pow(scalar_t(-4.0) * iPow(q, 2) + scalar_t(6.0) * q - scalar_t(2.0), scalar_t(scalar_t(0.75))) * (scalar_t(2.0) * iPow(q, 2) - scalar_t(3.0) * q + scalar_t(1.0)))
    return scalar_t(0.0)

@wp.func
def adhesionKernel_C_d(dim: wp.int32):
    if dim == 1: return scalar_t(0.007)
    elif dim == 2: return scalar_t(0.007)
    else: return scalar_t(0.007)

@wp.func
def adhesionKernel_kernelScale(dim: wp.int32 = 2):
    return scalar_t(1.0)

@wp.func
def adhesionKernel_packingRatio():
    return scalar_t(1.0)