import numpy as np
from ..utils import cpow_warp, iPow, bpow_warp
from ...types import *
from typing import Any
import warp as wp
from warp.types import vector

@wp.func
def poly6_k(q: scalar_t, dim: wp.int32 = 2):  
    return iPow(scalar_t(1.0) - iPow(q, 2), 3)

@wp.func
def poly6_dkdq(q: scalar_t, dim: wp.int32 = 2):    
    return scalar_t(-6.0) * q * iPow(scalar_t(1.0) - iPow(q, 2), 2)

@wp.func
def poly6_d2kdq2(q: scalar_t, dim: wp.int32 = 2):        
    return scalar_t(-6.0) * (scalar_t(5.0) * iPow(q, 4) - scalar_t(6.0) * iPow(q, 2) + scalar_t(1.0))

@wp.func
def poly6_d3kdq3(q: scalar_t, dim: wp.int32 = 2):        
    return scalar_t(-6.0) * (scalar_t(20.0) * iPow(q, 3) - scalar_t(12.0) * q)

@wp.func
def poly6_C_d(dim: wp.int32):
    if dim == 1: return scalar_t(35.0)/scalar_t(16.0)
    elif dim == 2: return scalar_t(35.0) / (scalar_t(32.0) * scalar_t(np.pi))
    else: return scalar_t(315.0) / (scalar_t(64.0) * scalar_t(np.pi))

@wp.func
def poly6_kernelScale(dim: wp.int32 = 2):
    return scalar_t(1.0)

@wp.func
def poly6_packingRatio():
    return scalar_t(1.0)