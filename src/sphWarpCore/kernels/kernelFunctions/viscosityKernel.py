import numpy as np
from ...math import cpow_warp, iPow, bpow_warp
from ...type_config import *
from typing import Any
import warp as wp
from warp.types import vector

@wp.func
def viscosityKernel_k(q: scalar_t, dim: wp.int32 = 2):  
    return scalar_t(-0.5) * iPow(q, 3) + iPow(q, 2) + scalar_t(0.5) / q - scalar_t(1.0)

@wp.func
def viscosityKernel_dkdq(q: scalar_t, dim: wp.int32 = 2):    
    return scalar_t(-1.5) * iPow(q, 2) - scalar_t(0.5) / iPow(q, 2) + scalar_t(2.0) * q

@wp.func
def viscosityKernel_d2kdq2(q: scalar_t, dim: wp.int32 = 2):        
    return scalar_t(1.0) / iPow(q, 3) - scalar_t(3.0) * q + scalar_t(2.0)

@wp.func
def viscosityKernel_d3kdq3(q: scalar_t, dim: wp.int32 = 2):
    return scalar_t(-3.0) / iPow(q, 4) - scalar_t(3.0)

@wp.func
def viscosityKernel_C_d(dim: wp.int32):
    if dim == 1: return scalar_t(15.0) / (scalar_t(2.0) * scalar_t(np.pi))
    elif dim == 2: return scalar_t(10.0) / (scalar_t(9.0) * scalar_t(np.pi))
    else: return scalar_t(15.0) / (scalar_t(2.0) * scalar_t(np.pi))

@wp.func
def viscosityKernel_kernelScale(dim: wp.int32 = 2):
    return scalar_t(1.0)

@wp.func
def viscosityKernel_packingRatio():
    return scalar_t(1.0)