import numpy as np
from ...math import cpow_warp, iPow, bpow_warp
from ...type_config import *
from typing import Any
import warp as wp
from warp.types import vector

@wp.func
def spiky_k(q: scalar_t, dim: wp.int32 = 2):  
    return cpow_warp(scalar_t(1.0) - q, 3)

@wp.func
def spiky_dkdq(q: scalar_t, dim: wp.int32 = 2):    
    return scalar_t(-3.0) * cpow_warp(scalar_t(1.0) - q, 2)

@wp.func
def spiky_d2kdq2(q: scalar_t, dim: wp.int32 = 2):        
    return scalar_t(6.0) * (scalar_t(1.0) - q)

@wp.func
def spiky_d3kdq3(q: scalar_t, dim: wp.int32 = 2):
    return scalar_t(-6.0)

@wp.func
def spiky_C_d(dim: wp.int32):
    if dim == 1: return scalar_t(2.0)
    elif dim == 2: return scalar_t(10.0) / scalar_t(np.pi)
    else: return scalar_t(15.0) / scalar_t(np.pi)

@wp.func
def spiky_kernelScale(dim: wp.int32 = 2):
    return scalar_t(1.0)

@wp.func
def spiky_packingRatio():
    return scalar_t(1.0)