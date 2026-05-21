import numpy as np
from ...types import *
from ..utils import cpow_warp, iPow
from typing import Any
import warp as wp
from warp.types import vector

@wp.func
def wendland2_k(q : scalar_t, dim: wp.int32 = 2):        
    if dim == 1:
        return cpow_warp(scalar_t(1.0) - q, 3) * (scalar_t(1.0) + scalar_t(3.0) * q)
    else:
        return cpow_warp(scalar_t(1.0) - q, 4) * (scalar_t(1.0) + scalar_t(4.0) * q)
@wp.func
def wendland2_dkdq(q : scalar_t, dim: wp.int32 = 2):        
    if dim == 1:
        return scalar_t(-12.0) * q * cpow_warp(scalar_t(1.0) - q, 2)
    else:
        return scalar_t(-20.0) * q * cpow_warp(scalar_t(1.0) - q, 3)
    
@wp.func
def wendland2_d2kdq2(q : scalar_t, dim: wp.int32 = 2):        
    if dim == 1:
        return scalar_t(-12.0) * (scalar_t(3.0) * iPow(q, 2) - scalar_t(4.0) * q + scalar_t(1.0))
    else:
        return scalar_t(20.0) * (scalar_t(4.0) * q - scalar_t(1.0)) * cpow_warp(scalar_t(1.0) - q, 2)

@wp.func
def wendland2_d3kdq3(q : scalar_t, dim: wp.int32 = 2):
    if dim == 1:
        return scalar_t(48.0) - scalar_t(72.0) * q
    else:
        return scalar_t(120.0) * (scalar_t(2.0) * iPow(q, 2) - scalar_t(3.0) * q + scalar_t(1.0))

@wp.func
def wendland2_C_d(dim : wp.int32):
    if dim == 1: return      scalar_t(5.0) / scalar_t(4.0)
    elif dim == 2: return    scalar_t(7.0) / scalar_t(np.pi)
    else: return            scalar_t(21.0) / (scalar_t(2.0) * scalar_t(np.pi))

@wp.func # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations
def wendland2_kernelScale(dim: wp.int32 = 2):
    if dim == 1: return scalar_t(1.620185)
    elif dim == 2: return scalar_t(1.897367)
    else: return scalar_t(1.936492)

@wp.func
def wendland2_packingRatio(): # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations Table 2
    return scalar_t(1.487) 