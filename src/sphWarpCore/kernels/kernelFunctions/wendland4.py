import numpy as np
from ...math import cpow_warp, iPow, bpow_warp
from ...types import *
from typing import Any
import warp as wp
from warp.types import vector

@wp.func
def wendland4_k(q: scalar_t, dim: wp.int32 = 2):        
    if dim == 1:
        return cpow_warp(scalar_t(1.0) - q, 5) * (scalar_t(1.0) + scalar_t(5.0) * q + scalar_t(8.0) * iPow(q, 2))
    else:
        return cpow_warp(scalar_t(1.0) - q, 6) * (scalar_t(1.0) + scalar_t(6.0) * q + scalar_t(35.0)/scalar_t(3.0) * iPow(q, 2))

@wp.func
def wendland4_dkdq(q: scalar_t, dim: wp.int32 = 2):        
    if dim == 1:
        return scalar_t(-14.0) * q * (scalar_t(4.0) * q + scalar_t(1.0)) * cpow_warp(scalar_t(1.0) - q, 4)
    else:
        return scalar_t(-56.0)/scalar_t(3.0) * q * (scalar_t(5.0) * q + scalar_t(1.0)) * cpow_warp(scalar_t(1.0) - q, 5)

@wp.func
def wendland4_d2kdq2(q: scalar_t, dim: wp.int32 = 2):        
    if dim == 1:
        return scalar_t(14.0) * (scalar_t(24.0) * iPow(q, 2) - scalar_t(3.0) * q - scalar_t(1.0)) * cpow_warp(scalar_t(1.0) - q, 3)
    else:
        return scalar_t(56.0)/scalar_t(3.0) * (scalar_t(35.0) * iPow(q, 2) - scalar_t(4.0) * q - scalar_t(1.0)) * cpow_warp(scalar_t(1.0) - q, 4)

@wp.func
def wendland4_d3kdq3(q: scalar_t, dim: wp.int32 = 2):
    if dim == 1:
        return scalar_t(-840.0) * q * (scalar_t(2.0) * q - scalar_t(1.0)) * cpow_warp(scalar_t(1.0) - q, 2)
    else:
        return scalar_t(-560.0) * q * (scalar_t(7.0) * q - scalar_t(3.0)) * cpow_warp(scalar_t(1.0) - q, 3)

@wp.func
def wendland4_C_d(dim: wp.int32):
    if dim == 1: return scalar_t(3.0)/scalar_t(2.0)
    elif dim == 2: return scalar_t(9.0) / scalar_t(np.pi)
    else: return scalar_t(495.0) / (scalar_t(32.0) * scalar_t(np.pi))

@wp.func # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations
def wendland4_kernelScale(dim: wp.int32 = 2):
    if dim == 1: return scalar_t(1.936492)
    elif dim == 2: return scalar_t(2.171239)
    else: return scalar_t(2.207940)

@wp.func
def wendland4_packingRatio(): # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations Table 2
    return scalar_t(1.643)