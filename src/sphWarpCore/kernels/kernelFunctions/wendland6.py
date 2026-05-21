import numpy as np
from ..utils import cpow_warp, iPow, bpow_warp
from ...types import *
from typing import Any
import warp as wp
from warp.types import vector

@wp.func
def wendland6_k(q: scalar_t, dim: wp.int32 = 2):        
    if dim == 1:
        return cpow_warp(scalar_t(1.0) - q, 7) * (scalar_t(1.0) + scalar_t(7.0) * q + scalar_t(19.0) * iPow(q, 2) + scalar_t(21.0) * iPow(q, 3))
    else:
        return cpow_warp(scalar_t(1.0) - q, 8) * (scalar_t(1.0) + scalar_t(8.0) * q + scalar_t(25.0) * iPow(q, 2) + scalar_t(32.0) * iPow(q, 3))

@wp.func
def wendland6_dkdq(q: scalar_t, dim: wp.int32 = 2):        
    if dim == 1:
        return scalar_t(-6.0) * q * (scalar_t(35.0) * iPow(q, 2) + scalar_t(18.0) * q + scalar_t(3.0)) * cpow_warp(scalar_t(1.0) - q, 6)
    else:
        return scalar_t(-22.0) * q * (scalar_t(16.0) * iPow(q, 2) + scalar_t(7.0) * q + scalar_t(1.0)) * cpow_warp(scalar_t(1.0) - q, 7)

@wp.func
def wendland6_d2kdq2(q: scalar_t, dim: wp.int32 = 2):        
    if dim == 1:
        return scalar_t(18.0) * (scalar_t(105.0) * iPow(q, 3) + scalar_t(13.0) * iPow(q, 2) - scalar_t(5.0) * q - scalar_t(1.0)) * cpow_warp(scalar_t(1.0) - q, 5)
    else:
        return scalar_t(22.0) * (scalar_t(160.0) * iPow(q, 3) + scalar_t(15.0) * iPow(q, 2) - scalar_t(6.0) * q - scalar_t(1.0)) * cpow_warp(scalar_t(1.0) - q, 6)

@wp.func
def wendland6_d3kdq3(q: scalar_t, dim: wp.int32 = 2):
    if dim == 1:
        return scalar_t(-1008.0) * q * (scalar_t(15.0) * iPow(q, 2) - scalar_t(4.0) * q - scalar_t(1.0)) * cpow_warp(scalar_t(1.0) - q, 4)
    else:
        return scalar_t(-1584.0) * q * (scalar_t(20.0) * iPow(q, 2) - scalar_t(5.0) * q - scalar_t(1.0)) * cpow_warp(scalar_t(1.0) - q, 5)

@wp.func
def wendland6_C_d(dim: wp.int32):
    if dim == 1: return scalar_t(55.0)/scalar_t(32.0)
    elif dim == 2: return scalar_t(78.0) / (scalar_t(7.0) * scalar_t(np.pi))
    else: return scalar_t(1365.0) / (scalar_t(64.0) * scalar_t(np.pi))

@wp.func # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations
def wendland6_kernelScale(dim: wp.int32 = 2):
    if dim == 1: return scalar_t(2.207940)
    elif dim == 2: return scalar_t(2.415230)
    else: return scalar_t(2.449490)

@wp.func
def wendland6_packingRatio(): # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations Table 2
    return scalar_t(1.866)