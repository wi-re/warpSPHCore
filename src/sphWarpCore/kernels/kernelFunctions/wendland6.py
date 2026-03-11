import numpy as np
from ..utils import cpow_warp, iPow, bpow_warp
from typing import Any
import warp as wp
from warp.types import vector

@wp.func
def wendland6_k(q: wp.float32, dim: wp.int32 = 2):        
    if dim == 1:
        return cpow_warp(1.0 - q, 7) * (1.0 + 7.0 * q + 19.0 * iPow(q, 2) + 21.0 * iPow(q, 3))
    else:
        return cpow_warp(1.0 - q, 8) * (1.0 + 8.0 * q + 25.0 * iPow(q, 2) + 32.0 * iPow(q, 3))

@wp.func
def wendland6_dkdq(q: wp.float32, dim: wp.int32 = 2):        
    if dim == 1:
        return -6.0 * q * (35.0 * iPow(q, 2) + 18.0 * q + 3.0) * cpow_warp(1.0 - q, 6)
    else:
        return -22.0 * q * (16.0 * iPow(q, 2) + 7.0 * q + 1.0) * cpow_warp(1.0 - q, 7)

@wp.func
def wendland6_d2kdq2(q: wp.float32, dim: wp.int32 = 2):        
    if dim == 1:
        return 18.0 * (105.0 * iPow(q, 3) + 13.0 * iPow(q, 2) - 5.0 * q - 1.0) * cpow_warp(1.0 - q, 5)
    else:
        return 22.0 * (160.0 * iPow(q, 3) + 15.0 * iPow(q, 2) - 6.0 * q - 1.0) * cpow_warp(1.0 - q, 6)

@wp.func
def wendland6_d3kdq3(q: wp.float32, dim: wp.int32 = 2):
    if dim == 1:
        return -1008.0 * q * (15.0 * iPow(q, 2) - 4.0 * q - 1.0) * cpow_warp(1.0 - q, 4)
    else:
        return -1584.0 * q * (20.0 * iPow(q, 2) - 5.0 * q - 1.0) * cpow_warp(1.0 - q, 5)

@wp.func
def wendland6_C_d(dim: wp.int32):
    if dim == 1: return 55.0/32.0
    elif dim == 2: return 78.0 / (7.0 * np.pi)
    else: return 1365.0 / (64.0 * np.pi)

@wp.func # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations
def wendland6_kernelScale(dim: wp.int32 = 2):
    if dim == 1: return 2.207940
    elif dim == 2: return 2.415230
    else: return 2.449490

@wp.func
def wendland6_packingRatio(): # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations Table 2
    return 1.866