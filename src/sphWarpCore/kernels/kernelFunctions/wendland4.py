import numpy as np
from ..utils import cpow_warp, iPow, bpow_warp
from typing import Any
import warp as wp
from warp.types import vector

@wp.func
def wendland4_k(q: wp.float32, dim: wp.int32 = 2):        
    if dim == 1:
        return cpow_warp(1.0 - q, 5) * (1.0 + 5.0 * q + 8.0 * iPow(q, 2))
    else:
        return cpow_warp(1.0 - q, 6) * (1.0 + 6.0 * q + 35.0/3.0 * iPow(q, 2))

@wp.func
def wendland4_dkdq(q: wp.float32, dim: wp.int32 = 2):        
    if dim == 1:
        return -14.0 * q * (4.0 * q + 1.0) * cpow_warp(1.0 - q, 4)
    else:
        return -56.0/3.0 * q * (5.0 * q + 1.0) * cpow_warp(1.0 - q, 5)

@wp.func
def wendland4_d2kdq2(q: wp.float32, dim: wp.int32 = 2):        
    if dim == 1:
        return 14.0 * (24.0 * iPow(q, 2) - 3.0 * q - 1.0) * cpow_warp(1.0 - q, 3)
    else:
        return 56.0/3.0 * (35.0 * iPow(q, 2) - 4.0 * q - 1.0) * cpow_warp(1.0 - q, 4)

@wp.func
def wendland4_d3kdq3(q: wp.float32, dim: wp.int32 = 2):
    if dim == 1:
        return -840.0 * q * (2.0 * q - 1.0) * cpow_warp(1.0 - q, 2)
    else:
        return -560.0 * q * (7.0 * q - 3.0) * cpow_warp(1.0 - q, 3)

@wp.func
def wendland4_C_d(dim: wp.int32):
    if dim == 1: return 3.0/2.0
    elif dim == 2: return 9.0 / np.pi
    else: return 495.0 / (32.0 * np.pi)

@wp.func # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations
def wendland4_kernelScale(dim: wp.int32 = 2):
    if dim == 1: return 1.936492
    elif dim == 2: return 2.171239
    else: return 2.207940

@wp.func
def wendland4_packingRatio(): # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations Table 2
    return 1.643