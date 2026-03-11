import numpy as np
from ..utils import cpow_warp, iPow, bpow_warp
from typing import Any
import warp as wp
from warp.types import vector

@wp.func
def quinticSpline_k(q: wp.float32, dim: wp.int32 = 2):        
    return cpow_warp(1.0 - q, 5) - 6.0 * cpow_warp(2.0/3.0 - q, 5) + 15.0 * cpow_warp(1.0/3.0 - q, 5)

@wp.func
def quinticSpline_dkdq(q: wp.float32, dim: wp.int32 = 2):     
    return -5.0 * cpow_warp(1.0 - q, 4) + 30.0 * cpow_warp(2.0/3.0 - q, 4) - 75.0 * cpow_warp(1.0/3.0 - q, 4)

@wp.func
def quinticSpline_d2kdq2(q: wp.float32, dim: wp.int32 = 2):        
    return 20.0 * cpow_warp(1.0 - q, 3) - 120.0 * cpow_warp(2.0/3.0 - q, 3) + 300.0 * cpow_warp(1.0/3.0 - q, 3)

@wp.func
def quinticSpline_d3kdq3(q: wp.float32, dim: wp.int32 = 2):
    return -60.0 * cpow_warp(1.0 - q, 2) + 360.0 * cpow_warp(2.0/3.0 - q, 2) - 900.0 * cpow_warp(1.0/3.0 - q, 2)

@wp.func
def quinticSpline_C_d(dim: wp.int32):
    if dim == 1: return 243.0/40.0
    elif dim == 2: return 15309.0 / (478.0 * np.pi)
    else: return 2187.0 / (40.0 * np.pi)

@wp.func # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations
def quinticSpline_kernelScale(dim: wp.int32 = 2):
    if dim == 1: return 2.121321
    elif dim == 2: return 2.158131
    else: return 2.195775

@wp.func
def quinticSpline_packingRatio(): # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations Table 2
    return 1.595 * 1.1425 # Factor to be in line CRKSPH