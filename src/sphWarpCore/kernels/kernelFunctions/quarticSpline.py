import numpy as np
from ..utils import cpow_warp, iPow, bpow_warp
from typing import Any
import warp as wp
from warp.types import vector

@wp.func
def quarticSpline_k(q: wp.float32, dim: wp.int32 = 2):   
    return cpow_warp(1.0 - q, 4) - 5.0 * cpow_warp(0.6 - q, 4) + 10.0 * cpow_warp(0.2 - q, 4)

@wp.func
def quarticSpline_dkdq(q: wp.float32, dim: wp.int32 = 2):    
    return -4.0 * cpow_warp(1.0 - q, 3) + 20.0 * cpow_warp(0.6 - q, 3) - 40.0 * cpow_warp(0.2 - q, 3)

@wp.func
def quarticSpline_d2kdq2(q: wp.float32, dim: wp.int32 = 2):        
    return 12.0 * cpow_warp(1.0 - q, 2) - 60.0 * cpow_warp(0.6 - q, 2) + 120.0 * cpow_warp(0.2 - q, 2)

@wp.func
def quarticSpline_d3kdq3(q: wp.float32, dim: wp.int32 = 2):
    return -24.0 * (1.0 - q) + 120.0 * cpow_warp(0.6 - q, 1) - 240.0 * cpow_warp(0.2 - q, 1)

@wp.func
def quarticSpline_C_d(dim: wp.int32):
    if dim == 1: return 3125.0/768.0
    elif dim == 2: return 46875.0 / (2398.0 * np.pi)
    else: return 15625.0 / (512.0 * np.pi)

@wp.func # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations
def quarticSpline_kernelScale(dim: wp.int32 = 2):
    if dim == 1: return 1.936492
    elif dim == 2: return 1.977173
    else: return 2.018932

@wp.func
def quarticSpline_packingRatio(): # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations Table 2
    return 1.203