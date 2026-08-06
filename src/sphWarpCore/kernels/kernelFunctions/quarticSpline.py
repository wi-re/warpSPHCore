import numpy as np
from ...math import cpow_warp, iPow, bpow_warp
from ...types import *
from typing import Any
import warp as wp
from warp.types import vector

@wp.func
def quarticSpline_k(q: scalar_t, dim: wp.int32 = 2):   
    return cpow_warp(scalar_t(1.0) - q, 4) - scalar_t(5.0) * cpow_warp(scalar_t(0.6) - q, 4) + scalar_t(10.0) * cpow_warp(scalar_t(0.2) - q, 4)

@wp.func
def quarticSpline_dkdq(q: scalar_t, dim: wp.int32 = 2):    
    return scalar_t(-4.0) * cpow_warp(scalar_t(1.0) - q, 3) + scalar_t(20.0) * cpow_warp(scalar_t(0.6) - q, 3) - scalar_t(40.0) * cpow_warp(scalar_t(0.2) - q, 3)

@wp.func
def quarticSpline_d2kdq2(q: scalar_t, dim: wp.int32 = 2):        
    return scalar_t(12.0) * cpow_warp(scalar_t(1.0) - q, 2) - scalar_t(60.0) * cpow_warp(scalar_t(0.6) - q, 2) + scalar_t(120.0) * cpow_warp(scalar_t(0.2) - q, 2)

@wp.func
def quarticSpline_d3kdq3(q: scalar_t, dim: wp.int32 = 2):
    return scalar_t(-24.0) * (scalar_t(1.0) - q) + scalar_t(120.0) * cpow_warp(scalar_t(0.6) - q, 1) - scalar_t(240.0) * cpow_warp(scalar_t(0.2) - q, 1)

@wp.func
def quarticSpline_C_d(dim: wp.int32):
    if dim == 1: return scalar_t(3125.0)/scalar_t(768.0)
    elif dim == 2: return scalar_t(46875.0) / (scalar_t(2398.0) * scalar_t(np.pi))
    else: return scalar_t(15625.0) / (scalar_t(512.0) * scalar_t(np.pi))

@wp.func # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations
def quarticSpline_kernelScale(dim: wp.int32 = 2):
    if dim == 1: return scalar_t(1.936492)
    elif dim == 2: return scalar_t(1.977173)
    else: return scalar_t(2.018932)

@wp.func
def quarticSpline_packingRatio(): # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations Table 2
    return scalar_t(1.203)