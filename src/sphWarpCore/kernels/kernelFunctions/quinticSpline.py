import numpy as np
from ...math import cpow_warp, iPow, bpow_warp
from ...types import *
from typing import Any
import warp as wp
from warp.types import vector

@wp.func
def quinticSpline_k(q: scalar_t, dim: wp.int32 = 2):        
    return cpow_warp(scalar_t(scalar_t(1.0)) - q, 5) - scalar_t(scalar_t(6.0)) * cpow_warp(scalar_t(scalar_t(2.0))/scalar_t(scalar_t(3.0)) - q, 5) + scalar_t(scalar_t(15.0)) * cpow_warp(scalar_t(scalar_t(1.0))/scalar_t(scalar_t(3.0)) - q, 5)

@wp.func
def quinticSpline_dkdq(q: scalar_t, dim: wp.int32 = 2):     
    return scalar_t(scalar_t(-5.0)) * cpow_warp(scalar_t(scalar_t(1.0)) - q, 4) + scalar_t(scalar_t(30.0)) * cpow_warp(scalar_t(scalar_t(2.0))/scalar_t(scalar_t(3.0)) - q, 4) - scalar_t(scalar_t(75.0)) * cpow_warp(scalar_t(scalar_t(1.0))/scalar_t(scalar_t(3.0)) - q, 4)

@wp.func
def quinticSpline_d2kdq2(q: scalar_t, dim: wp.int32 = 2):        
    return scalar_t(scalar_t(20.0)) * cpow_warp(scalar_t(scalar_t(1.0)) - q, 3) - scalar_t(scalar_t(120.0)) * cpow_warp(scalar_t(scalar_t(2.0))/scalar_t(scalar_t(3.0)) - q, 3) + scalar_t(scalar_t(300.0)) * cpow_warp(scalar_t(scalar_t(1.0))/scalar_t(scalar_t(3.0)) - q, 3)

@wp.func
def quinticSpline_d3kdq3(q: scalar_t, dim: wp.int32 = 2):
    return scalar_t(scalar_t(-60.0)) * cpow_warp(scalar_t(scalar_t(1.0)) - q, 2) + scalar_t(scalar_t(360.0)) * cpow_warp(scalar_t(scalar_t(2.0))/scalar_t(scalar_t(3.0)) - q, 2) - scalar_t(scalar_t(900.0)) * cpow_warp(scalar_t(scalar_t(1.0))/scalar_t(scalar_t(3.0)) - q, 2)

@wp.func
def quinticSpline_C_d(dim: wp.int32):
    if dim == 1: return scalar_t(scalar_t(243.0))/scalar_t(scalar_t(40.0))
    elif dim == 2: return scalar_t(scalar_t(15309.0)) / (scalar_t(scalar_t(478.0)) * scalar_t(np.pi))
    else: return scalar_t(scalar_t(2187.0)) / (scalar_t(scalar_t(40.0)) * scalar_t(np.pi))

@wp.func # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations
def quinticSpline_kernelScale(dim: wp.int32 = 2):
    if dim == 1: return scalar_t(scalar_t(2.121321))
    elif dim == 2: return scalar_t(scalar_t(2.158131))
    else: return scalar_t(scalar_t(2.195775))

@wp.func
def quinticSpline_packingRatio(): # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations Table 2
    return scalar_t(scalar_t(1.595)) * scalar_t(scalar_t(1.1425)) # Factor to be in line CRKSPH