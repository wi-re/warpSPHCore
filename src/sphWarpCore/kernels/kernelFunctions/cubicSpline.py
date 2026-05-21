import numpy as np
from ..utils import cpow_warp, iPow, bpow_warp
from ...types import *
from typing import Any
import warp as wp
from warp.types import vector

@wp.func
def cubicSpline_k(q: scalar_t, dim: wp.int32 = 2):  
    return cpow_warp(scalar_t(1.0) - q, 3) - scalar_t(4.0) * cpow_warp(scalar_t(0.5) - q, 3)

@wp.func
def cubicSpline_dkdq(q: scalar_t, dim: wp.int32 = 2):    
    return scalar_t(-3.0) * cpow_warp(scalar_t(1.0) - q, 2) + scalar_t(12.0) * cpow_warp(scalar_t(0.5) - q, 2)

@wp.func
def cubicSpline_d2kdq2(q: scalar_t, dim: wp.int32 = 2):        
    if q >= scalar_t(0.5):
        return scalar_t(6.0) * (scalar_t(1.0) - q)
    else:
        return scalar_t(6.0) * (scalar_t(1.0) - q) - scalar_t(24.0) * (scalar_t(0.5) - q)

@wp.func
def cubicSpline_d3kdq3(q: scalar_t, dim: wp.int32 = 2):
    if q >= scalar_t(0.5):
        return scalar_t(-6.0)
    else:
        return scalar_t(18.0)

@wp.func
def cubicSpline_C_d(dim: wp.int32):
    if dim == 1: return scalar_t(8.0)/scalar_t(3.0)
    elif dim == 2: return scalar_t(80.0) / (scalar_t(7.0) * scalar_t(np.pi))
    else: return scalar_t(16.0) / scalar_t(np.pi)

@wp.func # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations
def cubicSpline_kernelScale(dim: wp.int32 = 2):
    if dim == 1: return scalar_t(1.732051)
    elif dim == 2: return scalar_t(1.778002)
    else: return scalar_t(1.825742)

@wp.func
def cubicSpline_packingRatio(): # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations Table 2
    return scalar_t(1.292) * scalar_t(1.0175) # Correction to match DJ Price 2012 with scalar_t(57.9) neighbors in 3D see page 776
    # return scalar_t(1.292) # scalar_t(1.181) for Astrophysics