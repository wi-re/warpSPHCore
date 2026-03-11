import numpy as np
from ..utils import cpow_warp, iPow, bpow_warp
from typing import Any
import warp as wp
from warp.types import vector

@wp.func
def cubicSpline_k(q: wp.float32, dim: wp.int32 = 2):  
    return cpow_warp(1.0 - q, 3) - 4.0 * cpow_warp(0.5 - q, 3)

@wp.func
def cubicSpline_dkdq(q: wp.float32, dim: wp.int32 = 2):    
    return -3.0 * cpow_warp(1.0 - q, 2) + 12.0 * cpow_warp(0.5 - q, 2)

@wp.func
def cubicSpline_d2kdq2(q: wp.float32, dim: wp.int32 = 2):        
    if q >= 0.5:
        return 6.0 * (1.0 - q)
    else:
        return 6.0 * (1.0 - q) - 24.0 * (0.5 - q)

@wp.func
def cubicSpline_d3kdq3(q: wp.float32, dim: wp.int32 = 2):
    if q >= 0.5:
        return -6.0
    else:
        return 18.0

@wp.func
def cubicSpline_C_d(dim: wp.int32):
    if dim == 1: return 8.0/3.0
    elif dim == 2: return 80.0 / (7.0 * np.pi)
    else: return 16.0 / np.pi

@wp.func # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations
def cubicSpline_kernelScale(dim: wp.int32 = 2):
    if dim == 1: return 1.732051
    elif dim == 2: return 1.778002
    else: return 1.825742

@wp.func
def cubicSpline_packingRatio(): # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations Table 2
    return 1.292 * 1.0175 # Correction to match DJ Price 2012 with 57.9 neighbors in 3D see page 776
    # return 1.292 # 1.181 for Astrophysics