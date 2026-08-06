import numpy as np
from ...math import cpow_warp, iPow, bpow_warp
from ...types import *
from typing import Any
import warp as wp
from warp.types import vector

@wp.func
def B7_k(q: scalar_t, dim: wp.int32 = 2):        
    return scalar_t(56.0) * bpow_warp(q - scalar_t(0.25), 7) - scalar_t(28.0) * bpow_warp(q - scalar_t(0.5), 7) + scalar_t(8.0) * bpow_warp(q - scalar_t(0.75), 7) - bpow_warp(q - scalar_t(1.0), 7)

@wp.func
def B7_dkdq(q: scalar_t, dim: wp.int32 = 2):        
    return (scalar_t(56.0) * bpow_warp(q - scalar_t(0.25), 6) - scalar_t(28.0) * bpow_warp(q - scalar_t(0.5), 6) + scalar_t(8.0) * bpow_warp(q - scalar_t(0.75), 6) - bpow_warp(q - scalar_t(1.0), 6)) * scalar_t(7.0)

@wp.func
def B7_d2kdq2(q: scalar_t, dim: wp.int32 = 2):        
    return (scalar_t(56.0) * bpow_warp(q - scalar_t(0.25), 5) - scalar_t(28.0) * bpow_warp(q - scalar_t(0.5), 5) + scalar_t(8.0) * bpow_warp(q - scalar_t(0.75), 5) - bpow_warp(q - scalar_t(1.0), 5)) * scalar_t(42.0)

@wp.func
def B7_d3kdq3(q: scalar_t, dim: wp.int32 = 2):        
    return (scalar_t(56.0) * bpow_warp(q - scalar_t(0.25), 4) - scalar_t(28.0) * bpow_warp(q - scalar_t(0.5), 4) + scalar_t(8.0) * bpow_warp(q - scalar_t(0.75), 4) - bpow_warp(q - scalar_t(1.0), 4)) * scalar_t(210.0)

@wp.func
def B7_C_d(dim: wp.int32):
    if dim == 1: return scalar_t(4096.0)/scalar_t(315.0)
    elif dim == 2: return scalar_t(589824.0) / (scalar_t(7435.0) * scalar_t(np.pi))
    else: return scalar_t(1024.0) / (scalar_t(105.0) * scalar_t(np.pi))

@wp.func # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations
def B7_kernelScale(dim: wp.int32 = 2):
    if dim == 1: return scalar_t(2.121321)
    elif dim == 2: return scalar_t(2.158131)
    else: return scalar_t(2.195775)

@wp.func
def B7_packingRatio(): # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations Table 2
    return scalar_t(1.595) * scalar_t(1.1425) # Factor to be in line CRKSPH