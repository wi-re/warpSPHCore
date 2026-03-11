import numpy as np
from ..utils import cpow_warp, iPow, bpow_warp
from typing import Any
import warp as wp
from warp.types import vector

@wp.func
def B7_k(q: wp.float32, dim: wp.int32 = 2):        
    return 56.0 * bpow_warp(q - 0.25, 7) - 28.0 * bpow_warp(q - 0.5, 7) + 8.0 * bpow_warp(q - 0.75, 7) - bpow_warp(q - 1.0, 7)

@wp.func
def B7_dkdq(q: wp.float32, dim: wp.int32 = 2):        
    return (56.0 * bpow_warp(q - 0.25, 6) - 28.0 * bpow_warp(q - 0.5, 6) + 8.0 * bpow_warp(q - 0.75, 6) - bpow_warp(q - 1.0, 6)) * 7.0

@wp.func
def B7_d2kdq2(q: wp.float32, dim: wp.int32 = 2):        
    return (56.0 * bpow_warp(q - 0.25, 5) - 28.0 * bpow_warp(q - 0.5, 5) + 8.0 * bpow_warp(q - 0.75, 5) - bpow_warp(q - 1.0, 5)) * 42.0

@wp.func
def B7_d3kdq3(q: wp.float32, dim: wp.int32 = 2):        
    return (56.0 * bpow_warp(q - 0.25, 4) - 28.0 * bpow_warp(q - 0.5, 4) + 8.0 * bpow_warp(q - 0.75, 4) - bpow_warp(q - 1.0, 4)) * 210.0

@wp.func
def B7_C_d(dim: wp.int32):
    if dim == 1: return 4096.0/315.0
    elif dim == 2: return 589824.0 / (7435.0 * np.pi)
    else: return 1024.0 / (105.0 * np.pi)

@wp.func # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations
def B7_kernelScale(dim: wp.int32 = 2):
    if dim == 1: return 2.121321
    elif dim == 2: return 2.158131
    else: return 2.195775

@wp.func
def B7_packingRatio(): # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations Table 2
    return 1.595 * 1.1425 # Factor to be in line CRKSPH