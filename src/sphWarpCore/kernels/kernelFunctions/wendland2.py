import numpy as np
from ..utils import cpow_warp, iPow
from typing import Any
import warp as wp
from warp.types import vector

@wp.func
def wendland2_k(q : wp.float32, dim: wp.int32 = 2):        
    if dim == 1:
        return cpow_warp(1.0 - q, 3) * (1.0 + 3.0 * q)
    else:
        return cpow_warp(1.0 - q, 4) * (1.0 + 4.0 * q)
@wp.func
def wendland2_dkdq(q : wp.float32, dim: wp.int32 = 2):        
    if dim == 1:
        return -12.0 * q * cpow_warp(1.0 - q, 2)
    else:
        return -20.0 * q * cpow_warp(1.0 - q, 3)
    
@wp.func
def wendland2_d2kdq2(q : wp.float32, dim: wp.int32 = 2):        
    if dim == 1:
        return -12.0 * (3.0 * iPow(q, 2) - 4.0 * q + 1)
    else:
        return 20.0 * (4.0 * q - 1.0) * cpow_warp(1.0 - q, 2)

@wp.func
def wendland2_d3kdq3(q : wp.float32, dim: wp.int32 = 2):
    if dim == 1:
        return 48.0 - 72.0 * q
    else:
        return 120.0 * (2.0 * iPow(q, 2) - 3.0 * q + 1.0)

@wp.func
def wendland2_C_d(dim : wp.int32):
    if dim == 1: return      5.0 / 4.0
    elif dim == 2: return    7.0 / np.pi
    else: return            21.0 / (2.0 * np.pi)

@wp.func
def wendland2_kernel(rij: wp.float32, hij: wp.float32, dim : wp.int32 = 2):
    return wendland2_k(rij, dim) * wendland2_C_d(dim) / iPow(hij, dim)
    
# formerly kernelGradient
@wp.func
def wendland2_gradient(rij: wp.float32, xij: vector(dtype=wp.float32, length=Any), hij: wp.float32, dim : wp.int32 = 2):
    return xij * (wendland2_dkdq(rij, dim) * wendland2_C_d(dim) / iPow(hij, dim + 1))[:,None]

#formerly kernelLaplacian
@wp.func
def wendland2_laplacian(rij: wp.float32, hij: wp.float32, dim : wp.int32 = 2):
    term_a = wendland2_dkdq(rij + 1e-7, dim) * hij
    term_b = wendland2_d2kdq2(rij + 1e-7, dim)

    fac = (dim - 1) / (rij * hij + 1e-7 * hij)

    return (fac * term_a + term_b) * wendland2_C_d(dim) / iPow(hij, dim + 2)

@wp.func # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations
def wendland2_kernelScale(dim: wp.int32 = 2):
    if dim == 1: return 1.620185
    elif dim == 2: return 1.897367
    else: return 1.936492


@wp.func
def wendland2_packingRatio(): # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations Table 2
    return 1.487 