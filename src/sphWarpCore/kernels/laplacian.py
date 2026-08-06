from typing import Any
from ..types import *
import warp as wp
from warp.types import vector, matrix
from .properties import eval_C_d
from .eval_kernel import *
import numpy as np
from ..math import *
from ..type_config import scalar_t, dim_t
from .kernelFunctions import *
from ..utils.support import computePairwiseSupport

@wp.func 
def sphKernelLaplacian_(x: vector(dtype=scalar_t, length=dim_t), h: scalar_t, kernel: wp.int32):
    dim = wp.int32(x.length)
    r = vectorNorm_warp(x)
    q = r / h
    eps = scalar_t(1e-5)
    r_eps = r + eps * h
    
    k1 = eval_dkdq(q, dim, kernel)   * eval_C_d(dim, kernel) / iPow(h, dim + 1)
    k2 = eval_d2kdq2(q, dim, kernel) * eval_C_d(dim, kernel) / iPow(h, dim + 2)
    
    s = wp.dot(x,x) / iPow(r_eps, 2)
    if q < eps:
        s = scalar_t(1.0)
    t = - wp.dot(x,x) / iPow(r_eps, 3)
    t += scalar_t(dim) / r_eps
    
    laplacian = s * k2 + t * k1
    if q < eps or q > scalar_t(1.0):
        laplacian = scalar_t(0.0)
    return laplacian

@wp.func
def sphKernelLaplacian(
    xi: vector(dtype=scalar_t, length=dim_t),
    xj: vector(dtype=scalar_t, length=dim_t),
    hi: scalar_t,
    hj: scalar_t,
    kernel: wp.int32,
    mode: wp.uint32,
    periodic: wp.array(dtype = wp.bool),
    minDomain: wp.array(dtype = scalar_t),
    maxDomain: wp.array(dtype = scalar_t),
):
    hij = computePairwiseSupport(hi, hj, mode)
    xij = computeDistanceVec(xi, xj, periodic, minDomain, maxDomain)
    if mode == wp.static(SupportScheme.SuperSymmetric.value): # SuperSymmetric
        return (sphKernelLaplacian_(xij,hi,kernel) + sphKernelLaplacian_(xij,hj,kernel))/scalar_t(2.0)
    
    return sphKernelLaplacian_(xij, hij, kernel)
