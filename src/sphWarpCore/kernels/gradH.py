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
def sphKernelDkDh_(x: vector(dtype=scalar_t, length=dim_t), h: scalar_t, kernel: wp.int32):
    dim = wp.int32(x.length)
    r = vectorNorm_warp(x)
    q = r/h
    
    k = eval_k(q, dim, kernel)
    dkdq = eval_dkdq(q, dim, kernel)
    
    normConstant = - eval_C_d(dim, kernel) / iPow(h, dim + 2)
    
    return normConstant * (scalar_t(dim) * h * k + r * dkdq)

@wp.func
def sphKernelDkDh(
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
        return (sphKernelDkDh_(xij,hi,kernel) + sphKernelDkDh_(xij,hj,kernel))/scalar_t(2.0)
    
    return sphKernelDkDh_(xij, hij, kernel)
    