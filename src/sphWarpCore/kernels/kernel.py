from typing import Any
from ..type_config import *
import warp as wp
from warp.types import vector, matrix
from .properties import eval_C_d
from .eval_kernel import *
import numpy as np
from ..math import *
from ..type_config import scalar_t, dim_t
from .kernelFunctions import *
from ..util.support import computePairwiseSupport

@wp.func
def sphKernel_(x: vector(dtype=scalar_t, length=dim_t), h: scalar_t, kernel: wp.int32):
    dim = wp.int32(x.length)
    r = vectorNorm_warp(x)
    # r = safe_sqrt(wp.dot(x,x))
    q = r / h
    if q > scalar_t(1.0):
        return scalar_t(0.0)
    return eval_k(q, dim, kernel) * eval_C_d(dim, kernel) / iPow(h, dim)

@wp.func
def sphKernel(
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
    if mode == wp.static(SupportScheme.KernelMeanSymmetric.value) or mode == wp.static(SupportScheme.SuperSymmetric.value): # KernelMeanSymmetric or SuperSymmetric
        return (sphKernel_(xij,hi,kernel) + sphKernel_(xij,hj,kernel))/scalar(2.0)
    return sphKernel_(xij, hij, kernel)

@wp.func
def sphKernel_ij(
    xij: vector(dtype=scalar_t, length=dim_t),
    hi: scalar_t,
    hj: scalar_t,
    kernel: wp.int32,
    mode: wp.uint32,
    periodic: wp.array(dtype = wp.bool),
    minDomain: wp.array(dtype = scalar_t),
    maxDomain: wp.array(dtype = scalar_t),
):
    hij = computePairwiseSupport(hi, hj, mode)
    if mode == wp.static(SupportScheme.KernelMeanSymmetric.value) or mode == wp.static(SupportScheme.SuperSymmetric.value): # KernelMeanSymmetric or SuperSymmetric
        return (sphKernel_(xij,hi,kernel) + sphKernel_(xij,hj,kernel))/scalar(2.0)
    return sphKernel_(xij, hij, kernel)
