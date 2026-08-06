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
def sphKernelDerivative_(x: vector(dtype=scalar_t, length=dim_t), h: scalar_t, kernel: wp.int32):
    dim = wp.int32(x.length)
    r = vectorNorm_warp(x)
    q = r / h
    if q > scalar_t(1.0):
        return scalar_t(0.0)
    return eval_dkdq(q, dim, kernel) * eval_C_d(dim, kernel) / iPow(h, dim + 1)

@wp.func
def sphKernelDerivative(
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
        return (sphKernelDerivative_(xij,hi,kernel) + sphKernelDerivative_(xij,hj,kernel))/scalar(2.0)
    return sphKernelDerivative_(xij, hij, kernel)
    