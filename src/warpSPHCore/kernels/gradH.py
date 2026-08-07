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
from ..dataTypes.domain_t import domainData
from ..dataTypes.kernelState_t import kernelState

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
    kernelProperties: kernelState,
    domainState: domainData,
):
    hij = computePairwiseSupport(hi, hj, kernelProperties.supportMode)
    xij = computeDistanceVec(xi, xj, domainState)
    if kernelProperties.supportMode == wp.static(SupportScheme.SuperSymmetric.value): # SuperSymmetric
        return (sphKernelDkDh_(xij,hi,kernelProperties.kernelFunction) + sphKernelDkDh_(xij,hj,kernelProperties.kernelFunction))/scalar_t(2.0)

    return sphKernelDkDh_(xij, hij, kernelProperties.kernelFunction)
    