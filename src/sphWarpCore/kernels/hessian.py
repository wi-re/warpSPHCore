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
def sphKernelHessian_(x: vector(dtype=scalar_t, length=dim_t), h: scalar_t, kernel: wp.int32):
    r = vectorNorm_warp(x)
    dim = wp.int32(x.length)
    q = r / h
    eps = scalar_t(1e-5)
    
    k1 = eval_dkdq(q, dim, kernel)   * eval_C_d(dim, kernel) / iPow(h, dim + 1)
    k2 = eval_d2kdq2(q, dim, kernel) * eval_C_d(dim, kernel) / iPow(h, dim + 2)
    s = (iPow(r, 2) + iPow(eps,2) *iPow(h,2))
    
    factorA = wp.outer(x, x) / s
    if q < eps:
        for i in range(dim):
            factorA[i,i] = scalar_t(1.0)
    
    factorB = - wp.outer(x,x) /  (iPow(r, 3) + iPow(eps, 3) *iPow(h, 3))
    factorB += warp_eye(x) / (r + iPow(eps, 2) * h)
    
    hessian = factorA * k2 + factorB * k1
    return hessian
    
@wp.func
def sphKernelHessian(
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
        return (sphKernelHessian_(xij,hi,kernelProperties.kernelFunction) + sphKernelHessian_(xij,hj,kernelProperties.kernelFunction))/scalar_t(2.0)
    return sphKernelHessian_(xij, hij, kernelProperties.kernelFunction)
    