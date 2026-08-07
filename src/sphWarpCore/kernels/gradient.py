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
def sphGradient_(x: vector(dtype=scalar_t, length=dim_t), h: scalar_t, kernel: wp.int32):
    dim = wp.int32(x.length)
    r = vectorNorm_warp(x)
    q = r / h
    if q > scalar_t(1.0):
        return type(x)(scalar_t(0.0))
    grad = vectorNormalize_warp(input = x)
    normalizationTerm = eval_C_d(dim, kernel) / iPow(h, dim + 1)
    kernelTerm = eval_dkdq(q, dim, kernel)
    normalizedKernelTerm = kernelTerm * normalizationTerm
    return grad * normalizedKernelTerm



@wp.func
def sphKernelGradient(
    xi: vector(dtype=scalar_t, length=dim_t),
    xj: vector(dtype=scalar_t, length=dim_t),
    hi: scalar_t,
    hj: scalar_t,
    kernelProperties: kernelState,
    domainState: domainData,
):
    hij = computePairwiseSupport(hi, hj, kernelProperties.supportMode)
    xij = computeDistanceVec(xi, xj, domainState)
    if kernelProperties.supportMode == wp.static(SupportScheme.KernelMeanSymmetric.value): # KernelMeanSymmetric
        return (sphGradient_(xij,hi,kernelProperties.kernelFunction) + sphGradient_(xij,hj,kernelProperties.kernelFunction))/scalar_t(2.0)
    elif kernelProperties.supportMode == wp.static(SupportScheme.SuperSymmetric.value): # SuperSymmetric
        return (sphGradient_(xij,hi,kernelProperties.kernelFunction) - sphGradient_(-xij,hj,kernelProperties.kernelFunction))/scalar_t(2.0)
    return sphGradient_(xij, hij, kernelProperties.kernelFunction)


@wp.func
def sphKernelGradient_ij(
    xij: vector(dtype=scalar_t, length=dim_t),
    hi: scalar_t,
    hj: scalar_t,
    kernelProperties: kernelState,
    domainState: domainData,
):
    hij = computePairwiseSupport(hi, hj, kernelProperties.supportMode)
    if kernelProperties.supportMode == wp.static(SupportScheme.KernelMeanSymmetric.value): # KernelMeanSymmetric
        return (sphGradient_(xij,hi,kernelProperties.kernelFunction) + sphGradient_(xij,hj,kernelProperties.kernelFunction))/scalar_t(2.0)
    if kernelProperties.supportMode == wp.static(SupportScheme.SuperSymmetric.value): # SuperSymmetric
        return (sphGradient_(xij,hi,kernelProperties.kernelFunction) - sphGradient_(-xij,hj,kernelProperties.kernelFunction))/scalar_t(2.0)
    return sphGradient_(xij, hij, kernelProperties.kernelFunction)