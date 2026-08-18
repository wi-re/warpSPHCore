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
    if q > scalar_t(1.0):
        return scalar_t(0.0)

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


@wp.func
def sphGradientDkDh_(x: vector(dtype=scalar_t, length=dim_t), h: scalar_t, kernel: wp.int32):
    """d(sphGradient_)/dh: the mixed partial d(grad W)/dh, needed to propagate
    a support-length tangent through any operator built on sphGradient_
    (Gradient/Divergence/Curl/Laplacian's kernelTerm) rather than on the
    kernel value alone (which sphKernelDkDh_ already covers).

    Derivation: sphGradient_(x,h) = x_hat * K1(q,h), K1 = C_d*dkdq(q)/h^(dim+1),
    q = r/h, x_hat independent of h. Differentiating K1 w.r.t. h (product +
    chain rule through q(h) = r/h, dq/dh = -q/h) gives
        dK1/dh = -C_d/h^(dim+2) * (q*d2kdq2(q) + (dim+1)*dkdq(q))
    which collapses to sphKernelDkDh_'s own -C_d/h^(dim+2)*(dim*h*k + r*dkdq)
    under the analogous derivation one order down -- same technique, one
    derivative higher. Verified against wp.Tape (scripts/kernel_sanity_native.py
    Section J) rather than trusted from the derivation alone.
    """
    dim = wp.int32(x.length)
    r = vectorNorm_warp(x)
    q = r / h
    if q > scalar_t(1.0):
        return type(x)(scalar_t(0.0))

    direction = vectorNormalize_warp(input=x)
    dkdq = eval_dkdq(q, dim, kernel)
    d2kdq2 = eval_d2kdq2(q, dim, kernel)
    normConstant = - eval_C_d(dim, kernel) / iPow(h, dim + 2)

    return direction * (normConstant * (q * d2kdq2 + scalar_t(dim + 1) * dkdq))


@wp.func
def sphGradientDkDh(
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
        return (sphGradientDkDh_(xij,hi,kernelProperties.kernelFunction) + sphGradientDkDh_(xij,hj,kernelProperties.kernelFunction))/scalar_t(2.0)
    elif kernelProperties.supportMode == wp.static(SupportScheme.SuperSymmetric.value): # SuperSymmetric
        return (sphGradientDkDh_(xij,hi,kernelProperties.kernelFunction) - sphGradientDkDh_(-xij,hj,kernelProperties.kernelFunction))/scalar_t(2.0)
    return sphGradientDkDh_(xij, hij, kernelProperties.kernelFunction)
