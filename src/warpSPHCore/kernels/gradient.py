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
from .hessian import sphKernelHessian_
from .gradH import sphGradientDkDh_

# `sphGradient_` needs a custom reverse-mode adjoint (see adj_sphGradient_'s
# docstring below), and Warp's @wp.func_grad rejects functions with a
# dim_t-generic vector argument ("functions with generic input arguments are
# not yet supported") -- the exact same constraint math/wp_normalize.py's
# vectorNormalize_warp/vectorNorm_warp already work around, by splitting into
# concrete-length _1D/_2D/_3D implementations (each individually eligible for
# a custom grad, bodies duplicated rather than factored into a shared plain-
# Python helper -- Warp requires every callee reached from a @wp.func to
# itself be a @wp.func, so a plain Python helper isn't callable from here) and
# dispatching to them from a dim_t-generic wrapper that carries no adjoint of
# its own -- Warp resolves the dispatcher's overload purely from the concrete
# argument type at every call site, so the wrapper adds no extra indirection.
@wp.func
def sphGradient_1D(x: vector(dtype=scalar_t, length=1), h: scalar_t, kernel: wp.int32):
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
def sphGradient_2D(x: vector(dtype=scalar_t, length=2), h: scalar_t, kernel: wp.int32):
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
def sphGradient_3D(x: vector(dtype=scalar_t, length=3), h: scalar_t, kernel: wp.int32):
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


# Warp's automatically-composed adjoint of sphGradient_{1,2,3}D (chaining
# through vectorNormalize_warp's own hand-derived, math/wp_normalize.py
# adjoint, which is built on norm_hess_warp's eps-regularized Hessian-of-|x|)
# is correct for r > 0 but wrong at an exact self-pair (x == 0, r == 0):
# norm_hess_warp's regularized value blows up like O(1/eps) there, and gets
# multiplied by kernelTerm(q=0), which is exactly 0.0 in floating point for a
# smooth kernel's derivative at its peak -- "huge * 0.0" silently collapses to
# 0.0 instead of the true finite limit (a removable 0-times-infinity
# singularity, like sin(x)/x at x=0).
#
# sphKernelHessian_/sphGradientDkDh_ (kernels/hessian.py, kernels/gradH.py)
# already derive this function's true Jacobian in closed form, with an
# explicit near-origin branch (sphKernelHessian_'s `if q < eps` case) that
# gets the r=0 limit right -- validated independently at r=0 (see
# coreOperations/wp_densityHVP.py's docstring: sphKernelHessian_'s value there
# is finite, physically meaningful -- the kernel's own curvature at its peak,
# not noise) and at r>0 (kernel_sanity_native.py Section 6/8, both checked
# against wp.Tape's automatic derivative of this very function). Using them
# directly here replaces the buggy automatic composition with the already-
# validated closed form everywhere this function's gradient is taken, rather
# than patching norm_hess_warp itself (whose Jacobian-of-normalize is
# genuinely, unavoidably direction-dependent at x=0 in dim > 1 -- only the
# specific combination sphKernelHessian_ computes is direction-independent in
# the limit).
@wp.func_grad(sphGradient_1D)
def adj_sphGradient_1D(x: vector(dtype=scalar_t, length=1), h: scalar_t, kernel: wp.int32, adj_ret: vector(dtype=scalar_t, length=1)):
    H = sphKernelHessian_(x, h, kernel)
    dGdh = sphGradientDkDh_(x, h, kernel)
    wp.adjoint[x] += matmul(H, adj_ret)
    wp.adjoint[h] += wp.dot(dGdh, adj_ret)
@wp.func_grad(sphGradient_2D)
def adj_sphGradient_2D(x: vector(dtype=scalar_t, length=2), h: scalar_t, kernel: wp.int32, adj_ret: vector(dtype=scalar_t, length=2)):
    H = sphKernelHessian_(x, h, kernel)
    dGdh = sphGradientDkDh_(x, h, kernel)
    wp.adjoint[x] += matmul(H, adj_ret)
    wp.adjoint[h] += wp.dot(dGdh, adj_ret)
@wp.func_grad(sphGradient_3D)
def adj_sphGradient_3D(x: vector(dtype=scalar_t, length=3), h: scalar_t, kernel: wp.int32, adj_ret: vector(dtype=scalar_t, length=3)):
    H = sphKernelHessian_(x, h, kernel)
    dGdh = sphGradientDkDh_(x, h, kernel)
    wp.adjoint[x] += matmul(H, adj_ret)
    wp.adjoint[h] += wp.dot(dGdh, adj_ret)


@wp.func
def sphGradient_(x: vector(dtype=scalar_t, length=1), h: scalar_t, kernel: wp.int32):
    return sphGradient_1D(x, h, kernel)
@wp.func
def sphGradient_(x: vector(dtype=scalar_t, length=2), h: scalar_t, kernel: wp.int32):
    return sphGradient_2D(x, h, kernel)
@wp.func
def sphGradient_(x: vector(dtype=scalar_t, length=3), h: scalar_t, kernel: wp.int32):
    return sphGradient_3D(x, h, kernel)



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