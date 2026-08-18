"""Tier-2 JVP of the pairwise kernel value `W_ij` w.r.t. positions/supports
(`warpier_adjoint.md` Tier 2.1): `sphKernelJVP`/`sphKernelJVP_ij` mirror
`sphKernel`/`sphKernel_ij` (`kernel.py`) field-for-field and branch-for-branch
on `SupportScheme`, returning `(W_ij, dW_ij)` instead of just `W_ij`. Built
entirely from already-validated Tier 2.0 building blocks (`sphKernel_`,
`sphGradient_`, `sphKernelDkDh_`) plus `computePairwiseSupportJVP` (ordinary
calculus, not kernel math) -- no new kernel math here, only the chain rule
through `x_ij = x_i - x_j` and `h_ij = computePairwiseSupport(h_i, h_j, mode)`.

Field-value tangents are Tier 1 territory (a re-launch of the same kernel on
the tangent array) and have no place here; this only ever produces the
position/support tangent.
"""

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
from ..util.support import computePairwiseSupport, computePairwiseSupportJVP
from ..dataTypes.domain_t import domainData
from ..dataTypes.kernelState_t import kernelState
from .kernel import sphKernel_
from .gradient import sphGradient_
from .gradH import sphKernelDkDh_


@wp.func
def sphKernelJVP_ij(
    xij: vector(dtype=scalar_t, length=dim_t),
    hi: scalar_t,
    hj: scalar_t,
    dxij: vector(dtype=scalar_t, length=dim_t),
    dhi: scalar_t,
    dhj: scalar_t,
    kernelProperties: kernelState,
    domainState: domainData,
):
    if kernelProperties.supportMode == wp.static(SupportScheme.KernelMeanSymmetric.value) or kernelProperties.supportMode == wp.static(SupportScheme.SuperSymmetric.value):
        Wi = sphKernel_(xij, hi, kernelProperties.kernelFunction)
        Wj = sphKernel_(xij, hj, kernelProperties.kernelFunction)
        gWi = sphGradient_(xij, hi, kernelProperties.kernelFunction)
        gWj = sphGradient_(xij, hj, kernelProperties.kernelFunction)
        dWdhi = sphKernelDkDh_(xij, hi, kernelProperties.kernelFunction)
        dWdhj = sphKernelDkDh_(xij, hj, kernelProperties.kernelFunction)
        W = (Wi + Wj) * scalar_t(0.5)
        dW = scalar_t(0.5) * (wp.dot(gWi, dxij) + wp.dot(gWj, dxij) + dWdhi * dhi + dWdhj * dhj)
        return W, dW
    hij = computePairwiseSupport(hi, hj, kernelProperties.supportMode)
    dhij = computePairwiseSupportJVP(hi, hj, dhi, dhj, kernelProperties.supportMode)
    W = sphKernel_(xij, hij, kernelProperties.kernelFunction)
    gW = sphGradient_(xij, hij, kernelProperties.kernelFunction)
    dWdh = sphKernelDkDh_(xij, hij, kernelProperties.kernelFunction)
    dW = wp.dot(gW, dxij) + dWdh * dhij
    return W, dW


@wp.func
def sphKernelJVP(
    xi: vector(dtype=scalar_t, length=dim_t),
    xj: vector(dtype=scalar_t, length=dim_t),
    hi: scalar_t,
    hj: scalar_t,
    dxi: vector(dtype=scalar_t, length=dim_t),
    dxj: vector(dtype=scalar_t, length=dim_t),
    dhi: scalar_t,
    dhj: scalar_t,
    kernelProperties: kernelState,
    domainState: domainData,
):
    xij = computeDistanceVec(xi, xj, domainState)
    dxij = dxi - dxj  # ordinary difference -- periodic-wrap tangent
                       # discontinuity is out of scope (warpier_adjoint.md
                       # Tier 2.1), matching sphKernelJVP_ij's own scope.
    return sphKernelJVP_ij(xij, hi, hj, dxij, dhi, dhj, kernelProperties, domainState)
