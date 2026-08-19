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
from .gradH import sphKernelDkDh_, sphGradientDkDh_
from .hessian import sphKernelHessian_
from .laplacian import sphKernelLaplacian_, sphKernelLaplacianGradient_, sphKernelLaplacianDkDh_


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


@wp.func
def sphKernelGradientJVP_ij(
    xij: vector(dtype=scalar_t, length=dim_t),
    hi: scalar_t,
    hj: scalar_t,
    dxij: vector(dtype=scalar_t, length=dim_t),
    dhi: scalar_t,
    dhj: scalar_t,
    kernelProperties: kernelState,
    domainState: domainData,
):
    """JVP of `sphKernelGradient_ij` (`kernels/gradient.py`), i.e.
    `d(nabla_i W_ij)/d{x,h}` (`warpier_adjoint.md` Tier 2.2): mirrors that
    function's three-way `SupportScheme` dispatch byte-for-byte, built
    entirely from already-validated Tier 2.0 building blocks (`sphGradient_`,
    `sphKernelHessian_` = `d(sphGradient_)/dx`, `sphGradientDkDh_` =
    `d(sphGradient_)/dh`) plus `computePairwiseSupport`/
    `computePairwiseSupportJVP` for the non-KernelMeanSymmetric/SuperSymmetric
    branches. Validated in dense all-pairs form by
    `scripts/spike_forward_mode_tier2_gradient.py`'s `_kernelGradientJVP`
    (`rel_err ~1e-9` in float64 against `warpOperation`'s own reverse-mode
    Jacobian, for every `GradientScheme`/`SupportScheme` combination) --
    ported here byte-for-byte (only the vector/matrix types are the
    module's fixed `dim_t`/`scalar_t` rather than the spike's generic `Any`,
    matching every other function in this file)."""
    if kernelProperties.supportMode == wp.static(SupportScheme.KernelMeanSymmetric.value) or kernelProperties.supportMode == wp.static(SupportScheme.SuperSymmetric.value):
        Gi = sphGradient_(xij, hi, kernelProperties.kernelFunction)
        Gj = sphGradient_(xij, hj, kernelProperties.kernelFunction)
        Hi = sphKernelHessian_(xij, hi, kernelProperties.kernelFunction)
        Hj = sphKernelHessian_(xij, hj, kernelProperties.kernelFunction)
        dGdhi = sphGradientDkDh_(xij, hi, kernelProperties.kernelFunction)
        dGdhj = sphGradientDkDh_(xij, hj, kernelProperties.kernelFunction)
        G = (Gi + Gj) * scalar_t(0.5)
        dG = (matmul(Hi, dxij) + matmul(Hj, dxij) + dGdhi * dhi + dGdhj * dhj) * scalar_t(0.5)
        return G, dG
    hij = computePairwiseSupport(hi, hj, kernelProperties.supportMode)
    dhij = computePairwiseSupportJVP(hi, hj, dhi, dhj, kernelProperties.supportMode)
    G = sphGradient_(xij, hij, kernelProperties.kernelFunction)
    H = sphKernelHessian_(xij, hij, kernelProperties.kernelFunction)
    dGdh = sphGradientDkDh_(xij, hij, kernelProperties.kernelFunction)
    dG = matmul(H, dxij) + dGdh * dhij
    return G, dG


@wp.func
def sphKernelGradientJVP(
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
    dxij = dxi - dxj
    return sphKernelGradientJVP_ij(xij, hi, hj, dxij, dhi, dhj, kernelProperties, domainState)


@wp.func
def sphKernelLaplacianJVP_ij(
    xij: vector(dtype=scalar_t, length=dim_t),
    hi: scalar_t,
    hj: scalar_t,
    dxij: vector(dtype=scalar_t, length=dim_t),
    dhi: scalar_t,
    dhj: scalar_t,
    kernelProperties: kernelState,
    domainState: domainData,
):
    """JVP of `sphKernelLaplacian` (`kernels/laplacian.py`), i.e.
    `d(sphKernelLaplacian)/d{x,h}` (`warpier_adjoint.md` Tier 2.3): mirrors
    that function's own **two-branch** `SupportScheme` dispatch (SuperSymmetric
    explicit, everything else -- including `KernelMeanSymmetric` -- falling
    through to `computePairwiseSupport`'s own max-fallback dispatch), NOT the
    three-branch KernelMeanSymmetric/SuperSymmetric-together dispatch
    `sphKernelJVP_ij`/`sphKernelGradientJVP_ij` use above -- `sphKernelLaplacian`
    itself never gained an explicit `KernelMeanSymmetric` branch when those two
    did (`warpier_adjoint.md` Tier 2.3's own structural finding: the two
    schemes are provably identical for the kernel value/gradient but *not*
    for the Naive Laplacian). Built from `sphKernelLaplacianGradient_`
    (`d(sphKernelLaplacian_)/dx`) and `sphKernelLaplacianDkDh_`
    (`d(sphKernelLaplacian_)/dh`), both already validated against `wp.Tape`
    (`kernel_sanity_native.py` Section K). Ported byte-for-byte from
    `scripts/spike_forward_mode_tier2_laplacian_naive.py`'s already-validated
    `_kernelLaplacianJVP` (`rel_err ~1e-9` in float64 against `warpOperation`'s
    own reverse-mode Jacobian, for every `GradientScheme`/`SupportScheme`
    combination, including the KernelMeanSymmetric-vs-SuperSymmetric
    genuinely-differ check)."""
    if kernelProperties.supportMode == wp.static(SupportScheme.SuperSymmetric.value):
        Li = sphKernelLaplacian_(xij, hi, kernelProperties.kernelFunction)
        Lj = sphKernelLaplacian_(xij, hj, kernelProperties.kernelFunction)
        L = (Li + Lj) * scalar_t(0.5)
        Gi = sphKernelLaplacianGradient_(xij, hi, kernelProperties.kernelFunction)
        Gj = sphKernelLaplacianGradient_(xij, hj, kernelProperties.kernelFunction)
        DhI = sphKernelLaplacianDkDh_(xij, hi, kernelProperties.kernelFunction)
        DhJ = sphKernelLaplacianDkDh_(xij, hj, kernelProperties.kernelFunction)
        dL = (wp.dot(Gi, dxij) + DhI * dhi + wp.dot(Gj, dxij) + DhJ * dhj) * scalar_t(0.5)
        return L, dL
    hij = computePairwiseSupport(hi, hj, kernelProperties.supportMode)
    dhij = computePairwiseSupportJVP(hi, hj, dhi, dhj, kernelProperties.supportMode)
    L = sphKernelLaplacian_(xij, hij, kernelProperties.kernelFunction)
    G = sphKernelLaplacianGradient_(xij, hij, kernelProperties.kernelFunction)
    Dh = sphKernelLaplacianDkDh_(xij, hij, kernelProperties.kernelFunction)
    dL = wp.dot(G, dxij) + Dh * dhij
    return L, dL


@wp.func
def sphKernelLaplacianJVP(
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
    dxij = dxi - dxj
    return sphKernelLaplacianJVP_ij(xij, hi, hj, dxij, dhi, dhj, kernelProperties, domainState)
