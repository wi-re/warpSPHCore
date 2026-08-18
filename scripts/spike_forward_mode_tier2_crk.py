#!/usr/bin/env python3
"""Tier 2.5 forward-mode spike: CRK correction (warpier_adjoint.md).

**The question this answers.** Tier 2.5's plan claims the CRK correction's Tier-2 JVP
needs no new *kernel* math beyond Tier 2.0/2.2 -- just working through crk_moments.py's
moment-sum construction and crk_terms.py's matrix-calculus solve for A/B/gradA/gradB,
plus product-rule assembly through crk/kernel.py's correctGradientCRK. This script
writes that assembly down end to end, in four validated stages (each checked
independently against the corresponding production function's own reverse-mode
Jacobian, before being chained into the next):

  Stage 1 (crk_volume.py):  positions/supports        -> apparentVolume V_i, dV_i
  Stage 2 (crk_moments.py): positions/supports, V/dV  -> m_0,m_1,m_2,dm_*dgamma + tangents
  Stage 3 (crk_terms.py):   moments + tangents         -> A,B,gradA,gradB + tangents
  Stage 4 (crk/kernel.py):  Stage 3's output + kernelGradient's own JVP (Tier 2.2)
                            -> corrected kernelGradient + tangent -> Gradient operator JVP

**Stage 3 is a deliberate, justified departure from every earlier tier's "no
torch.autograd.functional.jvp" rule.** That rule (warpier_adjoint.md's validation
methodology section) exists because a Warp kernel wrapped as a torch.autograd.Function
only implements a *first-order* backward -- reading gradients off a wp.Tape is not
itself differentiable, so torch's double-backward trick for jvp silently returns a zero
tangent whenever any Warp call sits in the graph. `computeCRKTermsWarp` (crk_terms.py)
is the one link in this whole pipeline with **no Warp call anywhere in it** -- pure
`torch.einsum`/`torch.linalg.pinv`/`torch.where` on the six moment tensors, already a
plain differentiable PyTorch function (that's how gradcheck_crk_native.py's reverse-mode
gradcheck already gets through it today). Double-backward is therefore exactly as valid
here as ordinary backward is -- confirmed empirically before relying on it (see "Stage 3
validation" below: torch.autograd.functional.jvp on computeCRKTermsWarp alone matches
central finite differences to ~1e-10, both on synthetic well-conditioned moment data and
on the real moments Stage 2 produces). Using it sidesteps hand-deriving gradA/gradB's
four-index-tensor matrix calculus (`gradATerm4`'s `nil,nklm,nmj,nj,ni->nk` contraction and
its structural twins) by hand -- a large amount of error-prone algebra for no additional
correctness margin, since the exact JVP of a plain differentiable torch function is
already available for free. This is the same spirit as Tier 2.4 consuming production's
own `L`/`num_nbrs` rather than re-deriving them: reuse an already-correct piece instead of
re-proving it from scratch when reuse is available and honest.

Stages 1, 2, and 4 all go through Warp kernels (crk_volume.py/crk_moments.py/the
kernelGradient dispatch respectively), so they get the usual hand-assembled treatment,
built from Tier 2.0's `sphKernel_`/`sphGradient_`/`sphKernelHessian_`/`sphGradientDkDh_`
and Tier 2.1/2.2's `_pairwiseSupportTangent`/`_kernelGradientJVP` dispatch functions
(duplicated here verbatim, the standing per-script convention -- see
spike_forward_mode_tier2_renorm.py's module docstring).

**Mathematical derivation.**

1. `computeCRKVolume_Func_i`/`_Kernel` (crk_volume.py): `V_i = 1/wsum_i`,
   `wsum_i = Sum_j W_ij`, always called with `supportMode=SupportScheme.Gather`
   (crk_wrapper.py's `volumeProperties`) -- i.e. `h_ij = h_i` always, `dh_ij = dh_i`
   (Tier 2.1's Gather branch). So `dwsum_i = Sum_j [gradW_ij.dx_ij + dW/dh(x_ij,h_i)*dh_i]`
   (Tier 2.1's single-h branch, Gather case only -- KernelMeanSymmetric/SuperSymmetric
   never arise here since the mode is hardcoded) and `dV_i = -dwsum_i/wsum_i^2`.

2. `computeCRKMoments_Func_i`/`_Kernel` (crk_moments.py): always called with
   `supportMode=SupportScheme.Scatter` (crk_wrapper.py's `momentsProperties`) -- i.e.
   `h_ij = h_j`, `dh_ij = dh_j`, for BOTH the kernel value `w_ij` (Tier 2.1's single-h
   branch, Scatter case) and the kernel gradient `gradw_ij` (Tier 2.2's `kernelGradient`
   dispatch, Scatter case). `V_j` is Stage 1's `apparentVolume[j]` -- note this is a
   DIFFERENT quantity from Tier 2.2/2.4's `Vj = mass_j/density_j` despite the same
   letter; CRK's `V_j` never involves mass or density at all. Ordinary product rule
   through the six accumulators (`m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma`)
   gives their tangents -- see `_dense_crk_moments_jvp`'s docstring for the exact
   per-term formulas; `eye = warp_eye(...)` is a literal constant (its only argument is
   used purely to select a compile-time overload by vector length, see math/wp_eye.py),
   so contributes no term of its own. `dm_2dgamma`'s flattened index
   `gamma*dim*dim + alpha*dim + beta` (crk_moments.py) becomes axis order
   `[gamma, alpha, beta]` after `_computeCRKMoments_stateBackend`'s own
   `.view(-1, dim, dim, dim)` reshape -- matched exactly here so the assembled tensor
   can be fed into the *same* `computeCRKTermsWarp` Stage 3 uses.

3. `computeCRKTermsWarp` (crk_terms.py): JVP obtained via `torch.autograd.functional.jvp`
   directly on the function, using Stage 2's assembled moments/tangents as primals/
   tangents (see the departure-from-the-rule note above). `num_nbrs`/`supports` are
   taken directly from production's own output, the same "consume a non-differentiable
   or already-produced value rather than re-derive it" pattern every earlier tier used
   for `SupportScheme` dispatch decisions and Tier 2.4's `num_nbrs`/pinv rank cutoff.
   Test geometries here are the same well-conditioned line/grid cases every earlier
   tier used, so `is_singular`/`num_nbrs<2`'s masking branch is never engaged --
   consistent with Tier 2.4's choice not to manufacture a near-singular case on
   purpose (the plan asks for the identity to be validated, not the discontinuity
   itself characterized).

4. `correctGradientCRK`/`computeKernelGradientCRK` (crk/kernel.py), assembled by the
   ordinary product rule directly on the four-term formula:

       term1 = (Ai*Wij)*Bi
       term2 = (Ai*(1+dot(Bi,xij)))*gradWij
       term3 = ((1+dot(Bi,xij))*Wij)*gradAi
       term4 = (Ai*Wij)*(gradBi^T @ xij)

   with `Wij, gradWij` and their tangents from Tier 2.1/2.2's dispatch (evaluated at
   whatever `SupportScheme` the *consuming operator* uses -- `SupportScheme.Gather` in
   this script's test, matching gradcheck_crk_correction_native.py) and
   `Ai, Bi, gradAi, gradBi` (and their tangents) Stage 3's per-QUERY-particle output,
   broadcast over the neighbor index `j` (confirmed from `util/stateUtil.py`'s
   `getCRK_i`: `correctionData.queryA[i]` etc., indexed at `i` only, constant across the
   `j` loop -- these are NOT the same as `gradA_i`/`gradB_i` being re-differentiated
   again; they are simply held fixed while summing over neighbors, exactly like `fi` in
   every earlier tier's field-value coefficient). `d(term4)`'s `gradBi^T @ xij` product
   needs the SAME first-axis-vs-second-axis contraction care crk/kernel.py's own
   docstring already flags (`matmul(wp.transpose(gradBi), x_ij)`) -- matched here via
   `einsum('icl,ijc->ijl', gradBi, x_ij)` (contract gradBi's first/component axis `c`
   against `x_ij`, leave the differentiation axis `l` free), and its tangent needs BOTH
   `dgradBi` and `dx_ij` terms (ordinary product rule over a bilinear contraction).
   `Gradient_i = Sum_j coeff_ij * correctedG_ij` then reuses Tier 2.2's
   `_gradient_weights` (mass/density-based `coeff_ij`, unrelated to CRK's `A_i/B_i`)
   verbatim, exactly as Tier 2.2/2.4 did -- no new derivation for this last step.

Two independent code paths at every stage (both exact analytic derivatives, no finite
differences on either side, except the one-time bring-up sanity check against central
finite differences noted in the Stage 3 departure above) -- same dense-all-pairs-is-safe
argument as every earlier tier (every kernel-derivative building block is exactly zero
for q=|x|/h>1).

    python scripts/spike_forward_mode_tier2_crk.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys
from typing import Any

import torch
import warp as wp
from warp.types import vector

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, grid_case_2d, line_case, make_domain
from warpSPHCore import OperationProperties, ParticleState, warpOperation
from warpSPHCore.crk import computeCRKFactors
from warpSPHCore.crk.crk_moments import _computeCRKMoments_stateBackend
from warpSPHCore.crk.crk_terms import computeCRKTermsWarp
from warpSPHCore.crk.crk_volume import _computeCRKVolume_stateBackend
from warpSPHCore.enumTypes import GradientScheme, OperationDirection, SupportScheme, WarpOperation
from warpSPHCore.type_config import scalar_t
from warpSPHCore.kernels.kernel import sphKernel_
from warpSPHCore.kernels.gradient import sphGradient_
from warpSPHCore.kernels.hessian import sphKernelHessian_
from warpSPHCore.kernels.gradH import sphKernelDkDh_, sphGradientDkDh_
from warpSPHCore.util.support import computePairwiseSupport
from warpSPHCore.math import matmul

TOL = 1e-9  # float64, both sides exact analytic derivatives -- round-off only

vec1_t = vector(dtype=scalar_t, length=1)
vec2_t = vector(dtype=scalar_t, length=2)


# --------------------------------------------------------------------------
# Tier 2.1's / Tier 2.2's building blocks, reused verbatim (see
# spike_forward_mode_tier2_renorm.py's module docstring for the standing
# per-script duplication convention).
# --------------------------------------------------------------------------

@wp.func
def _pairwiseSupportTangent(hi: scalar_t, hj: scalar_t, dhi: scalar_t, dhj: scalar_t, mode: wp.uint32):
    if mode == wp.static(SupportScheme.Gather.value):
        return dhi
    elif mode == wp.static(SupportScheme.Scatter.value):
        return dhj
    elif mode == wp.static(SupportScheme.MeanSymmetric.value):
        return (dhi + dhj) / scalar_t(2.0)
    else:
        if hi >= hj:
            return dhi
        return dhj


@wp.func
def _kernelValueJVP(
    xij: vector(dtype=scalar_t, length=Any),  # type: ignore
    hi: scalar_t, hj: scalar_t,
    dxij: vector(dtype=scalar_t, length=Any),  # type: ignore
    dhi: scalar_t, dhj: scalar_t,
    mode: wp.uint32, kernel_id: wp.int32,
):
    """Tier 2.1's dW_ij dispatch, factored into a standalone @wp.func (Tier
    2.1's own script inlined this in its per-pair kernel; Tier 2.5 needs it
    called from two different SupportSchemes -- Gather for apparentVolume,
    Scatter for the moments -- and again for the consuming operator's own
    mode in Stage 4, so it is worth its own function here)."""
    if mode == wp.static(SupportScheme.KernelMeanSymmetric.value) or mode == wp.static(SupportScheme.SuperSymmetric.value):
        Wi = sphKernel_(xij, hi, kernel_id)
        Wj = sphKernel_(xij, hj, kernel_id)
        W = (Wi + Wj) * scalar_t(0.5)
        gWi = sphGradient_(xij, hi, kernel_id)
        gWj = sphGradient_(xij, hj, kernel_id)
        dWdhi = sphKernelDkDh_(xij, hi, kernel_id)
        dWdhj = sphKernelDkDh_(xij, hj, kernel_id)
        dW = scalar_t(0.5) * (wp.dot(gWi, dxij) + wp.dot(gWj, dxij) + dWdhi * dhi + dWdhj * dhj)
    else:
        hij = computePairwiseSupport(hi, hj, mode)
        dhij = _pairwiseSupportTangent(hi, hj, dhi, dhj, mode)
        W = sphKernel_(xij, hij, kernel_id)
        gW = sphGradient_(xij, hij, kernel_id)
        dWdh = sphKernelDkDh_(xij, hij, kernel_id)
        dW = wp.dot(gW, dxij) + dWdh * dhij
    return W, dW


@wp.func
def _kernelGradientJVP(
    xij: vector(dtype=scalar_t, length=Any),  # type: ignore
    hi: scalar_t, hj: scalar_t,
    dxij: vector(dtype=scalar_t, length=Any),  # type: ignore
    dhi: scalar_t, dhj: scalar_t,
    mode: wp.uint32, kernel_id: wp.int32,
):
    if mode == wp.static(SupportScheme.KernelMeanSymmetric.value) or mode == wp.static(SupportScheme.SuperSymmetric.value):
        gi = sphGradient_(xij, hi, kernel_id)
        gj = sphGradient_(xij, hj, kernel_id)
        grad = (gi + gj) * scalar_t(0.5)
        Hi = sphKernelHessian_(xij, hi, kernel_id)
        Hj = sphKernelHessian_(xij, hj, kernel_id)
        dhdhi = sphGradientDkDh_(xij, hi, kernel_id)
        dhdhj = sphGradientDkDh_(xij, hj, kernel_id)
        dgrad = (matmul(Hi, dxij) + matmul(Hj, dxij) + dhdhi * dhi + dhdhj * dhj) * scalar_t(0.5)
    else:
        hij = computePairwiseSupport(hi, hj, mode)
        dhij = _pairwiseSupportTangent(hi, hj, dhi, dhj, mode)
        grad = sphGradient_(xij, hij, kernel_id)
        H = sphKernelHessian_(xij, hij, kernel_id)
        dHdh = sphGradientDkDh_(xij, hij, kernel_id)
        dgrad = matmul(H, dxij) + dHdh * dhij
    return grad, dgrad


@wp.kernel
def _pair_jvp_1d(
    xi: wp.array(dtype=vec1_t), xj: wp.array(dtype=vec1_t),
    hi: wp.array(dtype=scalar_t), hj: wp.array(dtype=scalar_t),
    dxi: wp.array(dtype=vec1_t), dxj: wp.array(dtype=vec1_t),
    dhi: wp.array(dtype=scalar_t), dhj: wp.array(dtype=scalar_t),
    mode: wp.uint32, kernel_id: wp.int32,
    W_out: wp.array(dtype=scalar_t), dW_out: wp.array(dtype=scalar_t),
    G_out: wp.array(dtype=vec1_t), dG_out: wp.array(dtype=vec1_t),
):
    p = wp.tid()
    xij = xi[p] - xj[p]
    dxij = dxi[p] - dxj[p]
    w, dw = _kernelValueJVP(xij, hi[p], hj[p], dxij, dhi[p], dhj[p], mode, kernel_id)
    g, dg = _kernelGradientJVP(xij, hi[p], hj[p], dxij, dhi[p], dhj[p], mode, kernel_id)
    W_out[p] = w
    dW_out[p] = dw
    G_out[p] = g
    dG_out[p] = dg


@wp.kernel
def _pair_jvp_2d(
    xi: wp.array(dtype=vec2_t), xj: wp.array(dtype=vec2_t),
    hi: wp.array(dtype=scalar_t), hj: wp.array(dtype=scalar_t),
    dxi: wp.array(dtype=vec2_t), dxj: wp.array(dtype=vec2_t),
    dhi: wp.array(dtype=scalar_t), dhj: wp.array(dtype=scalar_t),
    mode: wp.uint32, kernel_id: wp.int32,
    W_out: wp.array(dtype=scalar_t), dW_out: wp.array(dtype=scalar_t),
    G_out: wp.array(dtype=vec2_t), dG_out: wp.array(dtype=vec2_t),
):
    p = wp.tid()
    xij = xi[p] - xj[p]
    dxij = dxi[p] - dxj[p]
    w, dw = _kernelValueJVP(xij, hi[p], hj[p], dxij, dhi[p], dhj[p], mode, kernel_id)
    g, dg = _kernelGradientJVP(xij, hi[p], hj[p], dxij, dhi[p], dhj[p], mode, kernel_id)
    W_out[p] = w
    dW_out[p] = dw
    G_out[p] = g
    dG_out[p] = dg


_PAIR_JVP_BY_DIM = {1: _pair_jvp_1d, 2: _pair_jvp_2d}
_VEC_BY_DIM = {1: vec1_t, 2: vec2_t}


def _dense_pair_WG(pos, sup, dpos, dsup, dim, mode, kernel_id):
    """All-pairs (i, j) including i==j: W_ij, dW_ij ((n,n)) and G_ij, dG_ij
    ((n,n,dim)) together, for a single SupportScheme mode. Safe for the same
    reason every earlier tier's dense loop was (kernel building blocks are
    exactly zero for q>1)."""
    n = pos.shape[0]
    vec_t = _VEC_BY_DIM[dim]
    idx_i = torch.arange(n).repeat_interleave(n)
    idx_j = torch.arange(n).repeat(n)

    xi = wp.from_torch(pos[idx_i].contiguous(), dtype=vec_t)
    xj = wp.from_torch(pos[idx_j].contiguous(), dtype=vec_t)
    hi = wp.from_torch(sup[idx_i].contiguous(), dtype=scalar_t)
    hj = wp.from_torch(sup[idx_j].contiguous(), dtype=scalar_t)
    dxi = wp.from_torch(dpos[idx_i].contiguous(), dtype=vec_t)
    dxj = wp.from_torch(dpos[idx_j].contiguous(), dtype=vec_t)
    dhi = wp.from_torch(dsup[idx_i].contiguous(), dtype=scalar_t)
    dhj = wp.from_torch(dsup[idx_j].contiguous(), dtype=scalar_t)

    W_out = wp.zeros(n * n, dtype=scalar_t, device=DEVICE.type)
    dW_out = wp.zeros(n * n, dtype=scalar_t, device=DEVICE.type)
    G_out = wp.zeros(n * n, dtype=vec_t, device=DEVICE.type)
    dG_out = wp.zeros(n * n, dtype=vec_t, device=DEVICE.type)
    wp.launch(_PAIR_JVP_BY_DIM[dim], dim=n * n,
              inputs=[xi, xj, hi, hj, dxi, dxj, dhi, dhj, mode, kernel_id],
              outputs=[W_out, dW_out, G_out, dG_out], device=DEVICE.type)

    W = wp.to_torch(W_out).reshape(n, n)
    dW = wp.to_torch(dW_out).reshape(n, n)
    G = wp.to_torch(G_out).reshape(n, n, dim)
    dG = wp.to_torch(dG_out).reshape(n, n, dim)
    return W, dW, G, dG


# --------------------------------------------------------------------------
# Stage 1: apparentVolume (crk_volume.py). Gather mode, hardcoded (module
# docstring point 1) -- Tier 2.1's single-h/Gather branch plus the reciprocal.
# --------------------------------------------------------------------------

def assembled_apparent_volume_jvp(pos, sup, dpos, dsup, dim, kernel_id):
    mode = SupportScheme.Gather.value
    W, dW, _, _ = _dense_pair_WG(pos, sup, dpos, dsup, dim, mode, kernel_id)
    wsum = W.sum(dim=1)
    dwsum = dW.sum(dim=1)
    V = 1.0 / wsum
    dV = -dwsum / wsum**2
    return V, dV


# --------------------------------------------------------------------------
# Stage 2: CRK moments (crk_moments.py). Scatter mode, hardcoded (module
# docstring point 2). Ordinary product rule over the six accumulators, using
# Stage 1's V/dV as the per-j weight.
# --------------------------------------------------------------------------

def assembled_crk_moments_jvp(pos, sup, V, dpos, dsup, dV, dim, kernel_id):
    """Returns ((m0,m1,m2,dm0g,dm1g,dm2g), (dm0,dm1,dm2,d_dm0g,d_dm1g,d_dm2g)).
    dm2g/d_dm2g have axes [i, gamma, alpha, beta], matching
    _computeCRKMoments_stateBackend's own `.view(-1,dim,dim,dim)` reshape of
    crk_moments.py's `gamma*dim*dim+alpha*dim+beta` flatten order."""
    mode = SupportScheme.Scatter.value
    W, dW, G, dG = _dense_pair_WG(pos, sup, dpos, dsup, dim, mode, kernel_id)

    x_ij = pos.unsqueeze(1) - pos.unsqueeze(0)        # (n,n,dim), row=i, col=j
    dx_ij = dpos.unsqueeze(1) - dpos.unsqueeze(0)

    Vj = V.unsqueeze(0)          # (1,n) -- V is per-j, broadcasts over query i
    dVj = dV.unsqueeze(0)

    VW = Vj * W                  # (n,n)
    dVW = dVj * W + Vj * dW

    m_0 = VW.sum(dim=1)
    dm_0 = dVW.sum(dim=1)

    m_1 = (x_ij * VW.unsqueeze(-1)).sum(dim=1)
    dm_1 = (dx_ij * VW.unsqueeze(-1) + x_ij * dVW.unsqueeze(-1)).sum(dim=1)

    outer_xx = x_ij.unsqueeze(-1) * x_ij.unsqueeze(-2)          # [...,alpha,beta]
    d_outer_xx = dx_ij.unsqueeze(-1) * x_ij.unsqueeze(-2) + x_ij.unsqueeze(-1) * dx_ij.unsqueeze(-2)
    m_2 = (outer_xx * VW.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
    dm_2 = (d_outer_xx * VW.unsqueeze(-1).unsqueeze(-1) + outer_xx * dVW.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)

    dm_0dgamma = (Vj.unsqueeze(-1) * G).sum(dim=1)
    d_dm_0dgamma = (dVj.unsqueeze(-1) * G + Vj.unsqueeze(-1) * dG).sum(dim=1)

    # dm_1dgamma[k,l] = Vj*(x_ij[k]*G[l] + w_ij*eye[k,l])  (wp.outer(x_ij,G)[k,l]=x_ij[k]*G[l])
    outer_xG = x_ij.unsqueeze(-1) * G.unsqueeze(-2)                          # [...,k(alpha),l(gamma)]
    d_outer_xG = dx_ij.unsqueeze(-1) * G.unsqueeze(-2) + x_ij.unsqueeze(-1) * dG.unsqueeze(-2)
    eye = torch.eye(dim, dtype=pos.dtype, device=pos.device)
    pairTerm = outer_xG + W.unsqueeze(-1).unsqueeze(-1) * eye
    d_pairTerm = d_outer_xG + dW.unsqueeze(-1).unsqueeze(-1) * eye
    dm_1dgamma = (Vj.unsqueeze(-1).unsqueeze(-1) * pairTerm).sum(dim=1)
    d_dm_1dgamma = (dVj.unsqueeze(-1).unsqueeze(-1) * pairTerm + Vj.unsqueeze(-1).unsqueeze(-1) * d_pairTerm).sum(dim=1)

    # dm_2dgamma[g,a,b] = Vj*( x_ij[a]*x_ij[b]*G[g] + w_ij*(x_ij[a]*delta(b,g) + delta(a,g)*x_ij[b]) )
    gradTerm = torch.einsum("ija,ijb,ijg->ijgab", x_ij, x_ij, G)
    d_gradTerm = (torch.einsum("ija,ijb,ijg->ijgab", dx_ij, x_ij, G)
                  + torch.einsum("ija,ijb,ijg->ijgab", x_ij, dx_ij, G)
                  + torch.einsum("ija,ijb,ijg->ijgab", x_ij, x_ij, dG))

    termA = torch.einsum("ija,bg->ijgab", x_ij, eye)          # x_ij[a]*delta(b,g)
    termB = torch.einsum("ag,ijb->ijgab", eye, x_ij)          # delta(a,g)*x_ij[b]
    d_termA = torch.einsum("ija,bg->ijgab", dx_ij, eye)
    d_termB = torch.einsum("ag,ijb->ijgab", eye, dx_ij)

    kernelTerm = W[..., None, None, None] * (termA + termB)
    d_kernelTerm = dW[..., None, None, None] * (termA + termB) + W[..., None, None, None] * (d_termA + d_termB)

    pairContribution = gradTerm + kernelTerm
    d_pairContribution = d_gradTerm + d_kernelTerm

    Vj5 = Vj[..., None, None, None]     # (1,n,1,1,1)
    dVj5 = dVj[..., None, None, None]
    dm_2dgamma = (Vj5 * pairContribution).sum(dim=1)
    d_dm_2dgamma = (dVj5 * pairContribution + Vj5 * d_pairContribution).sum(dim=1)

    values = (m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma)
    tangents = (dm_0, dm_1, dm_2, d_dm_0dgamma, d_dm_1dgamma, d_dm_2dgamma)
    return values, tangents


# --------------------------------------------------------------------------
# Stage 3: crk_terms.py's computeCRKTermsWarp. Pure torch (no Warp anywhere
# in it) -- torch.autograd.functional.jvp's double-backward trick is exact
# here, unlike every Warp-kernel-backed operator earlier tiers had to
# hand-derive around (see module docstring's "deliberate departure" note).
# --------------------------------------------------------------------------

def crk_terms_jvp(m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma,
                   dm_0, dm_1, dm_2, d_dm_0dgamma, d_dm_1dgamma, d_dm_2dgamma,
                   num_nbrs, supports):
    primals = (m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma)
    tangents = (dm_0, dm_1, dm_2, d_dm_0dgamma, d_dm_1dgamma, d_dm_2dgamma)

    def f(m0, m1, m2, dm0g, dm1g, dm2g):
        return computeCRKTermsWarp(m0, m1, m2, dm0g, dm1g, dm2g, num_nbrs, supports)

    values, jvps = torch.autograd.functional.jvp(f, primals, tangents)
    return values, jvps  # (A,B,gradA,gradB), (dA,dB,dgradA,dgradB)


# --------------------------------------------------------------------------
# Stage 4: crk/kernel.py's correctGradientCRK, assembled by the ordinary
# product rule (module docstring point 4), then Tier 2.2's field-value
# coefficient to get the CRK-corrected Gradient operator's own JVP.
# --------------------------------------------------------------------------

def _gradient_weights(mass, density, dmass, ddensity, scheme: GradientScheme):
    """Tier 2.2's _gradient_weights, reused verbatim -- coeff_ij's mass/
    density-based Vj is unrelated to CRK's apparentVolume V despite sharing a
    symbol; see module docstring point 2."""
    density_i = density.unsqueeze(1)
    ddensity_i = ddensity.unsqueeze(1)
    mass_j = mass.unsqueeze(0)
    density_j = density.unsqueeze(0)
    dmass_j = dmass.unsqueeze(0)
    ddensity_j = ddensity.unsqueeze(0)

    Vj = mass_j / density_j
    dVj = dmass_j / density_j - mass_j * ddensity_j / density_j**2

    if scheme == GradientScheme.Naive:
        A, dA = torch.zeros_like(Vj), torch.zeros_like(Vj)
        B, dB = Vj, dVj
    elif scheme == GradientScheme.Difference:
        A, dA = -Vj, -dVj
        B, dB = Vj, dVj
    elif scheme == GradientScheme.Summation:
        A, dA = Vj, dVj
        B, dB = Vj, dVj
    elif scheme == GradientScheme.Symmetric:
        A = mass_j / density_i
        dA = dmass_j / density_i - mass_j * ddensity_i / density_i**2
        B = mass_j * density_i / density_j**2
        dB = (dmass_j * density_i / density_j**2
              + mass_j * ddensity_i / density_j**2
              - 2.0 * mass_j * density_i * ddensity_j / density_j**3)
    else:
        raise ValueError(scheme)
    return A, B, dA, dB


def assembled_correctedGradient_jvp(pos, sup, mass, density, dpos, dsup, dmass, ddensity,
                                     Ai, Bi, gradAi, gradBi, dAi, dBi, dgradAi, dgradBi,
                                     dim, mode, kernel_id, scheme):
    """Ai,Bi,gradAi,gradBi (and tangents): (n,), (n,dim), (n,dim), (n,dim,dim) --
    per-QUERY-particle (Stage 3's output), broadcast over neighbor j exactly
    like getCRK_i (util/stateUtil.py) does in production."""
    W, dW, G, dG = _dense_pair_WG(pos, sup, dpos, dsup, dim, mode, kernel_id)

    x_ij = pos.unsqueeze(1) - pos.unsqueeze(0)
    dx_ij = dpos.unsqueeze(1) - dpos.unsqueeze(0)

    Ai_b = Ai.unsqueeze(1)                       # (n,1)
    dAi_b = dAi.unsqueeze(1)
    Bi_b = Bi.unsqueeze(1)                       # (n,1,dim)
    dBi_b = dBi.unsqueeze(1)
    gradAi_b = gradAi.unsqueeze(1)                # (n,1,dim)
    dgradAi_b = dgradAi.unsqueeze(1)

    dot_Bx = (Bi_b * x_ij).sum(-1)                # (n,n)
    d_dot_Bx = (dBi_b * x_ij).sum(-1) + (Bi_b * dx_ij).sum(-1)

    term1 = (Ai_b * W).unsqueeze(-1) * Bi_b
    dterm1 = (dAi_b * W + Ai_b * dW).unsqueeze(-1) * Bi_b + (Ai_b * W).unsqueeze(-1) * dBi_b

    factor2 = Ai_b * (1.0 + dot_Bx)
    dfactor2 = dAi_b * (1.0 + dot_Bx) + Ai_b * d_dot_Bx
    term2 = factor2.unsqueeze(-1) * G
    dterm2 = dfactor2.unsqueeze(-1) * G + factor2.unsqueeze(-1) * dG

    factor3 = (1.0 + dot_Bx) * W
    dfactor3 = d_dot_Bx * W + (1.0 + dot_Bx) * dW
    term3 = factor3.unsqueeze(-1) * gradAi_b
    dterm3 = dfactor3.unsqueeze(-1) * gradAi_b + factor3.unsqueeze(-1) * dgradAi_b

    # product[i,j,l] = sum_c gradBi[i,c,l]*x_ij[i,j,c] -- contract gradBi's
    # FIRST (component) axis against x_ij, leave the differentiation axis l
    # free, matching crk/kernel.py's matmul(wp.transpose(gradBi), x_ij).
    product = torch.einsum("icl,ijc->ijl", gradBi, x_ij)
    d_product = torch.einsum("icl,ijc->ijl", dgradBi, x_ij) + torch.einsum("icl,ijc->ijl", gradBi, dx_ij)

    factor4 = Ai_b * W
    dfactor4 = dAi_b * W + Ai_b * dW
    term4 = factor4.unsqueeze(-1) * product
    dterm4 = dfactor4.unsqueeze(-1) * product + factor4.unsqueeze(-1) * d_product

    correctedG = term1 + term2 + term3 + term4
    d_correctedG = dterm1 + dterm2 + dterm3 + dterm4

    A, B, dA, dB = _gradient_weights(mass, density, dmass, ddensity, scheme)
    return correctedG, d_correctedG, A, B, dA, dB


def assembled_crk_gradient_jvp(pos, sup, mass, density, fv_q, fv_r, dpos, dsup, dmass, ddensity,
                                Ai, Bi, gradAi, gradBi, dAi, dBi, dgradAi, dgradBi,
                                dim, mode, kernel_id, scheme):
    correctedG, d_correctedG, A, B, dA, dB = assembled_correctedGradient_jvp(
        pos, sup, mass, density, dpos, dsup, dmass, ddensity,
        Ai, Bi, gradAi, gradBi, dAi, dBi, dgradAi, dgradBi, dim, mode, kernel_id, scheme,
    )
    fi = fv_q.unsqueeze(1)
    fj = fv_r.unsqueeze(0)
    coeff = fi * A + fj * B
    dcoeff = fi * dA + fj * dB
    out = (coeff.unsqueeze(-1) * correctedG).sum(dim=1)
    d_out = (dcoeff.unsqueeze(-1) * correctedG + coeff.unsqueeze(-1) * d_correctedG).sum(dim=1)
    return out, d_out


# --------------------------------------------------------------------------
# Reference: reverse-mode Jacobian of the PRODUCTION function, contracted
# with the tangent -- every earlier tier's pattern.
# --------------------------------------------------------------------------

def _reference_jvp(f, primals, tangents):
    J = torch.autograd.functional.jacobian(f, primals, vectorize=False)
    out = f(*primals)
    if isinstance(out, tuple):
        outs = out
    else:
        outs = (out,)
        J = (J,)
    accs = []
    for o, Jo in zip(outs, J):
        n_out = o.numel()
        acc = torch.zeros(n_out, dtype=DTYPE, device=DEVICE)
        for Jk, vk in zip(Jo, tangents):
            acc = acc + Jk.reshape(n_out, -1) @ vk.reshape(-1)
        accs.append(acc.reshape(o.shape))
    return accs if isinstance(out, tuple) else accs[0]


def check(name, assembled, reference, tol=TOL):
    assembled_flat, reference_flat = assembled.reshape(-1), reference.reshape(-1)
    assert assembled_flat.numel() == reference_flat.numel(), (
        f"{name}: shape mismatch assembled={tuple(assembled.shape)} reference={tuple(reference.shape)}"
    )
    scale = max(float(reference_flat.abs().max()), 1e-300)
    err = float((assembled_flat - reference_flat).abs().max()) / scale
    ok = err <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:70s} rel_err={err:.3e}")
    return ok


def _compute_densities(pos, sup, mass, kinds, domain, adjacency):
    p = ParticleState(positions=pos.detach(), supports=sup.detach(), masses=mass.detach(), densities=None, kinds=kinds)
    rho = warpOperation(
        p, OperationProperties(kernel=KERNEL, operation=WarpOperation.Density,
                                supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll),
        domain, adjacency=adjacency,
    )
    return rho.detach().clone()


def _perturbed_case(n, dim, seed):
    torch.manual_seed(seed)
    if dim == 1:
        pos0, sup0, mass0 = line_case(n)
    else:
        pos0, sup0, mass0 = grid_case_2d(n)
    pos0, sup0, mass0 = pos0.detach(), sup0.detach(), mass0.detach()
    sup0 = sup0 * (1.0 + 0.15 * torch.linspace(-1, 1, sup0.shape[0], dtype=DTYPE))
    domain = make_domain(dim=dim)
    adjacency, kinds = build_adjacency(pos0, sup0, mass0, domain, mode=SupportScheme.KernelMeanSymmetric)
    density0 = _compute_densities(pos0, sup0, mass0, kinds, domain, adjacency)
    return pos0, sup0, mass0, density0, domain, adjacency, kinds


# --------------------------------------------------------------------------
# Stage 1 validation: apparentVolume.
# --------------------------------------------------------------------------

def run_volume_case(n, dim, kernel_id, seed=0):
    pos0, sup0, mass0, density0, domain, adjacency, kinds = _perturbed_case(n, dim, seed)

    def f(pos, sup):
        p = ParticleState(positions=pos, supports=sup, masses=mass0, densities=None, kinds=kinds)
        return _computeCRKVolume_stateBackend(p, OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather,
                                                                       operationMode=OperationDirection.AllToAll),
                                               domain, adjacency=adjacency)

    pos0.requires_grad_(True); sup0.requires_grad_(True)
    dpos, dsup = torch.randn_like(pos0), torch.randn_like(sup0) * 0.1

    reference = _reference_jvp(f, (pos0, sup0), (dpos, dsup))
    _, d = assembled_apparent_volume_jvp(pos0.detach(), sup0.detach(), dpos, dsup, dim, kernel_id)
    return d, reference


# --------------------------------------------------------------------------
# Stage 2 validation: moments, using Stage 1's V/dV.
# --------------------------------------------------------------------------

def run_moments_case(n, dim, kernel_id, seed=0):
    pos0, sup0, mass0, density0, domain, adjacency, kinds = _perturbed_case(n, dim, seed)

    def f(pos, sup):
        p = ParticleState(positions=pos, supports=sup, masses=mass0, densities=None, kinds=kinds)
        V = _computeCRKVolume_stateBackend(p, OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather,
                                                                    operationMode=OperationDirection.AllToAll),
                                            domain, adjacency=adjacency)
        m_0, m_1, m_2, dm0g, dm1g, dm2g, _ = _computeCRKMoments_stateBackend(
            p, OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Scatter, operationMode=OperationDirection.AllToAll),
            domain, queryVolumes=V, referenceVolumes=V, adjacency=adjacency,
        )
        return m_0, m_1, m_2, dm0g, dm1g, dm2g

    pos0.requires_grad_(True); sup0.requires_grad_(True)
    dpos, dsup = torch.randn_like(pos0), torch.randn_like(sup0) * 0.1

    reference = _reference_jvp(f, (pos0, sup0), (dpos, dsup))
    V, dV = assembled_apparent_volume_jvp(pos0.detach(), sup0.detach(), dpos, dsup, dim, kernel_id)
    values, tangents = assembled_crk_moments_jvp(pos0.detach(), sup0.detach(), V, dpos, dsup, dV, dim, kernel_id)
    return tangents, reference


# --------------------------------------------------------------------------
# Stage 3 validation: A, B, gradA, gradB, vs. the FULL production
# computeCRKFactors pipeline (positions/supports -> A/B/gradA/gradB), the
# same scope gradcheck_crk_native.py checks (minus crk_density -- see module
# docstring's scope note).
# --------------------------------------------------------------------------

def run_crk_factors_case(n, dim, kernel_id, seed=0):
    pos0, sup0, mass0, density0, domain, adjacency, kinds = _perturbed_case(n, dim, seed)

    def f(pos, sup):
        p = ParticleState(positions=pos, supports=sup, masses=mass0, densities=None, kinds=kinds)
        _, _, crk = computeCRKFactors(queryParticles=p, domain=domain, kernel=KERNEL,
                                       operationMode=OperationDirection.AllToAll, adjacency=adjacency)
        return crk.A, crk.B, crk.gradA, crk.gradB

    pos0.requires_grad_(True); sup0.requires_grad_(True)
    dpos, dsup = torch.randn_like(pos0), torch.randn_like(sup0) * 0.1

    reference = _reference_jvp(f, (pos0, sup0), (dpos, dsup))

    V, dV = assembled_apparent_volume_jvp(pos0.detach(), sup0.detach(), dpos, dsup, dim, kernel_id)
    values, tangents = assembled_crk_moments_jvp(pos0.detach(), sup0.detach(), V, dpos, dsup, dV, dim, kernel_id)

    p = ParticleState(positions=pos0.detach(), supports=sup0.detach(), masses=mass0, densities=None, kinds=kinds)
    _, _, _, _, _, _, num_nbrs = _computeCRKMoments_stateBackend(
        p, OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Scatter, operationMode=OperationDirection.AllToAll),
        domain, queryVolumes=V.detach(), referenceVolumes=V.detach(), adjacency=adjacency,
    )
    crk_values, crk_jvps = crk_terms_jvp(*values, *tangents, num_nbrs, sup0.detach())
    return crk_values, crk_jvps, reference


# --------------------------------------------------------------------------
# Stage 4 validation: CRK-corrected Gradient operator, vs.
# gradcheck_crk_correction_native.py's exact scenario (crkState computed
# from the SAME leaf positions/supports the operator itself perturbs).
# --------------------------------------------------------------------------

def run_crk_gradient_case(n, dim, mode: SupportScheme, scheme: GradientScheme, kernel_id, seed=0):
    pos0, sup0, mass0, density0, domain, adjacency, kinds = _perturbed_case(n, dim, seed)
    n_actual = pos0.shape[0]  # n above is n_per_side for the 2D grid case, not the particle count
    fv_q = torch.randn(n_actual, dtype=DTYPE)
    fv_r = torch.randn(n_actual, dtype=DTYPE)

    def f(pos, sup, mass, density):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=density, kinds=kinds)
        _, _, crkState = computeCRKFactors(queryParticles=p, domain=domain, kernel=KERNEL,
                                            operationMode=OperationDirection.AllToAll, adjacency=adjacency)
        return warpOperation(
            p, OperationProperties(kernel=KERNEL, operation=WarpOperation.Gradient,
                                    supportMode=mode, operationMode=OperationDirection.AllToAll, gradientMode=scheme),
            domain, queryValues=fv_q, referenceValues=fv_r, adjacency=adjacency, crkState=crkState,
        )

    for t in (pos0, sup0, mass0, density0):
        t.requires_grad_(True)
    dpos, dsup, dmass, ddensity = (torch.randn_like(pos0), torch.randn_like(sup0) * 0.1,
                                    torch.randn_like(mass0), torch.randn_like(density0) * 0.1)

    reference = _reference_jvp(f, (pos0, sup0, mass0, density0), (dpos, dsup, dmass, ddensity))

    # Assemble Stage 1 -> 2 -> 3 with detached leaves, exactly mirroring the
    # production call graph f() builds above.
    V, dV = assembled_apparent_volume_jvp(pos0.detach(), sup0.detach(), dpos, dsup, dim, kernel_id)
    values, tangents = assembled_crk_moments_jvp(pos0.detach(), sup0.detach(), V, dpos, dsup, dV, dim, kernel_id)
    p = ParticleState(positions=pos0.detach(), supports=sup0.detach(), masses=mass0.detach(), densities=None, kinds=kinds)
    _, _, _, _, _, _, num_nbrs = _computeCRKMoments_stateBackend(
        p, OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Scatter, operationMode=OperationDirection.AllToAll),
        domain, queryVolumes=V.detach(), referenceVolumes=V.detach(), adjacency=adjacency,
    )
    (A, B, gradA, gradB), (dA, dB, dgradA, dgradB) = crk_terms_jvp(*values, *tangents, num_nbrs, sup0.detach())

    _, d = assembled_crk_gradient_jvp(pos0.detach(), sup0.detach(), mass0.detach(), density0.detach(), fv_q, fv_r,
                                       dpos, dsup, dmass, ddensity,
                                       A, B, gradA, gradB, dA, dB, dgradA, dgradB,
                                       dim, mode.value, kernel_id, scheme)
    return d, reference


def main():
    wp.init()
    print(__doc__.split("\n\n")[0])
    print()

    kernel_id = KERNEL.value
    ok = True

    print("Stage 1 (apparentVolume), 1D line of 7 / 2D 3x3 grid, non-uniform supports:")
    for dim, n in ((1, 7), (2, 3)):
        d, r = run_volume_case(n, dim, kernel_id)
        ok &= check(f"apparentVolume JVP (dim={dim})", d, r)

    print("\nStage 2 (CRK moments m_0/m_1/m_2/dm_*dgamma), 1D line of 7 / 2D 3x3 grid:")
    names = ("m_0", "m_1", "m_2", "dm_0dgamma", "dm_1dgamma", "dm_2dgamma")
    for dim, n in ((1, 7), (2, 3)):
        tangents, reference = run_moments_case(n, dim, kernel_id)
        for name, d, r in zip(names, tangents, reference):
            ok &= check(f"{name} JVP (dim={dim})", d, r)

    print("\nStage 3 (A, B, gradA, gradB via computeCRKTermsWarp's own JVP), 1D line of 7 / 2D 3x3 grid:")
    names = ("A", "B", "gradA", "gradB")
    for dim, n in ((1, 7), (2, 3)):
        crk_values, crk_jvps, reference = run_crk_factors_case(n, dim, kernel_id)
        for name, d, r in zip(names, crk_jvps, reference):
            ok &= check(f"{name} JVP (dim={dim})", d, r)

    print("\nStage 4 (CRK-corrected Gradient operator, end to end), 1D line of 7, GradientScheme x SupportScheme.Gather:")
    for scheme in GradientScheme:
        d, r = run_crk_gradient_case(7, 1, SupportScheme.Gather, scheme, kernel_id)
        ok &= check(f"CRK-corrected Gradient JVP ({scheme.name})", d, r)

    print("\nStage 4, 2D 3x3 grid, GradientScheme x SupportScheme.Gather:")
    for scheme in GradientScheme:
        d, r = run_crk_gradient_case(3, 2, SupportScheme.Gather, scheme, kernel_id)
        ok &= check(f"CRK-corrected Gradient JVP ({scheme.name})", d, r)

    print()
    if ok:
        print("ALL PASSED -- Tier 2.5's assembled JVP (apparentVolume's Gather-mode kernel-")
        print("  value JVP, chain-ruled through the moments' Scatter-mode product rule, then")
        print("  computeCRKTermsWarp's own exact double-backward JVP -- justified since it is")
        print("  pure PyTorch with no Warp call in it -- then correctGradientCRK's product-")
        print("  rule assembly and Tier 2.2's field-value coefficient) matches the production")
        print("  CRK pipeline's own reverse-mode derivative at every stage, end to end.")
    else:
        print("FAILED -- see the individual case(s) above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
