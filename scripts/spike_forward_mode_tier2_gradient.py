#!/usr/bin/env python3
"""Tier 2.2 forward-mode spike: Gradient, Divergence, Curl, Laplacian(Brookshaw)
(warpier_adjoint.md).

**The question this answers.** Tier 2.2's plan claims Gradient/Divergence/Curl
(and Laplacian's Brookshaw scheme) all route through the *same* `kernelGradient
= sphKernelGradient_ij(...)` building block (useCRK=False, no renormalization
-- those are Tier 2.4/2.5), so their Tier-2 JVP needs only ONE new kernel-level
derivation (`d(kernelGradient)/d{x,h}`, assembled from Tier 2.0's already-
validated `sphGradient_`/`sphKernelHessian_`/`sphGradientDkDh_`) plus ordinary
chain rule through each operator's field-value coefficient and (for Laplacian)
the `n_ij`/`r_ij` regularized-distance algebra. This script writes that down
and checks it end to end, same two-level methodology as Tier 2.1:

  * "assembled" side: `_kernelGradientJVP` (a new @wp.func, the direct analog
    of `sphKernelGradient_ij` -- see kernels/gradient.py) dispatches on
    SupportScheme exactly like that function, built only from already-
    validated kernel_sanity_native.py functions
    (`sphGradient_`/`sphKernelHessian_`/`sphGradientDkDh_`) plus Tier 2.1's
    `_pairwiseSupportTangent`. Everything downstream of that (the per-
    GradientScheme field-value coefficient, and Laplacian's n_ij/r_ij chain)
    is *ordinary* vector calculus -- done directly in torch on the dense
    (n,n) pair grid, not new kernel math, matching the plan's Tier 2.2 entry.
  * "reference" side: `torch.autograd.functional.jacobian` on the actual
    production `warpOperation(Gradient/Divergence/Curl/Laplacian)` call,
    contracted with the tangent -- identical pattern to Tier 2.1.

**Mathematical derivation.**

1. `kernelGradient` JVP (new kernel-level piece, mirrors sphKernelGradient_ij's
   three-way SupportScheme dispatch in kernels/gradient.py):

       KernelMeanSymmetric/SuperSymmetric (provably identical -- see below):
         G_ij = 0.5*(sphGradient_(x,hi) + sphGradient_(x,hj))
         dG_ij = 0.5*[ sphKernelHessian_(x,hi)@dx + sphKernelHessian_(x,hj)@dx
                       + sphGradientDkDh_(x,hi)*dhi + sphGradientDkDh_(x,hj)*dhj ]
       else (Gather/Scatter/MeanSymmetric/max-fallback):
         h_ij = computePairwiseSupport(hi,hj,mode), dh_ij = _pairwiseSupportTangent(...)
         G_ij = sphGradient_(x,h_ij)
         dG_ij = sphKernelHessian_(x,h_ij) @ dx + sphGradientDkDh_(x,h_ij)*dh_ij

   **SuperSymmetric is provably identical to KernelMeanSymmetric here too, not
   just at the value level (Tier 2.1's finding).** `sphKernelGradient_ij`'s
   SuperSymmetric branch is `(sphGradient_(x,hi) - sphGradient_(-x,hj))/2`.
   `sphGradient_` is odd in its position argument (direction = normalize(x),
   magnitude depends only on |x|), so `sphGradient_(-x,hj) = -sphGradient_(x,hj)`,
   collapsing the branch to `(sphGradient_(x,hi)+sphGradient_(x,hj))/2` --
   bit-for-bit KernelMeanSymmetric. The same oddness/evenness argument carries
   through the JVP: `sphKernelHessian_` (the Jacobian of an odd function) is
   *even* in x, and `sphGradientDkDh_` is *odd* in x (same structure as
   `sphGradient_`, one derivative order up in h) -- both confirmed directly
   from their closed forms below, not assumed. Differentiating the SuperSymmetric
   branch's literal formula (sign-flipped x argument and all) through these
   parities reduces algebraically to the exact same KernelMeanSymmetric JVP
   formula above. Checked explicitly (bit-for-bit assert), not just claimed.

2. Field-value coefficient (ordinary calculus -- the SAME weights are reused
   verbatim by Gradient, Divergence, Curl, and Laplacian, which is itself
   worth recording): every GradientScheme's per-neighbor coefficient reduces
   to `coeff_ij = fi*A_ij + fj*B_ij` (fi, fj frozen -- Tier 1 territory), where

       Vj = mass_j/density_j,  dVj = dmass_j/density_j - mass_j*ddensity_j/density_j^2
       Naive:      A=0,               B=Vj
       Difference: A=-Vj,             B=Vj
       Summation:  A=Vj,              B=Vj
       Symmetric:  A=mass_j/density_i,  B=mass_j*density_i/density_j^2

   (A, B differentiated the same way -- product/quotient rule, mass_j/density_i/
   density_j all differentiable, fi/fj frozen so contribute no term of their
   own). Gradient combines `coeff_ij` with `G_ij` via scalar multiplication
   (scalar field); Divergence via `dot(coeff_ij, G_ij)` (vector field,
   dotMode=False); Curl via the 2D scalar cross `G_ij.x*coeff_ij.y -
   G_ij.y*coeff_ij.x` (exactly `curlProduct`'s 2D formula, wp_cross.py) --
   all three are bilinear in `(coeff_ij, G_ij)`, so the JVP is just the
   product rule through that bilinear form: `d(coeff⋅G) = dcoeff⋅G + coeff⋅dG`.
   **Laplacian's `q_ij` (wp_laplacian.py) is `(fj-fi)*B_ij` -- literally the
   SAME `B_ij` as Gradient's coefficient, for every GradientScheme** (Naive/
   Difference/Summation's B is already Vj; Symmetric's B already matches
   `mass_j*density_i/density_j^2` exactly). Not a coincidence worth re-deriving
   -- this script computes B/dB once and reuses it for both.

3. Laplacian(Brookshaw)'s regularized-distance chain (ordinary calculus, no
   kernel math, but genuinely new to this tier):
   `D_ij = r_ij + eps*h_ij`, `n_ij = x_ij/D_ij`, `L_ij = -2*q_ij*dot(G_ij,n_ij)/D_ij`
   (`eps=1e-8`, matching wp_laplacian.py's literal constant -- not
   `get_epsilon(r)`, the two-argument form some other kernel files use; this
   script reuses the SAME constant, not the other convention, since it is
   checking THIS formula). `h_ij` here is `computePairwiseSupport` evaluated
   directly (max-fallback for KernelMeanSymmetric/SuperSymmetric too --
   `computePairwiseSupport` itself has no branch for either, see Tier 2.1's
   note 2 on `PartialSymmetric` for the same kind of silent fallthrough),
   independent of which branch `kernelGradient` itself took.
       dr_ij = dot(x_ij,dx_ij)/r_ij   -- EXCEPT at self-pairs (r_ij=0, the
               dense all-pairs loop includes i==j): production computes r_ij
               via `safe_sqrt(dot(x_ij,x_ij))`, whose custom `@wp.func_grad`
               (math/wp_sqrt.py) contributes NOTHING to the adjoint when its
               argument is <=0 -- i.e. dr_ij is defined to be exactly 0 at
               r_ij=0 by production's own convention, not a 0/0 this script
               invents. Guarded here the same way (`torch.where(r_ij>0, ...,
               0)`), the exact analog of Tier 2.1's `SupportScheme` subgradient
               guard. (Self-pairs end up contributing exactly zero to L_ij and
               dL_ij regardless, since G_ii=0 and dx_ii=0 identically -- this
               guard only avoids a spurious NaN from 0*inf on the way there.)
       dh_ij = _pairwiseSupportTangent(...) (Tier 2.1's building block, reused
               verbatim since `computePairwiseSupport`'s dispatch is unchanged)
       dD_ij = dr_ij + eps*dh_ij
       dn_ij = (dx_ij - n_ij*dD_ij)/D_ij         -- standard d(x/D)/dx identity
       dL_ij = -2*[ dq_ij*dot(G,n)/D + q_ij*((dot(dG,n)+dot(G,dn))/D
                                              - dot(G,n)*dD/D^2) ]

Two independent code paths, both exact (no finite differences on either side)
-- see the module docstring pattern in spike_forward_mode_tier2_density.py for
why the dense-all-pairs assembled side is safe to compare against the true
(adjacency-based) production reference: every kernel-derivative building block
here is exactly zero for q=|x|/h>1, so a pair outside the true support radius
contributes nothing to either side regardless of whether the real neighbor
search would have found it.

    python scripts/spike_forward_mode_tier2_gradient.py
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
from warpSPHCore.enumTypes import GradientScheme, LaplacianScheme, OperationDirection, SupportScheme, WarpOperation
from warpSPHCore.type_config import scalar_t
from warpSPHCore.kernels.gradient import sphGradient_
from warpSPHCore.kernels.hessian import sphKernelHessian_
from warpSPHCore.kernels.gradH import sphGradientDkDh_
from warpSPHCore.util.support import computePairwiseSupport
from warpSPHCore.math import matmul

TOL = 1e-9  # float64, both sides exact analytic derivatives -- round-off only
LAPLACIAN_EPS = 1e-8  # matches wp_laplacian.py's literal constant, not get_epsilon(r)

vec1_t = vector(dtype=scalar_t, length=1)
vec2_t = vector(dtype=scalar_t, length=2)


# --------------------------------------------------------------------------
# Tier 2.1's building block, reused verbatim (computePairwiseSupport's
# dispatch is unchanged by this tier).
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


# --------------------------------------------------------------------------
# New building block: the JVP of sphKernelGradient_ij itself (kernels/gradient.py).
# Generic over vector length (Any), like the production math funcs it calls.
# --------------------------------------------------------------------------

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
def _pair_jvp_grad_1d(
    xi: wp.array(dtype=vec1_t), xj: wp.array(dtype=vec1_t),
    hi: wp.array(dtype=scalar_t), hj: wp.array(dtype=scalar_t),
    dxi: wp.array(dtype=vec1_t), dxj: wp.array(dtype=vec1_t),
    dhi: wp.array(dtype=scalar_t), dhj: wp.array(dtype=scalar_t),
    mode: wp.uint32, kernel_id: wp.int32,
    G_out: wp.array(dtype=vec1_t), dG_out: wp.array(dtype=vec1_t),
):
    p = wp.tid()
    xij = xi[p] - xj[p]
    dxij = dxi[p] - dxj[p]
    g, dg = _kernelGradientJVP(xij, hi[p], hj[p], dxij, dhi[p], dhj[p], mode, kernel_id)
    G_out[p] = g
    dG_out[p] = dg


@wp.kernel
def _pair_jvp_grad_2d(
    xi: wp.array(dtype=vec2_t), xj: wp.array(dtype=vec2_t),
    hi: wp.array(dtype=scalar_t), hj: wp.array(dtype=scalar_t),
    dxi: wp.array(dtype=vec2_t), dxj: wp.array(dtype=vec2_t),
    dhi: wp.array(dtype=scalar_t), dhj: wp.array(dtype=scalar_t),
    mode: wp.uint32, kernel_id: wp.int32,
    G_out: wp.array(dtype=vec2_t), dG_out: wp.array(dtype=vec2_t),
):
    p = wp.tid()
    xij = xi[p] - xj[p]
    dxij = dxi[p] - dxj[p]
    g, dg = _kernelGradientJVP(xij, hi[p], hj[p], dxij, dhi[p], dhj[p], mode, kernel_id)
    G_out[p] = g
    dG_out[p] = dg


_PAIR_JVP_GRAD_BY_DIM = {1: _pair_jvp_grad_1d, 2: _pair_jvp_grad_2d}
_VEC_BY_DIM = {1: vec1_t, 2: vec2_t}


def _dense_kernelGradient_jvp(pos, sup, dpos, dsup, dim, mode, kernel_id):
    """All-pairs (i, j) including i==j, G_ij=kernelGradient and dG_ij as
    (n, n, dim) torch tensors. Safe for the same reason Tier 2.1's dense loop
    was: every building block is exactly zero for q>1, so pairs outside the
    true support radius contribute nothing on either side of the comparison."""
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

    G_out = wp.zeros(n * n, dtype=vec_t, device=DEVICE.type)
    dG_out = wp.zeros(n * n, dtype=vec_t, device=DEVICE.type)
    wp.launch(_PAIR_JVP_GRAD_BY_DIM[dim], dim=n * n,
              inputs=[xi, xj, hi, hj, dxi, dxj, dhi, dhj, mode, kernel_id],
              outputs=[G_out, dG_out], device=DEVICE.type)

    G = wp.to_torch(G_out).reshape(n, n, dim)
    dG = wp.to_torch(dG_out).reshape(n, n, dim)
    return G, dG


# --------------------------------------------------------------------------
# Ordinary calculus from here on: field-value coefficients (shared by all
# four operators) and Laplacian's n_ij/r_ij chain. Pure torch, no warp.
# --------------------------------------------------------------------------

def _gradient_weights(mass, density, dmass, ddensity, scheme: GradientScheme):
    """coeff_ij = fi*A_ij + fj*B_ij (fi, fj frozen -- Tier 1). Returns
    (A, B, dA, dB) as (1,n) or (n,n) torch tensors (broadcastable against a
    (n,1)-shaped fi and (1,n)-shaped fj). B doubles as Laplacian's q_ij
    coefficient -- see module docstring point 2."""
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


def _h_ij_and_tangent(sup, dsup, mode):
    """computePairwiseSupport's JVP, dense (n,n) -- the SAME dispatch as
    Tier 2.1's _pairwiseSupportTangent, just evaluated in torch instead of
    warp since it only ever feeds Laplacian's ordinary-calculus eps term
    here (not re-derived, just re-expressed). `mode` arrives as the raw
    SupportScheme.value int (every call site in this script passes mode.value
    for the warp kernel launch) -- coerce back to the enum, since comparing
    an int against SupportScheme members is always False and silently
    falls through to the `else` branch otherwise (caught this exact bug
    while writing this script: Gather/MeanSymmetric were silently getting
    the max-fallback h_ij instead, a small ~1e-9 relative error since h_ij
    only enters through the tiny eps=1e-8 regularization term)."""
    mode = SupportScheme(mode)
    hi, hj = sup.unsqueeze(1), sup.unsqueeze(0)
    dhi, dhj = dsup.unsqueeze(1), dsup.unsqueeze(0)
    if mode == SupportScheme.Gather:
        return hi.expand(-1, sup.shape[0]), dhi.expand(-1, sup.shape[0])
    elif mode == SupportScheme.Scatter:
        return hj.expand(sup.shape[0], -1), dhj.expand(sup.shape[0], -1)
    elif mode == SupportScheme.MeanSymmetric:
        return (hi + hj) / 2.0, (dhi + dhj) / 2.0
    else:
        h_ij = torch.maximum(hi, hj * torch.ones_like(hi))
        dh_ij = torch.where(hi >= hj, dhi * torch.ones_like(h_ij), dhj * torch.ones_like(h_ij))
        return h_ij, dh_ij


def assembled_gradient_jvp(pos, sup, mass, density, fv_q, fv_r, dpos, dsup, dmass, ddensity, dim, mode, kernel_id, scheme):
    G, dG = _dense_kernelGradient_jvp(pos, sup, dpos, dsup, dim, mode, kernel_id)
    A, B, dA, dB = _gradient_weights(mass, density, dmass, ddensity, scheme)
    fi = fv_q.unsqueeze(1)
    fj = fv_r.unsqueeze(0)
    coeff = fi * A + fj * B
    dcoeff = fi * dA + fj * dB
    out = (coeff.unsqueeze(-1) * G).sum(dim=1)
    d_out = (dcoeff.unsqueeze(-1) * G + coeff.unsqueeze(-1) * dG).sum(dim=1)
    return out, d_out


def assembled_divergence_jvp(pos, sup, mass, density, fv_q, fv_r, dpos, dsup, dmass, ddensity, dim, mode, kernel_id, scheme):
    """fv_q, fv_r: (n, dim) vector field. dotMode=False (production's
    divergenceProduct outputElements=1 case -- see wp_divdot.py)."""
    G, dG = _dense_kernelGradient_jvp(pos, sup, dpos, dsup, dim, mode, kernel_id)
    A, B, dA, dB = _gradient_weights(mass, density, dmass, ddensity, scheme)
    fi = fv_q.unsqueeze(1)  # (n,1,dim)
    fj = fv_r.unsqueeze(0)  # (1,n,dim)
    coeff = fi * A.unsqueeze(-1) + fj * B.unsqueeze(-1)
    dcoeff = fi * dA.unsqueeze(-1) + fj * dB.unsqueeze(-1)
    out = (coeff * G).sum(-1).sum(dim=1)
    d_out = ((dcoeff * G).sum(-1) + (coeff * dG).sum(-1)).sum(dim=1)
    return out, d_out


def assembled_curl_jvp(pos, sup, mass, density, fv_q, fv_r, dpos, dsup, dmass, ddensity, dim, mode, kernel_id, scheme):
    """2D only: production's curlProduct 2D overload gives
    R[0] = G.x*coeff.y - G.y*coeff.x (wp_cross.py)."""
    assert dim == 2
    G, dG = _dense_kernelGradient_jvp(pos, sup, dpos, dsup, dim, mode, kernel_id)
    A, B, dA, dB = _gradient_weights(mass, density, dmass, ddensity, scheme)
    fi = fv_q.unsqueeze(1)
    fj = fv_r.unsqueeze(0)
    coeff = fi * A.unsqueeze(-1) + fj * B.unsqueeze(-1)
    dcoeff = fi * dA.unsqueeze(-1) + fj * dB.unsqueeze(-1)

    def cross(g, c):
        return g[..., 0] * c[..., 1] - g[..., 1] * c[..., 0]

    out = cross(G, coeff).sum(dim=1)
    d_out = (cross(dG, coeff) + cross(G, dcoeff)).sum(dim=1)
    return out, d_out


def assembled_laplacian_brookshaw_jvp(pos, sup, mass, density, fv_q, fv_r, dpos, dsup, dmass, ddensity, dim, mode, kernel_id, scheme):
    """Scalar field. q_ij = (fj-fi)*B_ij (B from _gradient_weights, see
    module docstring point 2), L_ij = -2*q_ij*dot(G_ij,n_ij)/D_ij."""
    G, dG = _dense_kernelGradient_jvp(pos, sup, dpos, dsup, dim, mode, kernel_id)
    _, B, _, dB = _gradient_weights(mass, density, dmass, ddensity, scheme)
    fi = fv_q.unsqueeze(1)
    fj = fv_r.unsqueeze(0)
    q = (fj - fi) * B
    dq = (fj - fi) * dB

    x_ij = pos.unsqueeze(1) - pos.unsqueeze(0)          # (n,n,dim), row=i, col=j
    dx_ij = dpos.unsqueeze(1) - dpos.unsqueeze(0)
    r_ij = x_ij.norm(dim=-1)
    r_ij_safe = torch.where(r_ij > 0, r_ij, torch.ones_like(r_ij))
    # safe_sqrt's custom adjoint (math/wp_sqrt.py) contributes 0 when its
    # argument is <=0 -- i.e. production defines dr_ij=0 exactly at r_ij=0
    # (self-pairs), not 0/0. Matched here, not just NaN-avoided.
    dr_ij = torch.where(r_ij > 0, (x_ij * dx_ij).sum(-1) / r_ij_safe, torch.zeros_like(r_ij))

    h_ij, dh_ij = _h_ij_and_tangent(sup, dsup, mode)
    D_ij = r_ij + LAPLACIAN_EPS * h_ij
    dD_ij = dr_ij + LAPLACIAN_EPS * dh_ij
    n_ij = x_ij / D_ij.unsqueeze(-1)
    dn_ij = (dx_ij - n_ij * dD_ij.unsqueeze(-1)) / D_ij.unsqueeze(-1)

    dot_Gn = (G * n_ij).sum(-1)
    d_dot_Gn = (dG * n_ij).sum(-1) + (G * dn_ij).sum(-1)
    P = dot_Gn / D_ij
    dP = d_dot_Gn / D_ij - dot_Gn * dD_ij / D_ij**2

    L = -2.0 * q * P
    dL = -2.0 * (dq * P + q * dP)
    out = L.sum(dim=1)
    d_out = dL.sum(dim=1)
    return out, d_out


# --------------------------------------------------------------------------
# Reference: reverse-mode Jacobian of the PRODUCTION operator (Tier 2.1's pattern).
# --------------------------------------------------------------------------

def _reference_jvp(f, primals, tangents):
    J = torch.autograd.functional.jacobian(f, primals, vectorize=False)
    out = f(*primals).detach()
    n_out = out.numel()
    acc = torch.zeros(n_out, dtype=DTYPE, device=DEVICE)
    for Jk, vk in zip(J, tangents):
        acc = acc + Jk.reshape(n_out, -1) @ vk.reshape(-1)
    return acc.reshape(out.shape)


def check(name, assembled, reference):
    # Flatten rather than subtract as-is: a shape mismatch (e.g. Curl's
    # production output is (n,1), not (n,)) would otherwise broadcast to an
    # (n,n) matrix and silently compare the wrong elements against each
    # other instead of raising -- caught exactly this way once while writing
    # this script (Curl's rel_err ~1.5-1.8, not >1e9 as a real formula bug
    # near a division would produce, was the tell).
    assembled_flat, reference_flat = assembled.reshape(-1), reference.reshape(-1)
    assert assembled_flat.numel() == reference_flat.numel(), (
        f"{name}: shape mismatch assembled={tuple(assembled.shape)} reference={tuple(reference.shape)}"
    )
    scale = max(float(reference_flat.abs().max()), 1e-300)
    err = float((assembled_flat - reference_flat).abs().max()) / scale
    ok = err <= 1e-9
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


def run_gradient_case(n, dim, mode: SupportScheme, scheme: GradientScheme, kernel_id, seed=0):
    pos0, sup0, mass0, density0, domain, adjacency, kinds = _perturbed_case(n, dim, seed)

    def f(pos, sup, mass, density, qval, rval):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=density, kinds=kinds)
        return warpOperation(
            p, OperationProperties(kernel=KERNEL, operation=WarpOperation.Gradient,
                                    supportMode=mode, operationMode=OperationDirection.AllToAll, gradientMode=scheme),
            domain, queryValues=qval, referenceValues=rval, adjacency=adjacency,
        )

    for t in (pos0, sup0, mass0, density0):
        t.requires_grad_(True)
    dpos, dsup, dmass, ddensity = (torch.randn_like(pos0), torch.randn_like(sup0) * 0.1,
                                    torch.randn_like(mass0), torch.randn_like(density0) * 0.1)
    fv_q = torch.randn(pos0.shape[0], dtype=DTYPE)
    fv_r = torch.randn(pos0.shape[0], dtype=DTYPE)

    reference = _reference_jvp(f, (pos0, sup0, mass0, density0, fv_q, fv_r), (dpos, dsup, dmass, ddensity, torch.zeros_like(fv_q), torch.zeros_like(fv_r)))
    _, d = assembled_gradient_jvp(pos0.detach(), sup0.detach(), mass0.detach(), density0.detach(), fv_q, fv_r,
                                   dpos, dsup, dmass, ddensity, dim, mode.value, kernel_id, scheme)
    return d, reference


def run_divergence_case(n, dim, mode: SupportScheme, scheme: GradientScheme, kernel_id, seed=0):
    pos0, sup0, mass0, density0, domain, adjacency, kinds = _perturbed_case(n, dim, seed)

    def f(pos, sup, mass, density, qval, rval):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=density, kinds=kinds)
        return warpOperation(
            p, OperationProperties(kernel=KERNEL, operation=WarpOperation.Divergence,
                                    supportMode=mode, operationMode=OperationDirection.AllToAll,
                                    gradientMode=scheme, divergenceDotMode=False),
            domain, queryValues=qval, referenceValues=rval, adjacency=adjacency,
        )

    for t in (pos0, sup0, mass0, density0):
        t.requires_grad_(True)
    dpos, dsup, dmass, ddensity = (torch.randn_like(pos0), torch.randn_like(sup0) * 0.1,
                                    torch.randn_like(mass0), torch.randn_like(density0) * 0.1)
    fv_q = torch.randn(pos0.shape[0], dim, dtype=DTYPE)
    fv_r = torch.randn(pos0.shape[0], dim, dtype=DTYPE)

    reference = _reference_jvp(f, (pos0, sup0, mass0, density0, fv_q, fv_r), (dpos, dsup, dmass, ddensity, torch.zeros_like(fv_q), torch.zeros_like(fv_r)))
    _, d = assembled_divergence_jvp(pos0.detach(), sup0.detach(), mass0.detach(), density0.detach(), fv_q, fv_r,
                                     dpos, dsup, dmass, ddensity, dim, mode.value, kernel_id, scheme)
    return d, reference


def run_curl_case(n, dim, mode: SupportScheme, scheme: GradientScheme, kernel_id, seed=0):
    pos0, sup0, mass0, density0, domain, adjacency, kinds = _perturbed_case(n, dim, seed)

    def f(pos, sup, mass, density, qval, rval):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=density, kinds=kinds)
        return warpOperation(
            p, OperationProperties(kernel=KERNEL, operation=WarpOperation.Curl,
                                    supportMode=mode, operationMode=OperationDirection.AllToAll, gradientMode=scheme),
            domain, queryValues=qval, referenceValues=rval, adjacency=adjacency,
        )

    for t in (pos0, sup0, mass0, density0):
        t.requires_grad_(True)
    dpos, dsup, dmass, ddensity = (torch.randn_like(pos0), torch.randn_like(sup0) * 0.1,
                                    torch.randn_like(mass0), torch.randn_like(density0) * 0.1)
    fv_q = torch.randn(pos0.shape[0], dim, dtype=DTYPE)
    fv_r = torch.randn(pos0.shape[0], dim, dtype=DTYPE)

    reference = _reference_jvp(f, (pos0, sup0, mass0, density0, fv_q, fv_r), (dpos, dsup, dmass, ddensity, torch.zeros_like(fv_q), torch.zeros_like(fv_r)))
    _, d = assembled_curl_jvp(pos0.detach(), sup0.detach(), mass0.detach(), density0.detach(), fv_q, fv_r,
                               dpos, dsup, dmass, ddensity, dim, mode.value, kernel_id, scheme)
    return d, reference


def run_laplacian_case(n, dim, mode: SupportScheme, scheme: GradientScheme, kernel_id, seed=0):
    pos0, sup0, mass0, density0, domain, adjacency, kinds = _perturbed_case(n, dim, seed)

    def f(pos, sup, mass, density, qval, rval):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=density, kinds=kinds)
        return warpOperation(
            p, OperationProperties(kernel=KERNEL, operation=WarpOperation.Laplacian,
                                    supportMode=mode, operationMode=OperationDirection.AllToAll,
                                    gradientMode=scheme, laplacianMode=LaplacianScheme.Brookshaw),
            domain, queryValues=qval, referenceValues=rval, adjacency=adjacency,
        )

    for t in (pos0, sup0, mass0, density0):
        t.requires_grad_(True)
    dpos, dsup, dmass, ddensity = (torch.randn_like(pos0), torch.randn_like(sup0) * 0.1,
                                    torch.randn_like(mass0), torch.randn_like(density0) * 0.1)
    fv_q = torch.randn(pos0.shape[0], dtype=DTYPE)
    fv_r = torch.randn(pos0.shape[0], dtype=DTYPE)

    reference = _reference_jvp(f, (pos0, sup0, mass0, density0, fv_q, fv_r), (dpos, dsup, dmass, ddensity, torch.zeros_like(fv_q), torch.zeros_like(fv_r)))
    _, d = assembled_laplacian_brookshaw_jvp(pos0.detach(), sup0.detach(), mass0.detach(), density0.detach(), fv_q, fv_r,
                                              dpos, dsup, dmass, ddensity, dim, mode.value, kernel_id, scheme)
    return d, reference


def main():
    wp.init()
    print(__doc__.split("\n\n")[0])
    print()

    kernel_id = KERNEL.value
    ok = True

    print("Gradient, 1D line of 7 particles, non-uniform supports, all GradientScheme x all SupportScheme:")
    for scheme in GradientScheme:
        for mode in (SupportScheme.Gather, SupportScheme.Scatter, SupportScheme.MeanSymmetric,
                     SupportScheme.KernelMeanSymmetric, SupportScheme.SuperSymmetric, SupportScheme.PartialSymmetric):
            d, r = run_gradient_case(7, 1, mode, scheme, kernel_id)
            ok &= check(f"Gradient JVP ({scheme.name}/{mode.name})", d, r)

    print("\nGradient, 2D 3x3 grid, non-uniform supports:")
    for scheme in GradientScheme:
        for mode in (SupportScheme.Gather, SupportScheme.MeanSymmetric, SupportScheme.KernelMeanSymmetric):
            d, r = run_gradient_case(3, 2, mode, scheme, kernel_id)
            ok &= check(f"Gradient JVP ({scheme.name}/{mode.name})", d, r)

    print("\nSuperSymmetric == KernelMeanSymmetric for kernelGradient's JVP too (not just its value -- see docstring):")
    _, d_kms = run_gradient_case(7, 1, SupportScheme.KernelMeanSymmetric, GradientScheme.Naive, kernel_id, seed=1)
    _, d_ss = run_gradient_case(7, 1, SupportScheme.SuperSymmetric, GradientScheme.Naive, kernel_id, seed=1)
    identical = bool(torch.allclose(d_kms, d_ss, atol=1e-12))
    print(f"  [{'PASS' if identical else 'FAIL'}] assembled JVP bit-identical across the two schemes: {identical}")
    ok &= identical

    print("\nDivergence, 2D 3x3 grid, vector field, dotMode=False:")
    for scheme in GradientScheme:
        for mode in (SupportScheme.Gather, SupportScheme.MeanSymmetric, SupportScheme.KernelMeanSymmetric):
            d, r = run_divergence_case(3, 2, mode, scheme, kernel_id)
            ok &= check(f"Divergence JVP ({scheme.name}/{mode.name})", d, r)

    print("\nCurl, 2D 3x3 grid, vector field:")
    for scheme in GradientScheme:
        for mode in (SupportScheme.Gather, SupportScheme.KernelMeanSymmetric):
            d, r = run_curl_case(3, 2, mode, scheme, kernel_id)
            ok &= check(f"Curl JVP ({scheme.name}/{mode.name})", d, r)

    print("\nLaplacian (Brookshaw), 1D line of 7 particles, scalar field:")
    for scheme in GradientScheme:
        for mode in (SupportScheme.Gather, SupportScheme.MeanSymmetric):
            d, r = run_laplacian_case(7, 1, mode, scheme, kernel_id)
            ok &= check(f"Laplacian/Brookshaw JVP ({scheme.name}/{mode.name})", d, r)

    print("\nLaplacian (Brookshaw), 2D 3x3 grid, scalar field:")
    for scheme in GradientScheme:
        for mode in (SupportScheme.Gather, SupportScheme.MeanSymmetric, SupportScheme.KernelMeanSymmetric):
            d, r = run_laplacian_case(3, 2, mode, scheme, kernel_id)
            ok &= check(f"Laplacian/Brookshaw JVP ({scheme.name}/{mode.name})", d, r)

    print()
    if ok:
        print("ALL PASSED -- Tier 2.2's assembled JVP (a single new kernelGradient-JVP")
        print("  dispatch function, chain-ruled through each operator's field-value")
        print("  coefficient and, for Laplacian, its n_ij/r_ij regularized-distance")
        print("  algebra) matches the production operators' own reverse-mode derivative,")
        print("  for Gradient/Divergence/Curl/Laplacian(Brookshaw).")
    else:
        print("FAILED -- see the individual case(s) above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
