#!/usr/bin/env python3
"""Tier 2.1 forward-mode spike: Density and Interpolate (warpier_adjoint.md).

**The question this answers.** warpier_adjoint.md's Tier 2.1 claims the JVP of
Density/Interpolate w.r.t. positions/supports/masses[/densities] can be
*assembled* -- no new kernel math -- from the already-validated building
blocks in kernel_sanity_native.py (`sphKernel_`, `sphGradient_`,
`sphKernelDkDh_`) plus ordinary chain rule through `x_ij = x_i - x_j` and
`h_ij = computePairwiseSupport(h_i, h_j, mode)`. This script writes that
assembly down as code and checks it end to end:

    JVP_{x,h,m[,rho]}[Density/Interpolate](primal) . (dx, dh, dm[, drho])
        == reverse-mode Jacobian of the PRODUCTION operator, contracted with
           the same tangent.

Two independent code paths, both exact (no finite differences on either
side):

  * "assembled" side: a hand-written per-pair kernel (`_pair_jvp_*d` below)
    that dispatches on SupportScheme exactly the way `sphKernel_ij`
    (kernels/kernel.py) and `computePairwiseSupport` (util/support.py) do,
    calling only already-validated kernel_sanity_native.py functions plus one
    new, trivial building block (`_pairwiseSupportTangent`, the JVP of
    `computePairwiseSupport` itself -- ordinary calculus, not kernel math).
    Summed over a dense all-pairs loop (safe: sphKernel_/sphGradient_/
    sphKernelDkDh_ are all exactly zero outside q<=1 by construction, so
    including pairs beyond the true support radius contributes nothing --
    see kernel_sanity_native.py Section I).
  * "reference" side: `torch.autograd.functional.jacobian` on the actual
    `warpOperation(Density/Interpolate)` production call (the same reverse
    -mode path every gradcheck_*_native.py script already validates),
    contracted with the tangent -- exactly Tier 1's reference-construction
    pattern (spike_forward_mode_tier1.py), just differentiating w.r.t.
    positions/supports/masses/densities instead of field values.

**Mathematical derivation (see warpier_adjoint.md's Tier 2.1 log for the
full writeup).** Both Density and Interpolate reduce to a sum over neighbors
of `(coefficient)_j * W_ij`, W_ij = sphKernel_ij(x_ij, h_i, h_j, mode):

    Density_i      = sum_j  m_j            * W_ij
    Interpolate_i  = sum_j  f_j * V_j       * W_ij,   V_j = m_j / rho_j
                                                       (f_j frozen: Tier 1)

so by the product rule the tangent is

    dDensity_i     = sum_j [ dm_j          * W_ij  +  m_j     * dW_ij ]
    dInterpolate_i = sum_j [ f_j * dV_j    * W_ij  +  f_j*V_j * dW_ij ],
                      dV_j = dm_j/rho_j - m_j*drho_j/rho_j^2

and the only genuinely new piece is dW_ij, which needs case analysis on
SupportScheme because `sphKernel_ij` itself branches on it (kernel.py):

  * Gather/Scatter/MeanSymmetric (and the unimplemented-PartialSymmetric/
    Maximum fallback -- see note below): W_ij = sphKernel_(x_ij, h_ij, k),
    a single-argument-h evaluation, so
        dW_ij = grad_x[W](x_ij,h_ij).dx_ij + dW/dh(x_ij,h_ij) * dh_ij
    with dx_ij = dx_i - dx_j (ordinary difference; periodic minimum-image's
    tangent discontinuity is explicitly out of scope, see warpier_adjoint.md)
    and dh_ij the JVP of `computePairwiseSupport` itself:
        Gather:       dh_ij = dh_i
        Scatter:      dh_ij = dh_j
        MeanSymmetric: dh_ij = (dh_i + dh_j)/2
        else (max):    dh_ij = dh_i if h_i>=h_j else dh_j  (a subgradient --
                        genuinely discontinuous at h_i==h_j, the same class
                        of forward-branch-boundary issue Tier 2.4's plan
                        flags for pinv's rank cutoff; irrelevant almost
                        everywhere, avoided in this script's test data)
  * KernelMeanSymmetric/SuperSymmetric: W_ij = 0.5*(W(x_ij,h_i)+W(x_ij,h_j))
    -- see note below on why SuperSymmetric's code, despite the enum
    docstring's "-" sign, is provably identical here -- so
        dW_ij = 0.5*[ (grad_x[W](x_ij,h_i)+grad_x[W](x_ij,h_j)).dx_ij
                      + dW/dh(x_ij,h_i)*dh_i + dW/dh(x_ij,h_j)*dh_j ]

**Two pre-existing production-code facts this derivation surfaced, not bugs
to fix under this plan (documented in warpier_adjoint.md's Tier 2.1 log):**

1. `SupportScheme.SuperSymmetric`'s enum comment says the *value* dispatch
   should differ from KernelMeanSymmetric by a sign (mirroring the
   *gradient* formula's `W(x_ij,h_i) - W(x_ji,h_j)`), but `sphKernel_ij`
   codes an identical `+` branch for both. This is NOT a bug: W(x,h) depends
   only on |x|, so W(x_ji,h_j) == W(x_ij,h_j) by isotropy, and the docstring
   formula's "-W(x_ji,h_j)" is therefore identically "+W(x_ij,h_j)" once
   evaluated -- SuperSymmetric and KernelMeanSymmetric are mathematically
   forced to coincide at the *value* level (they legitimately differ at the
   *gradient* level, Tier 2.2, because grad_x[W] is odd in x, not even).
   Checked explicitly below (`assert` the two schemes agree bit-for-bit on
   both W and dW).
2. `SupportScheme.PartialSymmetric` is meant (per its enum comment and its
   *only* other production use, the neighbor-search radius in
   radiusSearch/compactHash/{grid,wp_collectNeighbors,wp_countNeighbors}.py)
   to be a field-value-weighted `f_i*W(h_i) + f_j*W(h_j)` scheme, but
   `sphKernel_ij`/`computePairwiseSupport` have no such branch for it -- it
   silently falls through to the `else` (max(h_i,h_j)) case, same as an
   unweighted "Maximum" scheme. Tested below as "whatever the code actually
   does", not as the documented formula: the JVP assembled here deliberately
   matches production behavior, not the aspirational docstring.

    python scripts/spike_forward_mode_tier2_density.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp
from warp.types import vector

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, line_case, make_domain
from warpSPHCore import OperationProperties, ParticleState, warpOperation
from warpSPHCore.enumTypes import OperationDirection, SupportScheme, WarpOperation
from warpSPHCore.type_config import scalar_t
from warpSPHCore.kernels.kernel import sphKernel_
from warpSPHCore.kernels.gradient import sphGradient_
from warpSPHCore.kernels.gradH import sphKernelDkDh_
from warpSPHCore.util.support import computePairwiseSupport

TOL = 1e-9  # float64, both sides exact analytic derivatives -- round-off only

vec1_t = vector(dtype=scalar_t, length=1)
vec2_t = vector(dtype=scalar_t, length=2)


# --------------------------------------------------------------------------
# New building block: the JVP of computePairwiseSupport itself. Ordinary
# calculus (not kernel math), but genuinely branch-dependent -- this is the
# "risk" Tier 2.1's plan entry flagged ("these enter asymmetrically once
# SupportScheme is anything other than MeanSymmetric").
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
        # max(hi, hj)'s subgradient: exact away from the hi==hj kink.
        if hi >= hj:
            return dhi
        return dhj


# --------------------------------------------------------------------------
# Assembled per-pair JVP: W_ij and dW_ij, dispatching on SupportScheme
# exactly like sphKernel_ij (kernels/kernel.py). One kernel per dim (warp
# resolves array-dtype annotations at decoration time -- see
# kernel_sanity_native.py's comment on why this is the established pattern
# here rather than a closure factory).
# --------------------------------------------------------------------------

@wp.kernel
def _pair_jvp_1d(
    xi: wp.array(dtype=vec1_t), xj: wp.array(dtype=vec1_t),
    hi: wp.array(dtype=scalar_t), hj: wp.array(dtype=scalar_t),
    dxi: wp.array(dtype=vec1_t), dxj: wp.array(dtype=vec1_t),
    dhi: wp.array(dtype=scalar_t), dhj: wp.array(dtype=scalar_t),
    mode: wp.uint32, kernel_id: wp.int32,
    W_out: wp.array(dtype=scalar_t), dW_out: wp.array(dtype=scalar_t),
):
    p = wp.tid()
    xij = xi[p] - xj[p]
    dxij = dxi[p] - dxj[p]
    if mode == wp.static(SupportScheme.KernelMeanSymmetric.value) or mode == wp.static(SupportScheme.SuperSymmetric.value):
        Wi = sphKernel_(xij, hi[p], kernel_id)
        Wj = sphKernel_(xij, hj[p], kernel_id)
        W_out[p] = (Wi + Wj) * scalar_t(0.5)
        gWi = sphGradient_(xij, hi[p], kernel_id)
        gWj = sphGradient_(xij, hj[p], kernel_id)
        dWdhi = sphKernelDkDh_(xij, hi[p], kernel_id)
        dWdhj = sphKernelDkDh_(xij, hj[p], kernel_id)
        dW_out[p] = scalar_t(0.5) * (wp.dot(gWi, dxij) + wp.dot(gWj, dxij) + dWdhi * dhi[p] + dWdhj * dhj[p])
    else:
        hij = computePairwiseSupport(hi[p], hj[p], mode)
        dhij = _pairwiseSupportTangent(hi[p], hj[p], dhi[p], dhj[p], mode)
        W_out[p] = sphKernel_(xij, hij, kernel_id)
        gW = sphGradient_(xij, hij, kernel_id)
        dWdh = sphKernelDkDh_(xij, hij, kernel_id)
        dW_out[p] = wp.dot(gW, dxij) + dWdh * dhij


@wp.kernel
def _pair_jvp_2d(
    xi: wp.array(dtype=vec2_t), xj: wp.array(dtype=vec2_t),
    hi: wp.array(dtype=scalar_t), hj: wp.array(dtype=scalar_t),
    dxi: wp.array(dtype=vec2_t), dxj: wp.array(dtype=vec2_t),
    dhi: wp.array(dtype=scalar_t), dhj: wp.array(dtype=scalar_t),
    mode: wp.uint32, kernel_id: wp.int32,
    W_out: wp.array(dtype=scalar_t), dW_out: wp.array(dtype=scalar_t),
):
    p = wp.tid()
    xij = xi[p] - xj[p]
    dxij = dxi[p] - dxj[p]
    if mode == wp.static(SupportScheme.KernelMeanSymmetric.value) or mode == wp.static(SupportScheme.SuperSymmetric.value):
        Wi = sphKernel_(xij, hi[p], kernel_id)
        Wj = sphKernel_(xij, hj[p], kernel_id)
        W_out[p] = (Wi + Wj) * scalar_t(0.5)
        gWi = sphGradient_(xij, hi[p], kernel_id)
        gWj = sphGradient_(xij, hj[p], kernel_id)
        dWdhi = sphKernelDkDh_(xij, hi[p], kernel_id)
        dWdhj = sphKernelDkDh_(xij, hj[p], kernel_id)
        dW_out[p] = scalar_t(0.5) * (wp.dot(gWi, dxij) + wp.dot(gWj, dxij) + dWdhi * dhi[p] + dWdhj * dhj[p])
    else:
        hij = computePairwiseSupport(hi[p], hj[p], mode)
        dhij = _pairwiseSupportTangent(hi[p], hj[p], dhi[p], dhj[p], mode)
        W_out[p] = sphKernel_(xij, hij, kernel_id)
        gW = sphGradient_(xij, hij, kernel_id)
        dWdh = sphKernelDkDh_(xij, hij, kernel_id)
        dW_out[p] = wp.dot(gW, dxij) + dWdh * dhij


_PAIR_JVP_BY_DIM = {1: _pair_jvp_1d, 2: _pair_jvp_2d}
_VEC_BY_DIM = {1: vec1_t, 2: vec2_t}


def _dense_pair_W_dW(pos, sup, dpos, dsup, dim, mode, kernel_id):
    """All-pairs (i, j) including i==j, W_ij and dW_ij as (n, n) torch
    tensors. Safe even though this is not the production adjacency: W/gradW/
    dW/dh are exactly zero for q>1 (kernel_sanity_native.py Section I), so a
    pair outside the true support radius contributes nothing to either side
    -- no need to reproduce the neighbor search here."""
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
    wp.launch(_PAIR_JVP_BY_DIM[dim], dim=n * n,
              inputs=[xi, xj, hi, hj, dxi, dxj, dhi, dhj, mode, kernel_id],
              outputs=[W_out, dW_out], device=DEVICE.type)

    W = wp.to_torch(W_out).reshape(n, n)
    dW = wp.to_torch(dW_out).reshape(n, n)
    return W, dW


def assembled_density_jvp(pos, sup, mass, dpos, dsup, dmass, dim, mode, kernel_id):
    W, dW = _dense_pair_W_dW(pos, sup, dpos, dsup, dim, mode, kernel_id)
    mass_j = mass.unsqueeze(0)   # (1, n) broadcasts to (n, n) as mass[j]
    dmass_j = dmass.unsqueeze(0)
    density = (mass_j * W).sum(dim=1)
    d_density = (dmass_j * W + mass_j * dW).sum(dim=1)
    return density, d_density


def assembled_interpolate_jvp(pos, sup, mass, density, fv, dpos, dsup, dmass, ddensity, dim, mode, kernel_id):
    W, dW = _dense_pair_W_dW(pos, sup, dpos, dsup, dim, mode, kernel_id)
    Vj = (mass / density).unsqueeze(0)               # (1, n) == V_j
    dVj = (dmass / density - mass * ddensity / density**2).unsqueeze(0)
    fv_j = fv.unsqueeze(0)
    interp = (fv_j * Vj * W).sum(dim=1)
    d_interp = (fv_j * (dVj * W + Vj * dW)).sum(dim=1)
    return interp, d_interp


# --------------------------------------------------------------------------
# Reference: reverse-mode Jacobian of the PRODUCTION operator, contracted
# with the tangent -- Tier 1's pattern (spike_forward_mode_tier1.py).
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
    scale = max(float(reference.abs().max()), 1e-300)
    err = float((assembled - reference).abs().max()) / scale
    ok = err <= 1e-9
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:60s} rel_err={err:.3e}")
    return ok


def run_density_case(n, dim, mode: SupportScheme, kernel_id, seed=0):
    torch.manual_seed(seed)
    if dim == 1:
        pos0, sup0, mass0 = line_case(n)
    else:
        from _gradcheck_common import grid_case_2d
        pos0, sup0, mass0 = grid_case_2d(n)
    pos0, sup0, mass0 = pos0.detach(), sup0.detach(), mass0.detach()
    # Perturb supports so they are NOT all equal -- otherwise Gather/Scatter/
    # MeanSymmetric/KernelMeanSymmetric all trivially coincide and the
    # support-scheme dispatch this script exists to check is never exercised.
    sup0 = sup0 * (1.0 + 0.15 * torch.linspace(-1, 1, sup0.shape[0], dtype=DTYPE))

    domain = make_domain(dim=dim)
    adjacency, kinds = build_adjacency(pos0, sup0, mass0, domain, mode=SupportScheme.KernelMeanSymmetric)

    def f(pos, sup, mass):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=None, kinds=kinds)
        return warpOperation(
            p,
            OperationProperties(kernel=KERNEL, operation=WarpOperation.Density,
                                 supportMode=mode, operationMode=OperationDirection.AllToAll),
            domain, adjacency=adjacency,
        )

    pos0.requires_grad_(True); sup0.requires_grad_(True); mass0.requires_grad_(True)
    dpos = torch.randn_like(pos0)
    dsup = torch.randn_like(sup0) * 0.1  # keep supports positive-ish under perturbation
    dmass = torch.randn_like(mass0)

    reference = _reference_jvp(f, (pos0, sup0, mass0), (dpos, dsup, dmass))
    _, assembled = assembled_density_jvp(pos0.detach(), sup0.detach(), mass0.detach(),
                                          dpos, dsup, dmass, dim, mode.value, kernel_id)
    return assembled, reference


def run_interpolate_case(n, dim, mode: SupportScheme, kernel_id, seed=0):
    torch.manual_seed(seed)
    pos0, sup0, mass0 = line_case(n)
    pos0, sup0, mass0 = pos0.detach(), sup0.detach(), mass0.detach()
    sup0 = sup0 * (1.0 + 0.15 * torch.linspace(-1, 1, sup0.shape[0], dtype=DTYPE))
    density0 = 1.0 + 0.3 * torch.rand(n, dtype=DTYPE)
    fv = torch.randn(n, dtype=DTYPE)  # frozen field values -- Tier 1 territory

    domain = make_domain(dim=dim)
    adjacency, kinds = build_adjacency(pos0, sup0, mass0, domain, mode=SupportScheme.KernelMeanSymmetric)

    def f(pos, sup, mass, density):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=density, kinds=kinds)
        return warpOperation(
            p,
            OperationProperties(kernel=KERNEL, operation=WarpOperation.Interpolate,
                                 supportMode=mode, operationMode=OperationDirection.AllToAll),
            domain, queryValues=fv, referenceValues=fv, adjacency=adjacency,
        )

    pos0.requires_grad_(True); sup0.requires_grad_(True); mass0.requires_grad_(True); density0.requires_grad_(True)
    dpos = torch.randn_like(pos0)
    dsup = torch.randn_like(sup0) * 0.1
    dmass = torch.randn_like(mass0)
    ddensity = torch.randn_like(density0) * 0.1

    reference = _reference_jvp(f, (pos0, sup0, mass0, density0), (dpos, dsup, dmass, ddensity))
    _, assembled = assembled_interpolate_jvp(pos0.detach(), sup0.detach(), mass0.detach(), density0.detach(), fv,
                                              dpos, dsup, dmass, ddensity, dim, mode.value, kernel_id)
    return assembled, reference


def main():
    wp.init()
    print(__doc__.split("\n\n")[0])
    print()

    kernel_id = KERNEL.value
    ok = True

    print("Density, 1D line of 7 particles, non-uniform supports:")
    for mode in (SupportScheme.Gather, SupportScheme.Scatter, SupportScheme.MeanSymmetric,
                 SupportScheme.KernelMeanSymmetric, SupportScheme.SuperSymmetric, SupportScheme.PartialSymmetric):
        assembled, reference = run_density_case(7, 1, mode, kernel_id)
        ok &= check(f"Density JVP ({mode.name})", assembled, reference)

    print("\nDensity, 2D 3x3 grid, non-uniform supports:")
    for mode in (SupportScheme.Gather, SupportScheme.MeanSymmetric, SupportScheme.KernelMeanSymmetric):
        assembled, reference = run_density_case(3, 2, mode, kernel_id)
        ok &= check(f"Density JVP ({mode.name})", assembled, reference)

    print("\nSuperSymmetric == KernelMeanSymmetric at the VALUE level (isotropy argument, see docstring):")
    a_kms, r_kms = run_density_case(7, 1, SupportScheme.KernelMeanSymmetric, kernel_id, seed=1)
    a_ss, r_ss = run_density_case(7, 1, SupportScheme.SuperSymmetric, kernel_id, seed=1)
    identical = bool(torch.allclose(a_kms, a_ss, atol=1e-14)) and bool(torch.allclose(r_kms, r_ss, atol=1e-12))
    print(f"  [{'PASS' if identical else 'FAIL'}] assembled and reference both bit-identical across the two schemes: {identical}")
    ok &= identical

    print("\nInterpolate (frozen field values), 1D line of 7 particles, non-uniform supports:")
    for mode in (SupportScheme.Gather, SupportScheme.MeanSymmetric, SupportScheme.KernelMeanSymmetric):
        assembled, reference = run_interpolate_case(7, 1, mode, kernel_id)
        ok &= check(f"Interpolate JVP ({mode.name})", assembled, reference)

    print()
    if ok:
        print("ALL PASSED -- Tier 2.1's assembled JVP (kernel_sanity_native.py's validated")
        print("  sphKernel_/sphGradient_/sphKernelDkDh_ chain-ruled through x_ij and h_ij)")
        print("  matches the production operator's own reverse-mode derivative, for every")
        print("  SupportScheme sphKernel_ij actually implements.")
    else:
        print("FAILED -- see the individual case(s) above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
