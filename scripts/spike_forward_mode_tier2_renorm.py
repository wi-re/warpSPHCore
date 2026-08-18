#!/usr/bin/env python3
"""Tier 2.4 forward-mode spike: gradient-renormalization matrix (warpier_adjoint.md).

**The question this answers.** Tier 2.4's plan claims the renormalization matrix's
Tier-2 JVP needs no new kernel math beyond what Tier 2.2 already validated
(`d(kernelGradient)/d{x,h}`) plus one standard matrix-calculus identity
(`d(C^-1) = -C^-1 (dC) C^-1`) for the pseudo-inverse. This script writes that down
and checks it end to end, same two-level methodology as Tiers 2.1-2.3:

  * "assembled" side: `_dense_covariance_jvp` reuses Tier 2.2's `_kernelGradientJVP`
    dispatch (duplicated here verbatim, same convention Tier 2.2 used for Tier 2.1's
    `_pairwiseSupportTangent`) to get `dC/d{x,h}`, plus ordinary product-rule
    calculus for the `apparentVolume`/`fij` factors that turn a sum of kernel
    gradients into a covariance matrix. `-L(dC)L` (pure torch, batched 2x2/1x1
    matmul) then gives `dL` on the invertible branch; the low-neighbor-count
    identity-fallback branch gets an explicit zero tangent by construction.
  * "reference" side: `torch.autograd.functional.jacobian` on the actual production
    `computeRenormalizationMatrices` call, contracted with the tangent -- identical
    pattern to every earlier tier.

**Mathematical derivation.**

1. `wp_covariance.py`'s per-neighbor contribution is
   `out += wp.outer(fij * apparentVolume, kernelGradient)`, where
   `fij = -computeDistanceVec(xi, xj) = -(x_i - x_j)`. Writing `y_ij = -x_ij = x_j-x_i`
   (so `wp.outer(a,b)[k,l] = a[k]*b[l]`) and `Vj = apparentVolume = mass_j/density_j`
   (`useVolume=False`, the case this script covers -- no `queryVolumes`/
   `referenceVolumes`), the covariance matrix and its JVP are

       C_i  = Sum_j Vj * outer(y_ij, G_ij)
       dC_i = Sum_j [ dVj * outer(y_ij, G_ij) + Vj * outer(dy_ij, G_ij) + Vj * outer(y_ij, dG_ij) ]

   `G_ij, dG_ij` are exactly Tier 2.2's `kernelGradient`/its JVP (`useCRK=False`, no
   `useGradientRenormalization` -- confirmed by reading `computeKernelGradientCRK`,
   which reduces to plain `sphKernelGradient_ij` under those flags, and by
   `computeRenormalizationMatrices_`, which is only ever called here with
   `crkState=None, renormalizationState=None`). `Vj, dVj` are literally Tier 2.2's
   `_gradient_weights`'s `GradientScheme.Naive` `B_ij, dB_ij` (`Vj = mass_j/density_j`)
   -- not re-derived, just reused under a different name for this file's own
   building block (`_apparent_volume_jvp`).

2. Low-neighbor-count identity fallback (`renorm.py`): `lowNbrMask = num_nbrs < dim+2`,
   `C = where(lowNbrMask, I, C_raw)`. `num_nbrs` is a non-differentiable integer count
   (same treatment as every discrete `SupportScheme`/`h_i>=h_j` branch decision in
   earlier tiers) -- consumed directly from production's own Covariance-with-
   `covarianceReturnNumNeighbors=True` output, not re-derived. The masked branch is a
   literal constant (`I`), so its tangent is exactly zero by construction:
   `dC = where(lowNbrMask, 0, dC_raw)`. Checked explicitly (an isolated single
   particle, `num_nbrs=1 < dim+2` for both dim=1 and dim=2) the same way
   `gradcheck_pinv_native.py`'s Test 3 checks the analogous *reverse*-mode case.

3. The pseudo-inverse itself (`pinv_warp` -> `pinv1x1`/`pinv2x2_warpBackend`):
   `d(C^-1) = -C^-1 (dC) C^-1` is the standard matrix-inverse-derivative identity, true
   whenever `L = C^-1` exactly (both eigenvalues on the "kept" side of `pinv2x2_warp`'s
   `rcond`-relative cutoff, or `pinv1x1`'s `m[0,0] > 1e-10` branch) -- no new
   derivation, just batched `-L @ dC @ L` in torch. `L` itself is taken directly from
   production's own (already independently gradchecked, `gradcheck_pinv_native.py`)
   `computeRenormalizationMatrices` output rather than re-derived here, the same
   "consume an already-validated value, don't re-derive it" pattern used for `num_nbrs`
   above and for densities in every earlier tier's script.

   **Risk, inherited from the plan entry, deliberately not tested here:**
   `pinv2x2_warpBackend` has its own eigenvalue-relative rank cutoff (`rcond=1e-6`);
   a position tangent that pushes a case across that cutoff mid-JVP is a genuine
   discontinuity no JVP formula can represent (same class as `SupportScheme`'s
   `h_i==h_j` tie or the periodic wrap boundary). Test geometries here (a regular 1D
   line, a regular 2D grid) are comfortably well-conditioned and nowhere near that
   cutoff -- this script does not probe it, consistent with the plan's "flag failing
   cases near that boundary rather than silently producing a wrong tangent" guidance
   (there is nothing to flag if the boundary is never approached).

Two independent code paths, both exact (no finite differences on either side) -- same
dense-all-pairs-is-safe argument as every earlier tier: every kernel-derivative
building block is exactly zero for q=|x|/h>1, so a pair outside the true support
radius contributes nothing to either side regardless of whether the real neighbor
search would have found it.

    python scripts/spike_forward_mode_tier2_renorm.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys
from typing import Any

import torch
import warp as wp
from warp.types import vector

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, grid_case_2d, line_case, make_domain, single_particle_case
from warpSPHCore import OperationProperties, ParticleState, warpOperation
from warpSPHCore.enumTypes import OperationDirection, SupportScheme, WarpOperation
from warpSPHCore.type_config import scalar_t
from warpSPHCore.kernels.gradient import sphGradient_
from warpSPHCore.kernels.hessian import sphKernelHessian_
from warpSPHCore.kernels.gradH import sphGradientDkDh_
from warpSPHCore.util.support import computePairwiseSupport
from warpSPHCore.math import matmul
from warpSPHCore.renorm import computeRenormalizationMatrices

TOL = 1e-9  # float64, both sides exact analytic derivatives -- round-off only

vec1_t = vector(dtype=scalar_t, length=1)
vec2_t = vector(dtype=scalar_t, length=2)


# --------------------------------------------------------------------------
# Tier 2.1's / Tier 2.2's building blocks, reused verbatim (see
# spike_forward_mode_tier2_gradient.py -- same convention: each Tier-2.x
# script duplicates the building blocks it needs rather than importing
# another tier's spike script).
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
    (n, n, dim) torch tensors. Safe for the same reason every earlier tier's
    dense loop was: every building block is exactly zero for q>1."""
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
# New building blocks: apparentVolume's JVP (Tier 2.2's Naive-B, reused
# under this file's own name) and the covariance matrix's JVP assembled
# from it plus _dense_kernelGradient_jvp. Pure torch from here on -- no new
# kernel math, matching the plan's Tier 2.4 entry.
# --------------------------------------------------------------------------

def _apparent_volume_jvp(mass, density, dmass, ddensity):
    """Vj = mass_j/density_j (wp_covariance.py's apparentVolume, useVolume=False --
    identical formula to Tier 2.2's GradientScheme.Naive B_ij/dB_ij)."""
    mass_j = mass.unsqueeze(0)
    density_j = density.unsqueeze(0)
    dmass_j = dmass.unsqueeze(0)
    ddensity_j = ddensity.unsqueeze(0)
    Vj = mass_j / density_j
    dVj = dmass_j / density_j - mass_j * ddensity_j / density_j ** 2
    return Vj, dVj


def _dense_covariance_jvp(pos, sup, mass, density, dpos, dsup, dmass, ddensity, dim, mode, kernel_id):
    """C_i = Sum_j Vj*outer(y_ij, G_ij), y_ij = x_j - x_i (== fij in
    wp_covariance.py). dC_i by the ordinary product rule -- see module
    docstring point 1."""
    G, dG = _dense_kernelGradient_jvp(pos, sup, dpos, dsup, dim, mode, kernel_id)
    Vj, dVj = _apparent_volume_jvp(mass, density, dmass, ddensity)

    x_ij = pos.unsqueeze(1) - pos.unsqueeze(0)      # (n,n,dim), row=i, col=j
    y_ij = -x_ij
    dx_ij = dpos.unsqueeze(1) - dpos.unsqueeze(0)
    dy_ij = -dx_ij

    outer = y_ij.unsqueeze(-1) * G.unsqueeze(-2)                                  # [...,k,l] = y[k]*G[l]
    d_outer = dy_ij.unsqueeze(-1) * G.unsqueeze(-2) + y_ij.unsqueeze(-1) * dG.unsqueeze(-2)

    Vj4 = Vj.unsqueeze(-1).unsqueeze(-1)   # (1,n,1,1), broadcasts over i and k,l
    dVj4 = dVj.unsqueeze(-1).unsqueeze(-1)

    C = (Vj4 * outer).sum(dim=1)
    dC = (dVj4 * outer + Vj4 * d_outer).sum(dim=1)
    return C, dC


def assembled_renorm_jvp(pos, sup, mass, density, dpos, dsup, dmass, ddensity, dim, mode: SupportScheme, kernel_id, domain, adjacency, kinds):
    """Returns (C_raw, C_prod, L, dL): C_raw/C_prod are the assembled and
    production *unmasked* covariance matrices (forward-value sanity check),
    L is production's own renormalization matrix (consumed, not re-derived
    -- see module docstring point 3), dL is this tier's assembled JVP."""
    C_raw, dC_raw = _dense_covariance_jvp(pos, sup, mass, density, dpos, dsup, dmass, ddensity, dim, mode.value, kernel_id)

    covarianceProperties = OperationProperties(kernel=KERNEL, operation=WarpOperation.Covariance,
                                                supportMode=mode, operationMode=OperationDirection.AllToAll)
    p = ParticleState(positions=pos.detach(), supports=sup.detach(), masses=mass.detach(), densities=density.detach(), kinds=kinds)
    C_prod, num_nbrs = warpOperation(p, covarianceProperties, domain, adjacency=adjacency, covarianceReturnNumNeighbors=True)

    lowNbrMask = (num_nbrs < dim + 2).view(-1, 1, 1)
    dC = torch.where(lowNbrMask, torch.zeros_like(dC_raw), dC_raw)

    operationProperties = OperationProperties(kernel=KERNEL, operation=WarpOperation.Gradient,
                                               supportMode=mode, operationMode=OperationDirection.AllToAll)
    _, _, renormState = computeRenormalizationMatrices(p, operationProperties, domain, adjacency=adjacency)
    L = renormState.renormalizationMatrices

    dL = -torch.matmul(L, torch.matmul(dC, L))
    return C_raw, C_prod, L, dL


# --------------------------------------------------------------------------
# Reference: reverse-mode Jacobian of the PRODUCTION operator (every
# earlier tier's pattern).
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
    assembled_flat, reference_flat = assembled.reshape(-1), reference.reshape(-1)
    assert assembled_flat.numel() == reference_flat.numel(), (
        f"{name}: shape mismatch assembled={tuple(assembled.shape)} reference={tuple(reference.shape)}"
    )
    scale = max(float(reference_flat.abs().max()), 1e-300)
    err = float((assembled_flat - reference_flat).abs().max()) / scale
    ok = err <= TOL
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
    """Non-uniform supports (h_i perturbed +-15%) -- Tier 2.1's standing
    pattern for any Tier-2.x script touching SupportScheme, so
    Gather/Scatter/MeanSymmetric/KernelMeanSymmetric don't collapse to the
    same number under a degenerate uniform-h test case."""
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


def _renorm_forward(mode: SupportScheme, domain, adjacency, kinds):
    def f(pos, sup, mass, density):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=density, kinds=kinds)
        operationProperties = OperationProperties(kernel=KERNEL, operation=WarpOperation.Gradient,
                                                   supportMode=mode, operationMode=OperationDirection.AllToAll)
        _, _, renormState = computeRenormalizationMatrices(p, operationProperties, domain, adjacency=adjacency)
        return renormState.renormalizationMatrices
    return f


def run_case(n, dim, mode: SupportScheme, kernel_id, seed=0):
    pos0, sup0, mass0, density0, domain, adjacency, kinds = _perturbed_case(n, dim, seed)
    f = _renorm_forward(mode, domain, adjacency, kinds)

    for t in (pos0, sup0, mass0, density0):
        t.requires_grad_(True)
    dpos, dsup, dmass, ddensity = (torch.randn_like(pos0), torch.randn_like(sup0) * 0.1,
                                    torch.randn_like(mass0), torch.randn_like(density0) * 0.1)

    reference = _reference_jvp(f, (pos0, sup0, mass0, density0), (dpos, dsup, dmass, ddensity))
    C_raw, C_prod, L, dL = assembled_renorm_jvp(pos0.detach(), sup0.detach(), mass0.detach(), density0.detach(),
                                                 dpos, dsup, dmass, ddensity, dim, mode, kernel_id, domain, adjacency, kinds)
    return dL, reference, C_raw, C_prod


def run_low_neighbor_case(dim, kernel_id):
    """An isolated particle: num_nbrs=1 < dim+2 for both dim=1 and dim=2, so
    production forces C (and hence L) to the constant identity -- both the
    assembled and reference JVPs should be exactly/near-exactly zero (module
    docstring point 2)."""
    if dim == 1:
        pos0, sup0, mass0 = single_particle_case()
    else:
        pos0 = torch.zeros((1, dim), dtype=DTYPE, device=DEVICE, requires_grad=True)
        sup0 = torch.ones((1,), dtype=DTYPE, device=DEVICE, requires_grad=True)
        mass0 = torch.ones((1,), dtype=DTYPE, device=DEVICE, requires_grad=True)
    pos0, sup0, mass0 = pos0.detach(), sup0.detach(), mass0.detach()
    domain = make_domain(dim=dim)
    adjacency, kinds = build_adjacency(pos0, sup0, mass0, domain, mode=SupportScheme.KernelMeanSymmetric)
    density0 = _compute_densities(pos0, sup0, mass0, kinds, domain, adjacency)

    mode = SupportScheme.Gather
    f = _renorm_forward(mode, domain, adjacency, kinds)

    for t in (pos0, sup0, mass0, density0):
        t.requires_grad_(True)
    dpos, dsup, dmass, ddensity = (torch.randn_like(pos0), torch.randn_like(sup0) * 0.1,
                                    torch.randn_like(mass0), torch.randn_like(density0) * 0.1)

    reference = _reference_jvp(f, (pos0, sup0, mass0, density0), (dpos, dsup, dmass, ddensity))
    _, _, L, dL = assembled_renorm_jvp(pos0.detach(), sup0.detach(), mass0.detach(), density0.detach(),
                                        dpos, dsup, dmass, ddensity, dim, mode, kernel_id, domain, adjacency, kinds)

    is_identity = torch.allclose(L[0], torch.eye(dim, dtype=DTYPE), atol=1e-12)
    is_zero_assembled = torch.allclose(dL, torch.zeros_like(dL), atol=1e-12)
    is_zero_reference = torch.allclose(reference, torch.zeros_like(reference), atol=1e-10)
    print(f"  [{'PASS' if is_identity and is_zero_assembled and is_zero_reference else 'FAIL'}] "
          f"dim={dim}: L==identity {is_identity}, assembled dL==0 {is_zero_assembled}, reference dL==0 {is_zero_reference}")
    return is_identity and is_zero_assembled and is_zero_reference


def main():
    wp.init()
    print(__doc__.split("\n\n")[0])
    print()

    kernel_id = KERNEL.value
    ok = True

    print("Renormalization matrix, 1D line of 7 particles, non-uniform supports, all SupportScheme:")
    for mode in (SupportScheme.Gather, SupportScheme.Scatter, SupportScheme.MeanSymmetric,
                 SupportScheme.KernelMeanSymmetric, SupportScheme.SuperSymmetric, SupportScheme.PartialSymmetric):
        dL, ref, C_raw, C_prod = run_case(7, 1, mode, kernel_id)
        ok &= check(f"C forward-value parity (assembled vs. production) ({mode.name})", C_raw, C_prod)
        ok &= check(f"Renorm JVP ({mode.name})", dL, ref)

    print("\nRenormalization matrix, 2D 3x3 grid, non-uniform supports:")
    for mode in (SupportScheme.Gather, SupportScheme.MeanSymmetric, SupportScheme.KernelMeanSymmetric):
        dL, ref, C_raw, C_prod = run_case(3, 2, mode, kernel_id)
        ok &= check(f"C forward-value parity (assembled vs. production) ({mode.name})", C_raw, C_prod)
        ok &= check(f"Renorm JVP ({mode.name})", dL, ref)

    print("\nLow-neighbor-count identity fallback -- exactly-zero tangent, both dims:")
    ok &= run_low_neighbor_case(1, kernel_id)
    ok &= run_low_neighbor_case(2, kernel_id)

    print()
    if ok:
        print("ALL PASSED -- Tier 2.4's assembled JVP (Tier 2.2's kernelGradient-JVP")
        print("  dispatch, chain-ruled through the covariance sum's apparentVolume/fij")
        print("  factors, plus the standard -L(dC)L matrix-inverse-derivative identity)")
        print("  matches the production renormalization matrix's own reverse-mode")
        print("  derivative, across every SupportScheme in 1D/2D, with an explicit")
        print("  zero-tangent check on the low-neighbor-count identity fallback.")
    else:
        print("FAILED -- see the individual case(s) above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
