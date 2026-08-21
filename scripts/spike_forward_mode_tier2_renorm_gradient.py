#!/usr/bin/env python3
"""Tier 2.4b forward-mode spike: renormalization-corrected Gradient JVP
(`warpier_adjoint.md`, `warpier_tier2_correction_jvp_plan.md` phase (d), Step
2 -- "the one piece of this entire plan needing a fresh derivation
write-up").

**The question this answers.** `wp_gradient.py`'s renormalization composition
is `kernelGradient_final = L_i @ kernelGradient_corrected` (`matmul` of the
per-query renormalization matrix against whatever the CRK dispatch already
produced -- `renorm alone` here, `kernelGradient_corrected == kernelGradient`,
matching this plan phase's own "scope to renorm alone (CRK off) first"
decision). Its JVP is the ordinary product rule on that matrix-vector
product:

    dKernelGradient_final = dL_i @ kernelGradient_corrected + L_i @ dKernelGradient_corrected

`dL_i` is Tier 2.4's own already-validated JVP
(`spike_forward_mode_tier2_renorm.py`: the covariance matrix's JVP,
masked by the low-neighbor-count identity fallback, pushed through
`d(C^-1) = -C^-1(dC)C^-1`); `(kernelGradient_corrected, dKernelGradient_corrected)`
is Tier 2.2's already-validated plain (non-CRK) kernel-gradient JVP. No new
matrix calculus beyond what Tier 2.4 already proved for `dL` itself -- this
script's only new content is chaining that matrix-vector product rule onto
Tier 2.2's Gradient-operator coefficient assembly (`coeff_ij = fi*A_ij +
fj*B_ij`, unchanged from Tier 2.2/2.4/2.5 -- renormalization does not touch
the coefficient, only the kernel-gradient factor it multiplies).

Both Tier 2.2's dense `(G, dG)` builder and Tier 2.4's covariance/pinv-
derivative assembly are duplicated here verbatim rather than imported from
their own spike scripts -- the standing per-script convention
(`spike_forward_mode_tier2_renorm.py`'s own module docstring: "each Tier-2.x
script duplicates the building blocks it needs rather than importing another
tier's spike script").

**Reference.** `torch.autograd.functional.jacobian` on the actual production
`warpOperation(Gradient, renormalizationState=...)`, with `renormalizationState`
itself computed by production's own `computeRenormalizationMatrices` from the
SAME leaf positions/supports the Gradient operator differentiates through
(mirrors `spike_forward_mode_tier2_crk.py`'s `run_crk_gradient_case`, which
builds its `crkState` from the same leaves for the identical reason: the
production call graph really does differentiate through both stages when a
caller supplies a correction state computed from the geometry being moved).

**Carried forward, not solved here** (same as `spike_forward_mode_tier2_renorm.py`'s
own final section): `pinv2x2_warpBackend`'s `rcond=1e-6` eigenvalue-relative
cutoff is a genuine JVP discontinuity if a tangent pushes a case across it
mid-JVP -- test geometries below are the same well-conditioned line/grid
cases every earlier Tier-2 script used, nowhere near that cutoff, so this
script does not probe it.

    python scripts/spike_forward_mode_tier2_renorm_gradient.py
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
from warpSPHCore.enumTypes import GradientScheme, OperationDirection, SupportScheme, WarpOperation
from warpSPHCore.type_config import scalar_t
from warpSPHCore.kernels.gradient import sphGradient_
from warpSPHCore.kernels.hessian import sphKernelHessian_
from warpSPHCore.kernels.gradH import sphGradientDkDh_
from warpSPHCore.util.support import computePairwiseSupport
from warpSPHCore.math import matmul
from warpSPHCore.renorm import computeRenormalizationMatrices

TOL = 1e-8  # float64, both sides exact analytic derivatives -- round-off only
             # (slightly looser than Tier 2.4's own 1e-9: this script chains
             # TWO already-1e-9-validated JVPs together through a product
             # rule, so round-off compounds a little further).

vec1_t = vector(dtype=scalar_t, length=1)
vec2_t = vector(dtype=scalar_t, length=2)


# --------------------------------------------------------------------------
# Tier 2.1's / Tier 2.2's building blocks, duplicated verbatim (see module
# docstring's duplication-convention note).
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
    """All-pairs (i, j) including i==j, G_ij and dG_ij as (n, n, dim) torch
    tensors -- safe for the usual dense-all-pairs-is-safe reason (kernel
    building blocks are exactly zero for q=|x|/h>1)."""
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


def _gradient_weights(mass, density, dmass, ddensity, scheme: GradientScheme):
    """Tier 2.2's _gradient_weights, duplicated verbatim -- coeff_ij's
    mass/density-based Vj is untouched by renormalization."""
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


# --------------------------------------------------------------------------
# Tier 2.4's building blocks, duplicated verbatim from
# spike_forward_mode_tier2_renorm.py (module docstring's duplication
# convention) -- apparentVolume/covariance/pinv-derivative assembly.
# --------------------------------------------------------------------------

def _apparent_volume_jvp(mass, density, dmass, ddensity):
    """Vj = mass_j/density_j (wp_covariance.py's apparentVolume, useVolume=False)."""
    mass_j = mass.unsqueeze(0)
    density_j = density.unsqueeze(0)
    dmass_j = dmass.unsqueeze(0)
    ddensity_j = ddensity.unsqueeze(0)
    Vj = mass_j / density_j
    dVj = dmass_j / density_j - mass_j * ddensity_j / density_j ** 2
    return Vj, dVj


def _dense_covariance_jvp(pos, sup, mass, density, dpos, dsup, dmass, ddensity, dim, mode, kernel_id):
    """C_i = Sum_j Vj*outer(y_ij, G_ij), y_ij = x_j - x_i. dC_i by the
    ordinary product rule -- see spike_forward_mode_tier2_renorm.py's module
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


def assembled_L_dL(pos, sup, mass, density, dpos, dsup, dmass, ddensity, dim, mode: SupportScheme, kernel_id, domain, adjacency, kinds):
    """Tier 2.4's own assembled `(L, dL)`, duplicated from
    spike_forward_mode_tier2_renorm.py's `assembled_renorm_jvp` (dropping the
    forward-value-parity return values this script doesn't need)."""
    C_raw, dC_raw = _dense_covariance_jvp(pos, sup, mass, density, dpos, dsup, dmass, ddensity, dim, mode.value, kernel_id)

    covarianceProperties = OperationProperties(kernel=KERNEL, operation=WarpOperation.Covariance,
                                                supportMode=mode, operationMode=OperationDirection.AllToAll)
    p = ParticleState(positions=pos.detach(), supports=sup.detach(), masses=mass.detach(), densities=density.detach(), kinds=kinds)
    _, num_nbrs = warpOperation(p, covarianceProperties, domain, adjacency=adjacency, covarianceReturnNumNeighbors=True)

    lowNbrMask = (num_nbrs < dim + 2).view(-1, 1, 1)
    dC = torch.where(lowNbrMask, torch.zeros_like(dC_raw), dC_raw)

    operationProperties = OperationProperties(kernel=KERNEL, operation=WarpOperation.Gradient,
                                               supportMode=mode, operationMode=OperationDirection.AllToAll)
    _, _, renormState = computeRenormalizationMatrices(p, operationProperties, domain, adjacency=adjacency)
    L = renormState.renormalizationMatrices

    dL = -torch.matmul(L, torch.matmul(dC, L))
    return L, dL


# --------------------------------------------------------------------------
# NEW (this script): the renormalization-corrected Gradient operator's own
# JVP -- the matrix-vector product rule on `kernelGradient_final = L_i @ G_ij`
# chained into Tier 2.2's coeff_ij assembly. See module docstring.
# --------------------------------------------------------------------------

def assembled_renorm_gradient_jvp(pos, sup, mass, density, fv_q, fv_r, dpos, dsup, dmass, ddensity,
                                   L, dL, dim, mode, kernel_id, scheme):
    """`L`/`dL`: (n,dim,dim), per-QUERY-particle (Tier 2.4's output),
    broadcast over neighbor `j` exactly like `getL_i` (util/stateUtil.py)
    does in production -- constant across the `j` sum, same convention as
    every other per-query correction term in this codebase."""
    G, dG = _dense_kernelGradient_jvp(pos, sup, dpos, dsup, dim, mode, kernel_id)

    # kernelGradient_final_ij = L_i @ G_ij; dKernelGradient_final_ij =
    # dL_i @ G_ij + L_i @ dG_ij -- the ordinary product rule this script
    # exists to validate (module docstring). einsum contracts L's second
    # (column) axis against G, matching wp.matmul(L, G)'s convention.
    G_final = torch.einsum("ikl,ijl->ijk", L, G)
    dG_final = torch.einsum("ikl,ijl->ijk", dL, G) + torch.einsum("ikl,ijl->ijk", L, dG)

    A, B, dA, dB = _gradient_weights(mass, density, dmass, ddensity, scheme)
    fi = fv_q.unsqueeze(1)
    fj = fv_r.unsqueeze(0)
    coeff = fi * A + fj * B
    dcoeff = fi * dA + fj * dB

    out = (coeff.unsqueeze(-1) * G_final).sum(dim=1)
    d_out = (dcoeff.unsqueeze(-1) * G_final + coeff.unsqueeze(-1) * dG_final).sum(dim=1)
    return out, d_out


# --------------------------------------------------------------------------
# Reference: reverse-mode Jacobian of the PRODUCTION operator, with
# renormalizationState computed from the SAME leaf positions/supports being
# differentiated -- see module docstring.
# --------------------------------------------------------------------------

def _reference_jvp(f, primals, tangents):
    J = torch.autograd.functional.jacobian(f, primals, vectorize=False)
    out = f(*primals).detach()
    n_out = out.numel()
    acc = torch.zeros(n_out, dtype=DTYPE, device=DEVICE)
    for Jk, vk in zip(J, tangents):
        acc = acc + Jk.reshape(n_out, -1) @ vk.reshape(-1)
    return acc.reshape(out.shape)


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


def run_renorm_gradient_case(n, dim, mode: SupportScheme, scheme: GradientScheme, kernel_id, seed=0):
    pos0, sup0, mass0, density0, domain, adjacency, kinds = _perturbed_case(n, dim, seed)
    n_actual = pos0.shape[0]
    fv_q = torch.randn(n_actual, dtype=DTYPE)
    fv_r = torch.randn(n_actual, dtype=DTYPE)

    def f(pos, sup, mass, density):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=density, kinds=kinds)
        renormProperties = OperationProperties(kernel=KERNEL, operation=WarpOperation.Gradient,
                                                supportMode=mode, operationMode=OperationDirection.AllToAll)
        _, _, renormState = computeRenormalizationMatrices(p, renormProperties, domain, adjacency=adjacency)
        return warpOperation(
            p, OperationProperties(kernel=KERNEL, operation=WarpOperation.Gradient,
                                    supportMode=mode, operationMode=OperationDirection.AllToAll, gradientMode=scheme),
            domain, queryValues=fv_q, referenceValues=fv_r, adjacency=adjacency, renormalizationState=renormState,
        )

    for t in (pos0, sup0, mass0, density0):
        t.requires_grad_(True)
    dpos, dsup, dmass, ddensity = (torch.randn_like(pos0), torch.randn_like(sup0) * 0.1,
                                    torch.randn_like(mass0), torch.randn_like(density0) * 0.1)

    reference = _reference_jvp(f, (pos0, sup0, mass0, density0), (dpos, dsup, dmass, ddensity))

    L, dL = assembled_L_dL(pos0.detach(), sup0.detach(), mass0.detach(), density0.detach(),
                            dpos, dsup, dmass, ddensity, dim, mode, kernel_id, domain, adjacency, kinds)
    _, d = assembled_renorm_gradient_jvp(pos0.detach(), sup0.detach(), mass0.detach(), density0.detach(), fv_q, fv_r,
                                          dpos, dsup, dmass, ddensity, L, dL, dim, mode.value, kernel_id, scheme)
    return d, reference


def run_low_neighbor_case(dim, kernel_id):
    """An isolated particle: dL is exactly zero (Tier 2.4's own identity-
    fallback check), so dKernelGradient_final should reduce to plain
    L_i @ dG_ij with L_i == identity -- i.e. exactly Tier 2.2's un-
    renormalized JVP. Mirrors spike_forward_mode_tier2_renorm.py's own
    low-neighbor-count check."""
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
    scheme = GradientScheme.Difference
    kernel_id = KERNEL.value
    fv_q = torch.randn(1, dtype=DTYPE)
    fv_r = torch.randn(1, dtype=DTYPE)

    def f(pos, sup, mass, density):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=density, kinds=kinds)
        renormProperties = OperationProperties(kernel=KERNEL, operation=WarpOperation.Gradient,
                                                supportMode=mode, operationMode=OperationDirection.AllToAll)
        _, _, renormState = computeRenormalizationMatrices(p, renormProperties, domain, adjacency=adjacency)
        return warpOperation(
            p, OperationProperties(kernel=KERNEL, operation=WarpOperation.Gradient,
                                    supportMode=mode, operationMode=OperationDirection.AllToAll, gradientMode=scheme),
            domain, queryValues=fv_q, referenceValues=fv_r, adjacency=adjacency, renormalizationState=renormState,
        )

    for t in (pos0, sup0, mass0, density0):
        t.requires_grad_(True)
    dpos, dsup, dmass, ddensity = (torch.randn_like(pos0), torch.randn_like(sup0) * 0.1,
                                    torch.randn_like(mass0), torch.randn_like(density0) * 0.1)

    reference = _reference_jvp(f, (pos0, sup0, mass0, density0), (dpos, dsup, dmass, ddensity))
    L, dL = assembled_L_dL(pos0.detach(), sup0.detach(), mass0.detach(), density0.detach(),
                            dpos, dsup, dmass, ddensity, dim, mode, kernel_id, domain, adjacency, kinds)
    _, d = assembled_renorm_gradient_jvp(pos0.detach(), sup0.detach(), mass0.detach(), density0.detach(), fv_q, fv_r,
                                          dpos, dsup, dmass, ddensity, L, dL, dim, mode.value, kernel_id, scheme)

    is_identity = torch.allclose(L[0], torch.eye(dim, dtype=DTYPE), atol=1e-12)
    is_zero_dL = torch.allclose(dL, torch.zeros_like(dL), atol=1e-12)
    ok = check(f"dim={dim} low-neighbor renorm-gradient JVP", d, reference)
    print(f"       (L==identity {is_identity}, dL==0 {is_zero_dL})")
    return ok and is_identity and is_zero_dL


def main():
    wp.init()
    print(__doc__.split("\n\n")[0])
    print()

    kernel_id = KERNEL.value
    ok = True

    print("Renormalization-corrected Gradient JVP, 1D line of 7 particles, all GradientScheme:")
    for scheme in (GradientScheme.Naive, GradientScheme.Difference, GradientScheme.Summation, GradientScheme.Symmetric):
        d, ref = run_renorm_gradient_case(7, 1, SupportScheme.KernelMeanSymmetric, scheme, kernel_id)
        ok &= check(f"1D renorm-Gradient JVP ({scheme.name})", d, ref)

    print("\nRenormalization-corrected Gradient JVP, 2D 3x3 grid, all GradientScheme:")
    for scheme in (GradientScheme.Naive, GradientScheme.Difference, GradientScheme.Summation, GradientScheme.Symmetric):
        d, ref = run_renorm_gradient_case(3, 2, SupportScheme.KernelMeanSymmetric, scheme, kernel_id)
        ok &= check(f"2D renorm-Gradient JVP ({scheme.name})", d, ref)

    print("\nRenormalization-corrected Gradient JVP, other SupportSchemes (2D 3x3 grid, Symmetric):")
    for mode in (SupportScheme.Gather, SupportScheme.Scatter, SupportScheme.MeanSymmetric, SupportScheme.SuperSymmetric):
        d, ref = run_renorm_gradient_case(3, 2, mode, GradientScheme.Symmetric, kernel_id)
        ok &= check(f"2D renorm-Gradient JVP ({mode.name})", d, ref)

    print("\nLow-neighbor-count identity fallback -- dL==0, reduces to Tier 2.2's plain JVP, both dims:")
    ok &= run_low_neighbor_case(1, kernel_id)
    ok &= run_low_neighbor_case(2, kernel_id)

    print()
    if ok:
        print("ALL PASSED -- the renormalization-corrected Gradient operator's JVP")
        print("  (the ordinary product rule dKernelGradient_final = dL_i @ G_ij + L_i @ dG_ij,")
        print("  chaining Tier 2.4's already-validated dL into Tier 2.2's already-validated dG)")
        print("  matches the production Gradient operator's own reverse-mode derivative when")
        print("  called with renormalizationState=..., across every GradientScheme/SupportScheme")
        print("  combination tested, with an explicit low-neighbor-count identity-fallback check.")
    else:
        print("FAILED -- see the individual case(s) above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
