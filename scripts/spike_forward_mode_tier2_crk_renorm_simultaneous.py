#!/usr/bin/env python3
"""Combined spike for CRK and gradient-renormalization tangent support
applied SIMULTANEOUSLY (`warpier_tier2_correction_jvp_plan.md` follow-up,
2026-08-21, after the plan's own phases (a1)-(f) closed, which deliberately
left this combination out of scope as "a documented fast follow-up").

**Why this needed no new derivation.** Every primal kernel here composes CRK
and renormalization in the exact same fixed order already, unconditionally,
regardless of whether one, the other, or both are enabled: `kernelGradient =
computeKernelGradientCRK(...)`, then `if useGradientRenormalization:
kernelGradient = matmul(renormalizationMatrix, kernelGradient)` (confirmed by
grep across `wp_gradient.py`/`wp_divergence.py`/`wp_curl.py`/`wp_laplacian.py`
-- identical shape in all four). Every JVP formula here mirrors that exactly:
`G, dG = computeKernelGradientCRKJVP(...)` (a no-op when CRK is off), then `if
useGradientRenormalization: dG = matmul(dL,G) + matmul(L,dG); G = matmul(L,G)`
(phase (d)'s product rule, landed for Gradient in phase (d), extended to
Divergence/Curl/Laplacian(Brookshaw) in phase (f), extended again to
Laplacian(Dot/Default) in this same follow-up as CRK's own extension --
`spike_forward_mode_tier2_laplacian_dot_default_extension.py`). Since `L`
(`renormalizationMatrices`) is computed from raw geometry only -- confirmed
`computeRenormalizationMatrices_`'s own internal covariance call is always
`crkState=None` -- there is no cross-term between the two corrections'
*values*; the only question this spike answers is whether the JVP
*composition* (CRK-then-renorm, both differentiated) is correct when chained
together for real, which was never validated end-to-end before now
(`operations.py` unconditionally rejected the combination until this
follow-up).

**A red herring investigated and ruled out while writing this spike**: an
early version compared against a jacobian reference on an UNPERTURBED,
perfectly uniform 2D grid and saw a large mismatch at exactly one particle --
the grid's symmetric center. This is `spike_forward_mode_tier2_renorm.py`'s
own already-documented `pinv2x2_warpBackend` degeneracy (a perfectly uniform
neighborhood's covariance matrix sits exactly at the eigenvalue-relative
`rcond` cutoff), not a CRK+renorm interaction bug -- confirmed by re-running
with the same `+-15%` non-uniform-support perturbation every other
Tier-2.4-touching script already applies for this exact reason: the mismatch
vanished (`rel_err` dropped from ~7e-3 to ~1e-16). This spike uses that
perturbation throughout, so it never rediscovers the same red herring.

    python scripts/spike_forward_mode_tier2_crk_renorm_simultaneous.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, grid_case_2d, line_case, make_domain
from warpSPHCore import OperationProperties, ParticleState, ParticleTangentState, warpOperation, warpOperationJVP
from warpSPHCore.crk import computeCRKFactors, computeCRKFactorsJVP
from warpSPHCore.enumTypes import GradientScheme, LaplacianScheme, OperationDirection, SupportScheme, WarpOperation
from warpSPHCore.renorm import computeRenormalizationMatrices, computeRenormalizationMatricesJVP


def compute_densities(positions, supports, masses, kinds, domain, adjacency):
    p = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=None, kinds=kinds)
    rho = warpOperation(
        p,
        OperationProperties(kernel=KERNEL, operation=WarpOperation.Density, supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll),
        domain, adjacency=adjacency,
    )
    return rho.detach().clone().requires_grad_(True)


def _reference_jvp(f, primals, tangents):
    J = torch.autograd.functional.jacobian(f, primals, vectorize=False)
    out = f(*primals)
    acc = torch.zeros(out.numel(), dtype=DTYPE, device=DEVICE)
    for Jk, vk in zip(J, tangents):
        acc = acc + Jk.reshape(out.numel(), -1) @ vk.reshape(-1)
    return acc.reshape(out.shape)


def run_case(op: WarpOperation, label: str, domain, positions, supports, masses, kinds, adjacency, densities, extra_props=None) -> bool:
    extra_props = extra_props or {}
    n = positions.shape[0]
    scalarField = op in (WarpOperation.Gradient, WarpOperation.Laplacian)
    torch.manual_seed(hash((op, label)) % (2 ** 31))
    qv = torch.randn(n, dtype=DTYPE, device=DEVICE) if scalarField else torch.randn(n, 2, dtype=DTYPE, device=DEVICE)
    rv = torch.randn(n, dtype=DTYPE, device=DEVICE) if scalarField else torch.randn(n, 2, dtype=DTYPE, device=DEVICE)
    props = OperationProperties(kernel=KERNEL, operation=op, supportMode=SupportScheme.Gather,
                                 operationMode=OperationDirection.AllToAll, gradientMode=GradientScheme.Naive, **extra_props)

    def f(pos, sup, mass, dens):
        pp = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        _, _, crkState = computeCRKFactors(pp, domain, KERNEL, operationMode=OperationDirection.AllToAll, adjacency=adjacency)
        renormProps = OperationProperties(kernel=KERNEL, operation=op, supportMode=SupportScheme.Gather,
                                           operationMode=OperationDirection.AllToAll, **extra_props)
        _, _, renormState = computeRenormalizationMatrices(pp, renormProps, domain, adjacency=adjacency)
        return warpOperation(pp, props, domain, queryValues=qv, referenceValues=rv, adjacency=adjacency,
                              crkState=crkState, renormalizationState=renormState)

    pos0 = positions.detach().clone().requires_grad_(True)
    sup0 = supports.detach().clone().requires_grad_(True)
    mass0 = masses.detach().clone().requires_grad_(True)
    dens0 = densities.detach().clone().requires_grad_(True)
    dpos, dsup = torch.randn_like(pos0), torch.randn_like(sup0) * 0.1
    dmass, ddens = torch.randn_like(mass0), torch.randn_like(dens0) * 0.1

    reference = _reference_jvp(f, (pos0, sup0, mass0, dens0), (dpos, dsup, dmass, ddens))

    p_now = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=densities.detach(), kinds=kinds)
    qts = ParticleTangentState(positions=dpos, supports=dsup, masses=None)
    rts = ParticleTangentState(positions=dpos, supports=dsup, masses=dmass, densities=ddens)
    _, _, crkState_now, crkTangentState_now = computeCRKFactorsJVP(
        p_now, domain, KERNEL, queryTangentState=qts, referenceTangentState=qts,
        operationMode=OperationDirection.AllToAll, adjacency=adjacency,
    )
    renormProps_now = OperationProperties(kernel=KERNEL, operation=op, supportMode=SupportScheme.Gather,
                                           operationMode=OperationDirection.AllToAll, **extra_props)
    _, _, renormState_now, renormTangentState_now = computeRenormalizationMatricesJVP(
        p_now, renormProps_now, domain, queryTangentState=qts, referenceTangentState=rts, adjacency=adjacency,
    )
    assembled = warpOperationJVP(
        p_now, props, domain, adjacency=adjacency,
        queryTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=None, densities=ddens),
        referenceTangentState=rts,
        queryValues=qv, referenceValues=rv,
        crkState=crkState_now, crkTangentState=crkTangentState_now,
        renormalizationState=renormState_now, renormalizationTangentState=renormTangentState_now,
    )
    rel_err = float((assembled - reference).detach().abs().max()) / max(float(reference.detach().abs().max()), 1e-300)
    ok = rel_err <= 1e-8
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:32s} rel_err={rel_err:.3e}")
    return ok


def main():
    wp.init()
    ok = True

    # 2D, non-uniform (+-15%) supports for Gradient/Divergence/Curl -- avoids the
    # uniform-grid pinv degeneracy documented above, matching every
    # Tier-2.4-touching script's own standing discipline.
    domain2d = make_domain(dim=2)
    pos2d, sup2d, mass2d = grid_case_2d(n_per_side=3)
    n2d = pos2d.shape[0]
    sup2d = (sup2d.detach() * (1.0 + 0.15 * torch.linspace(-1, 1, n2d, dtype=DTYPE))).requires_grad_(True)
    adjacency2d, kinds2d = build_adjacency(pos2d, sup2d, mass2d, domain2d)
    dens2d = compute_densities(pos2d, sup2d, mass2d, kinds2d, domain2d, adjacency2d)

    print("CRK + renormalization applied simultaneously, production warpOperationJVP vs. jacobian on primal warpOperation(..., crkState=..., renormalizationState=...):")
    ok &= run_case(WarpOperation.Gradient, "Gradient", domain2d, pos2d, sup2d, mass2d, kinds2d, adjacency2d, dens2d)
    ok &= run_case(WarpOperation.Divergence, "Divergence", domain2d, pos2d, sup2d, mass2d, kinds2d, adjacency2d, dens2d)
    ok &= run_case(WarpOperation.Curl, "Curl", domain2d, pos2d, sup2d, mass2d, kinds2d, adjacency2d, dens2d)

    # 1D line_case for Laplacian's three corrected schemes -- Dot's own
    # dim-block restriction needs dim == 1 for a scalar field (matching every
    # other Dot script's convention); also exercises the self-pair-adjoint-bug
    # regression guard (Brookshaw/Dot/Default all had one, phases (e)/(f) and
    # this follow-up's own Dot/Default extension).
    domain1d = make_domain(dim=1)
    pos1d, sup1d, mass1d = line_case(6)
    adjacency1d, kinds1d = build_adjacency(pos1d, sup1d, mass1d, domain1d)
    dens1d = compute_densities(pos1d, sup1d, mass1d, kinds1d, domain1d, adjacency1d)

    ok &= run_case(WarpOperation.Laplacian, "Laplacian(Brookshaw)", domain1d, pos1d, sup1d, mass1d, kinds1d, adjacency1d, dens1d,
                    extra_props={"laplacianMode": LaplacianScheme.Brookshaw})
    ok &= run_case(WarpOperation.Laplacian, "Laplacian(Dot)", domain1d, pos1d, sup1d, mass1d, kinds1d, adjacency1d, dens1d,
                    extra_props={"laplacianMode": LaplacianScheme.Dot})
    ok &= run_case(WarpOperation.Laplacian, "Laplacian(Default)", domain1d, pos1d, sup1d, mass1d, kinds1d, adjacency1d, dens1d,
                    extra_props={"laplacianMode": LaplacianScheme.Default})

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- CRK+renormalization applied simultaneously has a wrong Jacobian for at least one operator.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
