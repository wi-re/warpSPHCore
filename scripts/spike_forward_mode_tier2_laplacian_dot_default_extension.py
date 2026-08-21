#!/usr/bin/env python3
"""Combined spike for CRK and gradient-renormalization tangent EXTENSION to
Laplacian's Dot/Default schemes (`warpier_tier2_correction_jvp_plan.md`
follow-up, 2026-08-21, after the plan's own phases (a1)-(f) closed) --
mirrors how phases (e)/(f) were each one spike covering three operators
(`spike_forward_mode_tier2_crk_extension.py`/
`spike_forward_mode_tier2_renorm_extension.py`).

**Why Dot/Default, not Naive.** `wp_laplacian.py`'s single primal kernel
(`computeSPHLaplacianTensor_Func_i`) computes one CRK/renorm-corrected
`kernelGradient` unconditionally, then dispatches on `laplacianMode`:
Brookshaw/Dot/Default all consume that corrected `kernelGradient` (confirmed
by reading the dispatch directly -- Dot calls `computeLaplacianDot2(...,
kernelGradient, ...)`, Default calls `computeDotLaplacian(...,
kernelGradient, ...)`), but Naive's branch calls `sphKernelLaplacian(...)`
directly on raw positions/supports, never touching `kernelGradient` at all.
So CRK/renorm correction is mathematically inapplicable to Naive by
construction (nothing to derive, not just "not yet implemented") -- this
spike only covers Dot/Default, matching `operations.py`'s own
`_LAPLACIAN_CORRECTION_SCHEMES` restriction. The JVP side already had the
right building block for both: `wp_laplacianJVP.py`'s `_laplacianGeometryChainJVP`
(shared by Brookshaw/Dot/Default) already produced a CRK/renorm-corrected
`(G, dG)` pair unconditionally since phases (e)/(f) -- Dot/Default's own
`computeSPHLaplacian{Dot,Default}GeometryJVP` just never exposed
`crkState`/`renormalizationState` parameters to reach it. Extending them was
parameter-threading, not new derivation.

**Found and fixed a genuine, pre-existing reverse-mode adjoint bug while
writing this spike, in BOTH the primal kernel and this JVP, not introduced by
this follow-up**: Dot's own `F_ab = dot(G,n_ij)/D_ij` is bit-for-bit
Brookshaw's own `P` (same `n_ij`/`D_ij`, same `eps=1e-8`) -- so it inherits
the identical self-pair (`r_ij == 0`) reverse-mode-adjoint hazard phase (e)
already found and fixed for Brookshaw (`correctGradientCRK`'s value at
`x_ij == 0` is generically nonzero once CRK is enabled, and Warp's
reverse-mode through "a nonzero G dotted against an exactly-zero n_ij,
divided by D_ij again" produces a wrong adjoint there). Default's own
`n_ij2 = n_ij/D2_ij` divides by a *second* regularized distance on top of
`n_ij`'s own division -- one quotient-rule level deeper than Brookshaw/Dot,
same hazard. Confirmed via `torch.autograd.gradcheck` failing directly on
`warpOperation(Laplacian, Dot/Default, crkState=...)` and on
`computeSPHLaplacian{Dot,Default}GeometryJVP(..., crkState=..., crkTangentState=...)`
before the fix -- both failures isolated to exactly the self-pair (diagonal)
Jacobian entries, the same signature Brookshaw's own bug had. **Fixed** in
`wp_laplacian.py` (primal, both branches) and `wp_laplacianJVP.py` (this JVP,
both `_Func_i`s) by guarding each scheme's own kernelGradient-consuming
contribution with an explicit `if r_ij > 0:`, mirroring Brookshaw's own fix
exactly -- the true contribution at `r_ij == 0` is always exactly 0 for both
schemes (`n_ij == 0` there by construction, CRK or not), so this changes no
forward value anywhere (confirmed: `operation_matrix.py` stayed bit-identical
at `OK=258, HIGH=0, ERR=0, NAN=0`, and every existing non-CRK gradcheck script
stayed green with unchanged output).

    python scripts/spike_forward_mode_tier2_laplacian_dot_default_extension.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, line_case, make_domain
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


def run_crk_case(laplacianMode: LaplacianScheme, label: str, domain, positions, supports, masses, kinds, adjacency, densities) -> bool:
    n = positions.shape[0]
    torch.manual_seed(hash(("crk", laplacianMode, label)) % (2 ** 31))
    qv = torch.randn(n, dtype=DTYPE, device=DEVICE)
    rv = torch.randn(n, dtype=DTYPE, device=DEVICE)
    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Laplacian, supportMode=SupportScheme.Gather,
                                 operationMode=OperationDirection.AllToAll, gradientMode=GradientScheme.Naive,
                                 laplacianMode=laplacianMode)

    def f(pos, sup, mass, dens):
        pp = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        _, _, crkState = computeCRKFactors(pp, domain, KERNEL, operationMode=OperationDirection.AllToAll, adjacency=adjacency)
        return warpOperation(pp, props, domain, queryValues=qv, referenceValues=rv, adjacency=adjacency, crkState=crkState)

    pos0 = positions.detach().clone().requires_grad_(True)
    sup0 = supports.detach().clone().requires_grad_(True)
    mass0 = masses.detach().clone().requires_grad_(True)
    dens0 = densities.detach().clone().requires_grad_(True)
    dpos, dsup = torch.randn_like(pos0), torch.randn_like(sup0) * 0.1
    dmass, ddens = torch.randn_like(mass0), torch.randn_like(dens0) * 0.1

    reference = _reference_jvp(f, (pos0, sup0, mass0, dens0), (dpos, dsup, dmass, ddens))

    p_now = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=densities.detach(), kinds=kinds)
    _, _, crkState_now, crkTangentState_now = computeCRKFactorsJVP(
        p_now, domain, KERNEL,
        queryTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=None),
        referenceTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=None),
        operationMode=OperationDirection.AllToAll, adjacency=adjacency,
    )
    assembled = warpOperationJVP(
        p_now, props, domain, adjacency=adjacency,
        queryTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=None, densities=ddens),
        referenceTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=dmass, densities=ddens),
        queryValues=qv, referenceValues=rv,
        crkState=crkState_now, crkTangentState=crkTangentState_now,
    )
    rel_err = float((assembled - reference).detach().abs().max()) / max(float(reference.detach().abs().max()), 1e-300)
    ok = rel_err <= 1e-8
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:32s} rel_err={rel_err:.3e}")
    return ok


def run_renorm_case(laplacianMode: LaplacianScheme, label: str, domain, positions, supports, masses, kinds, adjacency, densities) -> bool:
    n = positions.shape[0]
    torch.manual_seed(hash(("renorm", laplacianMode, label)) % (2 ** 31))
    qv = torch.randn(n, dtype=DTYPE, device=DEVICE)
    rv = torch.randn(n, dtype=DTYPE, device=DEVICE)
    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Laplacian, supportMode=SupportScheme.Gather,
                                 operationMode=OperationDirection.AllToAll, gradientMode=GradientScheme.Naive,
                                 laplacianMode=laplacianMode)

    def f(pos, sup, mass, dens):
        pp = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        renormProps = OperationProperties(kernel=KERNEL, operation=WarpOperation.Laplacian, supportMode=SupportScheme.Gather,
                                           operationMode=OperationDirection.AllToAll)
        _, _, renormState = computeRenormalizationMatrices(pp, renormProps, domain, adjacency=adjacency)
        return warpOperation(pp, props, domain, queryValues=qv, referenceValues=rv, adjacency=adjacency, renormalizationState=renormState)

    pos0 = positions.detach().clone().requires_grad_(True)
    sup0 = supports.detach().clone().requires_grad_(True)
    mass0 = masses.detach().clone().requires_grad_(True)
    dens0 = densities.detach().clone().requires_grad_(True)
    dpos, dsup = torch.randn_like(pos0), torch.randn_like(sup0) * 0.1
    dmass, ddens = torch.randn_like(mass0), torch.randn_like(dens0) * 0.1

    reference = _reference_jvp(f, (pos0, sup0, mass0, dens0), (dpos, dsup, dmass, ddens))

    p_now = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=densities.detach(), kinds=kinds)
    renormProps_now = OperationProperties(kernel=KERNEL, operation=WarpOperation.Laplacian, supportMode=SupportScheme.Gather,
                                           operationMode=OperationDirection.AllToAll)
    _, _, renormState_now, renormTangentState_now = computeRenormalizationMatricesJVP(
        p_now, renormProps_now, domain,
        queryTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=None),
        referenceTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=dmass, densities=ddens),
        adjacency=adjacency,
    )
    assembled = warpOperationJVP(
        p_now, props, domain, adjacency=adjacency,
        queryTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=None, densities=ddens),
        referenceTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=dmass, densities=ddens),
        queryValues=qv, referenceValues=rv,
        renormalizationState=renormState_now, renormalizationTangentState=renormTangentState_now,
    )
    rel_err = float((assembled - reference).detach().abs().max()) / max(float(reference.detach().abs().max()), 1e-300)
    ok = rel_err <= 1e-8
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:32s} rel_err={rel_err:.3e}")
    return ok


def main():
    wp.init()
    ok = True

    # 1D line_case: Dot's own dim-block restriction (`flatInputShape % dim == 0`)
    # makes a scalar field valid only at dim == 1 -- matching every existing
    # Dot gradcheck script's own convention -- and always includes at least one
    # exact self-pair per query particle (the self-pair-adjoint-bug regression
    # guard, same discipline `gradcheck_tier2_jvp_laplacian_brookshaw_crk.py` uses).
    domain = make_domain(dim=1)
    n = 6
    positions, supports, masses = line_case(n)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    print("CRK tangent extension to Laplacian Dot/Default, production warpOperationJVP vs. jacobian on primal warpOperation(..., crkState=...):")
    ok &= run_crk_case(LaplacianScheme.Dot, "Laplacian(Dot)", domain, positions, supports, masses, kinds, adjacency, densities)
    ok &= run_crk_case(LaplacianScheme.Default, "Laplacian(Default)", domain, positions, supports, masses, kinds, adjacency, densities)

    print("\nRenormalization tangent extension to Laplacian Dot/Default, production warpOperationJVP vs. jacobian on primal warpOperation(..., renormalizationState=...):")
    ok &= run_renorm_case(LaplacianScheme.Dot, "Laplacian(Dot)", domain, positions, supports, masses, kinds, adjacency, densities)
    ok &= run_renorm_case(LaplacianScheme.Default, "Laplacian(Default)", domain, positions, supports, masses, kinds, adjacency, densities)

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- CRK/renormalization tangent extension to Laplacian Dot/Default has a wrong Jacobian.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
