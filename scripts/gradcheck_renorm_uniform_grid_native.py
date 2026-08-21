#!/usr/bin/env python3
"""Regression guard for the `pinv2x2_warp` isotropic-eigenvalue reverse-mode
adjoint bug (`warpier_tier2_correction_jvp_plan.md` follow-up, 2026-08-21) --
end-to-end, on an EXACTLY uniform/regular 2D grid, no support perturbation.

See `scripts/gradcheck_pinv_native.py`'s `run_pinv2x2_isotropic` for the bug
itself, isolated to the pinv function. This script closes the loop at the
level users actually care about: every renorm-touching gradcheck/spike
script in this repo used to need a deliberate `+-15%` non-uniform-support
perturbation to dodge this bug, meaning gradient-renormalization was silently
untested (and silently wrong under reverse-mode autodiff) on the single most
common test geometry for numerical SPH analysis -- a perfectly regular grid.
That workaround is no longer needed; this script proves it on the exact
geometry the workaround used to avoid.

Three layers, matching this repo's own standing "prove it, don't assume it"
convention:
  1. `torch.autograd.gradcheck` directly on `computeRenormalizationMatrices`
     (renorm.py) at a perfectly regular, unperturbed 3x3 grid.
  2. `torch.autograd.gradcheck` on `warpOperation(Gradient,
     renormalizationState=...)` -- the full production consumer.
  3. A JVP-vs-jacobian identity check via `warpOperationJVP` with CRK and
     renormalization applied SIMULTANEOUSLY (the hardest combination this
     pinv fix needed to support), also on the unperturbed grid.

    python scripts/gradcheck_renorm_uniform_grid_native.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, grid_case_2d
from warpSPHCore import OperationProperties, ParticleState, ParticleTangentState, warpOperation, warpOperationJVP
from warpSPHCore.crk import computeCRKFactors, computeCRKFactorsJVP
from warpSPHCore.dataTypes import DomainDescription
from warpSPHCore.enumTypes import GradientScheme, OperationDirection, SupportScheme, WarpOperation
from warpSPHCore.renorm import computeRenormalizationMatrices, computeRenormalizationMatricesJVP


def _make_domain(dim: int = 2, margin: float = 10.0) -> DomainDescription:
    return DomainDescription(
        min=torch.tensor([-margin] * dim, dtype=DTYPE, device=DEVICE),
        max=torch.tensor([margin] * dim, dtype=DTYPE, device=DEVICE),
        periodic=torch.tensor([False] * dim, device=DEVICE),
        dim=dim,
    )


def compute_densities(positions, supports, masses, kinds, domain, adjacency):
    p = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=None, kinds=kinds)
    rho = warpOperation(
        p,
        OperationProperties(kernel=KERNEL, operation=WarpOperation.Density, supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll),
        domain, adjacency=adjacency,
    )
    return rho.detach().clone().requires_grad_(True)


def _uniform_grid():
    # Deliberately NO support perturbation -- grid_case_2d's own supports are
    # already perfectly uniform; every other renorm-touching script in this
    # repo re-perturbs them by +-15% before use. This script is the one place
    # that must NOT do that, or it isn't testing what it claims to.
    return grid_case_2d(n_per_side=3)


def run_renorm_gradcheck() -> bool:
    domain = _make_domain()
    positions, supports, masses = _uniform_grid()
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Gradient, supportMode=SupportScheme.Gather,
                                 operationMode=OperationDirection.AllToAll)

    def f(pos, sup, mass, dens):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        _, _, renormState = computeRenormalizationMatrices(p, props, domain, adjacency=adjacency)
        return renormState.renormalizationMatrices

    print("\n=== computeRenormalizationMatrices: torch.autograd.gradcheck on an unperturbed 3x3 grid ===")
    ok = torch.autograd.gradcheck(f, (positions, supports, masses, densities), eps=1e-6, atol=1e-5)
    print("PASSED" if ok else "FAILED")
    return bool(ok)


def run_gradient_renorm_gradcheck() -> bool:
    domain = _make_domain()
    positions, supports, masses = _uniform_grid()
    n = positions.shape[0]
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Gradient, supportMode=SupportScheme.Gather,
                                 operationMode=OperationDirection.AllToAll)
    torch.manual_seed(0)
    qv = torch.randn(n, dtype=DTYPE, device=DEVICE)
    rv = torch.randn(n, dtype=DTYPE, device=DEVICE)

    def f(pos, sup, mass, dens):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        _, _, renormState = computeRenormalizationMatrices(p, props, domain, adjacency=adjacency)
        return warpOperation(p, props, domain, queryValues=qv, referenceValues=rv, adjacency=adjacency, renormalizationState=renormState)

    print("\n=== warpOperation(Gradient, renormalizationState=...): torch.autograd.gradcheck on an unperturbed 3x3 grid ===")
    ok = torch.autograd.gradcheck(f, (positions, supports, masses, densities), eps=1e-6, atol=1e-5)
    print("PASSED" if ok else "FAILED")
    return bool(ok)


def run_crk_renorm_jvp_identity() -> bool:
    domain = _make_domain()
    positions, supports, masses = _uniform_grid()
    n = positions.shape[0]
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    torch.manual_seed(1)
    qv = torch.randn(n, dtype=DTYPE, device=DEVICE)
    rv = torch.randn(n, dtype=DTYPE, device=DEVICE)
    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Gradient, supportMode=SupportScheme.Gather,
                                 operationMode=OperationDirection.AllToAll, gradientMode=GradientScheme.Naive)

    def f(pos, sup, mass, dens):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        _, _, crkState = computeCRKFactors(p, domain, KERNEL, operationMode=OperationDirection.AllToAll, adjacency=adjacency)
        _, _, renormState = computeRenormalizationMatrices(p, props, domain, adjacency=adjacency)
        return warpOperation(p, props, domain, queryValues=qv, referenceValues=rv, adjacency=adjacency,
                              crkState=crkState, renormalizationState=renormState)

    pos0 = positions.detach().clone().requires_grad_(True)
    sup0 = supports.detach().clone().requires_grad_(True)
    mass0 = masses.detach().clone().requires_grad_(True)
    dens0 = densities.detach().clone().requires_grad_(True)
    dpos, dsup = torch.randn_like(pos0), torch.randn_like(sup0) * 0.1
    dmass, ddens = torch.randn_like(mass0), torch.randn_like(dens0) * 0.1

    J = torch.autograd.functional.jacobian(f, (pos0, sup0, mass0, dens0), vectorize=False)
    out = f(pos0, sup0, mass0, dens0)
    acc = torch.zeros(out.numel(), dtype=DTYPE, device=DEVICE)
    for Jk, vk in zip(J, (dpos, dsup, dmass, ddens)):
        acc = acc + Jk.reshape(out.numel(), -1) @ vk.reshape(-1)
    reference = acc.reshape(out.shape)

    p_now = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=densities.detach(), kinds=kinds)
    qts = ParticleTangentState(positions=dpos, supports=dsup, masses=None)
    rts = ParticleTangentState(positions=dpos, supports=dsup, masses=dmass, densities=ddens)
    _, _, crkState_now, crkTangentState_now = computeCRKFactorsJVP(
        p_now, domain, KERNEL, queryTangentState=qts, referenceTangentState=qts,
        operationMode=OperationDirection.AllToAll, adjacency=adjacency,
    )
    _, _, renormState_now, renormTangentState_now = computeRenormalizationMatricesJVP(
        p_now, props, domain, queryTangentState=qts, referenceTangentState=rts, adjacency=adjacency,
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
    print("\n=== warpOperationJVP(Gradient, crkState=..., renormalizationState=...): JVP-vs-jacobian on an unperturbed 3x3 grid ===")
    print(f"rel_err={rel_err:.3e}")
    print("PASSED" if ok else "FAILED")
    return ok


def main():
    wp.init()

    ok = True
    ok &= run_renorm_gradcheck()
    ok &= run_gradient_renorm_gradcheck()
    ok &= run_crk_renorm_jvp_identity()

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- the pinv2x2 isotropic-eigenvalue adjoint bug (warpier_tier2_correction_jvp_plan.md follow-up, 2026-08-21) has regressed.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
