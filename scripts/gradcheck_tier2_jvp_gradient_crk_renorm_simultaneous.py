#!/usr/bin/env python3
"""Native torch.autograd.gradcheck checks against Gradient's Tier-2
(geometry) JVP with CRK correction and gradient-renormalization applied
SIMULTANEOUSLY (`warpier_tier2_correction_jvp_plan.md` follow-up, 2026-08-21)
-- `computeSPHGradientGeometryJVP(..., crkState=..., renormalizationState=...)`
via `warpOperationJVP` (the public entry point, now that `operations.py` no
longer rejects the combination).

See `spike_forward_mode_tier2_crk_renorm_simultaneous.py`'s module docstring
for why no new derivation was needed (both corrections already compose in
the same fixed CRK-then-renorm order in every primal kernel and every JVP
formula here, unconditionally) -- that spike validates the JVP-vs-jacobian
identity for all six operator/scheme combinations that individually support
CRK and renorm; this script complements it with reverse-mode-through-the-JVP
gradcheck for one representative operator (Gradient), the same "spike proves
the formula, gradcheck proves the reverse-mode wiring" split every other
phase in this plan uses.

    python scripts/gradcheck_tier2_jvp_gradient_crk_renorm_simultaneous.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, grid_case_2d
from warpSPHCore import OperationProperties, ParticleState, ParticleTangentState, warpOperation
from warpSPHCore.crk import computeCRKFactorsJVP
from warpSPHCore.dataTypes import CRKState, CRKTangentState, DomainDescription, RenormalizationState, RenormalizationTangentState
from warpSPHCore.enumTypes import GradientScheme, OperationDirection, SupportScheme, WarpOperation
from warpSPHCore.coreOperations.wp_gradientJVP import computeSPHGradientGeometryJVP
from warpSPHCore.renorm import computeRenormalizationMatricesJVP


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


# ---------------------------------------------------------------------------
# 1. Targeted direct-tensor gradcheck: crkState/crkTangentState AND
#    renormalizationState/renormalizationTangentState as independent leaves,
#    both supplied at once -- isolates that _jvpCommon's flat_tensors wiring
#    for the two corrections doesn't interfere with each other when both are
#    active in the same launch.
# ---------------------------------------------------------------------------

def run_direct_gradcheck() -> bool:
    domain = _make_domain(dim=2)
    positions, supports, masses = grid_case_2d(n_per_side=3)
    n = positions.shape[0]
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    pos = positions.detach().clone().requires_grad_(True)
    sup = supports.detach().clone().requires_grad_(True)
    dens = densities.detach().clone().requires_grad_(True)
    tqp = (0.1 * torch.randn(n, 2, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    tqs = (0.01 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    qval = torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)
    rval = torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)

    crk_A = (0.5 + 0.1 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    crk_B = (0.1 * torch.randn(n, 2, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    crk_gradA = (0.1 * torch.randn(n, 2, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    crk_gradB = (0.1 * torch.randn(n, 2, 2, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    renorm_L = (1.0 + 0.1 * torch.randn(n, 2, 2, dtype=DTYPE, device=DEVICE)).requires_grad_(True)

    def f(pos, sup, dens, tqp, tqs, qval, rval, cA, cB, cgA, cgB, L):
        p = ParticleState(positions=pos, supports=sup, masses=masses.detach(), densities=dens, kinds=kinds)
        crkState = CRKState(A=cA, B=cB, gradA=cgA, gradB=cgB)
        renormState = RenormalizationState(renormalizationMatrices=L)
        return computeSPHGradientGeometryJVP(
            p, domain, KERNEL, SupportScheme.Gather, adjacency,
            queryTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            referenceTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            queryValues=qval, referenceValues=rval,
            crkState=crkState, renormalizationState=renormState,
            gradientMode=GradientScheme.Naive,
        )

    inputs = (pos, sup, dens, tqp, tqs, qval, rval, crk_A, crk_B, crk_gradA, crk_gradB, renorm_L)
    ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5, rtol=1e-4)
    print("Direct-tensor CRK+renorm-simultaneous gradcheck (both correction states as independent leaves):", ok)
    return bool(ok)


# ---------------------------------------------------------------------------
# 2. End-to-end gradcheck: crkState/crkTangentState AND
#    renormalizationState/renormalizationTangentState both derived from the
#    same leaf positions/supports gradcheck perturbs.
# ---------------------------------------------------------------------------

def run_end_to_end_gradcheck() -> bool:
    domain = _make_domain(dim=2)
    positions, supports, masses = grid_case_2d(n_per_side=3)
    # Non-uniform supports (+-15%): avoids the uniform-grid pinv degeneracy
    # spike_forward_mode_tier2_crk_renorm_simultaneous.py's own docstring
    # documents (a red herring investigated there, ruled out via this exact
    # perturbation).
    n = positions.shape[0]
    supports = (supports.detach() * (1.0 + 0.15 * torch.linspace(-1, 1, n, dtype=DTYPE))).requires_grad_(True)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    pos = positions.detach().clone().requires_grad_(True)
    sup = supports.detach().clone().requires_grad_(True)
    mass = masses.detach().clone().requires_grad_(True)
    dens = densities.detach().clone().requires_grad_(True)
    tqp = (0.1 * torch.randn(n, 2, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    tqs = (0.01 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    trm = (0.1 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    trd = (0.1 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    qval = torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)
    rval = torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def f(pos, sup, mass, dens, tqp, tqs, trm, trd, qval, rval):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        qts = ParticleTangentState(positions=tqp, supports=tqs, masses=None)
        rts = ParticleTangentState(positions=tqp, supports=tqs, masses=trm, densities=trd)
        _, _, crkState, crkTangentState = computeCRKFactorsJVP(
            p, domain, KERNEL, queryTangentState=qts, referenceTangentState=qts,
            operationMode=OperationDirection.AllToAll, adjacency=adjacency,
        )
        renormProperties = OperationProperties(kernel=KERNEL, operation=WarpOperation.Gradient, supportMode=SupportScheme.Gather,
                                                operationMode=OperationDirection.AllToAll)
        _, _, renormState, renormTangentState = computeRenormalizationMatricesJVP(
            p, renormProperties, domain, queryTangentState=qts, referenceTangentState=rts, adjacency=adjacency,
        )
        return computeSPHGradientGeometryJVP(
            p, domain, KERNEL, SupportScheme.Gather, adjacency,
            queryTangentState=qts, referenceTangentState=rts,
            queryValues=qval, referenceValues=rval,
            crkState=crkState, crkTangentState=crkTangentState,
            renormalizationState=renormState, renormalizationTangentState=renormTangentState,
            gradientMode=GradientScheme.Naive,
        )

    inputs = (pos, sup, mass, dens, tqp, tqs, trm, trd, qval, rval)
    ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5, rtol=1e-4)
    print("End-to-end CRK+renorm-simultaneous gradcheck (both correction states derived from positions):", ok)
    return bool(ok)


def main():
    wp.init()
    ok = True

    ok &= run_direct_gradcheck()
    ok &= run_end_to_end_gradcheck()

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- CRK+renormalization applied simultaneously (warpier_tier2_correction_jvp_plan.md follow-up, 2026-08-21) has a wrong Jacobian for Gradient.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
