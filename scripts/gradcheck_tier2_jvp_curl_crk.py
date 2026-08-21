#!/usr/bin/env python3
"""Native torch.autograd.gradcheck / JVP-identity checks against Curl's
Tier-2 (geometry) JVP with CRK correction enabled
(`warpier_tier2_correction_jvp_plan.md` phase (e)) --
`computeSPHCurlGeometryJVP(..., crkState=..., crkTangentState=...)`.

See `gradcheck_tier2_jvp_divergence_crk.py`'s docstring for the shared
rationale -- Curl's own combination formula (the 2D cross-product expansion)
is the only new piece; Stages 1-4 (`crk.computeKernelGradientCRKJVP`) are
reused verbatim from phase (c).

    python scripts/gradcheck_tier2_jvp_curl_crk.py
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
from warpSPHCore.dataTypes import CRKState, CRKTangentState, DomainDescription
from warpSPHCore.enumTypes import GradientScheme, OperationDirection, SupportScheme, WarpOperation
from warpSPHCore.coreOperations.wp_curlJVP import computeSPHCurlGeometryJVP


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
# 1. Targeted direct-tensor gradcheck: crkState/crkTangentState as
#    independent leaves, isolating _jvpCommon's flat_tensors wiring for
#    Curl's own new crkState/crkTangentState parameters.
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
    qval = torch.randn(n, 2, dtype=DTYPE, device=DEVICE, requires_grad=True)
    rval = torch.randn(n, 2, dtype=DTYPE, device=DEVICE, requires_grad=True)

    crk_A = (0.5 + 0.1 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    crk_B = (0.1 * torch.randn(n, 2, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    crk_gradA = (0.1 * torch.randn(n, 2, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    crk_gradB = (0.1 * torch.randn(n, 2, 2, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    d_crk_A = (0.1 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    d_crk_B = (0.1 * torch.randn(n, 2, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    d_crk_gradA = (0.1 * torch.randn(n, 2, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    d_crk_gradB = (0.1 * torch.randn(n, 2, 2, dtype=DTYPE, device=DEVICE)).requires_grad_(True)

    def f(pos, sup, dens, tqp, tqs, qval, rval, cA, cB, cgA, cgB, dcA, dcB, dcgA, dcgB):
        p = ParticleState(positions=pos, supports=sup, masses=masses.detach(), densities=dens, kinds=kinds)
        crkState = CRKState(A=cA, B=cB, gradA=cgA, gradB=cgB)
        crkTangentState = CRKTangentState(A=dcA, B=dcB, gradA=dcgA, gradB=dcgB)
        return computeSPHCurlGeometryJVP(
            p, domain, KERNEL, SupportScheme.Gather, adjacency,
            queryTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            referenceTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            queryValues=qval, referenceValues=rval,
            crkState=crkState, crkTangentState=crkTangentState,
            gradientMode=GradientScheme.Naive,
        )

    inputs = (pos, sup, dens, tqp, tqs, qval, rval, crk_A, crk_B, crk_gradA, crk_gradB, d_crk_A, d_crk_B, d_crk_gradA, d_crk_gradB)
    ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5, rtol=1e-4)
    print("Direct-tensor CRK gradcheck (crkState/crkTangentState as independent leaves):", ok)
    return bool(ok)


# ---------------------------------------------------------------------------
# 2. End-to-end gradcheck: crkState/crkTangentState derived from the same
#    leaf positions/supports gradcheck perturbs, via computeCRKFactorsJVP.
# ---------------------------------------------------------------------------

def run_end_to_end_gradcheck() -> bool:
    domain = _make_domain(dim=2)
    positions, supports, masses = grid_case_2d(n_per_side=3)
    n = positions.shape[0]
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    pos = positions.detach().clone().requires_grad_(True)
    sup = supports.detach().clone().requires_grad_(True)
    mass = masses.detach().clone().requires_grad_(True)
    dens = densities.detach().clone().requires_grad_(True)
    tqp = (0.1 * torch.randn(n, 2, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    tqs = (0.01 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    trm = (0.1 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    qval = torch.randn(n, 2, dtype=DTYPE, device=DEVICE, requires_grad=True)
    rval = torch.randn(n, 2, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def f(pos, sup, mass, dens, tqp, tqs, trm, qval, rval):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        _, _, crkState, crkTangentState = computeCRKFactorsJVP(
            p, domain, KERNEL,
            queryTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            referenceTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            operationMode=OperationDirection.AllToAll, adjacency=adjacency,
        )
        return computeSPHCurlGeometryJVP(
            p, domain, KERNEL, SupportScheme.Gather, adjacency,
            queryTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            referenceTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=trm, densities=None),
            queryValues=qval, referenceValues=rval,
            crkState=crkState, crkTangentState=crkTangentState,
            gradientMode=GradientScheme.Naive,
        )

    inputs = (pos, sup, mass, dens, tqp, tqs, trm, qval, rval)
    ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5, rtol=1e-4)
    print("End-to-end CRK gradcheck (crkState/crkTangentState derived from positions):", ok)
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
        print("FAILED -- CRK tangent extension for Curl (warpier_tier2_correction_jvp_plan.md phase (e)) has a wrong Jacobian.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
