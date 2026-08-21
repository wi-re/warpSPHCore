#!/usr/bin/env python3
"""Native torch.autograd.gradcheck / JVP-identity checks against Curl's
Tier-2 (geometry) JVP with gradient-renormalization correction enabled
(`warpier_tier2_correction_jvp_plan.md` phase (f)) --
`computeSPHCurlGeometryJVP(..., renormalizationState=..., renormalizationTangentState=...)`.

See `gradcheck_tier2_jvp_divergence_renorm.py`'s docstring for the shared
rationale -- Curl's own combination formula (the 2D cross-product expansion)
is the only new piece; the `dL_i @ G + L_i @ dG` product rule
(`wp_gradientJVP.py`, phase (d)) is reused verbatim.

    python scripts/gradcheck_tier2_jvp_curl_renorm.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, grid_case_2d
from warpSPHCore import OperationProperties, ParticleState, ParticleTangentState, warpOperation
from warpSPHCore.dataTypes import DomainDescription, RenormalizationState, RenormalizationTangentState
from warpSPHCore.enumTypes import GradientScheme, OperationDirection, SupportScheme, WarpOperation
from warpSPHCore.coreOperations.wp_curlJVP import computeSPHCurlGeometryJVP
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
# 1. Targeted direct-tensor gradcheck: renormalizationState/
#    renormalizationTangentState as independent leaves, isolating
#    _jvpCommon's flat_tensors wiring for Curl's own new
#    renormalizationState/renormalizationTangentState parameters.
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

    renorm_L = (1.0 + 0.1 * torch.randn(n, 2, 2, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    d_renorm_L = (0.1 * torch.randn(n, 2, 2, dtype=DTYPE, device=DEVICE)).requires_grad_(True)

    def f(pos, sup, dens, tqp, tqs, qval, rval, L, dL):
        p = ParticleState(positions=pos, supports=sup, masses=masses.detach(), densities=dens, kinds=kinds)
        renormState = RenormalizationState(renormalizationMatrices=L)
        renormTangentState = RenormalizationTangentState(renormalizationMatrices=dL)
        return computeSPHCurlGeometryJVP(
            p, domain, KERNEL, SupportScheme.Gather, adjacency,
            queryTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            referenceTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            queryValues=qval, referenceValues=rval,
            renormalizationState=renormState, renormalizationTangentState=renormTangentState,
            gradientMode=GradientScheme.Naive,
        )

    inputs = (pos, sup, dens, tqp, tqs, qval, rval, renorm_L, d_renorm_L)
    ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5, rtol=1e-4)
    print("Direct-tensor renorm gradcheck (renormalizationState/renormalizationTangentState as independent leaves):", ok)
    return bool(ok)


# ---------------------------------------------------------------------------
# 2. End-to-end gradcheck: renormalizationState/renormalizationTangentState
#    derived from the same leaf positions/supports gradcheck perturbs, via
#    computeRenormalizationMatricesJVP.
# ---------------------------------------------------------------------------

def run_end_to_end_gradcheck() -> bool:
    domain = _make_domain(dim=2)
    positions, supports, masses = grid_case_2d(n_per_side=3)
    # Non-uniform supports (+-15%): a perfectly uniform grid's covariance matrix
    # can be exactly singular/degenerate for pinv2x2_warpBackend (called via
    # computeRenormalizationMatricesJVP below), producing NaN unrelated to this
    # phase's own JVP wiring -- same discipline every Tier-2.4-touching script uses.
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
    qval = torch.randn(n, 2, dtype=DTYPE, device=DEVICE, requires_grad=True)
    rval = torch.randn(n, 2, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def f(pos, sup, mass, dens, tqp, tqs, trm, trd, qval, rval):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        renormProperties = OperationProperties(kernel=KERNEL, operation=WarpOperation.Curl, supportMode=SupportScheme.Gather,
                                                operationMode=OperationDirection.AllToAll)
        _, _, renormState, renormTangentState = computeRenormalizationMatricesJVP(
            p, renormProperties, domain,
            queryTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            referenceTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=trm, densities=trd),
            adjacency=adjacency,
        )
        return computeSPHCurlGeometryJVP(
            p, domain, KERNEL, SupportScheme.Gather, adjacency,
            queryTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            referenceTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=trm, densities=trd),
            queryValues=qval, referenceValues=rval,
            renormalizationState=renormState, renormalizationTangentState=renormTangentState,
            gradientMode=GradientScheme.Naive,
        )

    inputs = (pos, sup, mass, dens, tqp, tqs, trm, trd, qval, rval)
    ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5, rtol=1e-4)
    print("End-to-end renorm gradcheck (renormalizationState/renormalizationTangentState derived from positions):", ok)
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
        print("FAILED -- renormalization tangent extension for Curl (warpier_tier2_correction_jvp_plan.md phase (f)) has a wrong Jacobian.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
