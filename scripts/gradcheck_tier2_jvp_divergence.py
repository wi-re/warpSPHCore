#!/usr/bin/env python3
"""Native torch.autograd.gradcheck against Divergence's Tier-2 (geometry)
JVP -- `computeSPHDivergenceGeometryJVP` -- no workarounds.

See `gradcheck_tier2_jvp_density.py`'s docstring for the shared rationale
and its CAVEAT (self-referencing here only validates the *combined*
d(output)/dx_i + d(output)/dx_j sensitivity, not each role's individual
partial -- see `gradcheck_tier2_jvp_interpolate.py` for the known-failing
distinct-role repro of the pre-existing `dW`-family reverse-mode gap this
shares).

Differentiable inputs checked (self-referencing, `referenceParticles=None`,
2D so `queryValues`/`referenceValues` are genuine vector fields):
positions, supports, densities, tangentQueryPositions/
tangentReferencePositions/tangentQuerySupports/tangentReferenceSupports/
tangentReferenceMasses, and queryValues/referenceValues (frozen fi/fj).

    python scripts/gradcheck_tier2_jvp_divergence.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, grid_case_2d, make_domain
from warpSPHCore import OperationProperties, ParticleState, warpOperation
from warpSPHCore.enumTypes import GradientScheme, OperationDirection, SupportScheme, WarpOperation
from warpSPHCore.coreOperations.wp_divergenceJVP import computeSPHDivergenceGeometryJVP


def compute_densities(positions, supports, masses, kinds, domain, adjacency):
    p = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=None, kinds=kinds)
    rho = warpOperation(
        p,
        OperationProperties(kernel=KERNEL, operation=WarpOperation.Density, supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll),
        domain, adjacency=adjacency,
    )
    return rho.detach().clone().requires_grad_(True)


def main():
    wp.init()

    domain = make_domain(dim=2)
    positions, supports, masses = grid_case_2d(n_per_side=3)
    n = positions.shape[0]
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    tqp = (0.1 * torch.randn(n, 2, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    tqs = (0.01 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    trm = (0.1 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    qval = torch.randn(n, 2, dtype=DTYPE, device=DEVICE, requires_grad=True)
    rval = torch.randn(n, 2, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def f(pos, sup, dens, tqp, tqs, trm, qval, rval):
        p = ParticleState(positions=pos, supports=sup, masses=masses.detach(), densities=dens, kinds=kinds)
        return computeSPHDivergenceGeometryJVP(
            p, domain, KERNEL, SupportScheme.Gather, adjacency,
            tangentQueryPositions=tqp, tangentReferencePositions=tqp,
            tangentQuerySupports=tqs, tangentReferenceSupports=tqs,
            tangentReferenceMasses=trm,
            queryValues=qval, referenceValues=rval,
            gradientMode=GradientScheme.Symmetric,
        )

    pos = positions.detach().clone().requires_grad_(True)
    sup = supports.detach().clone().requires_grad_(True)
    dens = densities.detach().clone().requires_grad_(True)
    ok = torch.autograd.gradcheck(f, (pos, sup, dens, tqp, tqs, trm, qval, rval), eps=1e-6, atol=1e-5, rtol=1e-4)
    print("Divergence Tier-2 JVP gradcheck (self-referencing):", ok)
    assert ok

    print("ALL PASSED.")


if __name__ == "__main__":
    main()
