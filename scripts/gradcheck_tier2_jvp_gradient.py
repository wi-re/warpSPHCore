#!/usr/bin/env python3
"""Native torch.autograd.gradcheck against Gradient's Tier-2 (geometry) JVP
-- `computeSPHGradientGeometryJVP` -- no workarounds.

See `gradcheck_tier2_jvp_density.py`'s docstring for the shared rationale
(`_jvpCommon.launchGeometryJVP` bridging every Tier-2 JVP kernel through
`StateAwareWarpFunction`) and its CAVEAT about the self-referencing
construction used here only validating the *combined* d(output)/dx_i +
d(output)/dx_j sensitivity, not each role's individual partial -- a
genuinely distinct-role case currently fails for the same pre-existing
`dW`-family reverse-mode gap documented there and reproduced (for
Interpolate) in `gradcheck_tier2_jvp_interpolate.py`.

Differentiable inputs checked (self-referencing, `referenceParticles=None`):
positions, supports, densities, tangentQueryPositions/
tangentReferencePositions/tangentQuerySupports/tangentReferenceSupports/
tangentReferenceMasses, and queryValues/referenceValues (frozen fi/fj).

    python scripts/gradcheck_tier2_jvp_gradient.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, line_case, make_domain
from warpSPHCore import OperationProperties, ParticleState, warpOperation
from warpSPHCore.enumTypes import GradientScheme, OperationDirection, SupportScheme, WarpOperation
from warpSPHCore.coreOperations.wp_gradientJVP import computeSPHGradientGeometryJVP


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

    domain = make_domain(dim=1)
    n = 6
    positions, supports, masses = line_case(n)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    tqp = (0.1 * torch.randn(n, 1, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    tqs = (0.01 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    trm = (0.1 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    qval = torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)
    rval = torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def f(pos, sup, dens, tqp, tqs, trm, qval, rval):
        p = ParticleState(positions=pos, supports=sup, masses=masses.detach(), densities=dens, kinds=kinds)
        return computeSPHGradientGeometryJVP(
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
    print("Gradient Tier-2 JVP gradcheck (self-referencing):", ok)
    assert ok

    print("ALL PASSED.")


if __name__ == "__main__":
    main()
