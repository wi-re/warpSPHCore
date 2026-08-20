#!/usr/bin/env python3
"""Native torch.autograd.gradcheck against Laplacian's Tier-2 (geometry) JVP
-- `computeSPHLaplacianBrookshawGeometryJVP`/`computeSPHLaplacianNaiveGeometryJVP`
-- no workarounds.

See `gradcheck_tier2_jvp_density.py`'s docstring for the shared rationale
and its CAVEAT (self-referencing here only validates the *combined*
d(output)/dx_i + d(output)/dx_j sensitivity, not each role's individual
partial -- see `gradcheck_tier2_jvp_interpolate.py` for the known-failing
distinct-role repro of the pre-existing `dW`-family reverse-mode gap this
shares).

Differentiable inputs checked (self-referencing, `referenceParticles=None`),
for both Brookshaw and Naive schemes: positions, supports, densities,
tangentQueryPositions/tangentReferencePositions/tangentQuerySupports/
tangentReferenceSupports/tangentReferenceMasses, and queryValues/
referenceValues (frozen fi/fj, scalar fields).

    python scripts/gradcheck_tier2_jvp_laplacian.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, line_case, make_domain
from warpSPHCore import OperationProperties, ParticleState, warpOperation
from warpSPHCore.enumTypes import GradientScheme, OperationDirection, SupportScheme, WarpOperation
from warpSPHCore.coreOperations.wp_laplacianJVP import (
    computeSPHLaplacianBrookshawGeometryJVP,
    computeSPHLaplacianNaiveGeometryJVP,
)


def compute_densities(positions, supports, masses, kinds, domain, adjacency):
    p = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=None, kinds=kinds)
    rho = warpOperation(
        p,
        OperationProperties(kernel=KERNEL, operation=WarpOperation.Density, supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll),
        domain, adjacency=adjacency,
    )
    return rho.detach().clone().requires_grad_(True)


def run_scheme(name, fn, domain, positions, supports, masses, kinds, adjacency, densities):
    n = positions.shape[0]
    tqp = (0.1 * torch.randn(n, 1, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    tqs = (0.01 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    trm = (0.1 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    qval = torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)
    rval = torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def f(pos, sup, dens, tqp, tqs, trm, qval, rval):
        p = ParticleState(positions=pos, supports=sup, masses=masses.detach(), densities=dens, kinds=kinds)
        return fn(
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
    print(f"Laplacian({name}) Tier-2 JVP gradcheck (self-referencing):", ok)
    assert ok


def main():
    wp.init()

    domain = make_domain(dim=1)
    n = 6
    positions, supports, masses = line_case(n)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    run_scheme("Brookshaw", computeSPHLaplacianBrookshawGeometryJVP, domain, positions, supports, masses, kinds, adjacency, densities)
    run_scheme("Naive", computeSPHLaplacianNaiveGeometryJVP, domain, positions, supports, masses, kinds, adjacency, densities)

    print("ALL PASSED.")


if __name__ == "__main__":
    main()
