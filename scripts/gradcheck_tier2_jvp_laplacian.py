#!/usr/bin/env python3
"""Native torch.autograd.gradcheck against Laplacian's Tier-2 (geometry) JVP
-- `computeSPHLaplacianBrookshawGeometryJVP`/`computeSPHLaplacianNaiveGeometryJVP`/
`computeSPHLaplacianDotGeometryJVP`/`computeSPHLaplacianDefaultGeometryJVP`
-- no workarounds.

See `gradcheck_tier2_jvp_density.py`'s docstring for the shared rationale
and its CAVEAT (self-referencing here only validates the *combined*
d(output)/dx_i + d(output)/dx_j sensitivity, not each role's individual
partial -- see `gradcheck_tier2_jvp_interpolate.py` for the known-failing
distinct-role repro of the pre-existing `dW`-family reverse-mode gap this
shares).

Differentiable inputs checked (self-referencing, `referenceParticles=None`),
for all four schemes: positions, supports, densities,
tangentQueryPositions/tangentReferencePositions/tangentQuerySupports/
tangentReferenceSupports/tangentReferenceMasses, and queryValues/
referenceValues (frozen fi/fj, scalar fields -- Dot's own dim-block
restriction doesn't bite in this script's 1D domain, where block size is 1
and a scalar field is still in-scope, same as Brookshaw/Naive/Default).

**History: Dot was briefly excluded here, now fixed.** Writing Dot's own
JVP test initially surfaced a genuine reverse-mode adjoint bug -- gradcheck
differentiates a second time *through* `computeSPHLaplacianDotGeometryJVP`
(reverse-mode, via `wp.Tape`), and that second-order adjoint came out wrong.
Root cause: a loop-accumulated local (`proj = dot(q_block, n_ij)`, built via
a runtime loop) consumed by a further non-linear op (`proj * n_ij`) in the
*same* function silently drops part of its reverse-mode gradient -- a
recurring Warp code-generation footgun in this repo (confirmed via a minimal
standalone repro, and confirmed to be the *same* underlying issue as
`math/wp_laplaciandot.py`'s `computeLaplacianDot2`, the primal Dot kernel
this JVP differentiates). **Fixed** by moving the accumulation loop into its
own separate `@wp.func` that returns the accumulated value
(`_laplacianDotProjJVP` in `wp_laplacianJVP.py`), rather than leaving it as
a local reused in a non-linear op later in the same function -- confirmed
fixed by this script now passing for Dot too. Full write-up in
`docs/lessons_learned.md`.

    python scripts/gradcheck_tier2_jvp_laplacian.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, line_case, make_domain
from warpSPHCore import OperationProperties, ParticleState, ParticleTangentState, warpOperation
from warpSPHCore.enumTypes import GradientScheme, OperationDirection, SupportScheme, WarpOperation
from warpSPHCore.coreOperations.wp_laplacianJVP import (
    computeSPHLaplacianBrookshawGeometryJVP,
    computeSPHLaplacianNaiveGeometryJVP,
    computeSPHLaplacianDotGeometryJVP,
    computeSPHLaplacianDefaultGeometryJVP,
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
            queryTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            referenceTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=trm),
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
    run_scheme("Default", computeSPHLaplacianDefaultGeometryJVP, domain, positions, supports, masses, kinds, adjacency, densities)
    run_scheme("Dot", computeSPHLaplacianDotGeometryJVP, domain, positions, supports, masses, kinds, adjacency, densities)

    print("ALL PASSED.")


if __name__ == "__main__":
    main()
