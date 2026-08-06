#!/usr/bin/env python3
"""Native torch.autograd.gradcheck against computeRenormalizationMatrices (renorm.py)
across all three traversal inputs (explicit AdjacencyList, explicit CompactHashMap,
adjacency=None) -- closing a test-coverage gap, not a code gap: renorm.py's
NotImplementedError guard for grid-mode traversal has been commented out (dead, not
active) since the very first restructure commit, because computeRenormalizationMatrices_
already reads its neighbor count from Covariance's own per-particle kernel output
(covarianceReturnNumNeighbors=True), not from adjacency.numNeighbors -- the thing that
actually needed AdjacencyList. Nothing before this script ever exercised the grid/None
branches for renorm specifically: tests/operations/conftest.py's renorm_state() fixture
always passes an explicit AdjacencyList. See warpier_core.md's "Renormalization Grid-Mode
Coverage" note for the full story.

Differentiable inputs checked: positions, supports, masses, densities (same rationale as
gradcheck_covariance_native.py -- densities computed once via the already-verified
Density op, then detached and re-leafed).

    python scripts/gradcheck_renorm_native.py
"""

from __future__ import annotations

import os

os.environ.setdefault("SPHWARPCORE_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import KERNEL, build_adjacency, build_grid_adjacency, line_case, make_domain, single_particle_case
from sphWarpCore import OperationProperties, ParticleState, warpOperation
from sphWarpCore.enumTypes import OperationDirection, SupportScheme, WarpOperation
from sphWarpCore.renorm import computeRenormalizationMatrices


def compute_densities(positions, supports, masses, kinds, domain, adjacency):
    """See gradcheck_covariance_native.py's compute_densities -- identical rationale."""
    p = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=None, kinds=kinds)
    rho = warpOperation(
        p,
        OperationProperties(
            kernel=KERNEL,
            operation=WarpOperation.Density,
            supportMode=SupportScheme.Gather,
            operationMode=OperationDirection.AllToAll,
        ),
        domain,
        adjacency=adjacency,
    )
    return rho.detach().clone().requires_grad_(True)


def run_gradcheck(name: str, positions: torch.Tensor, supports: torch.Tensor, masses: torch.Tensor, traversal: str) -> bool:
    domain = make_domain()
    if traversal == "adjacency":
        adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    elif traversal == "grid":
        adjacency, kinds = build_grid_adjacency(positions, supports, masses, domain)
    else:  # "none" -- extractStateInfo auto-builds a CompactHashMap
        adjacency, kinds = None, torch.zeros(positions.shape[0], dtype=torch.int32, device=positions.device)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    operationProperties = OperationProperties(
        kernel=KERNEL,
        operation=WarpOperation.Gradient,
        supportMode=SupportScheme.Gather,
        operationMode=OperationDirection.AllToAll,
    )

    def f(pos, sup, mass, dens):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        _, _, renormState = computeRenormalizationMatrices(p, operationProperties, domain, adjacency=adjacency)
        return renormState.renormalizationMatrices

    print(f"\n=== {name} ({traversal} traversal): torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses, densities)
    try:
        ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a canary script
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def run_traversal_parity(name: str, positions: torch.Tensor, supports: torch.Tensor, masses: torch.Tensor) -> bool:
    """Same rationale as gradcheck_covariance_native.py's run_traversal_parity: the three
    traversal inputs sum neighbor contributions in different orders, so exact equality
    (not just closeness) here is a meaningful, tight check for these small deterministic
    cases."""
    domain = make_domain()
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    grid, _ = build_grid_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    operationProperties = OperationProperties(
        kernel=KERNEL,
        operation=WarpOperation.Gradient,
        supportMode=SupportScheme.Gather,
        operationMode=OperationDirection.AllToAll,
    )
    p = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=densities.detach(), kinds=kinds)

    _, _, r_adjacency = computeRenormalizationMatrices(p, operationProperties, domain, adjacency=adjacency)
    _, _, r_grid = computeRenormalizationMatrices(p, operationProperties, domain, adjacency=grid)
    _, _, r_none = computeRenormalizationMatrices(p, operationProperties, domain, adjacency=None)

    diff_grid = (r_adjacency.renormalizationMatrices - r_grid.renormalizationMatrices).abs().max().item()
    diff_none = (r_grid.renormalizationMatrices - r_none.renormalizationMatrices).abs().max().item()

    print(f"\n=== {name}: adjacency vs. grid vs. None forward-value parity ===")
    print(f"max|L_adjacency - L_grid| = {diff_grid:.3e}, max|L_grid - L_none| = {diff_none:.3e}")
    ok = diff_grid < 1e-10 and diff_none < 1e-10
    print("PASSED" if ok else "FAILED")
    return ok


def main():
    wp.init()
    torch.manual_seed(0)

    cases = [
        ("single particle (h=1)", single_particle_case()),
        ("line of 7 particles [-1, 1]", line_case(7)),
    ]

    ok = True
    for name, particles in cases:
        for traversal in ("adjacency", "grid", "none"):
            ok &= run_gradcheck(name, *particles, traversal=traversal)
        ok &= run_traversal_parity(name, *particles)

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- see warpier_core.md's renormalization grid-mode coverage note.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
