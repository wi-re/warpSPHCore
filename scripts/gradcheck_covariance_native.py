#!/usr/bin/env python3
"""Native torch.autograd.gradcheck against Covariance -- no workarounds.

Covariance (src/sphWarpCore/coreOperations/wp_covariance.py) was the operator the
canonical structured kernel ABI (queryState/referenceState/domainState/useAdjacency/
adjacencyState/gridState/correctionData/...) was originally modeled on (see
warpier_core.md's "Working Prototype -> Production"), and its kernel already branched
on useAdjacency internally. What it didn't have until this rework was a Python entry
point that actually let adjacency be None or a CompactHashMap --
computeRenormalizationMatrices (the only caller at the time) unconditionally raised
NotImplementedError unless given an explicit AdjacencyList. Covariance is now the
seventh operation dispatched through warpOperation(..., operation=WarpOperation.Covariance,
...) exactly like Density/Interpolate/Gradient/Divergence/Curl/Laplacian, rather than a
standalone computeCovarianceMatrix function -- so this script checks both traversal
branches agree and are both differentiable correctly through that same entry point.

Unlike every other gradcheck_*_native.py script, this one deliberately exercises BOTH
traversal branches (build_adjacency -> AdjacencyList/useAdjacency=True, and the new
build_grid_adjacency -> CompactHashMap/useAdjacency=False), plus a forward-value
parity check between them, since validating the grid path is the point of this script
-- every sibling script only ever exercises the (default) neighbor-list branch, see
build_grid_adjacency's docstring in _gradcheck_common.py for why.

Differentiable inputs checked: positions, supports, masses, densities. Densities are
computed once via the (already-verified) Density op for realistic magnitudes, then
detached and re-leafed as an independent gradcheck input -- see
gradcheck_interpolate_native.py's docstring for why (Density's own backward is a
different script's job, not chained through here). Covariance's per-neighbor term
(apparentVolume = mj/rhoj when useVolume=False) needs real, nonzero densities or the
forward pass itself is 0/0 = NaN, which is why this is not optional here the way it is
for e.g. gradcheck_density_native.py.

    python scripts/gradcheck_covariance_native.py
"""

from __future__ import annotations

import os

os.environ.setdefault("SPHWARPCORE_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, build_grid_adjacency, line_case, make_domain, single_particle_case
from sphWarpCore import OperationProperties, ParticleState, warpOperation
from sphWarpCore.enumTypes import OperationDirection, SupportScheme, WarpOperation


def compute_densities(positions, supports, masses, kinds, domain, adjacency):
    """See gradcheck_interpolate_native.py's compute_densities -- identical rationale:
    realistic magnitudes via the separately gradchecked Density op, detached and
    re-leafed rather than chained through its backward."""
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
    else:
        adjacency, kinds = build_grid_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    operationProperties = OperationProperties(
        kernel=KERNEL,
        operation=WarpOperation.Covariance,
        supportMode=SupportScheme.Gather,
        operationMode=OperationDirection.AllToAll,
    )

    def f(pos, sup, mass, dens):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        return warpOperation(p, operationProperties, domain, adjacency=adjacency)

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
    """The two traversal branches sum the same neighbor contributions in different
    orders (CSR neighbor order vs. cell-by-cell grid order); for these small,
    deterministic cases that should still agree exactly, not just approximately -- see
    warpier_core.md's Gradient section for why bit-exactness isn't guaranteed in
    general (it isn't asserted that strictly here), just checked to a tight tolerance."""
    domain = make_domain()
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    grid, _ = build_grid_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    operationProperties = OperationProperties(
        kernel=KERNEL,
        operation=WarpOperation.Covariance,
        supportMode=SupportScheme.Gather,
        operationMode=OperationDirection.AllToAll,
    )
    p = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=densities.detach(), kinds=kinds)

    C_adjacency = warpOperation(p, operationProperties, domain, adjacency=adjacency)
    C_grid = warpOperation(p, operationProperties, domain, adjacency=grid)
    C_none = warpOperation(p, operationProperties, domain, adjacency=None)

    diff_grid = (C_adjacency - C_grid).abs().max().item()
    diff_none = (C_grid - C_none).abs().max().item()

    print(f"\n=== {name}: adjacency vs. grid forward-value parity ===")
    print(f"max|C_adjacency - C_grid| = {diff_grid:.3e}, max|C_grid - C_none| = {diff_none:.3e}")
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
        for traversal in ("adjacency", "grid"):
            ok &= run_gradcheck(name, *particles, traversal=traversal)
        ok &= run_traversal_parity(name, *particles)

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- see warpier_core.md's covariance dual-path rework.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
