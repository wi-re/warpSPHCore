#!/usr/bin/env python3
"""Native torch.autograd.gradcheck against CRK factors (A, B, gradA, gradB) -- no
workarounds.

crk/crk_moments.py, crk/crk_volume.py, and crk/crk_density.py were migrated from the
legacy neighbor-list-only kernel style to the dual-path (useAdjacency branch,
neighbor-list + grid/CompactHashMap traversal in one kernel) style every operator in
coreOperations/ already uses -- see warpier_core.md's "Landing CRK's dual-path rework".
computeCRKFactors (crk/crk_wrapper.py) no longer raises NotImplementedError for
adjacency=None/CompactHashMap as a result.

Like gradcheck_covariance_native.py, this script deliberately exercises BOTH traversal
branches (build_adjacency -> AdjacencyList/useAdjacency=True, build_grid_adjacency ->
CompactHashMap/useAdjacency=False) plus a forward-value parity check between them, since
validating the grid path is the point of the migration this script guards.

Both the forward-value parity checks and every gradcheck call now PASS. Getting here
took two independent fixes, neither of which was "the traversal migration" itself --
see warpier_core.md's "Landing CRK's dual-path rework" and the
project_crk_dualpath_and_latent_bugs memory entry for the full writeup:

1. crk_terms.py's computeCRKTermsWarp low-neighbor/singular-matrix fallback used to do
   in-place `A[mask] = 1.0` (and similarly for B/gradA/gradB) on tensors that are
   themselves autograd-tracked -- `Tensor.__setitem__` bumps the version counter
   unconditionally, breaking backward() regardless of whether mask selects anything.
   Fixed by switching to `torch.where`.
2. computeCRKVolume_Func_Adjacency / computeCRKDensity_Func_Adjacency (crk_volume.py /
   crk_density.py) used to return an already-divided value (1/wsum, mDensity/vol1) from
   the same @wp.func that contains their dynamic `for o in range(numOffsets)` loop.
   Warp's adjoint for a dynamic-trip-count loop that accumulates into a local via `+=`
   and then feeds that local into a nonlinear op (division) *inside the same function*
   produces NaN gradients -- confirmed via a minimal standalone repro (see
   scripts/debug_crk_backward.py) that reproduces with no dependency on crk_terms.py,
   masking, or torch.where at all. No other migrated operator hit this because none of
   them apply a postprocessing transform to the loop-accumulated value inside the
   looped function -- they all return the raw accumulated sum directly. Fixed by
   returning the raw accumulated sum (plus a `masked` flag) from the looped function and
   applying the division one level up, in the @wp.kernel, outside the loop.

Both fixes were necessary; either alone still left the other's failure mode able to
surface (fixing only #1 leaves fixing #2 to still show NaN gradients, and vice versa).

This script IS registered in tests/operations/test_gradcheck_scripts.py's
GRADCHECK_SCRIPTS list, same as every other operator's gradcheck_*_native.py script.

    python scripts/gradcheck_crk_native.py
"""

from __future__ import annotations

import os

os.environ.setdefault("SPHWARPCORE_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, build_grid_adjacency, line_case, make_domain, single_particle_case
from sphWarpCore import ParticleState
from sphWarpCore.crk import computeCRKFactors
from sphWarpCore.enumTypes import OperationDirection


def run_gradcheck(name: str, positions: torch.Tensor, supports: torch.Tensor, masses: torch.Tensor, traversal: str) -> bool:
    domain = make_domain()
    if traversal == "adjacency":
        adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    else:
        adjacency, kinds = build_grid_adjacency(positions, supports, masses, domain)

    def f(pos, sup, mass):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=None, kinds=kinds)
        _, crk_density, crk = computeCRKFactors(
            queryParticles=p, domain=domain, kernel=KERNEL,
            operationMode=OperationDirection.AllToAll, adjacency=adjacency,
        )
        return crk.A, crk.B, crk.gradA, crk.gradB, crk_density

    print(f"\n=== {name} ({traversal} traversal): torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses)
    try:
        ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a canary script
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def run_traversal_parity(name: str, positions: torch.Tensor, supports: torch.Tensor, masses: torch.Tensor) -> bool:
    """Forward-value-only check (no backward involved): the two traversal branches sum
    the same neighbor contributions in different orders (CSR neighbor order vs.
    cell-by-cell grid order), so exact equality isn't guaranteed in general -- see
    warpier_core.md's Gradient section -- but should hold to a tight tolerance for
    these small, deterministic cases. adjacency=None (auto-built CompactHashMap) is
    checked against an explicit CompactHashMap too, and should be bit-exact."""
    domain = make_domain()
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    grid, _ = build_grid_adjacency(positions, supports, masses, domain)
    p = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=None, kinds=kinds)

    area_adj, dens_adj, crk_adj = computeCRKFactors(queryParticles=p, domain=domain, kernel=KERNEL, operationMode=OperationDirection.AllToAll, adjacency=adjacency)
    area_grid, dens_grid, crk_grid = computeCRKFactors(queryParticles=p, domain=domain, kernel=KERNEL, operationMode=OperationDirection.AllToAll, adjacency=grid)
    area_none, dens_none, crk_none = computeCRKFactors(queryParticles=p, domain=domain, kernel=KERNEL, operationMode=OperationDirection.AllToAll, adjacency=None)

    print(f"\n=== {name}: adjacency vs. grid forward-value parity ===")
    ok = True
    for label, a, b in [
        ("apparentArea", area_adj, area_grid),
        ("crk_density", dens_adj, dens_grid),
        ("A", crk_adj.A, crk_grid.A),
        ("B", crk_adj.B, crk_grid.B),
        ("gradA", crk_adj.gradA, crk_grid.gradA),
        ("gradB", crk_adj.gradB, crk_grid.gradB),
    ]:
        diff = (a - b).abs().max().item()
        step_ok = diff < 1e-6
        ok &= step_ok
        print(f"  {label}: max|adjacency - grid| = {diff:.3e} {'ok' if step_ok else 'FAILED'}")

    diff_none = (crk_grid.A - crk_none.A).abs().max().item()
    none_ok = diff_none < 1e-12
    ok &= none_ok
    print(f"  A: max|grid - None| = {diff_none:.3e} (should be bit-exact) {'ok' if none_ok else 'FAILED'}")

    print("PASSED" if ok else "FAILED")
    return ok


def main():
    wp.init()
    torch.manual_seed(0)

    cases = [
        ("single particle (h=1)", single_particle_case()),
        ("line of 7 particles [-1, 1]", line_case(7)),
    ]

    forward_ok = True
    backward_ok = True
    for name, particles in cases:
        forward_ok &= run_traversal_parity(name, *particles)
        for traversal in ("adjacency", "grid"):
            backward_ok &= run_gradcheck(name, *particles, traversal=traversal)

    print()
    print(f"Forward dual-path parity: {'PASSED' if forward_ok else 'FAILED'}")
    print(f"Backward gradcheck:       {'PASSED' if backward_ok else 'FAILED'}")

    sys.exit(0 if (forward_ok and backward_ok) else 1)


if __name__ == "__main__":
    main()
