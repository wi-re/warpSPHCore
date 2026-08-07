#!/usr/bin/env python3
"""Plain (non-gradcheck) forward + .backward() debugging harness for CRK factors.

gradcheck_crk_native.py's numerical-vs-analytical comparison tells you *that*
backward is broken but buries *why* under gradcheck's own machinery (it calls
backward() once per output element, catches/re-raises internally, and reports an
all-NaN analytical Jacobian rather than the originating op). This script instead does
exactly one plain `loss.backward()` per case, with torch.autograd.set_detect_anomaly
enabled by default, so PyTorch itself stops at the offending backward op and names it.

Also prints every intermediate CRK-terms quantity (m_0, m_2 determinant, the
low-neighbor/singular mask, A/B/gradA/gradB pre- and post-mask) so you can see
*where* a NaN/Inf first appears, not just that backward eventually fails.

As of this writing both cases below backward cleanly (finite gradients throughout) --
this script found two independent bugs while chasing the original NaN, both now fixed,
and stays here as a regression guard / template for debugging this pipeline again:

1. crk_terms.py's computeCRKTermsWarp used in-place `A[mask] = 1.0` on an
   autograd-tracked tensor -- fixed by switching to torch.where.
2. The ACTUAL primary cause, found by bisecting with a minimal standalone repro
   (isolating computeCRKVolume_Func_Adjacency from crk_terms.py, masking, and
   torch.where entirely -- a raw wp.Tape().backward() on just that function's output
   was already NaN): Warp's adjoint for a *dynamic* for-loop (`for o in range(numOffsets)`,
   numOffsets a runtime value) that accumulates into a local via `+=` and then feeds
   that local into a nonlinear op (division) *inside the same @wp.func* produces NaN
   gradients. computeCRKVolume_Func_Adjacency and computeCRKDensity_Func_Adjacency
   (crk_volume.py / crk_density.py) both used to do exactly this (return `1/wsum` /
   `mDensity/vol1` directly from the looped function); every other migrated operator's
   `_Func_Adjacency` was unaffected because none of them postprocess the
   loop-accumulated value inside the looped function -- they all return the raw
   accumulated sum. Fixed by returning `(wsum, masked)` / `(mDensity, vol1, masked)`
   from the looped function and applying the division one level up, in the
   `@wp.kernel`, outside the loop. This is a narrow, specific trigger, not a general
   "dynamic loops can't be differentiated" limitation -- Gradient/Covariance/Density's
   own dynamic offset-loops gradcheck cleanly precisely because they don't do this.

Both fixes were independently necessary -- fixing only one still left the other's
failure mode able to surface. See warpier_core.md's "Landing CRK's dual-path rework"
and the project_crk_dualpath_and_latent_bugs memory entry for the full writeup.

Usage:
    python scripts/debug_crk_backward.py                  # both cases, anomaly detection on
    python scripts/debug_crk_backward.py --case line       # only the line-of-7 case
    python scripts/debug_crk_backward.py --no-anomaly      # let the raw error surface unshaped
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import torch
import warp as wp

from _gradcheck_common import build_adjacency, line_case, make_domain, single_particle_case
from warpSPHCore import ParticleState
from warpSPHCore.crk.crk_moments import _computeCRKMoments_stateBackend
from warpSPHCore.crk.crk_terms import computeCRKTermsWarp
from warpSPHCore.crk.crk_volume import _computeCRKVolume_stateBackend
from warpSPHCore.dataTypes import OperationProperties
from warpSPHCore.enumTypes import KernelFunctions, OperationDirection, SupportScheme

KERNEL = KernelFunctions.Wendland2


def describe(name: str, t: torch.Tensor) -> None:
    finite = torch.isfinite(t)
    print(
        f"    {name}: shape={tuple(t.shape)} "
        f"has_nan={bool(torch.isnan(t).any())} has_inf={bool(torch.isinf(t).any())} "
        f"min={t[finite].min().item() if finite.any() else float('nan'):.4g} "
        f"max={t[finite].max().item() if finite.any() else float('nan'):.4g}"
    )


def run_case(name: str, positions: torch.Tensor, supports: torch.Tensor, masses: torch.Tensor) -> None:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")

    domain = make_domain()
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)

    p = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)

    volumeProperties = OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll)
    apparentArea = _computeCRKVolume_stateBackend(p, volumeProperties, domain, adjacency=adjacency)
    print("  apparentArea:")
    describe("apparentArea", apparentArea)

    momentsProperties = OperationProperties(kernel=KERNEL, supportMode=SupportScheme.Scatter, operationMode=OperationDirection.AllToAll)
    m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma, numNeighbors = _computeCRKMoments_stateBackend(
        p, momentsProperties, domain, queryVolumes=apparentArea, referenceVolumes=apparentArea, adjacency=adjacency,
    )
    print("  moments:")
    describe("m_0", m_0)
    describe("m_2", m_2)
    m_2_det = torch.det(m_2).abs()
    print(f"    m_2 determinant: min={m_2_det.min().item():.4g} max={m_2_det.max().item():.4g}")
    print(f"    numNeighbors: {numNeighbors.tolist()}")
    mask = (numNeighbors < 2) | (m_2_det < 1e-14)
    print(f"    low-neighbor/singular mask: {mask.tolist()}  ({int(mask.sum())}/{mask.numel()} particles)")

    A, B, gradA, gradB = computeCRKTermsWarp(m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma, num_nbrs=numNeighbors, supports=supports)
    print("  CRK terms (post-mask, as returned to callers):")
    describe("A", A)
    describe("B", B)
    describe("gradA", gradA)
    describe("gradB", gradB)

    loss = A.sum() + B.sum() + gradA.sum() + gradB.sum()
    print(f"\n  loss = {loss.item():.6g}  -- calling loss.backward() ...")
    try:
        loss.backward()
    except RuntimeError as exc:
        print(f"  loss.backward() RAISED: {type(exc).__name__}: {exc}")
        return

    print("  loss.backward() completed without raising.")
    for label, t in [("positions.grad", positions.grad), ("supports.grad", supports.grad), ("masses.grad", masses.grad)]:
        if t is None:
            print(f"    {label}: None (not populated -- did you pass a non-leaf tensor?)")
        else:
            describe(label, t)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=["single", "line", "both"], default="both")
    parser.add_argument("--no-anomaly", action="store_true", help="disable torch.autograd.set_detect_anomaly")
    args = parser.parse_args()

    wp.init()
    torch.manual_seed(0)
    torch.autograd.set_detect_anomaly(not args.no_anomaly)
    print(f"torch.autograd.set_detect_anomaly({not args.no_anomaly})")

    cases = []
    if args.case in ("single", "both"):
        cases.append(("single particle (h=1) -- self-term only, m_2 is exactly singular, forces the low-neighbor/singular mask branch", single_particle_case()))
    if args.case in ("line", "both"):
        cases.append(("line of 7 particles [-1, 1] -- well-conditioned, mask should be all-False", line_case(7)))

    for name, particles in cases:
        run_case(name, *particles)


if __name__ == "__main__":
    main()
