#!/usr/bin/env python3
"""Native torch.autograd.gradcheck against the Laplacian operator -- no workarounds.

Stage 6 of the Gradcheck Script Rollout Plan (see warpier_core.md's
"Backward-Mode (Reverse AD) Findings"). Follows gradcheck_curl_native.py's
pattern: ParticleState + warpOperation wired straight into
torch.autograd.gradcheck, no manual Jacobian, no per-call cloning.

Scalar field only (per the rollout plan), cross product of all four
GradientScheme x all four LaplacianScheme variants (Naive/Brookshaw/Dot/
Default), on the 2D grid case (grid_case_2d) rather than the 1D line_case --
deliberately, because LaplacianScheme.Dot's implementation
(computeLaplacianDot2, wp_laplacian.py) indexes q_ij[base+k] for
k in range(dim) where q_ij is a flatInputShape-sized vector; for a scalar
field flatInputShape==1, so this only reads within bounds when dim==1. A 1D
domain would never exercise the out-of-bounds path at all, so this script
deliberately uses dim=2 to actually test it.

Differentiable inputs checked: positions, supports, masses, densities,
queryValues, referenceValues, plus the shared query==reference-tensor
regression guard (LaplacianScheme.Default / GradientScheme.Naive only).

    python scripts/gradcheck_laplacian_native.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, grid_case_2d, make_domain
from warpSPHCore import OperationProperties, ParticleState, warpOperation
from warpSPHCore.enumTypes import GradientScheme, LaplacianScheme, OperationDirection, SupportScheme, WarpOperation


def compute_densities(positions, supports, masses, kinds, domain, adjacency):
    """See gradcheck_interpolate_native.py's compute_densities."""
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


def run_gradcheck(name: str, positions: torch.Tensor, supports: torch.Tensor, masses: torch.Tensor, gradient_mode: GradientScheme, laplacian_mode: LaplacianScheme, shared_values: bool) -> bool:
    domain = make_domain(dim=2)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    n = positions.shape[0]
    query_values = torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)
    reference_values = query_values if shared_values else torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def f(pos, sup, mass, dens, qval, rval):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        return warpOperation(
            p,
            OperationProperties(
                kernel=KERNEL,
                operation=WarpOperation.Laplacian,
                supportMode=SupportScheme.Gather,
                operationMode=OperationDirection.AllToAll,
                gradientMode=gradient_mode,
                laplacianMode=laplacian_mode,
            ),
            domain,
            queryValues=qval,
            referenceValues=rval,
            adjacency=adjacency,
        )

    tag = "shared query==reference tensor" if shared_values else "distinct query/reference tensors"
    print(f"\n=== {name} (scalar field, {gradient_mode.name}/{laplacian_mode.name}, {tag}): torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses, densities, query_values, reference_values)

    # LaplacianScheme.Dot is guarded off for scalar fields in >1D domains --
    # computeLaplacianDot2 indexes q_ij[block*dim+k] for k in range(dim),
    # which reads out of bounds when the field's flattened size (1, here)
    # isn't a multiple of dim. Confirmed via gradcheck itself before the
    # guard existed (real, silent wrong-gradient bug, not just theoretical --
    # see warpier_core.md's Stage 6 entry). Expect the ValueError here rather
    # than a gradient match.
    expect_dot_guard = laplacian_mode == LaplacianScheme.Dot and positions.shape[1] > 1
    try:
        ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5)
        if expect_dot_guard:
            print("FAILED: expected LaplacianScheme.Dot's scalar-field guard to raise, but gradcheck ran and returned", ok)
            return False
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except ValueError as exc:
        if expect_dot_guard and "computeLaplacianDot2" in str(exc):
            print(f"PASSED (scalar-field guard correctly raised: {exc})")
            return True
        print(f"FAILED: unexpected {type(exc).__name__}: {exc}")
        return False
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a canary script
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def main():
    wp.init()
    torch.manual_seed(0)

    cases = [
        ("single particle (h=1)", grid_case_2d(n_per_side=1)),
        ("3x3 grid", grid_case_2d(n_per_side=3)),
    ]

    ok = True
    for name, particles in cases:
        for gradient_mode in GradientScheme:
            for laplacian_mode in LaplacianScheme:
                ok &= run_gradcheck(name, *particles, gradient_mode=gradient_mode, laplacian_mode=laplacian_mode, shared_values=False)
        ok &= run_gradcheck(name, *particles, gradient_mode=GradientScheme.Naive, laplacian_mode=LaplacianScheme.Default, shared_values=True)

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- see warpier_core.md's Gradcheck Script Rollout Plan, Stage 6.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
