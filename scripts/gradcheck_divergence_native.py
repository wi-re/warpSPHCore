#!/usr/bin/env python3
"""Native torch.autograd.gradcheck against the Divergence operator -- no workarounds.

Stage 4 of the Gradcheck Script Rollout Plan (see warpier_core.md's
"Backward-Mode (Reverse AD) Findings"). Follows gradcheck_gradient_native.py's
pattern: ParticleState + warpOperation wired straight into
torch.autograd.gradcheck, no manual Jacobian, no per-call cloning.

Differentiable inputs checked: positions, supports, masses, densities,
queryValues, referenceValues, across all four GradientScheme variants.
Densities are computed once via the (already-verified) Density op for
realistic magnitudes, then detached and re-leafed -- see
gradcheck_interpolate_native.py's docstring for the rationale.

Field ranks tested:
  * "vector" (shape (n, D)) -> scalar output. For a rank-1 input,
    divergenceDotMode's two index formulas (`fij[i*dim+d]` vs.
    `fij[i+d*outputElements]`, see divergenceProduct in wp_divergence.py)
    are algebraically identical when outputElements == 1, so this case
    exercises the flag being threaded through correctly without testing a
    case where it actually changes the computation.
  * "matrix" (shape (n, D, D)) -> vector output. This is where dotMode's two
    conventions genuinely differ (see warpier_core.md's Reality Check entry
    on `divergenceDotMode`) -- both are checked. Note gradcheck only needs
    internal AD-vs-finite-difference consistency for whatever field it's
    given; it does not need the field pre-transposed to match a physical
    "correct" convention the way operation_matrix.py's analytic comparison
    does.

Uses a small 2D grid case (grid_case_2d) rather than the 1D line_case the
earlier stages used -- a 2D domain is what actually exercises the matrix/
dotMode paths (Curl will need one too).

    python scripts/gradcheck_divergence_native.py
"""

from __future__ import annotations

import os

os.environ.setdefault("SPHWARPCORE_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, grid_case_2d, make_domain
from sphWarpCore import OperationProperties, ParticleState, warpOperation
from sphWarpCore.enumTypes import GradientScheme, OperationDirection, SupportScheme, WarpOperation


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


def run_gradcheck(name: str, positions: torch.Tensor, supports: torch.Tensor, masses: torch.Tensor, field_rank: str, dot_mode: bool, gradient_mode: GradientScheme, shared_values: bool) -> bool:
    domain = make_domain(dim=2)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    n = positions.shape[0]
    D = positions.shape[1]
    value_shape = (n, D) if field_rank == "vector" else (n, D, D)
    query_values = torch.randn(*value_shape, dtype=DTYPE, device=DEVICE, requires_grad=True)
    reference_values = query_values if shared_values else torch.randn(*value_shape, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def f(pos, sup, mass, dens, qval, rval):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        return warpOperation(
            p,
            OperationProperties(
                kernel=KERNEL,
                operation=WarpOperation.Divergence,
                supportMode=SupportScheme.Gather,
                operationMode=OperationDirection.AllToAll,
                gradientMode=gradient_mode,
                divergenceDotMode=dot_mode,
            ),
            domain,
            queryValues=qval,
            referenceValues=rval,
            adjacency=adjacency,
        )

    tag = "shared query==reference tensor" if shared_values else "distinct query/reference tensors"
    print(f"\n=== {name} ({field_rank} field, dotMode={dot_mode}, {gradient_mode.name}, {tag}): torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses, densities, query_values, reference_values)
    try:
        ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
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
        # Vector field: dotMode doesn't change the formula here (rank-1 input),
        # so cover all GradientScheme variants with dotMode=False, and just
        # sanity-check dotMode=True threading with one scheme.
        for gradient_mode in GradientScheme:
            ok &= run_gradcheck(name, *particles, field_rank="vector", dot_mode=False, gradient_mode=gradient_mode, shared_values=False)
        ok &= run_gradcheck(name, *particles, field_rank="vector", dot_mode=True, gradient_mode=GradientScheme.Naive, shared_values=False)

        # Matrix field: dotMode genuinely changes the contraction here -- check both conventions, all schemes.
        for dot_mode in (False, True):
            for gradient_mode in GradientScheme:
                ok &= run_gradcheck(name, *particles, field_rank="matrix", dot_mode=dot_mode, gradient_mode=gradient_mode, shared_values=False)

        # Shared-tensor regression guard.
        ok &= run_gradcheck(name, *particles, field_rank="vector", dot_mode=False, gradient_mode=GradientScheme.Naive, shared_values=True)

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- see warpier_core.md's Gradcheck Script Rollout Plan, Stage 4.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
