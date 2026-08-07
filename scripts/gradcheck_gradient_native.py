#!/usr/bin/env python3
"""Native torch.autograd.gradcheck against the Gradient operator -- no workarounds.

Stage 3 of the Gradcheck Script Rollout Plan (see warpier_core.md's
"Backward-Mode (Reverse AD) Findings"). Follows gradcheck_interpolate_native.py's
pattern: ParticleState + warpOperation wired straight into
torch.autograd.gradcheck, no manual Jacobian, no per-call cloning.

This is a *backward*-mode check (d(gradient_output)/d(position|support|mass|
density|value)) -- distinct from the *forward*-value check the rollout plan
also calls for (does WarpOperation.Gradient's output match a hand-coded
reference SPH gradient sum for each GradientScheme?). That forward-value /
plot comparison is not implemented here; this script only exercises the
reverse-mode gradient.

Differentiable inputs checked: positions, supports, masses, densities,
queryValues, referenceValues, across all four GradientScheme variants
(Naive/Symmetric/Difference/Summation) and both scalar- and vector-rank input
fields (Gradient's output rank is always the input rank + 1, for the spatial
dimension). Densities are computed once via the (already-verified) Density op
for realistic magnitudes, then detached and re-leafed as an independent
gradcheck input -- see gradcheck_interpolate_native.py's docstring for why
(Density's own backward is Stage 1's job, not chained through here).

A "shared query==reference tensor" variant (same object passed as both
queryValues and referenceValues) is checked once per field rank, as a
standing regression guard for the AD-bridge shared-tensor class of bug found
in Stage 1 -- not run across all four schemes, since Stage 2 already
confirmed this class of bug is not scheme/operator-specific.

    python scripts/gradcheck_gradient_native.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, line_case, make_domain, single_particle_case
from warpSPHCore import OperationProperties, ParticleState, warpOperation
from warpSPHCore.enumTypes import GradientScheme, OperationDirection, SupportScheme, WarpOperation


def compute_densities(positions, supports, masses, kinds, domain, adjacency):
    """See gradcheck_interpolate_native.py's compute_densities -- identical
    rationale: realistic magnitudes via the separately gradchecked Density
    op, detached and re-leafed rather than chained through its backward."""
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


def run_gradcheck(name: str, positions: torch.Tensor, supports: torch.Tensor, masses: torch.Tensor, field_rank: str, gradient_mode: GradientScheme, shared_values: bool) -> bool:
    domain = make_domain()
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    n = positions.shape[0]
    value_shape = (n,) if field_rank == "scalar" else (n, 3)
    query_values = torch.randn(*value_shape, dtype=DTYPE, device=DEVICE, requires_grad=True)
    reference_values = query_values if shared_values else torch.randn(*value_shape, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def f(pos, sup, mass, dens, qval, rval):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        return warpOperation(
            p,
            OperationProperties(
                kernel=KERNEL,
                operation=WarpOperation.Gradient,
                supportMode=SupportScheme.Gather,
                operationMode=OperationDirection.AllToAll,
                gradientMode=gradient_mode,
            ),
            domain,
            queryValues=qval,
            referenceValues=rval,
            adjacency=adjacency,
        )

    tag = "shared query==reference tensor" if shared_values else "distinct query/reference tensors"
    print(f"\n=== {name} ({field_rank} field, {gradient_mode.name}, {tag}): torch.autograd.gradcheck ===")
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
        ("single particle (h=1)", single_particle_case()),
        ("line of 7 particles [-1, 1]", line_case(7)),
    ]

    ok = True
    for name, particles in cases:
        for gradient_mode in GradientScheme:
            for field_rank in ("scalar", "vector"):
                ok &= run_gradcheck(name, *particles, field_rank=field_rank, gradient_mode=gradient_mode, shared_values=False)
        # Shared-tensor regression guard: once per field rank, Naive scheme only.
        for field_rank in ("scalar", "vector"):
            ok &= run_gradcheck(name, *particles, field_rank=field_rank, gradient_mode=GradientScheme.Naive, shared_values=True)

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- see warpier_core.md's Gradcheck Script Rollout Plan, Stage 3.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
