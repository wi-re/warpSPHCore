#!/usr/bin/env python3
"""Native torch.autograd.gradcheck against the Interpolate operator -- no workarounds.

Stage 2 of the Gradcheck Script Rollout Plan (see warpier_core.md's
"Backward-Mode (Reverse AD) Findings"). Follows gradcheck_density_native.py's
pattern directly: ParticleState + warpOperation wired straight into
torch.autograd.gradcheck, no manual Jacobian, no per-call cloning.

Differentiable inputs checked: positions, supports, masses, densities,
queryValues, referenceValues -- each case runs both a scalar-rank and a
vector-rank field, and both a "distinct query/reference tensors" variant and
a "shared (same object) query/reference tensor" variant. The shared-tensor
variant specifically mirrors the self-interaction pattern
(referencePositions is queryPositions) that turned up the AD-bridge
gradient-doubling bug for Density -- see warpier_core.md -- so it stays here
as a standing regression guard for that class of bug on Interpolate too.

Densities: computeSPHInterpolation_Func's `rhoi = queryDensities[i]` is
unpacked but never read in the current kernel body (only
referenceDensities feeds the m_j/rho_j volume term) -- so
d(output)/d(queryDensities) is mathematically zero. This script does not
special-case that; gradcheck will simply see (and pass) a zero column
there. Density's own backward pass is Stage 1's job
(gradcheck_density_native.py) -- chaining through it here would turn this
into a second-derivative check against Density instead of a first-derivative
check against Interpolate. So: densities are computed once via the
already-verified Density operator (for realistic, physically-consistent
magnitudes) and then detached and re-leafed as their own independent
gradcheck input.

    python scripts/gradcheck_interpolate_native.py
"""

from __future__ import annotations

import os

os.environ.setdefault("SPHWARPCORE_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, line_case, make_domain, single_particle_case
from sphWarpCore import OperationProperties, ParticleState, warpOperation
from sphWarpCore.enumTypes import OperationDirection, SupportScheme, WarpOperation


def compute_densities(positions, supports, masses, kinds, domain, adjacency):
    """Realistic density magnitudes via the (separately gradchecked) Density
    op, then detached and re-leafed -- see module docstring for why this
    script doesn't chain gradients through Density itself."""
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


def run_gradcheck(name: str, positions: torch.Tensor, supports: torch.Tensor, masses: torch.Tensor, field_rank: str, shared_values: bool) -> bool:
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
                operation=WarpOperation.Interpolate,
                supportMode=SupportScheme.Gather,
                operationMode=OperationDirection.AllToAll,
            ),
            domain,
            queryValues=qval,
            referenceValues=rval,
            adjacency=adjacency,
        )

    tag = "shared query==reference tensor" if shared_values else "distinct query/reference tensors"
    print(f"\n=== {name} ({field_rank} field, {tag}): torch.autograd.gradcheck ===")
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
        for field_rank in ("scalar", "vector"):
            for shared_values in (False, True):
                ok &= run_gradcheck(name, *particles, field_rank=field_rank, shared_values=shared_values)

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- see warpier_core.md's Gradcheck Script Rollout Plan, Stage 2.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
