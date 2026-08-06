#!/usr/bin/env python3
"""Native torch.autograd.gradcheck against the Curl operator -- no workarounds.

Stage 5 of the Gradcheck Script Rollout Plan (see warpier_core.md's
"Backward-Mode (Reverse AD) Findings"). Follows gradcheck_divergence_native.py's
pattern: ParticleState + warpOperation wired straight into
torch.autograd.gradcheck, no manual Jacobian, no per-call cloning.

In 2D, Curl takes a vector field (shape (n, D), D=2) and produces a scalar
output (computeSPHCurl_warpBackend sets outputShape=[1] for a 2D vector-field
input -- see wp_curl.py). Uses the same small 2D grid case as Divergence
(grid_case_2d in _gradcheck_common.py).

Curl has a specific history worth calling out: a previous bug (see
warpier_core.md's Reality Check) was a bare `return` inside a non-void
@wp.func (computeSPHCurlTensor_Func's directionality-mask early-out), which
nvcc silently tolerated but the CPU/LLVM backend correctly rejected as
invalid IR -- caught by operation_matrix.py's forward-value matrix, not by
any gradient check (no gradcheck existed for Curl until this stage). This
script is a standing regression guard for that class of thing recurring
specifically in the backward pass, not just the forward path.

Differentiable inputs checked: positions, supports, masses, densities,
queryValues, referenceValues, across all four GradientScheme variants, plus
the shared query==reference-tensor regression guard.

    python scripts/gradcheck_curl_native.py
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


def run_gradcheck(name: str, positions: torch.Tensor, supports: torch.Tensor, masses: torch.Tensor, gradient_mode: GradientScheme, shared_values: bool) -> bool:
    domain = make_domain(dim=2)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    n = positions.shape[0]
    D = positions.shape[1]
    query_values = torch.randn(n, D, dtype=DTYPE, device=DEVICE, requires_grad=True)
    reference_values = query_values if shared_values else torch.randn(n, D, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def f(pos, sup, mass, dens, qval, rval):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        return warpOperation(
            p,
            OperationProperties(
                kernel=KERNEL,
                operation=WarpOperation.Curl,
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
    print(f"\n=== {name} (2D vector field, {gradient_mode.name}, {tag}): torch.autograd.gradcheck ===")
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
        for gradient_mode in GradientScheme:
            ok &= run_gradcheck(name, *particles, gradient_mode=gradient_mode, shared_values=False)
        ok &= run_gradcheck(name, *particles, gradient_mode=GradientScheme.Naive, shared_values=True)

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- see warpier_core.md's Gradcheck Script Rollout Plan, Stage 5.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
