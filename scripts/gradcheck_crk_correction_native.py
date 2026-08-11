#!/usr/bin/env python3
"""Native torch.autograd.gradcheck against an SPH operator with CRK correction
*enabled end-to-end* (crkState computed from positions via computeCRKFactors,
chained straight into WarpOperation.Gradient/Divergence/Curl -- i.e. the exact
shape of a real force-computation call site such as a pressure-gradient term,
not a synthetic/isolated one).

Why this script exists: gradcheck_crk_native.py already covers d(A, B, gradA,
gradB)/d(position) -- the CRK *factor* pipeline (crk_moments.py + crk_terms.py)
-- and passes. But none of the per-operator gradcheck scripts
(gradcheck_gradient_native.py, gradcheck_divergence_native.py, ...) ever pass
a crkState, so crk/kernel.py's correctGradientCRK (the function that actually
*applies* A/B/gradA/gradB to the raw kernel gradient via the CRK product-rule
expansion -- term1..term4) has never been exercised by any gradcheck in this
repo. A downstream user (compressibleSPH, scripts/gradcheck_crk.py) reported
gradcheck failing with a finite-but-wrong Jacobian for exactly this path, and
narrowed it to correctGradientCRK's handling of nonzero B/gradB (an identity
CRK correction -- A=1, B=0, gradA=0, gradB=0 -- passes; term1/term2/term3
vanish or reduce to the uncorrected kernel gradient in that case, so only
term4 is actually exercised by a nonzero-B configuration). This script
reproduces that end-to-end, inside this repo, against the real CRK pipeline
(not a synthetic one) so the bug -- and any fix -- can be verified here.

Root cause (confirmed via a from-scratch isolated single-neighbor-pair repro,
independent of this pipeline and of crk_moments.py/crk_terms.py entirely --
see the "correctGradientCRK isolated" scratch check in this investigation):
term4's original manual double `for row / for col: product[row] += x_ij[col]
* gradBi[row, col]` accumulation produced a wrong *adjoint* under Warp's
reverse-mode AD -- not a forward-value bug, and not an index-order/transpose
bug either (the failure reproduces even in 1D, where a transpose is a no-op
on a 1x1 matrix). Routing the exact same contraction through the existing
`matmul(wp.transpose(gradBi), x_ij)` @wp.func (already used elsewhere in this
codebase, e.g. wp_gradient.py's renormalization path) instead of a manual
loop fixes it. This is the same "index-accumulated loop can silently produce
a wrong adjoint" bug class flagged in docs/lessons_learned.md, but confirmed
here as a distinct instance from the previously-fixed dynamic-loop/division
and ternary-adjoint-zeroing bugs -- worth treating any new manual
accumulation loop over a Warp vector/matrix as suspect until gradchecked.

    python scripts/gradcheck_crk_correction_native.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, grid_case_2d, line_case, make_domain
from warpSPHCore import OperationProperties, ParticleState, warpOperation
from warpSPHCore.crk import computeCRKFactors
from warpSPHCore.enumTypes import GradientScheme, OperationDirection, SupportScheme, WarpOperation


def compute_densities(positions, supports, masses, kinds, domain, adjacency):
    """Realistic density magnitudes via the (separately gradchecked) Density
    op, then detached and re-leafed -- same rationale as every other
    gradcheck_*_native.py script's compute_densities."""
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


def run_gradcheck(name: str, positions: torch.Tensor, supports: torch.Tensor, masses: torch.Tensor, operation: WarpOperation, gradient_mode: GradientScheme) -> bool:
    domain = make_domain(dim=positions.shape[1])
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    n = positions.shape[0]
    query_values = torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)
    reference_values = torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def f(pos, sup, mass, dens, qval, rval):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        # crkState is a function of the SAME leaf positions/supports/masses gradcheck
        # perturbs -- this is what actually exercises correctGradientCRK's term1..term4
        # under a real (nonzero) B/gradB, matching a genuine force-computation call site.
        _, _, crkState = computeCRKFactors(
            queryParticles=p, domain=domain, kernel=KERNEL,
            operationMode=OperationDirection.AllToAll, adjacency=adjacency,
        )
        return warpOperation(
            p,
            OperationProperties(
                kernel=KERNEL,
                operation=operation,
                supportMode=SupportScheme.Gather,
                operationMode=OperationDirection.AllToAll,
                gradientMode=gradient_mode,
            ),
            domain,
            queryValues=qval,
            referenceValues=rval,
            adjacency=adjacency,
            crkState=crkState,
        )

    print(f"\n=== {name} ({operation.name}, {gradient_mode.name}, CRK-corrected): torch.autograd.gradcheck ===")
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
        ("line of 7 particles [-1, 1]", line_case(7)),
        ("3x3 grid (2D)", grid_case_2d(3)),
    ]

    ok = True
    for name, particles in cases:
        for gradient_mode in (GradientScheme.Naive, GradientScheme.Difference):
            ok &= run_gradcheck(name, *particles, operation=WarpOperation.Gradient, gradient_mode=gradient_mode)

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- CRK-corrected kernel gradient (crk/kernel.py: correctGradientCRK) has a wrong Jacobian.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
