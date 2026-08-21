#!/usr/bin/env python3
"""Native torch.autograd.gradcheck / JVP-identity checks against Laplacian's
Dot-scheme Tier-2 (geometry) JVP with CRK correction enabled
(`warpier_tier2_correction_jvp_plan.md` follow-up, 2026-08-21, extending
phase (e) past its own original Brookshaw-only scope) --
`computeSPHLaplacianDotGeometryJVP(..., crkState=..., crkTangentState=...)`.
Naive stays permanently out of CRK scope (its estimator never consumes the
CRK-corrected `kernelGradient` at all, so there is no formula to correct),
matching `operations.py`'s own restriction.

See `gradcheck_tier2_jvp_divergence_crk.py`'s docstring for the shared
rationale -- Dot's own `F_ab = dot(G,n_ij)/D_ij` (bit-for-bit Brookshaw's `P`)
is the only genuinely reused piece; Stages 1-4
(`crk.computeKernelGradientCRKJVP`) are reused verbatim from phase (c) via
the shared `_laplacianGeometryChainJVP` building block.

**Found and fixed a genuine, pre-existing reverse-mode adjoint bug while
writing this script, not introduced by this follow-up**: see
`spike_forward_mode_tier2_laplacian_dot_default_extension.py`'s module
docstring for the full root-cause writeup -- the identical self-pair
(`r_ij == 0`) hazard phase (e) already found and fixed for Brookshaw
(`spike_forward_mode_tier2_crk_extension.py`), since Dot's own `F_ab` is
bit-for-bit Brookshaw's `P`. Fixed in both `wp_laplacian.py` (primal) and
`wp_laplacianJVP.py` (this Tier-2 JVP) by guarding Dot's own contribution
with an explicit `if r_ij > 0:`, mirroring Brookshaw's own fix exactly -- the
true contribution at `r_ij == 0` is always exactly `0` regardless of CRK, so
this changes no forward value anywhere (confirmed: `operation_matrix.py` and
every existing non-CRK gradcheck script stay bit-identical).

    python scripts/gradcheck_tier2_jvp_laplacian_dot_crk.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, line_case, make_domain
from warpSPHCore import OperationProperties, ParticleState, ParticleTangentState, warpOperation
from warpSPHCore.crk import computeCRKFactorsJVP
from warpSPHCore.dataTypes import CRKState, CRKTangentState
from warpSPHCore.enumTypes import GradientScheme, OperationDirection, SupportScheme, WarpOperation
from warpSPHCore.coreOperations.wp_laplacianJVP import computeSPHLaplacianDotGeometryJVP


def compute_densities(positions, supports, masses, kinds, domain, adjacency):
    p = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=None, kinds=kinds)
    rho = warpOperation(
        p,
        OperationProperties(kernel=KERNEL, operation=WarpOperation.Density, supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll),
        domain, adjacency=adjacency,
    )
    return rho.detach().clone().requires_grad_(True)


# ---------------------------------------------------------------------------
# 1. Targeted direct-tensor gradcheck: crkState/crkTangentState as
#    independent leaves, isolating _jvpCommon's flat_tensors wiring for
#    Laplacian(Brookshaw)'s own new crkState/crkTangentState parameters --
#    also the self-pair-adjoint-bug regression guard (this repo's own
#    "confirmed fixed via torch.autograd.gradcheck" discipline): 1D line_case
#    always includes at least one exact self-pair per query particle.
# ---------------------------------------------------------------------------

def run_direct_gradcheck() -> bool:
    domain = make_domain(dim=1)
    n = 6
    positions, supports, masses = line_case(n)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    pos = positions.detach().clone().requires_grad_(True)
    sup = supports.detach().clone().requires_grad_(True)
    dens = densities.detach().clone().requires_grad_(True)
    tqp = (0.1 * torch.randn(n, 1, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    tqs = (0.01 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    qval = torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)
    rval = torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)

    crk_A = (0.5 + 0.1 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    crk_B = (0.1 * torch.randn(n, 1, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    crk_gradA = (0.1 * torch.randn(n, 1, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    crk_gradB = (0.1 * torch.randn(n, 1, 1, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    d_crk_A = (0.1 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    d_crk_B = (0.1 * torch.randn(n, 1, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    d_crk_gradA = (0.1 * torch.randn(n, 1, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    d_crk_gradB = (0.1 * torch.randn(n, 1, 1, dtype=DTYPE, device=DEVICE)).requires_grad_(True)

    def f(pos, sup, dens, tqp, tqs, qval, rval, cA, cB, cgA, cgB, dcA, dcB, dcgA, dcgB):
        p = ParticleState(positions=pos, supports=sup, masses=masses.detach(), densities=dens, kinds=kinds)
        crkState = CRKState(A=cA, B=cB, gradA=cgA, gradB=cgB)
        crkTangentState = CRKTangentState(A=dcA, B=dcB, gradA=dcgA, gradB=dcgB)
        return computeSPHLaplacianDotGeometryJVP(
            p, domain, KERNEL, SupportScheme.Gather, adjacency,
            queryTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            referenceTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            queryValues=qval, referenceValues=rval,
            crkState=crkState, crkTangentState=crkTangentState,
            gradientMode=GradientScheme.Naive,
        )

    inputs = (pos, sup, dens, tqp, tqs, qval, rval, crk_A, crk_B, crk_gradA, crk_gradB, d_crk_A, d_crk_B, d_crk_gradA, d_crk_gradB)
    ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5, rtol=1e-4)
    print("Direct-tensor CRK gradcheck (crkState/crkTangentState as independent leaves, self-pairs present):", ok)
    return bool(ok)


# ---------------------------------------------------------------------------
# 2. End-to-end gradcheck: crkState/crkTangentState derived from the same
#    leaf positions/supports gradcheck perturbs, via computeCRKFactorsJVP.
# ---------------------------------------------------------------------------

def run_end_to_end_gradcheck() -> bool:
    domain = make_domain(dim=1)
    n = 6
    positions, supports, masses = line_case(n)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    pos = positions.detach().clone().requires_grad_(True)
    sup = supports.detach().clone().requires_grad_(True)
    mass = masses.detach().clone().requires_grad_(True)
    dens = densities.detach().clone().requires_grad_(True)
    tqp = (0.1 * torch.randn(n, 1, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    tqs = (0.01 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    trm = (0.1 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    qval = torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)
    rval = torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def f(pos, sup, mass, dens, tqp, tqs, trm, qval, rval):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        _, _, crkState, crkTangentState = computeCRKFactorsJVP(
            p, domain, KERNEL,
            queryTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            referenceTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            operationMode=OperationDirection.AllToAll, adjacency=adjacency,
        )
        return computeSPHLaplacianDotGeometryJVP(
            p, domain, KERNEL, SupportScheme.Gather, adjacency,
            queryTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            referenceTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=trm, densities=None),
            queryValues=qval, referenceValues=rval,
            crkState=crkState, crkTangentState=crkTangentState,
            gradientMode=GradientScheme.Naive,
        )

    inputs = (pos, sup, mass, dens, tqp, tqs, trm, qval, rval)
    ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5, rtol=1e-4)
    print("End-to-end CRK gradcheck (crkState/crkTangentState derived from positions):", ok)
    return bool(ok)


def main():
    wp.init()
    ok = True

    ok &= run_direct_gradcheck()
    ok &= run_end_to_end_gradcheck()

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- CRK tangent extension for Laplacian(Dot) (warpier_tier2_correction_jvp_plan.md follow-up, 2026-08-21) has a wrong Jacobian.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
