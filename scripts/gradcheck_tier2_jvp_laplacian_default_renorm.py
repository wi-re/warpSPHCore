#!/usr/bin/env python3
"""Native torch.autograd.gradcheck / JVP-identity checks against Laplacian's
Default-scheme Tier-2 (geometry) JVP with gradient-renormalization correction
enabled (`warpier_tier2_correction_jvp_plan.md` follow-up, 2026-08-21,
extending phase (f) past its own original Brookshaw-only scope) --
`computeSPHLaplacianDefaultGeometryJVP(..., renormalizationState=..., renormalizationTangentState=...)`.
Naive stays permanently out of renorm scope (its estimator never consumes
the renorm-corrected `kernelGradient` at all, so there is no formula to
correct), matching `operations.py`'s own restriction (same boundary as CRK).

See `gradcheck_tier2_jvp_divergence_renorm.py`'s docstring for the shared
rationale -- Default's own `n_ij2 = n_ij/D2_ij` weighting is the only reused piece; the `dL_i @ G + L_i @ dG` product rule (`wp_gradientJVP.py`,
phase (d)) is reused verbatim through `_laplacianGeometryChainJVP`, applied
after the CRK swap (a no-op here, CRK is off). `line_case` (1D) always
includes at least one exact self-pair per query particle -- also exercises
the `if r_ij > 0:` guard this follow-up's own spike added to Default's
contribution (`wp_laplacianJVP.py`), confirming renormalization's own
`matmul(L, kernelGradient)` step does not reintroduce the self-pair adjoint
bug that CRK's nonzero-at-self-pair value did (see
`spike_forward_mode_tier2_laplacian_dot_default_extension.py`'s module
docstring for that bug's full root-cause writeup).

    python scripts/gradcheck_tier2_jvp_laplacian_default_renorm.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, line_case, make_domain
from warpSPHCore import OperationProperties, ParticleState, ParticleTangentState, warpOperation
from warpSPHCore.dataTypes import RenormalizationState, RenormalizationTangentState
from warpSPHCore.enumTypes import GradientScheme, OperationDirection, SupportScheme, WarpOperation
from warpSPHCore.coreOperations.wp_laplacianJVP import computeSPHLaplacianDefaultGeometryJVP
from warpSPHCore.renorm import computeRenormalizationMatricesJVP


def compute_densities(positions, supports, masses, kinds, domain, adjacency):
    p = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=None, kinds=kinds)
    rho = warpOperation(
        p,
        OperationProperties(kernel=KERNEL, operation=WarpOperation.Density, supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll),
        domain, adjacency=adjacency,
    )
    return rho.detach().clone().requires_grad_(True)


# ---------------------------------------------------------------------------
# 1. Targeted direct-tensor gradcheck: renormalizationState/
#    renormalizationTangentState as independent leaves, isolating
#    _jvpCommon's flat_tensors wiring for Laplacian(Brookshaw)'s own new
#    renormalizationState/renormalizationTangentState parameters -- also the
#    self-pair-adjoint-bug regression guard, same convention
#    gradcheck_tier2_jvp_laplacian_brookshaw_crk.py's own direct gradcheck uses.
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

    # dim=1 renormalization matrices are (n,1,1) -- an invertible 1x1 "matrix" is
    # just a nonzero scalar, matching gradcheck_tier2_jvp_gradient_renorm.py's own
    # dim=1 convention.
    renorm_L = (1.0 + 0.1 * torch.randn(n, 1, 1, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    d_renorm_L = (0.1 * torch.randn(n, 1, 1, dtype=DTYPE, device=DEVICE)).requires_grad_(True)

    def f(pos, sup, dens, tqp, tqs, qval, rval, L, dL):
        p = ParticleState(positions=pos, supports=sup, masses=masses.detach(), densities=dens, kinds=kinds)
        renormState = RenormalizationState(renormalizationMatrices=L)
        renormTangentState = RenormalizationTangentState(renormalizationMatrices=dL)
        return computeSPHLaplacianDefaultGeometryJVP(
            p, domain, KERNEL, SupportScheme.Gather, adjacency,
            queryTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            referenceTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            queryValues=qval, referenceValues=rval,
            renormalizationState=renormState, renormalizationTangentState=renormTangentState,
            gradientMode=GradientScheme.Naive,
        )

    inputs = (pos, sup, dens, tqp, tqs, qval, rval, renorm_L, d_renorm_L)
    ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5, rtol=1e-4)
    print("Direct-tensor renorm gradcheck (renormalizationState/renormalizationTangentState as independent leaves, self-pairs present):", ok)
    return bool(ok)


# ---------------------------------------------------------------------------
# 2. End-to-end gradcheck: renormalizationState/renormalizationTangentState
#    derived from the same leaf positions/supports gradcheck perturbs, via
#    computeRenormalizationMatricesJVP.
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
    trd = (0.1 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    qval = torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)
    rval = torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def f(pos, sup, mass, dens, tqp, tqs, trm, trd, qval, rval):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        renormProperties = OperationProperties(kernel=KERNEL, operation=WarpOperation.Laplacian, supportMode=SupportScheme.Gather,
                                                operationMode=OperationDirection.AllToAll)
        _, _, renormState, renormTangentState = computeRenormalizationMatricesJVP(
            p, renormProperties, domain,
            queryTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            referenceTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=trm, densities=trd),
            adjacency=adjacency,
        )
        return computeSPHLaplacianDefaultGeometryJVP(
            p, domain, KERNEL, SupportScheme.Gather, adjacency,
            queryTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            referenceTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=trm, densities=trd),
            queryValues=qval, referenceValues=rval,
            renormalizationState=renormState, renormalizationTangentState=renormTangentState,
            gradientMode=GradientScheme.Naive,
        )

    inputs = (pos, sup, mass, dens, tqp, tqs, trm, trd, qval, rval)
    ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5, rtol=1e-4)
    print("End-to-end renorm gradcheck (renormalizationState/renormalizationTangentState derived from positions):", ok)
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
        print("FAILED -- renormalization tangent extension for Laplacian(Default) (warpier_tier2_correction_jvp_plan.md follow-up, 2026-08-21) has a wrong Jacobian.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
