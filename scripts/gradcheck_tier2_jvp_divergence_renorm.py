#!/usr/bin/env python3
"""Native torch.autograd.gradcheck / JVP-identity checks against Divergence's
Tier-2 (geometry) JVP with gradient-renormalization correction enabled
(`warpier_tier2_correction_jvp_plan.md` phase (f)) --
`computeSPHDivergenceGeometryJVP(..., renormalizationState=..., renormalizationTangentState=...)`.

Mirrors `gradcheck_tier2_jvp_divergence_crk.py`'s two-layer pattern (phase
(e)) -- a much smaller diff than that script since the `dL_i @ G + L_i @ dG`
product rule (`wp_gradientJVP.py`, phase (d)) is already proven and
production-wired; what's new here is just Divergence's own
`dot(dcoeff,G) + dot(coeff,dG)` combination consuming the renorm-corrected
`(G, dG)` pair instead of the plain one. Two checks:

1. **Targeted direct-tensor gradcheck** (`run_direct_gradcheck`):
   `renormalizationState`/`renormalizationTangentState` as independent
   synthetic leaf tensors, isolating `_jvpCommon.launchGeometryJVP`'s
   `flat_tensors`/`build_fn` wiring for the renorm tensors -- already proven
   generically by phase (d)'s own script since the wiring is shared code, but
   re-verified here since Divergence's own `computeSPHDivergenceGeometryJVP`
   now exposes new `renormalizationState`/`renormalizationTangentState`
   parameters for the first time.
2. **End-to-end gradcheck** (`run_end_to_end_gradcheck`): `renormalizationState`/
   `renormalizationTangentState` derived from the same leaf positions/supports
   gradcheck perturbs, via `renorm.py`'s `computeRenormalizationMatricesJVP` --
   reverse-mode-through-the-JVP, matching `gradcheck_renorm_native.py`'s own
   "real force-computation call site" convention. The middle "JVP-vs-jacobian
   identity" layer phase (c)/(d)'s own scripts have is judged redundant here,
   same call phase (e)'s own three CRK-extension scripts made: the combined
   spike (`spike_forward_mode_tier2_renorm_extension.py`) already covers
   exactly that check for all three operators at once.

    python scripts/gradcheck_tier2_jvp_divergence_renorm.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, grid_case_2d
from warpSPHCore import OperationProperties, ParticleState, ParticleTangentState, warpOperation
from warpSPHCore.dataTypes import DomainDescription, RenormalizationState, RenormalizationTangentState
from warpSPHCore.enumTypes import GradientScheme, OperationDirection, SupportScheme, WarpOperation
from warpSPHCore.coreOperations.wp_divergenceJVP import computeSPHDivergenceGeometryJVP
from warpSPHCore.renorm import computeRenormalizationMatricesJVP


def _make_domain(dim: int = 2, margin: float = 10.0) -> DomainDescription:
    return DomainDescription(
        min=torch.tensor([-margin] * dim, dtype=DTYPE, device=DEVICE),
        max=torch.tensor([margin] * dim, dtype=DTYPE, device=DEVICE),
        periodic=torch.tensor([False] * dim, device=DEVICE),
        dim=dim,
    )


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
#    _jvpCommon's flat_tensors wiring for Divergence's own new
#    renormalizationState/renormalizationTangentState parameters.
# ---------------------------------------------------------------------------

def run_direct_gradcheck() -> bool:
    domain = _make_domain(dim=2)
    positions, supports, masses = grid_case_2d(n_per_side=3)
    n = positions.shape[0]
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    pos = positions.detach().clone().requires_grad_(True)
    sup = supports.detach().clone().requires_grad_(True)
    dens = densities.detach().clone().requires_grad_(True)
    tqp = (0.1 * torch.randn(n, 2, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    tqs = (0.01 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    qval = torch.randn(n, 2, dtype=DTYPE, device=DEVICE, requires_grad=True)
    rval = torch.randn(n, 2, dtype=DTYPE, device=DEVICE, requires_grad=True)

    renorm_L = (1.0 + 0.1 * torch.randn(n, 2, 2, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    d_renorm_L = (0.1 * torch.randn(n, 2, 2, dtype=DTYPE, device=DEVICE)).requires_grad_(True)

    def f(pos, sup, dens, tqp, tqs, qval, rval, L, dL):
        p = ParticleState(positions=pos, supports=sup, masses=masses.detach(), densities=dens, kinds=kinds)
        renormState = RenormalizationState(renormalizationMatrices=L)
        renormTangentState = RenormalizationTangentState(renormalizationMatrices=dL)
        return computeSPHDivergenceGeometryJVP(
            p, domain, KERNEL, SupportScheme.Gather, adjacency,
            queryTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            referenceTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            queryValues=qval, referenceValues=rval,
            renormalizationState=renormState, renormalizationTangentState=renormTangentState,
            gradientMode=GradientScheme.Naive,
        )

    inputs = (pos, sup, dens, tqp, tqs, qval, rval, renorm_L, d_renorm_L)
    ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5, rtol=1e-4)
    print("Direct-tensor renorm gradcheck (renormalizationState/renormalizationTangentState as independent leaves):", ok)
    return bool(ok)


# ---------------------------------------------------------------------------
# 2. End-to-end gradcheck: renormalizationState/renormalizationTangentState
#    derived from the same leaf positions/supports gradcheck perturbs, via
#    computeRenormalizationMatricesJVP.
# ---------------------------------------------------------------------------

def run_end_to_end_gradcheck() -> bool:
    domain = _make_domain(dim=2)
    positions, supports, masses = grid_case_2d(n_per_side=3)
    # Non-uniform supports (+-15%), matching every Tier-2.4-touching script's own
    # standing discipline (spike_forward_mode_tier2_renorm.py's `_perturbed_case`):
    # a perfectly uniform grid's covariance matrix can be exactly singular/
    # degenerate for pinv2x2_warpBackend (computeRenormalizationMatricesJVP below
    # calls through it), producing NaN unrelated to this phase's own JVP wiring.
    n = positions.shape[0]
    supports = (supports.detach() * (1.0 + 0.15 * torch.linspace(-1, 1, n, dtype=DTYPE))).requires_grad_(True)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    pos = positions.detach().clone().requires_grad_(True)
    sup = supports.detach().clone().requires_grad_(True)
    mass = masses.detach().clone().requires_grad_(True)
    dens = densities.detach().clone().requires_grad_(True)
    tqp = (0.1 * torch.randn(n, 2, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    tqs = (0.01 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    trm = (0.1 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    trd = (0.1 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    qval = torch.randn(n, 2, dtype=DTYPE, device=DEVICE, requires_grad=True)
    rval = torch.randn(n, 2, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def f(pos, sup, mass, dens, tqp, tqs, trm, trd, qval, rval):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        renormProperties = OperationProperties(kernel=KERNEL, operation=WarpOperation.Divergence, supportMode=SupportScheme.Gather,
                                                operationMode=OperationDirection.AllToAll)
        # Covariance's Vj = mass_j/density_j depends on the REFERENCE-side
        # mass/density tangent too (gradcheck_tier2_jvp_gradient_renorm.py's own
        # docstring finding) -- must be threaded here.
        _, _, renormState, renormTangentState = computeRenormalizationMatricesJVP(
            p, renormProperties, domain,
            queryTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            referenceTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=trm, densities=trd),
            adjacency=adjacency,
        )
        return computeSPHDivergenceGeometryJVP(
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
        print("FAILED -- renormalization tangent extension for Divergence (warpier_tier2_correction_jvp_plan.md phase (f)) has a wrong Jacobian.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
