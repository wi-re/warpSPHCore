#!/usr/bin/env python3
"""Native torch.autograd.gradcheck / JVP-identity checks against Gradient's
Tier-2 (geometry) JVP with gradient-renormalization correction enabled
(`warpier_tier2_correction_jvp_plan.md` phase (d)) --
`computeSPHGradientGeometryJVP(..., renormalizationState=..., renormalizationTangentState=...)`.

Mirrors `gradcheck_tier2_jvp_gradient_crk.py`'s three-layer pattern
(phase (c)), substituting CRK's `CRKState`/`CRKTangentState`/
`computeCRKFactors`/`computeCRKFactorsJVP` for renormalization's
`RenormalizationState`/`RenormalizationTangentState`/
`computeRenormalizationMatrices`/`computeRenormalizationMatricesJVP`:

1. **Targeted direct-tensor gradcheck** (`run_direct_gradcheck`):
   `renormalizationState`/`renormalizationTangentState` supplied as
   independent synthetic leaf tensors, not derived from positions at all --
   isolates `_jvpCommon.launchGeometryJVP`'s own `flat_tensors`/`build_fn`
   wiring for the two new renormalization/renormalization-tangent tensors
   (`warpier_tier2_correction_jvp_plan.md` phase (a2)'s "hard requirement").

2. **JVP-vs-jacobian identity** (`run_jvp_identity_case`): production's
   assembled JVP (`computeRenormalizationMatricesJVP` chained into
   `computeSPHGradientGeometryJVP`) against
   `torch.autograd.functional.jacobian` on primal
   `warpOperation(Gradient, renormalizationState=...)` -- the same check
   `scripts/spike_forward_mode_tier2_renorm_gradient.py` already validates by
   hand; this exercises it through the real production call graph instead
   (`computeRenormalizationMatricesJVP`'s own `wp_covarianceJVP.py` kernel,
   not a hand-assembled dense one).

3. **End-to-end gradcheck** (`run_end_to_end_gradcheck`):
   `renormalizationState`/`renormalizationTangentState` computed from the
   SAME leaf positions/supports gradcheck perturbs (via
   `computeRenormalizationMatricesJVP`), matching `gradcheck_renorm_native.py`'s
   own "real force-computation call site" convention -- reverse-mode-
   through-the-JVP, this repo's standing rule (never trust a hand Jacobian
   alone).

    python scripts/gradcheck_tier2_jvp_gradient_renorm.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, grid_case_2d, line_case, make_domain
from warpSPHCore import OperationProperties, ParticleState, ParticleTangentState, warpOperation
from warpSPHCore.dataTypes import RenormalizationState, RenormalizationTangentState
from warpSPHCore.enumTypes import GradientScheme, OperationDirection, SupportScheme, WarpOperation
from warpSPHCore.coreOperations.wp_gradientJVP import computeSPHGradientGeometryJVP
from warpSPHCore.renorm import computeRenormalizationMatrices, computeRenormalizationMatricesJVP


def compute_densities(positions, supports, masses, kinds, domain, adjacency):
    p = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=None, kinds=kinds)
    rho = warpOperation(
        p,
        OperationProperties(kernel=KERNEL, operation=WarpOperation.Density, supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll),
        domain,
        adjacency=adjacency,
    )
    return rho.detach().clone().requires_grad_(True)


# ---------------------------------------------------------------------------
# 1. Targeted direct-tensor gradcheck: renormalizationState/
#    renormalizationTangentState as independent leaves, isolating
#    _jvpCommon's flat_tensors wiring.
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

    # dim=1 renormalization matrices are (n,1,1) -- an invertible 1x1 "matrix"
    # is just a nonzero scalar, kept away from zero so pinv-adjacent numerics
    # (not exercised here directly, but the shape convention matches
    # RenormalizationState's real [N,D,D] contract) stay well away from any
    # degenerate case.
    renorm_L = (1.0 + 0.1 * torch.randn(n, 1, 1, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    d_renorm_L = (0.1 * torch.randn(n, 1, 1, dtype=DTYPE, device=DEVICE)).requires_grad_(True)

    def f(pos, sup, dens, tqp, tqs, qval, rval, L, dL):
        p = ParticleState(positions=pos, supports=sup, masses=masses.detach(), densities=dens, kinds=kinds)
        renormState = RenormalizationState(renormalizationMatrices=L)
        renormTangentState = RenormalizationTangentState(renormalizationMatrices=dL)
        return computeSPHGradientGeometryJVP(
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
# 2. JVP-vs-jacobian identity: production's assembled JVP against
#    torch.autograd.functional.jacobian on primal
#    warpOperation(Gradient, renormalizationState=...).
# ---------------------------------------------------------------------------

def _reference_jvp(f, primals, tangents):
    J = torch.autograd.functional.jacobian(f, primals, vectorize=False)
    out = f(*primals)
    acc = torch.zeros(out.numel(), dtype=DTYPE, device=DEVICE)
    for Jk, vk in zip(J, tangents):
        acc = acc + Jk.reshape(out.numel(), -1) @ vk.reshape(-1)
    return acc.reshape(out.shape)


def run_jvp_identity_case(n: int, dim: int, case_fn, mode: SupportScheme, scheme: GradientScheme) -> bool:
    torch.manual_seed(0)
    pos0, sup0, mass0 = case_fn(n)
    pos0, sup0, mass0 = pos0.detach(), sup0.detach(), mass0.detach()
    # Non-uniform supports (+-15%), matching every Tier-2.4-touching script's
    # own standing discipline (spike_forward_mode_tier2_renorm.py's
    # `_perturbed_case`): a perfectly uniform grid's covariance matrix can be
    # exactly singular/degenerate for `pinv2x2_warpBackend`, producing NaN
    # unrelated to this phase's own JVP wiring -- not the discontinuity the
    # plan already flags as out of scope (that's about a tangent crossing the
    # rcond cutoff mid-JVP; this is about the PRIMAL case being degenerate to
    # start with).
    sup0 = sup0 * (1.0 + 0.15 * torch.linspace(-1, 1, sup0.shape[0], dtype=DTYPE))
    domain = make_domain(dim=dim)
    adjacency, kinds = build_adjacency(pos0, sup0, mass0, domain, mode=SupportScheme.KernelMeanSymmetric)
    density0 = compute_densities(pos0, sup0, mass0, kinds, domain, adjacency).detach()
    n_actual = pos0.shape[0]
    fv_q = torch.randn(n_actual, dtype=DTYPE, device=DEVICE)
    fv_r = torch.randn(n_actual, dtype=DTYPE, device=DEVICE)

    def f(pos, sup, mass, dens):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        renormProperties = OperationProperties(kernel=KERNEL, operation=WarpOperation.Gradient, supportMode=mode,
                                                operationMode=OperationDirection.AllToAll)
        _, _, renormState = computeRenormalizationMatrices(p, renormProperties, domain, adjacency=adjacency)
        return warpOperation(
            p,
            OperationProperties(kernel=KERNEL, operation=WarpOperation.Gradient, supportMode=mode,
                                 operationMode=OperationDirection.AllToAll, gradientMode=scheme),
            domain, queryValues=fv_q, referenceValues=fv_r, adjacency=adjacency, renormalizationState=renormState,
        )

    pos = pos0.clone().requires_grad_(True)
    sup = sup0.clone().requires_grad_(True)
    mass = mass0.clone().requires_grad_(True)
    dens = density0.clone().requires_grad_(True)
    dpos, dsup = torch.randn_like(pos0), torch.randn_like(sup0) * 0.1
    dmass, ddens = torch.randn_like(mass0), torch.randn_like(density0) * 0.1

    reference = _reference_jvp(f, (pos, sup, mass, dens), (dpos, dsup, dmass, ddens))

    # Self-referencing (queryParticles is referenceParticles): every
    # differentiated primal tensor needs the SAME tangent fed to both the
    # query and reference role, matching gradcheck_tier2_jvp_gradient_crk.py's
    # own convention.
    p_now = ParticleState(positions=pos0, supports=sup0, masses=mass0, densities=density0, kinds=kinds)
    renormProperties = OperationProperties(kernel=KERNEL, operation=WarpOperation.Gradient, supportMode=mode,
                                            operationMode=OperationDirection.AllToAll)
    # Covariance's Vj = mass_j/density_j depends on the REFERENCE-side
    # mass/density tangent too (unlike CRK's Stage 1/2, which have no
    # mass/density term at all) -- must be threaded here or this assembled
    # dL silently omits the dmass/ddens contribution the reference Jacobian
    # (which differentiates warpOperation(Gradient, ...) w.r.t. mass/dens
    # too, since they flow into computeRenormalizationMatrices internally)
    # does include.
    _, _, renormState, renormTangentState = computeRenormalizationMatricesJVP(
        p_now, renormProperties, domain,
        queryTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=None),
        referenceTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=dmass, densities=ddens),
        adjacency=adjacency,
    )
    assembled = computeSPHGradientGeometryJVP(
        p_now, domain, KERNEL, mode, adjacency,
        queryTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=None, densities=ddens),
        referenceTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=dmass, densities=ddens),
        queryValues=fv_q, referenceValues=fv_r,
        renormalizationState=renormState, renormalizationTangentState=renormTangentState,
        gradientMode=scheme,
    )

    rel_err = float((assembled - reference).detach().abs().max()) / max(float(reference.detach().abs().max()), 1e-300)
    ok = rel_err <= 1e-8
    print(f"  [{'PASS' if ok else 'FAIL'}] JVP-vs-jacobian dim={dim} n={n_actual} {mode.name:18s} {scheme.name:10s} rel_err={rel_err:.3e}")
    return ok


# ---------------------------------------------------------------------------
# 3. End-to-end gradcheck: renormalizationState/renormalizationTangentState
#    derived from the SAME leaf positions/supports gradcheck perturbs, via
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
        renormProperties = OperationProperties(kernel=KERNEL, operation=WarpOperation.Gradient, supportMode=SupportScheme.Gather,
                                                operationMode=OperationDirection.AllToAll)
        _, _, renormState, renormTangentState = computeRenormalizationMatricesJVP(
            p, renormProperties, domain,
            queryTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            referenceTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            adjacency=adjacency,
        )
        return computeSPHGradientGeometryJVP(
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

    print("\nJVP-vs-jacobian identity, 1D line of 7 / 2D 3x3 grid, every GradientScheme:")
    for scheme in GradientScheme:
        ok &= run_jvp_identity_case(7, 1, line_case, SupportScheme.KernelMeanSymmetric, scheme)
    for scheme in GradientScheme:
        ok &= run_jvp_identity_case(3, 2, grid_case_2d, SupportScheme.KernelMeanSymmetric, scheme)

    print("\nJVP-vs-jacobian identity, other SupportSchemes (2D 3x3 grid, Symmetric):")
    for mode in (SupportScheme.Gather, SupportScheme.Scatter, SupportScheme.MeanSymmetric, SupportScheme.SuperSymmetric):
        ok &= run_jvp_identity_case(3, 2, grid_case_2d, mode, GradientScheme.Symmetric)

    ok &= run_end_to_end_gradcheck()

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- renormalization tangent wiring for Gradient (warpier_tier2_correction_jvp_plan.md phase (d)) has a wrong Jacobian.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
