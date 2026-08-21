#!/usr/bin/env python3
"""Native torch.autograd.gradcheck / JVP-identity checks against Gradient's
Tier-2 (geometry) JVP with CRK correction enabled
(`warpier_tier2_correction_jvp_plan.md` phase (c)) --
`computeSPHGradientGeometryJVP(..., crkState=..., crkTangentState=...)`.

Three checks, in increasing order of what they cover:

1. **Targeted direct-tensor gradcheck** (`run_direct_gradcheck`): `crkState`/
   `crkTangentState` are supplied as independent synthetic leaf tensors, not
   derived from positions at all. This isolates `_jvpCommon.launchGeometryJVP`'s
   own `flat_tensors`/`build_fn` wiring for the eight new CRK/CRK-tangent
   tensors -- per `warpier_tier2_correction_jvp_plan.md` phase (a2)'s "hard
   requirement", a tensor that bypasses that wiring would silently see a zero
   gradient here even though the forward value is correct (the same failure
   class `docs/lessons_learned.md` documents for tensors that bypass the
   autograd bridge).

2. **JVP-vs-jacobian identity** (`run_jvp_identity_case`): production's
   assembled JVP (`computeCRKFactorsJVP` chained into
   `computeSPHGradientGeometryJVP`) against
   `torch.autograd.functional.jacobian` contracted with the same tangent, on
   primal `warpOperation(Gradient, crkState=...)` -- the same pattern every
   prior Tier-2 JVP script uses, and the same check
   `scripts/spike_forward_mode_tier2_crk.py`'s Stage 4 already validates by
   hand; this exercises it through the real production call graph instead
   (`computeCRKFactorsJVP`'s own `torch.autograd.functional.jvp` hop through
   `crk_terms.py`, not a hand-assembled one).

3. **End-to-end gradcheck** (`run_end_to_end_gradcheck`): `crkState`/
   `crkTangentState` computed from the SAME leaf positions/supports gradcheck
   perturbs (via `computeCRKFactorsJVP`), matching
   `gradcheck_crk_correction_native.py`'s own "real force-computation call
   site" convention -- reverse-mode-through-the-JVP, this repo's standing
   rule (never trust a hand Jacobian alone).

    python scripts/gradcheck_tier2_jvp_gradient_crk.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, grid_case_2d, line_case, make_domain
from warpSPHCore import OperationProperties, ParticleState, ParticleTangentState, warpOperation
from warpSPHCore.crk import computeCRKFactors, computeCRKFactorsJVP
from warpSPHCore.dataTypes import CRKState, CRKTangentState
from warpSPHCore.enumTypes import GradientScheme, OperationDirection, SupportScheme, WarpOperation
from warpSPHCore.coreOperations.wp_gradientJVP import computeSPHGradientGeometryJVP


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
# 1. Targeted direct-tensor gradcheck: crkState/crkTangentState as
#    independent leaves, isolating _jvpCommon's flat_tensors wiring.
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
        return computeSPHGradientGeometryJVP(
            p, domain, KERNEL, SupportScheme.Gather, adjacency,
            queryTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            referenceTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            queryValues=qval, referenceValues=rval,
            crkState=crkState, crkTangentState=crkTangentState,
            gradientMode=GradientScheme.Naive,
        )

    inputs = (pos, sup, dens, tqp, tqs, qval, rval, crk_A, crk_B, crk_gradA, crk_gradB, d_crk_A, d_crk_B, d_crk_gradA, d_crk_gradB)
    ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5, rtol=1e-4)
    print("Direct-tensor CRK gradcheck (crkState/crkTangentState as independent leaves):", ok)
    return bool(ok)


# ---------------------------------------------------------------------------
# 2. JVP-vs-jacobian identity: production's assembled JVP against
#    torch.autograd.functional.jacobian on primal warpOperation(Gradient,
#    crkState=...).
# ---------------------------------------------------------------------------

def _reference_jvp(f, primals, tangents):
    J = torch.autograd.functional.jacobian(f, primals, vectorize=False)
    out = f(*primals)
    acc = torch.zeros(out.numel(), dtype=DTYPE, device=DEVICE)
    for Jk, vk in zip(J, tangents):
        acc = acc + Jk.reshape(out.numel(), -1) @ vk.reshape(-1)
    return acc.reshape(out.shape)


def run_jvp_identity_case(n: int, dim: int, case_fn, scheme: GradientScheme) -> bool:
    torch.manual_seed(0)
    pos0, sup0, mass0 = case_fn(n)
    pos0, sup0, mass0 = pos0.detach(), sup0.detach(), mass0.detach()
    domain = make_domain(dim=dim)
    adjacency, kinds = build_adjacency(pos0, sup0, mass0, domain, mode=SupportScheme.KernelMeanSymmetric)
    density0 = compute_densities(pos0, sup0, mass0, kinds, domain, adjacency).detach()
    n_actual = pos0.shape[0]
    fv_q = torch.randn(n_actual, dtype=DTYPE, device=DEVICE)
    fv_r = torch.randn(n_actual, dtype=DTYPE, device=DEVICE)

    def f(pos, sup, mass, dens):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        _, _, crkState = computeCRKFactors(
            queryParticles=p, domain=domain, kernel=KERNEL,
            operationMode=OperationDirection.AllToAll, adjacency=adjacency,
        )
        return warpOperation(
            p,
            OperationProperties(kernel=KERNEL, operation=WarpOperation.Gradient, supportMode=SupportScheme.Gather,
                                 operationMode=OperationDirection.AllToAll, gradientMode=scheme),
            domain, queryValues=fv_q, referenceValues=fv_r, adjacency=adjacency, crkState=crkState,
        )

    pos = pos0.clone().requires_grad_(True)
    sup = sup0.clone().requires_grad_(True)
    mass = mass0.clone().requires_grad_(True)
    dens = density0.clone().requires_grad_(True)
    dpos, dsup = torch.randn_like(pos0), torch.randn_like(sup0) * 0.1
    dmass, ddens = torch.randn_like(mass0), torch.randn_like(density0) * 0.1

    reference = _reference_jvp(f, (pos, sup, mass, dens), (dpos, dsup, dmass, ddens))

    # Self-referencing (queryParticles is referenceParticles): every
    # differentiated primal tensor -- including densities, which Symmetric's
    # own dDensityI term reads -- needs the SAME tangent fed to both the
    # query and reference role, matching every other self-referencing
    # gradcheck script's own convention (e.g. gradcheck_tier2_jvp_gradient.py).
    p_now = ParticleState(positions=pos0, supports=sup0, masses=mass0, densities=density0, kinds=kinds)
    _, _, crkState, crkTangentState = computeCRKFactorsJVP(
        p_now, domain, KERNEL,
        queryTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=None),
        referenceTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=None),
        operationMode=OperationDirection.AllToAll, adjacency=adjacency,
    )
    assembled = computeSPHGradientGeometryJVP(
        p_now, domain, KERNEL, SupportScheme.Gather, adjacency,
        queryTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=None, densities=ddens),
        referenceTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=dmass, densities=ddens),
        queryValues=fv_q, referenceValues=fv_r,
        crkState=crkState, crkTangentState=crkTangentState,
        gradientMode=scheme,
    )

    rel_err = float((assembled - reference).detach().abs().max()) / max(float(reference.detach().abs().max()), 1e-300)
    ok = rel_err <= 1e-8
    print(f"  [{'PASS' if ok else 'FAIL'}] JVP-vs-jacobian dim={dim} n={n_actual} {scheme.name:10s} rel_err={rel_err:.3e}")
    return ok


# ---------------------------------------------------------------------------
# 3. End-to-end gradcheck: crkState/crkTangentState derived from the SAME
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
    trd = (0.1 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    qval = torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)
    rval = torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def f(pos, sup, mass, dens, tqp, tqs, trm, trd, qval, rval):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        _, _, crkState, crkTangentState = computeCRKFactorsJVP(
            p, domain, KERNEL,
            queryTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            referenceTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            operationMode=OperationDirection.AllToAll, adjacency=adjacency,
        )
        return computeSPHGradientGeometryJVP(
            p, domain, KERNEL, SupportScheme.Gather, adjacency,
            queryTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            referenceTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=trm, densities=trd),
            queryValues=qval, referenceValues=rval,
            crkState=crkState, crkTangentState=crkTangentState,
            gradientMode=GradientScheme.Naive,
        )

    inputs = (pos, sup, mass, dens, tqp, tqs, trm, trd, qval, rval)
    ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5, rtol=1e-4)
    print("End-to-end CRK gradcheck (crkState/crkTangentState derived from positions):", ok)
    return bool(ok)


def main():
    wp.init()
    ok = True

    ok &= run_direct_gradcheck()

    print("\nJVP-vs-jacobian identity, 1D line of 7 / 2D 3x3 grid, every GradientScheme:")
    for scheme in GradientScheme:
        ok &= run_jvp_identity_case(7, 1, line_case, scheme)
    for scheme in GradientScheme:
        ok &= run_jvp_identity_case(3, 2, grid_case_2d, scheme)

    ok &= run_end_to_end_gradcheck()

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- CRK tangent promotion for Gradient (warpier_tier2_correction_jvp_plan.md phase (c)) has a wrong Jacobian.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
