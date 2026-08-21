#!/usr/bin/env python3
"""Combined spike for CRK tangent EXTENSION to Divergence/Curl/
Laplacian(Brookshaw) (`warpier_tier2_correction_jvp_plan.md` phase (e)) --
mirrors how Tier 2.2 itself was one spike covering three operators.

Unlike phase (c)'s own spike (`spike_forward_mode_tier2_crk.py`, which had to
derive and hand-assemble all four CRK JVP stages from scratch before any
production code existed), phase (e) reuses those stages verbatim -- Stages
1-4 (`crk.computeKernelGradientCRKJVP`) are operator-agnostic, already
production-wired and gradchecked for Gradient (phase (c)). What's new here is
purely each operator's own *combination* formula consuming the corrected
`(G, dG)` pair -- `wp_divergenceJVP.py`'s `dot(dcoeff,G) + dot(coeff,dG)`,
`wp_curlJVP.py`'s 2D cross-product expansion, and
`wp_laplacianJVP.py`'s Brookshaw `P = dot(G,n_ij)/D_ij` weighting -- so this
spike validates the already-wired production functions directly against
`torch.autograd.functional.jacobian` on primal
`warpOperation(<op>, crkState=...)`, rather than hand-deriving a formula
first.

**Found and fixed a genuine, pre-existing bug while writing this spike, not
introduced by this phase**: `wp_laplacian.py`'s (primal) and
`wp_laplacianJVP.py`'s (Tier-2 JVP) Brookshaw scheme both divide
`dot(kernelGradient, n_ij)` by `D_ij` a second time (on top of `n_ij`'s own
`x_ij/D_ij` division). At an exact self-pair (`r_ij == 0`, always present in
any `referenceParticles=None` self-referencing adjacency), `n_ij` is exactly
`0` by construction, forcing the *forward* contribution to exactly `0`
regardless of `kernelGradient` -- true with or without CRK. Without CRK,
`kernelGradient` is itself exactly `0` at `r_ij == 0` (a symmetric kernel's
gradient vanishes at its own peak, with a correct adjoint there too, thanks
to `sphGradient_`'s existing custom `@wp.func_grad`,
`project_tier2_jvp_distinct_role_adjoint_bug`'s fix). With CRK enabled,
though, `correctGradientCRK`'s own value at `x_ij == 0` is generically
**nonzero** (its `Ai*W_ij*Bi`/`W_ij*gradAi` terms don't vanish at the kernel's
own peak the way the plain gradient does) -- and Warp's reverse-mode through
"a nonzero vector dotted against an exactly-zero `n_ij`" at that exact point
produces a wrong adjoint (confirmed via a from-scratch minimal repro with no
dependency on this codebase's kernel structure, and via
`torch.autograd.gradcheck` failing directly on both
`warpOperation(Laplacian, Brookshaw, crkState=...)` and
`computeSPHLaplacianBrookshawGeometryJVP(..., crkState=..., crkTangentState=...)`
before the fix). **Fixed** in both kernels by guarding the Brookshaw
contribution with an explicit `if r_ij > 0:` -- the true contribution at
`r_ij == 0` is always exactly `0` (CRK or not), so skipping it outright
changes no forward value while sidestepping the bad adjoint, mirroring
`wp_densityHVP.py`'s own explicit self-pair `pairMask` precedent (a
different bug, same "don't rely on 0 falling out of the math naturally"
discipline). Divergence/Curl needed no such guard: their own combination
formulas (`dot(coeff,G)`, the 2D cross product) have no *second* division by
a quantity that itself vanishes at `r_ij == 0` the way Brookshaw's `n_ij`
does, so a CRK-nonzero `G` at the self-pair combines with a well-defined
(possibly nonzero) adjoint with no cancellation involved -- confirmed
empirically (both pass cleanly, CRK or not, no fix needed).

    python scripts/spike_forward_mode_tier2_crk_extension.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, grid_case_2d, make_domain
from warpSPHCore import OperationProperties, ParticleState, ParticleTangentState, warpOperation, warpOperationJVP
from warpSPHCore.crk import computeCRKFactors, computeCRKFactorsJVP
from warpSPHCore.enumTypes import GradientScheme, LaplacianScheme, OperationDirection, SupportScheme, WarpOperation


def compute_densities(positions, supports, masses, kinds, domain, adjacency):
    p = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=None, kinds=kinds)
    rho = warpOperation(
        p,
        OperationProperties(kernel=KERNEL, operation=WarpOperation.Density, supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll),
        domain, adjacency=adjacency,
    )
    return rho.detach().clone().requires_grad_(True)


def _reference_jvp(f, primals, tangents):
    J = torch.autograd.functional.jacobian(f, primals, vectorize=False)
    out = f(*primals)
    acc = torch.zeros(out.numel(), dtype=DTYPE, device=DEVICE)
    for Jk, vk in zip(J, tangents):
        acc = acc + Jk.reshape(out.numel(), -1) @ vk.reshape(-1)
    return acc.reshape(out.shape)


def run_case(op: WarpOperation, label: str, domain, positions, supports, masses, kinds, adjacency, densities, extra_props=None) -> bool:
    extra_props = extra_props or {}
    n = positions.shape[0]
    scalarField = op is WarpOperation.Laplacian
    torch.manual_seed(hash((op, label)) % (2 ** 31))
    qv = torch.randn(n, dtype=DTYPE, device=DEVICE) if scalarField else torch.randn(n, 2, dtype=DTYPE, device=DEVICE)
    rv = torch.randn(n, dtype=DTYPE, device=DEVICE) if scalarField else torch.randn(n, 2, dtype=DTYPE, device=DEVICE)

    def f(pos, sup, mass, dens):
        pp = ParticleState(positions=pos, supports=sup, masses=mass, densities=dens, kinds=kinds)
        _, _, crkState = computeCRKFactors(pp, domain, KERNEL, operationMode=OperationDirection.AllToAll, adjacency=adjacency)
        return warpOperation(
            pp, OperationProperties(kernel=KERNEL, operation=op, supportMode=SupportScheme.Gather,
                                     operationMode=OperationDirection.AllToAll, gradientMode=GradientScheme.Naive, **extra_props),
            domain, queryValues=qv, referenceValues=rv, adjacency=adjacency, crkState=crkState,
        )

    pos0 = positions.detach().clone().requires_grad_(True)
    sup0 = supports.detach().clone().requires_grad_(True)
    mass0 = masses.detach().clone().requires_grad_(True)
    dens0 = densities.detach().clone().requires_grad_(True)
    dpos, dsup = torch.randn_like(pos0), torch.randn_like(sup0) * 0.1
    dmass, ddens = torch.randn_like(mass0), torch.randn_like(dens0) * 0.1

    reference = _reference_jvp(f, (pos0, sup0, mass0, dens0), (dpos, dsup, dmass, ddens))

    p_now = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=densities.detach(), kinds=kinds)
    _, _, crkState_now, crkTangentState_now = computeCRKFactorsJVP(
        p_now, domain, KERNEL,
        queryTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=None),
        referenceTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=None),
        operationMode=OperationDirection.AllToAll, adjacency=adjacency,
    )
    props = OperationProperties(kernel=KERNEL, operation=op, supportMode=SupportScheme.Gather,
                                 operationMode=OperationDirection.AllToAll, gradientMode=GradientScheme.Naive, **extra_props)
    assembled = warpOperationJVP(
        p_now, props, domain, adjacency=adjacency,
        queryTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=None, densities=ddens),
        referenceTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=dmass, densities=ddens),
        queryValues=qv, referenceValues=rv,
        crkState=crkState_now, crkTangentState=crkTangentState_now,
    )
    rel_err = float((assembled - reference).detach().abs().max()) / max(float(reference.detach().abs().max()), 1e-300)
    ok = rel_err <= 1e-8
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:32s} rel_err={rel_err:.3e}")
    return ok


def main():
    wp.init()
    ok = True

    domain = make_domain(dim=2)
    positions, supports, masses = grid_case_2d(n_per_side=3)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    print("CRK tangent extension (phase (e)), production warpOperationJVP vs. jacobian on primal warpOperation(..., crkState=...):")
    ok &= run_case(WarpOperation.Divergence, "Divergence", domain, positions, supports, masses, kinds, adjacency, densities)
    ok &= run_case(WarpOperation.Curl, "Curl", domain, positions, supports, masses, kinds, adjacency, densities)
    ok &= run_case(WarpOperation.Laplacian, "Laplacian(Brookshaw)", domain, positions, supports, masses, kinds, adjacency, densities,
                    extra_props={"laplacianMode": LaplacianScheme.Brookshaw})

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- CRK tangent extension to Divergence/Curl/Laplacian(Brookshaw) (warpier_tier2_correction_jvp_plan.md phase (e)) has a wrong Jacobian.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
