#!/usr/bin/env python3
"""Combined spike for renormalization tangent EXTENSION to Divergence/Curl/
Laplacian(Brookshaw) (`warpier_tier2_correction_jvp_plan.md` phase (f)) --
mirrors how phase (e)'s own CRK extension was one spike covering the same
three operators (`spike_forward_mode_tier2_crk_extension.py`).

Phase (d) already did the one genuinely new piece for this correction: the
`dKernelGradient_final = dL_i @ kernelGradient_corrected + L_i @
dKernelGradient_corrected` product rule (`spike_forward_mode_tier2_renorm_gradient.py`).
That rule is operator-agnostic -- it produces a corrected `(kernelGradient,
dKernelGradient)` pair, nothing Gradient-specific -- so this phase reuses it
verbatim (landed in `wp_laplacianJVP.py`'s shared `_laplacianGeometryChainJVP`
and inline in `wp_divergenceJVP.py`/`wp_curlJVP.py`, right after the CRK swap,
matching every primal kernel's own fixed CRK-then-renorm composition order).
What's new here is purely each operator's own *combination* formula consuming
the renorm-corrected `(G, dG)` pair -- `wp_divergenceJVP.py`'s
`dot(dcoeff,G) + dot(coeff,dG)`, `wp_curlJVP.py`'s 2D cross-product expansion,
and `wp_laplacianJVP.py`'s Brookshaw `P = dot(G,n_ij)/D_ij` weighting -- so
this spike validates the already-wired production functions directly against
`torch.autograd.functional.jacobian` on primal
`warpOperation(<op>, renormalizationState=...)`, rather than hand-deriving a
formula first.

Renorm alone (CRK off), matching phase (d)'s own scope decision -- CRK+renorm
simultaneous stays an explicit fast follow-up (`operations.py`'s own
`NotImplementedError` for that combination), not attempted here.

Unlike phase (e)'s own CRK extension, no self-pair adjoint bug turned up here:
Brookshaw's second division by `D_ij` (`n_ij = x_ij/D_ij`, then `P =
dot(G,n_ij)/D_ij` again) is already guarded by the `if r_ij > 0:` phase (e)
added to both `wp_laplacian.py` and `wp_laplacianJVP.py` -- renormalization's
own `kernelGradient = matmul(L, kernelGradient)` does not reintroduce a
division, so no new discontinuity at `r_ij == 0` is possible; confirmed
empirically below (all three operators pass cleanly).

    python scripts/spike_forward_mode_tier2_renorm_extension.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, grid_case_2d, make_domain
from warpSPHCore import OperationProperties, ParticleState, ParticleTangentState, warpOperation, warpOperationJVP
from warpSPHCore.enumTypes import GradientScheme, LaplacianScheme, OperationDirection, SupportScheme, WarpOperation
from warpSPHCore.renorm import computeRenormalizationMatrices, computeRenormalizationMatricesJVP


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
        renormProps = OperationProperties(kernel=KERNEL, operation=op, supportMode=SupportScheme.Gather,
                                           operationMode=OperationDirection.AllToAll, **extra_props)
        _, _, renormState = computeRenormalizationMatrices(pp, renormProps, domain, adjacency=adjacency)
        return warpOperation(
            pp, OperationProperties(kernel=KERNEL, operation=op, supportMode=SupportScheme.Gather,
                                     operationMode=OperationDirection.AllToAll, gradientMode=GradientScheme.Naive, **extra_props),
            domain, queryValues=qv, referenceValues=rv, adjacency=adjacency, renormalizationState=renormState,
        )

    pos0 = positions.detach().clone().requires_grad_(True)
    sup0 = supports.detach().clone().requires_grad_(True)
    mass0 = masses.detach().clone().requires_grad_(True)
    dens0 = densities.detach().clone().requires_grad_(True)
    dpos, dsup = torch.randn_like(pos0), torch.randn_like(sup0) * 0.1
    dmass, ddens = torch.randn_like(mass0), torch.randn_like(dens0) * 0.1

    reference = _reference_jvp(f, (pos0, sup0, mass0, dens0), (dpos, dsup, dmass, ddens))

    p_now = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=densities.detach(), kinds=kinds)
    renormProps_now = OperationProperties(kernel=KERNEL, operation=op, supportMode=SupportScheme.Gather,
                                           operationMode=OperationDirection.AllToAll, **extra_props)
    # Covariance's Vj = mass_j/density_j depends on the REFERENCE-side mass/density
    # tangent too (unlike CRK's Stages 1-2), same finding
    # gradcheck_tier2_jvp_gradient_renorm.py's own docstring already documents --
    # must be threaded here or renormTangentState_now silently omits the
    # dmass/ddens contribution the reference jacobian (which differentiates
    # warpOperation(..., renormalizationState=...) w.r.t. mass/dens too, since they
    # flow into computeRenormalizationMatrices internally) does include.
    _, _, renormState_now, renormTangentState_now = computeRenormalizationMatricesJVP(
        p_now, renormProps_now, domain,
        queryTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=None),
        referenceTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=dmass, densities=ddens),
        adjacency=adjacency,
    )
    props = OperationProperties(kernel=KERNEL, operation=op, supportMode=SupportScheme.Gather,
                                 operationMode=OperationDirection.AllToAll, gradientMode=GradientScheme.Naive, **extra_props)
    assembled = warpOperationJVP(
        p_now, props, domain, adjacency=adjacency,
        queryTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=None, densities=ddens),
        referenceTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=dmass, densities=ddens),
        queryValues=qv, referenceValues=rv,
        renormalizationState=renormState_now, renormalizationTangentState=renormTangentState_now,
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
    # Non-uniform supports (+-15%), matching every Tier-2.4-touching script's own
    # standing discipline -- a perfectly uniform grid's covariance matrix can be
    # exactly singular/degenerate for pinv2x2_warpBackend, producing NaN unrelated
    # to this phase's own JVP wiring.
    supports = supports * (1.0 + 0.15 * torch.linspace(-1, 1, supports.shape[0], dtype=DTYPE))
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    print("Renormalization tangent extension (phase (f)), production warpOperationJVP vs. jacobian on primal warpOperation(..., renormalizationState=...):")
    ok &= run_case(WarpOperation.Divergence, "Divergence", domain, positions, supports, masses, kinds, adjacency, densities)
    ok &= run_case(WarpOperation.Curl, "Curl", domain, positions, supports, masses, kinds, adjacency, densities)
    ok &= run_case(WarpOperation.Laplacian, "Laplacian(Brookshaw)", domain, positions, supports, masses, kinds, adjacency, densities,
                    extra_props={"laplacianMode": LaplacianScheme.Brookshaw})

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- renormalization tangent extension to Divergence/Curl/Laplacian(Brookshaw) (warpier_tier2_correction_jvp_plan.md phase (f)) has a wrong Jacobian.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
