#!/usr/bin/env python3
"""Tier-2 JVP apparent-volume spike (`warpier_tier2_correction_jvp_plan.md`
phase b, `useVolume`).

**Why this spike is short.** Unlike Tier 2.2's kernelGradient JVP or the
CRK/renorm tiers, apparent-volume support needs no new derivation: `useVolume`
swaps a directly-supplied `referenceVolumes[j]` tensor in place of the
`mass_j/density_j` apparent-volume formula everywhere it appears (confirmed
against every primal kernel's own `apparentVolume = ... if not
correctionData.useVolume else correctionData.referenceVolumes[j]` branch --
`wp_gradient.py`/`wp_divergence.py`/`wp_curl.py`/`wp_laplacian.py`/
`wp_interpolate.py`), so its tangent is a pass-through of the caller's own
`tangentReferenceVolumes`, not a re-derivation. There is accordingly no
independent "assembled" formula to hand-derive here (contrast every other
`spike_forward_mode_tier2_*.py`, which build a second, independent
implementation of the JVP math to compare against production) -- this script
instead checks the already-wired production JVP entry points
(`computeSPH<Op>GeometryJVP(..., referenceVolumes=..., tangentReferenceVolumes=...)`)
directly against `torch.autograd.functional.jacobian` on the PRIMAL
`warpOperation(..., referenceVolumes=...)`, contracting with a tangent that
includes `dReferenceVolumes` alongside the usual position/support/mass/
density tangents. This is exactly the same "reference" side every other
Tier-2 spike already uses; only the "assembled" side collapses to "call the
production JVP function" since there is no new math underneath it to
independently re-implement.

`GradientScheme.Symmetric` is deliberately excluded from every case here:
its coefficient has no apparent-volume term at all (`gradientWeightsJVP`'s
own docstring), so it would exercise nothing.

    python scripts/spike_forward_mode_tier2_volume.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, grid_case_2d, line_case, make_domain
from warpSPHCore import OperationProperties, ParticleState, ParticleTangentState, warpOperation
from warpSPHCore.enumTypes import GradientScheme, LaplacianScheme, OperationDirection, SupportScheme, WarpOperation
from warpSPHCore.coreOperations.wp_gradientJVP import computeSPHGradientGeometryJVP
from warpSPHCore.coreOperations.wp_divergenceJVP import computeSPHDivergenceGeometryJVP
from warpSPHCore.coreOperations.wp_curlJVP import computeSPHCurlGeometryJVP
from warpSPHCore.coreOperations.wp_interpolateJVP import computeSPHInterpolateGeometryJVP
from warpSPHCore.coreOperations.wp_laplacianJVP import (
    computeSPHLaplacianBrookshawGeometryJVP,
    computeSPHLaplacianNaiveGeometryJVP,
    computeSPHLaplacianDotGeometryJVP,
    computeSPHLaplacianDefaultGeometryJVP,
)


def _reference_jvp(f, primals, tangents):
    """Same pattern every Tier-2 spike uses: reverse-mode Jacobian of the
    production forward operator, contracted with the tangent direction."""
    J = torch.autograd.functional.jacobian(f, primals, vectorize=False)
    out = f(*primals).detach()
    n_out = out.numel()
    acc = torch.zeros(n_out, dtype=DTYPE, device=DEVICE)
    for Jk, vk in zip(J, tangents):
        acc = acc + Jk.reshape(n_out, -1) @ vk.reshape(-1)
    return acc.reshape(out.shape)


def check(name, assembled, reference):
    assembled_flat, reference_flat = assembled.reshape(-1), reference.reshape(-1)
    assert assembled_flat.numel() == reference_flat.numel(), (
        f"{name}: shape mismatch assembled={tuple(assembled.shape)} reference={tuple(reference.shape)}"
    )
    scale = max(float(reference_flat.abs().max()), 1e-300)
    err = float((assembled_flat - reference_flat).abs().max()) / scale
    ok = err <= 1e-9
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:60s} rel_err={err:.3e}")
    return ok


def _compute_densities(pos, sup, mass, kinds, domain, adjacency):
    p = ParticleState(positions=pos.detach(), supports=sup.detach(), masses=mass.detach(), densities=None, kinds=kinds)
    rho = warpOperation(
        p, OperationProperties(kernel=KERNEL, operation=WarpOperation.Density,
                                supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll),
        domain, adjacency=adjacency,
    )
    return rho.detach().clone()


def _perturbed_case(n, dim, seed):
    torch.manual_seed(seed)
    pos0, sup0, mass0 = line_case(n) if dim == 1 else grid_case_2d(n)
    pos0, sup0, mass0 = pos0.detach(), sup0.detach(), mass0.detach()
    sup0 = sup0 * (1.0 + 0.15 * torch.linspace(-1, 1, sup0.shape[0], dtype=DTYPE))
    domain = make_domain(dim=dim)
    adjacency, kinds = build_adjacency(pos0, sup0, mass0, domain, mode=SupportScheme.Gather)
    density0 = _compute_densities(pos0, sup0, mass0, kinds, domain, adjacency)
    # Sized by the actual particle count, not `n` -- for dim==2, `n` is grid_case_2d's
    # n_per_side, not the total particle count (n_per_side**2), and this array must
    # match `nRef` or correctionData.referenceVolumes[j] reads out of bounds.
    volume0 = (0.5 + torch.rand(pos0.shape[0], dtype=DTYPE)).abs()
    return pos0, sup0, mass0, density0, volume0, domain, adjacency, kinds


def run_operator_case(op, jvp_fn, n, dim, seed, field_shape, op_kwargs=None, jvp_kwargs=None, has_query_values=True):
    pos0, sup0, mass0, density0, volume0, domain, adjacency, kinds = _perturbed_case(n, dim, seed)
    op_kwargs = op_kwargs or {}
    jvp_kwargs = jvp_kwargs or {}

    def f(pos, sup, mass, density, rv, qval, rval):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=density, kinds=kinds)
        return warpOperation(
            p, OperationProperties(kernel=KERNEL, operation=op, supportMode=SupportScheme.Gather,
                                    operationMode=OperationDirection.AllToAll, **op_kwargs),
            domain, queryValues=qval, referenceValues=rval, referenceVolumes=rv, adjacency=adjacency,
        )

    for t in (pos0, sup0, mass0, density0, volume0):
        t.requires_grad_(True)
    dpos, dsup, dmass, ddensity = (torch.randn_like(pos0), torch.randn_like(sup0) * 0.1,
                                    torch.randn_like(mass0), torch.randn_like(density0) * 0.1)
    drv = torch.randn_like(volume0) * 0.1
    fv_q = torch.randn(*field_shape, dtype=DTYPE)
    fv_r = torch.randn(*field_shape, dtype=DTYPE)

    reference = _reference_jvp(
        f, (pos0, sup0, mass0, density0, volume0, fv_q, fv_r),
        (dpos, dsup, dmass, ddensity, drv, torch.zeros_like(fv_q), torch.zeros_like(fv_r)),
    )

    p = ParticleState(positions=pos0.detach(), supports=sup0.detach(), masses=mass0.detach(), densities=density0.detach(), kinds=kinds)
    if has_query_values:
        jvp_kwargs = dict(jvp_kwargs, queryValues=fv_q)
    assembled = jvp_fn(
        p, domain, KERNEL, SupportScheme.Gather, adjacency,
        queryTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=None, densities=ddensity),
        referenceTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=dmass, densities=ddensity),
        referenceValues=fv_r,
        referenceVolumes=volume0.detach(), tangentReferenceVolumes=drv,
        **jvp_kwargs,
    )
    return assembled, reference


def main():
    wp.init()
    print(__doc__.split("\n\n")[0])
    print()

    ok = True
    n, dim = 6, 1

    d, r = run_operator_case(WarpOperation.Gradient, computeSPHGradientGeometryJVP, n, dim, 0, (n,),
                              op_kwargs=dict(gradientMode=GradientScheme.Naive),
                              jvp_kwargs=dict(gradientMode=GradientScheme.Naive))
    ok &= check("Gradient JVP (useVolume=True, Naive)", d, r)

    d, r = run_operator_case(WarpOperation.Divergence, computeSPHDivergenceGeometryJVP, n, dim, 0, (n, dim),
                              op_kwargs=dict(gradientMode=GradientScheme.Naive, divergenceDotMode=False),
                              jvp_kwargs=dict(gradientMode=GradientScheme.Naive))
    ok &= check("Divergence JVP (useVolume=True, Naive)", d, r)

    # n_per_side for grid_case_2d -- keeps the dense jacobian small, matching every
    # other Tier-2 spike's own 3x3-grid convention for 2D cases.
    n2 = 3
    d2, r2 = run_operator_case(WarpOperation.Curl, computeSPHCurlGeometryJVP, n2, 2, 0, (n2 * n2, 2),
                                op_kwargs=dict(gradientMode=GradientScheme.Naive),
                                jvp_kwargs=dict(gradientMode=GradientScheme.Naive))
    ok &= check("Curl JVP (useVolume=True, Naive, 2D)", d2, r2)

    dI, rI = run_operator_case(WarpOperation.Interpolate, computeSPHInterpolateGeometryJVP, n, dim, 0, (n,),
                                op_kwargs={}, jvp_kwargs={}, has_query_values=False)
    ok &= check("Interpolate JVP (useVolume=True)", dI, rI)

    for name, laplacianMode, jvp_fn in (
        ("Brookshaw", LaplacianScheme.Brookshaw, computeSPHLaplacianBrookshawGeometryJVP),
        ("Naive", LaplacianScheme.Naive, computeSPHLaplacianNaiveGeometryJVP),
        ("Dot", LaplacianScheme.Dot, computeSPHLaplacianDotGeometryJVP),
        ("Default", LaplacianScheme.Default, computeSPHLaplacianDefaultGeometryJVP),
    ):
        dL, rL = run_operator_case(WarpOperation.Laplacian, jvp_fn, n, dim, 0, (n,),
                                    op_kwargs=dict(gradientMode=GradientScheme.Naive, laplacianMode=laplacianMode),
                                    jvp_kwargs=dict(gradientMode=GradientScheme.Naive))
        ok &= check(f"Laplacian/{name} JVP (useVolume=True, Naive)", dL, rL)

    print()
    if ok:
        print("ALL PASSED -- useVolume=True's tangent (a direct pass-through of the")
        print("  caller's own referenceVolumes tangent, no new math) matches the production")
        print("  operators' own reverse-mode derivative, for every value-having operator.")
    else:
        print("FAILED -- see the individual case(s) above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
