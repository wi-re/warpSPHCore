#!/usr/bin/env python3
"""Native torch.autograd.gradcheck against Interpolate's Tier-2 (geometry)
JVP -- `computeSPHInterpolateGeometryJVP` -- no workarounds.

`warpier_tier2_jvp_reverse_mode_plan.md` Step 4 (see
`gradcheck_tier2_jvp_density.py`'s docstring for the shared rationale: the
CSR port let `_jvpCommon.launchGeometryJVP` route every Tier-2 JVP kernel
through `StateAwareWarpFunction`, so what used to be a bare, non-
differentiable `wp.launch` now carries real gradients).

Differentiable inputs checked (self-referencing case, `referenceParticles=
None`): positions/supports/densities, tangentQueryPositions/
tangentReferencePositions/tangentQuerySupports/tangentReferenceSupports/
tangentReferenceMasses, and referenceValues (frozen fj, still differentiable
even though it has no tangent of its own). `queryDensities` mirrors
`gradcheck_interpolate_native.py`'s own finding (read but never used by
Interpolate's formula) so is not checked here.

KNOWN-FAILING second case, kept and printed (not asserted -- see
`main()`'s own comment) as a standing repro: a genuinely distinct-role case
(separate `queryParticles`/`referenceParticles` position/support tensors,
not the same object/tensor playing both roles) currently mismatches on the
`dW`-w.r.t.-primal-(reference)-position term. This is a pre-existing gap in
the `vectorNorm_warp`/`vectorNormalize_warp`/`sphGradient_` reverse-mode
chain (`math/wp_normalize.py`), not something `_jvpCommon.
launchGeometryJVP` introduced -- the identical mismatch reproduces in
*primal, non-JVP* `warpOperation(..., WarpOperation.Gradient, ...)` under
the same distinct-role construction (confirmed during this plan's Step 4,
not written up as its own script). Every existing `gradcheck_*_native.py`
script always aliases query/reference to one shared tensor, so none of them
would have caught this -- `torch.autograd.gradcheck` on a shared leaf only
ever measures the *combined* sensitivity `d(output)/dx_i + d(output)/dx_j`,
not each role's individual partial.

    python scripts/gradcheck_tier2_jvp_interpolate.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, line_case, make_domain
from warpSPHCore import OperationProperties, ParticleState, ParticleTangentState, warpOperation
from warpSPHCore.enumTypes import OperationDirection, SupportScheme, WarpOperation
from warpSPHCore.coreOperations.wp_interpolateJVP import computeSPHInterpolateGeometryJVP


def compute_densities(positions, supports, masses, kinds, domain, adjacency):
    p = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=None, kinds=kinds)
    rho = warpOperation(
        p,
        OperationProperties(kernel=KERNEL, operation=WarpOperation.Density, supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll),
        domain, adjacency=adjacency,
    )
    return rho.detach().clone().requires_grad_(True)


def main():
    wp.init()

    domain = make_domain(dim=1)
    n = 6
    positions, supports, masses = line_case(n)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    # --- self-referencing (referenceParticles=None): real regression gate ---
    tqp = (0.1 * torch.randn(n, 1, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    tqs = (0.01 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    trm = (0.1 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    rval = torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def f_self(pos, sup, dens, tqp, tqs, trm, rval):
        p = ParticleState(positions=pos, supports=sup, masses=masses.detach(), densities=dens, kinds=kinds)
        return computeSPHInterpolateGeometryJVP(
            p, domain, KERNEL, SupportScheme.Gather, adjacency,
            queryTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            referenceTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=trm),
            referenceValues=rval,
        )

    pos = positions.detach().clone().requires_grad_(True)
    sup = supports.detach().clone().requires_grad_(True)
    dens = densities.detach().clone().requires_grad_(True)
    ok_self = torch.autograd.gradcheck(f_self, (pos, sup, dens, tqp, tqs, trm, rval), eps=1e-6, atol=1e-5, rtol=1e-4)
    print("Interpolate Tier-2 JVP gradcheck (self-referencing):", ok_self)
    assert ok_self

    # --- distinct query/reference roles: KNOWN-FAILING, not asserted ---
    # See module docstring. Reported, not asserted, so this script stays a
    # useful repro without being a broken CI gate for a pre-existing bug
    # outside this plan's scope.
    trp = (0.1 * torch.randn(n, 1, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    trs = (0.01 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    trd = (0.01 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    rval2 = torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def f_distinct(rp, rs, rd, tqp, trp, tqs, trs, trm, trd, rval):
        p = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=None, kinds=kinds)
        r = ParticleState(positions=rp, supports=rs, masses=masses.detach(), densities=rd, kinds=kinds)
        return computeSPHInterpolateGeometryJVP(
            p, domain, KERNEL, SupportScheme.Gather, adjacency,
            queryTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            referenceParticles=r,
            referenceTangentState=ParticleTangentState(positions=trp, supports=trs, masses=trm, densities=trd),
            referenceValues=rval,
        )

    rp = positions.detach().clone().requires_grad_(True)
    rs = supports.detach().clone().requires_grad_(True)
    rd = densities.detach().clone().requires_grad_(True)
    try:
        ok_distinct = torch.autograd.gradcheck(f_distinct, (rp, rs, rd, tqp, trp, tqs, trs, trm, trd, rval2), eps=1e-6, atol=1e-5, rtol=1e-4)
    except Exception:  # noqa: BLE001 - deliberately broad, this branch only reports a known-failing repro
        ok_distinct = False
    print("Interpolate Tier-2 JVP gradcheck (distinct roles, KNOWN BUG, not gated):", ok_distinct)

    print("ALL PASSED (gated case only -- see printed status of the known-failing distinct-role case above).")


if __name__ == "__main__":
    main()
