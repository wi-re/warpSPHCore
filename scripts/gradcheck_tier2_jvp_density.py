#!/usr/bin/env python3
"""Native torch.autograd.gradcheck against Density's Tier-2 (geometry) JVP --
`computeSPHDensityGeometryJVP` -- no workarounds.

`warpier_tier2_jvp_reverse_mode_plan.md` Step 4: once
`warpier_tier2_jvp_csr_backend_plan.md`'s CSR port let `_jvpCommon.
launchGeometryJVP` route every Tier-2 JVP kernel through
`StateAwareWarpFunction` instead of a bare `wp.launch`, `positions`/
`supports`/tangent counterparts/`tangentReferenceMasses` all carry real
gradients back through `computeSPHDensityGeometryJVP`'s output -- this
script is the gradcheck-based confirmation (replacing
`scripts/diagnostic_tier2_jvp_reverse_mode.py`'s `allow_unused=True`
reachability probe with an actual numerical-Jacobian PASS), following
`gradcheck_density_native.py`'s own "wired straight into
torch.autograd.gradcheck, no manual Jacobian" pattern.

Both cases below check `computeSPHDensityGeometryJVP` with `referenceParticles`
defaulting to `None` (so `queryParticles is referenceParticles`, one shared
`positions` tensor feeding both roles) -- the self-referencing pattern
`StateAwareWarpFunction.backward`'s `id(wa)`-keyed double-count guard exists
for (warpier_tier2_jvp_reverse_mode_plan.md Lookout 2), differing only in
whether the *tangent* position/support arrays are independent per role
(case 1) or also shared (case 2, `tangentQueryPositions is
tangentReferencePositions`).

CAVEAT, found while validating this plan: with `positions` shared across
roles, `torch.autograd.gradcheck` only ever measures the *combined*
sensitivity `d(output)/dx_i + d(output)/dx_j` at each shared index, not each
role's individual partial. A genuinely distinct-role case (separate
`queryParticles`/`referenceParticles` position tensors) exposes a real,
substantial mismatch between the analytical and numerical Jacobians in the
`dW`-w.r.t.-primal-position term -- and the identical bug reproduces in
*primal, non-JVP* `warpOperation(..., WarpOperation.Gradient, ...)` under
the same distinct-role construction, so it is a pre-existing gap in the
`vectorNorm_warp`/`vectorNormalize_warp`/`sphGradient_` reverse-mode chain
(`math/wp_normalize.py`), not something this plan's `_jvpCommon.
launchGeometryJVP` bridge introduced -- and not something every existing
`gradcheck_*_native.py` script would have caught, since each of those also
only ever aliases query/reference positions to one shared tensor. Not
reproduced here (this script stays a green regression guard for the
self-referencing path, which is unaffected); see
`warpier_tier2_jvp_reverse_mode_plan.md`'s follow-up notes for the
distinct-role repro.

    python scripts/gradcheck_tier2_jvp_density.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, line_case, make_domain
from warpSPHCore.dataTypes import ParticleState, ParticleTangentState
from warpSPHCore.enumTypes import SupportScheme
from warpSPHCore.coreOperations.wp_densityJVP import computeSPHDensityGeometryJVP


def main():
    wp.init()

    domain = make_domain(dim=1)
    n = 6
    positions, supports, masses = line_case(n)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)

    # --- shared primal positions, independent tangent positions/supports per role ---
    tqp = (0.1 * torch.randn(n, 1, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    trp = (0.1 * torch.randn(n, 1, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    tqs = (0.01 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    trs = (0.01 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    trm = (0.1 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)

    def f(pos, tqp, trp, tqs, trs, trm):
        p = ParticleState(positions=pos, supports=supports, masses=masses, densities=None, kinds=kinds)
        return computeSPHDensityGeometryJVP(
            p, domain, KERNEL, SupportScheme.Gather, adjacency,
            queryTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            referenceTangentState=ParticleTangentState(positions=trp, supports=trs, masses=trm),
        )

    pos_leaf = positions.detach().clone().requires_grad_(True)
    ok = torch.autograd.gradcheck(f, (pos_leaf, tqp, trp, tqs, trs, trm), eps=1e-6, atol=1e-5, rtol=1e-4)
    print("Density Tier-2 JVP gradcheck (shared position, independent tangents):", ok)
    assert ok

    # --- shared primal positions, also-shared tangent positions/supports ---
    tqp2 = (0.1 * torch.randn(n, 1, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    tqs2 = (0.01 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    trm2 = (0.1 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)

    def f_self(pos, tqp, tqs, trm):
        p = ParticleState(positions=pos, supports=supports, masses=masses, densities=None, kinds=kinds)
        return computeSPHDensityGeometryJVP(
            p, domain, KERNEL, SupportScheme.Gather, adjacency,
            queryTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            referenceTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=trm),
        )

    pos_leaf2 = positions.detach().clone().requires_grad_(True)
    ok_self = torch.autograd.gradcheck(f_self, (pos_leaf2, tqp2, tqs2, trm2), eps=1e-6, atol=1e-5, rtol=1e-4)
    print("Density Tier-2 JVP gradcheck (shared position, shared tangents):", ok_self)
    assert ok_self

    print("ALL PASSED.")


if __name__ == "__main__":
    main()
