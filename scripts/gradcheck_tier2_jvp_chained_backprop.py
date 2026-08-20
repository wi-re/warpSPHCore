#!/usr/bin/env python3
"""Item 3 Step 5 (`warpier_tier2_jvp_reverse_mode_plan.md` /
`warpier_tier2_jvp_remaining_work_plan.md`): the actual embedded-in-backprop
demonstration. `gradcheck_tier2_jvp_*.py`'s existing scripts (Steps 0-4)
already prove each `computeSPH<Op>GeometryJVP` is reverse-mode
differentiable w.r.t. its own inputs *in isolation*. What was never
demonstrated: the plan's actual motivating scenario -- a `warpOperationJVP`
call embedded *inside* a larger differentiable computation (a Newton-style
implicit solve, ultimately), where gradient has to flow through an upstream
operation, into `warpOperationJVP`, out through a reduction, and all the way
back to the original leaf inputs via one ordinary `loss.backward()` call.

Chain built here: `positions/supports/masses` -> `warpOperation(Density)` ->
`densities` -> re-packed into a `ParticleState` -> `warpOperationJVP(Gradient,
...)` (consuming that same `densities` tensor, still attached to the graph)
-> `(output ** 2).sum()` -> `loss.backward()`. This is two SPH operations
chained through ordinary torch autograd with a `warpOperationJVP` call in the
middle -- gradient reaching `positions`/`supports`/`masses` has to flow
through *both* Density's own backward and Gradient's Tier-2 JVP backward,
which per-operator scripts never exercise together.

Verified with `torch.autograd.gradcheck` (finite differences -- this repo's
own established ground truth, not a hand Jacobian; see
`gradcheck_tier2_jvp_gradient.py`'s docstring / `docs/lessons_learned.md`)
against the *whole chain* as one function, not just the JVP call alone. A
green result here proves gradients correctly compose end-to-end through
Density -> Gradient's Tier-2 JVP -> reduction, not just that each piece is
correct standing alone. An explicit `loss.backward()` + populated/finite
`.grad` check follows, matching the plan's literal ask.

Self-referencing construction (`referenceParticles=None`), same convention
every sibling `gradcheck_tier2_jvp_*.py` script uses. The distinct-role
construction is no longer required here either way -- the self-pair Hessian
bug that used to make it necessary to avoid distinct-role cases is fixed
(see `warpier_tier2_jvp_remaining_work_plan.md`'s "Related, tracked
elsewhere").

    python scripts/gradcheck_tier2_jvp_chained_backprop.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, line_case, make_domain
from warpSPHCore import OperationProperties, ParticleState, ParticleTangentState, warpOperation, warpOperationJVP
from warpSPHCore.enumTypes import GradientScheme, OperationDirection, SupportScheme, WarpOperation


def main():
    wp.init()

    domain = make_domain(dim=1)
    n = 6
    positions, supports, masses = line_case(n)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)

    densityProps = OperationProperties(
        kernel=KERNEL, operation=WarpOperation.Density,
        supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll,
    )
    gradientProps = OperationProperties(
        kernel=KERNEL, operation=WarpOperation.Gradient,
        supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll,
        gradientMode=GradientScheme.Symmetric,
    )

    tqp = (0.1 * torch.randn(n, 1, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    tqs = (0.01 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    trm = (0.1 * torch.randn(n, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    qval = torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)
    rval = torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def chain(pos, sup, mass, tqp, tqs, trm, qval, rval):
        # Op 1: Density -- production primal operator, already reverse-mode
        # differentiable on its own (gradcheck_density_native.py).
        densityParticles = ParticleState(positions=pos, supports=sup, masses=mass, densities=None, kinds=kinds)
        densities = warpOperation(densityParticles, densityProps, domain, adjacency=adjacency)

        # Op 2: Gradient's Tier-2 (geometry) JVP, embedded via the public
        # warpOperationJVP entry point -- consumes `densities` computed by
        # Op 1 above, still attached to the graph, so gradient must flow
        # positions -> Density -> densities -> Gradient JVP -> loss, not
        # just directly positions -> Gradient JVP the way every
        # gradcheck_tier2_jvp_*.py script tests it today.
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=densities, kinds=kinds)
        gradJVP = warpOperationJVP(
            p, gradientProps, domain, adjacency=adjacency,
            queryTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=None),
            referenceTangentState=ParticleTangentState(positions=tqp, supports=tqs, masses=trm),
            queryValues=qval, referenceValues=rval,
        )
        return (gradJVP ** 2).sum()

    pos = positions.detach().clone().requires_grad_(True)
    sup = supports.detach().clone().requires_grad_(True)
    mass = masses.detach().clone().requires_grad_(True)

    ok = torch.autograd.gradcheck(chain, (pos, sup, mass, tqp, tqs, trm, qval, rval), eps=1e-6, atol=1e-5, rtol=1e-4)
    print("Chained Density -> Gradient Tier-2 JVP embedded-in-backprop gradcheck:", ok)
    assert ok

    # Explicit loss.backward() demonstration, matching the plan's literal
    # ask: gradcheck above already proves correctness; this proves the
    # ergonomics (ordinary loss.backward(), no special-casing needed to
    # embed a warpOperationJVP call inside a larger computation) work too.
    loss = chain(pos, sup, mass, tqp, tqs, trm, qval, rval)
    loss.backward()
    for name, t in [
        ("positions", pos), ("supports", sup), ("masses", mass),
        ("tangentQueryPositions", tqp), ("tangentQuerySupports", tqs),
        ("tangentReferenceMasses", trm), ("queryValues", qval), ("referenceValues", rval),
    ]:
        assert t.grad is not None, f"{name}.grad is None -- backprop did not reach this leaf"
        assert torch.isfinite(t.grad).all(), f"{name}.grad has non-finite entries"

    print("ALL PASSED.")


if __name__ == "__main__":
    main()
