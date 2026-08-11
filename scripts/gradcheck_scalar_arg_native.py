#!/usr/bin/env python3
"""Native torch.autograd.gradcheck for the *differentiable-scalar* kernel
argument pattern (asScalarArg + a wp.array(dtype=scalar_t)-shaped parameter
read as param[0]), added to unblock passing a genuinely differentiable
scalar (e.g. an adaptive dt computed from particle state) into a Warp
kernel through warpWrapper2 -- see warpier_core.md / the AD-bridge design
note this script accompanies.

Before this, warpWrapper2's additionalArguments split
(isinstance(arg, torch.Tensor)) already routed *any* torch.Tensor -- including
a 1-element one -- through StateAwareWarpFunction's tracked-gradient path with
no changes needed. What never existed anywhere in this codebase was: (a) a
kernel parameter declared wp.array(dtype=scalar_t) and read as param[0]
instead of a by-value scalar_t parameter, and (b) a call-site helper that
normalizes a float/int/tensor into the shape that parameter expects. This
script proves both ends of that path together with a small demo kernel
(computeScalarArgDemo_Kernel below) built from the real, already-verified
Interpolate machinery (computeSPHInterpolation_Func_Adjacency) plus a `* dt[0]`
scale term -- not a from-scratch synthetic kernel.

The `dt[0]` read is a *broadcast*: every one of the N particle threads reads
the same array element, unlike existing scalar arrays (supports/masses) where
each thread reads its own index. That means the backward pass needs Warp's
tape to atomically accumulate N adjoint contributions into one grad element --
a pattern with no prior coverage in this codebase. gradcheck against `dt` is
the thing that actually proves this works, rather than assuming it does
because the ordinary per-index scalar-array path already works.

    python scripts/gradcheck_scalar_arg_native.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys
from typing import Any

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, line_case, make_domain, single_particle_case
from warpSPHCore import (
    OperationProperties,
    ParticleState,
    adjacencyData,
    asScalarArg,
    domainData,
    gridData,
    kernelState,
    launch_kernel,
    scalar_t,
    warpWrapper2,
)
from warpSPHCore.coreOperations.wp_interpolate import computeSPHInterpolation_Func_Adjacency
from warpSPHCore.enumTypes import OperationDirection, SupportScheme, WarpOperation
from warpSPHCore import warpOperation


@wp.kernel
def computeScalarArgDemo_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,  # type: ignore
    correctionData: Any,
    kernelProperties: kernelState,
    # end of the canonical structured kernel ABI prefix; the rest is specific to this demo.

    referenceValues: wp.array(dtype=scalar_t),  # type: ignore
    dt: wp.array(dtype=scalar_t),  # type: ignore -- the differentiable-scalar argument: array-shaped, read via [0]

    outputValues: wp.array(dtype=scalar_t),  # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    interpolated = computeSPHInterpolation_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        referenceValues,
        scalar_t(0.0),
    )
    outputValues[i] = interpolated * dt[0]


def scalarArgDemo(queryParticles, referenceParticles, domain, adjacency, referenceValues, dt):
    outputSize = queryParticles.positions.shape[0]
    operationProperties = OperationProperties(
        kernel=KERNEL,
        operation=WarpOperation.Interpolate,
        supportMode=SupportScheme.Gather,
        operationMode=OperationDirection.AllToAll,
    )
    return warpWrapper2(
        launcher=launch_kernel,
        kernel=computeScalarArgDemo_Kernel,
        outputSizes=outputSize,
        outputDtypes=scalar_t,
        defaultStateArguments=(
            queryParticles, operationProperties, domain,
            None, None,
            adjacency,
            referenceParticles,
            None, None, None,
        ),
        additionalArguments=(
            referenceValues,
            asScalarArg(dt, device=queryParticles.positions.device),
        ),
    )


def compute_densities(positions, supports, masses, kinds, domain, adjacency):
    """Realistic density magnitudes via the (separately gradchecked) Density
    op, then detached and re-leafed. Interpolate's vj = mass/density term
    would divide by zero for a dummy-filled density of 0.0 whenever a
    particle is its own neighbor (the self-interaction case) -- see
    gradcheck_interpolate_native.py's compute_densities for the same
    rationale; this script doesn't chain gradients through Density either."""
    p = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=None, kinds=kinds)
    rho = warpOperation(
        p,
        OperationProperties(
            kernel=KERNEL,
            operation=WarpOperation.Density,
            supportMode=SupportScheme.Gather,
            operationMode=OperationDirection.AllToAll,
        ),
        domain,
        adjacency=adjacency,
    )
    return rho.detach().clone()


def run_gradcheck(name: str, positions: torch.Tensor, supports: torch.Tensor, masses: torch.Tensor, dt_shape: str) -> bool:
    domain = make_domain()
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    n = positions.shape[0]
    reference_values = torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)
    dt = (
        torch.tensor(0.7, dtype=DTYPE, device=DEVICE, requires_grad=True)
        if dt_shape == "0-dim"
        else torch.tensor([0.7], dtype=DTYPE, device=DEVICE, requires_grad=True)
    )

    def f(pos, sup, mass, rval, dt_):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=densities, kinds=kinds)
        return scalarArgDemo(p, p, domain, adjacency, rval, dt_)

    print(f"\n=== {name} (dt {dt_shape}): torch.autograd.gradcheck ===")
    inputs = (positions, supports, masses, reference_values, dt)
    try:
        ok = torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a canary script
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def run_plain_float_regression() -> bool:
    """asScalarArg on a plain Python float must still resolve to a
    non-grad-tracked argument -- confirming the differentiable-array path
    doesn't force gradient tracking onto callers who don't ask for it."""
    print("\n=== plain-float dt regression: no grad_fn expected ===")
    positions, supports, masses = line_case(5)
    positions, supports, masses = positions.detach(), supports.detach(), masses.detach()
    domain = make_domain()
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)
    reference_values = torch.randn(5, dtype=DTYPE, device=DEVICE)

    p = ParticleState(positions=positions, supports=supports, masses=masses, densities=densities, kinds=kinds)
    result = scalarArgDemo(p, p, domain, adjacency, reference_values, 0.7)

    ok = not result.requires_grad and result.grad_fn is None
    print("PASSED" if ok else f"FAILED: requires_grad={result.requires_grad}, grad_fn={result.grad_fn}")
    return ok


def main():
    wp.init()
    torch.manual_seed(0)

    cases = [
        ("single particle (h=1)", single_particle_case()),
        ("line of 7 particles [-1, 1]", line_case(7)),
    ]

    ok = True
    for name, particles in cases:
        for dt_shape in ("0-dim", "1-element"):
            ok &= run_gradcheck(name, *particles, dt_shape=dt_shape)

    ok &= run_plain_float_regression()

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- see the differentiable-scalar AD-bridge design note.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
