#!/usr/bin/env python3
"""Native torch.autograd.gradcheck against the Density operator -- no workarounds.

Companion to scripts/gradcheck_density.py. This script is the "normal"
gradcheck call any user would reach for first -- ParticleState +
warpOperation wired directly into torch.autograd.gradcheck, no clone-per-call
trick, no manual Jacobian loop.

It was originally written to document a warp-lang Tape-reentrancy bug (a
second backward() against a retained tape -- exactly what gradcheck's
default multi-output Jacobian construction and its own reentrancy self-check
both do -- silently returned a wrong, typically-zero gradient). That has
since been fixed at the source in sphWarpCore's AD bridge
(WarpFunctionWrapper.backward / StateAwareWarpFunction.backward in
wp_autograd.py: clone the gradient tensor read out of Warp, and explicitly
zero the tape afterward, so a later call reusing the same underlying memory
never reads stale or aliased state). See warpier_core.md's "Backward-Mode
(Reverse AD) Findings" for the full diagnosis.

This script now PASSES and is kept as a regression guard for that fix --
re-run it after any change to wp_autograd.py or a warp-lang version bump to
confirm reentrancy hasn't broken again.

    python scripts/gradcheck_density_native.py
"""

from __future__ import annotations

import os

os.environ.setdefault("SPHWARPCORE_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import KERNEL, build_adjacency, line_case, make_domain, single_particle_case
from sphWarpCore import OperationProperties, ParticleState, warpOperation
from sphWarpCore.enumTypes import OperationDirection, SupportScheme, WarpOperation


def run_gradcheck(name: str, positions: torch.Tensor, supports: torch.Tensor, masses: torch.Tensor) -> bool:
    domain = make_domain()
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)

    def f(pos, sup, mass):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=None, kinds=kinds)
        return warpOperation(
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

    print(f"\n=== {name}: torch.autograd.gradcheck (native, no workarounds) ===")
    try:
        ok = torch.autograd.gradcheck(f, (positions, supports, masses), eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a canary script
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def main():
    wp.init()

    ok = True
    ok &= run_gradcheck("single particle (h=1)", *single_particle_case())
    ok &= run_gradcheck("line of 7 particles [-1, 1]", *line_case(7))

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- this is a regression: see warpier_core.md's AD-bridge reentrancy fix")
        print("(WarpFunctionWrapper.backward / StateAwareWarpFunction.backward in wp_autograd.py).")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
