#!/usr/bin/env python3
"""Run gradcheck twice in the same process -- the Step D/E reentrancy gate
warpier_fields.md's Section 5 (tests 1-2) asks for and that no other test
provides.

`tests/operations/test_gradcheck_scripts.py` auto-discovers and runs every
scripts/gradcheck_*.py, but each one in its own **subprocess** (see that
test's docstring), so cross-call state within a single process -- exactly
what a cached wp.array wrapper and its `.grad` buffer are -- is never
exercised there. A single `torch.autograd.gradcheck(...)` call already
repeats forward/backward many times over for its own numerical Jacobian,
but always as part of one gradcheck invocation against one fixed set of
leaf tensors; it doesn't cover two independent invocations sharing
warpSPHCore's process-level caches (the null-field registry, and as of
Step D/E, the view-reuse cache on both the no-grad and grad paths).

Two scenarios:

  1. Same leaf tensors, `torch.autograd.gradcheck` run twice back-to-back.
     The exact case Section 3.3 worries about: a cached wp.array wrapper
     (and its `.grad` buffer, which `wp.Tape.get_adjoint` returns directly,
     not a fresh per-tape copy) reused across two independent forward and
     backward passes on the same tensor objects, with nothing but
     zero-on-acquire and `tape.zero()` between them.
  2. Every gradcheck_*_native.py script's `main()`, run twice back-to-back
     in this one process (fresh leaf tensors each call, since each script
     builds its own case data inside `main()`) -- broader coverage across
     every operator/correction path, exercising the caches' handling of a
     *new* tensor object arriving right after a now-dead one from the first
     pass, not just same-object reuse.

Run as its own process (float64 precision, like every other gradcheck_*.py):

    python scripts/gradcheck_twice_in_process.py
"""

from __future__ import annotations

import importlib
import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import KERNEL, build_adjacency, line_case, make_domain
from warpSPHCore import OperationProperties, ParticleState, warpOperation
from warpSPHCore.enumTypes import OperationDirection, SupportScheme, WarpOperation

_NATIVE_SCRIPTS = [
    "gradcheck_density_native",
    "gradcheck_interpolate_native",
    "gradcheck_gradient_native",
    "gradcheck_divergence_native",
    "gradcheck_curl_native",
    "gradcheck_laplacian_native",
    "gradcheck_crk_native",
    "gradcheck_crk_correction_native",
    "gradcheck_renorm_native",
    "gradcheck_covariance_native",
    "gradcheck_pinv_native",
    "gradcheck_scalar_arg_native",
]


def _density_gradcheck(positions, supports, masses, kinds, domain, adjacency) -> bool:
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

    return bool(torch.autograd.gradcheck(f, (positions, supports, masses), eps=1e-6, atol=1e-5))


def scenario_same_tensors_twice() -> bool:
    print("=== Scenario 1: same leaf tensors, gradcheck run twice in-process ===")
    positions, supports, masses = line_case(7)
    domain = make_domain()
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)

    ok1 = _density_gradcheck(positions, supports, masses, kinds, domain, adjacency)
    print(f"  pass 1: {'PASSED' if ok1 else 'FAILED'}")
    ok2 = _density_gradcheck(positions, supports, masses, kinds, domain, adjacency)
    print(f"  pass 2 (same tensor objects): {'PASSED' if ok2 else 'FAILED'}")
    return ok1 and ok2


def scenario_full_suite_twice() -> bool:
    print("\n=== Scenario 2: every gradcheck_*_native.py, main() run twice in-process ===")
    modules = [importlib.import_module(name) for name in _NATIVE_SCRIPTS]

    ok = True
    for pass_num in (1, 2):
        print(f"\n--- pass {pass_num} ---")
        for name, mod in zip(_NATIVE_SCRIPTS, modules):
            try:
                mod.main()
                code = 0
            except SystemExit as exc:
                code = exc.code if exc.code is not None else 0
            passed = code == 0
            print(f"  {name}: {'PASSED' if passed else f'FAILED (exit {code})'}")
            ok &= passed
    return ok


def main():
    wp.init()

    ok = True
    ok &= scenario_same_tensors_twice()
    ok &= scenario_full_suite_twice()

    print()
    if ok:
        print("ALL PASSED (twice-in-process).")
    else:
        print("FAILED -- Step D/E's view-reuse gate did not hold up to repeated in-process calls.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
