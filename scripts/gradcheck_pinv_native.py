#!/usr/bin/env python3
"""Native torch.autograd.gradcheck against pinv1x1 and pinv2x2_warpBackend
(src/sphWarpCore/pinv/wp_pinv1x1.py, wp_pinv2x2.py) directly, as pure
matrix -> (inverse, eigenvalues) functions -- independent of the
renormalization pipeline that's their only production caller.

Closes the last "smaller open item" from warpier_core.md's "What's Next"
section: pinv2x2_warpBackend used to be a raw wp.launch on cast tensors,
with no torch.autograd.Function wrapping it at all -- silently
non-differentiable, not merely uncovered by a test. It has been ported to
the same warpWrapper/launch_kernel pattern pinv1x1 already used (see
wp_pinv2x2.py) so that it has a backward pass to gradcheck in the first
place. pinv1x1 was already on that path but had never been gradchecked in
isolation before this script -- only indirectly, via gradcheck_renorm_native
.py's dim=1 cases, which exercise it but don't gradcheck it as its own
function -- so it's included here too for direct, isolated coverage.

Cases:
  * pinv1x1: three independent 1x1 matrices, values well away from the
    "small(m) -> 0" threshold branch (`m[0,0] > 1e-10`) so the check
    exercises the differentiable branch, not the constant-zero one.
  * pinv2x2: two well-conditioned, deliberately non-symmetric 2x2 matrices
    (num_nbrs >= 4, the eigendecomposition branch -- see wp_pinv2x2.py's
    "if num_nbrs[i] < 4" low-neighbor-count fallback, which returns a
    constant identity and would gradcheck trivially/uninterestingly).
    Non-symmetric on purpose: pinv2x2_warp symmetrizes b = 0.5*(C01+C10)
    internally, so this also checks that gradient contributions to C01 and
    C10 are each threaded back out correctly rather than only checking the
    symmetric-input case.

    python scripts/gradcheck_pinv_native.py
"""

from __future__ import annotations

import os

os.environ.setdefault("SPHWARPCORE_PRECISION", "float64")

import sys

import torch
import warp as wp

from sphWarpCore.pinv import pinv1x1, pinv2x2_warpBackend

DEVICE = torch.device("cpu")
DTYPE = torch.float64


def run_pinv1x1() -> bool:
    C = torch.tensor(
        [[[2.0]], [[0.5]], [[5.0]]], dtype=DTYPE, device=DEVICE, requires_grad=True
    )

    def f(c):
        inv, ev = pinv1x1(c)
        return inv, ev

    print("\n=== pinv1x1: torch.autograd.gradcheck ===")
    try:
        ok = torch.autograd.gradcheck(f, (C,), eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a canary script
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def run_pinv2x2() -> bool:
    C = torch.tensor(
        [
            [[3.0, 0.4], [0.6, 1.0]],
            [[2.0, -0.3], [-0.2, 4.0]],
        ],
        dtype=DTYPE, device=DEVICE, requires_grad=True,
    )
    num_nbrs = torch.tensor([10, 10], dtype=torch.int32, device=DEVICE)

    def f(c):
        inv, ev = pinv2x2_warpBackend(c, num_nbrs)
        return inv, ev

    print("\n=== pinv2x2_warpBackend: torch.autograd.gradcheck ===")
    try:
        ok = torch.autograd.gradcheck(f, (C,), eps=1e-6, atol=1e-5)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a canary script
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def run_pinv2x2_low_neighbor_branch() -> bool:
    """num_nbrs < 4 takes the constant-identity fallback -- gradient w.r.t. C
    should be exactly zero, not just small. Sanity check that the branch
    doesn't leak a spurious nonzero adjoint."""
    C = torch.tensor(
        [[[3.0, 0.4], [0.6, 1.0]]], dtype=DTYPE, device=DEVICE, requires_grad=True
    )
    num_nbrs = torch.tensor([2], dtype=torch.int32, device=DEVICE)

    inv, ev = pinv2x2_warpBackend(C, num_nbrs)
    (inv.sum() + ev.sum()).backward()

    print("\n=== pinv2x2_warpBackend: low-neighbor-count fallback (num_nbrs < 4) ===")
    is_identity = torch.allclose(inv[0], torch.eye(2, dtype=DTYPE), atol=1e-12)
    is_zero_grad = torch.allclose(C.grad, torch.zeros_like(C.grad), atol=1e-12)
    print(f"inv == identity: {is_identity}, grad(C) == 0: {is_zero_grad}")
    ok = is_identity and is_zero_grad
    print("PASSED" if ok else "FAILED")
    return ok


def main():
    wp.init()
    torch.manual_seed(0)

    ok = True
    ok &= run_pinv1x1()
    ok &= run_pinv2x2()
    ok &= run_pinv2x2_low_neighbor_branch()

    print()
    if ok:
        print("ALL PASSED.")
    else:
        print("FAILED -- see warpier_core.md's pinv2x2 gradcheck note.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
