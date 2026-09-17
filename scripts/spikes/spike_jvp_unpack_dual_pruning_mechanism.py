#!/usr/bin/env python3
"""JFNK-jvp overhead spike, part 3: the mechanism `warpier_jvp_dual_argument_pruning_plan.md`'s
Phase 1 depends on -- can a callee several frames away from wherever
`fwAD.make_dual()` was invoked cheaply and correctly tell, per tensor, per
call, whether that specific tensor object is currently dual?

This is the empirical basis for "check the actual tensor object at runtime,
never classify a field as constant/differentiable from a static schema" (see
`spike_jvp_wave_case_overhead_profile.py`/`spike_jvp_dual_wrapping_alternatives.py`
for why the overhead exists and why static classification was rejected).
`torch.autograd.forward_ad.unpack_dual(t)` returns `(primal, tangent)` with
`tangent is None` iff `t` is not currently a dual tensor -- this script checks
that (a) it works correctly from inside `_launch`-depth call nesting, with no
knowledge of which `dual_level` (if any) created the tensor, correctly
reverting to "not dual" once that level has exited even if the tensor is
still held, and (b) it is cheap: no GPU sync, pure metadata inspection, far
below even the already-fast batched `hasLiveTangent`/`_liveTangentMask`
sync cost (`stateAwareWarpFunction.py`).

    python scripts/spikes/spike_jvp_unpack_dual_pruning_mechanism.py
"""

from __future__ import annotations

import time

import warp as wp
import torch
import torch.autograd.forward_ad as fwAD

wp.init()

from warpSPHCore.autograd.stateAwareWarpFunction import hasLiveTangent  # noqa: E402

DEVICE = "cuda:0"
N = 16384


def probe(t) -> bool:
    """What a callee several frames deep (e.g. inside _launch) would do --
    no knowledge of which level created the dual, if any."""
    primal, tangent = fwAD.unpack_dual(t)
    return tangent is not None


def level1(t):
    return level2(t)


def level2(t):
    return level3(t)


def level3(t):
    return probe(t)


def main() -> None:
    plain = torch.randn(N, device=DEVICE)
    dual_primal = torch.randn(N, device=DEVICE)
    dual_tangent = torch.randn(N, device=DEVICE)

    print("=== Correctness: dual-ness detection from a nested callee ===")
    assert level1(plain) is False
    print("  plain tensor, checked 3 frames deep: False (correct)")

    with fwAD.dual_level():
        d = fwAD.make_dual(dual_primal, dual_tangent)
        assert level1(d) is True
        print("  dual tensor, checked 3 frames deep, same active level: True (correct)")

    held = None
    with fwAD.dual_level():
        held = fwAD.make_dual(dual_primal, dual_tangent)
    assert probe(held) is False
    print("  dual tensor held past its creating dual_level exiting: False (correct -- reverts)")

    print("\n=== Cost: unpack_dual vs. hasLiveTangent's GPU->CPU sync ===")

    def bench(fn, n_warmup=50, n_iter=2000):
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_iter * 1e6  # us/call

    t_plain_check = bench(lambda: probe(plain))
    print(f"  unpack_dual on a PLAIN tensor: {t_plain_check:.3f} us/call")

    with fwAD.dual_level():
        d2 = fwAD.make_dual(dual_primal, dual_tangent)
        t_dual_check = bench(lambda: probe(d2))
        print(f"  unpack_dual on a DUAL tensor:  {t_dual_check:.3f} us/call")

    t_haslive = bench(lambda: hasLiveTangent(dual_tangent))
    print(f"  (for scale) hasLiveTangent's .abs().max()>0 sync: {t_haslive:.3f} us/call")
    print(f"\n  unpack_dual is ~{t_haslive / max(t_dual_check, t_plain_check):.0f}x cheaper than the sync"
          " it would let Phase 1 avoid paying for structurally-inert arguments.")


if __name__ == "__main__":
    main()
