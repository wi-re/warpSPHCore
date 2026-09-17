#!/usr/bin/env python3
"""JFNK-jvp overhead spike, part 1: what the wave-equation benchmark's `jvp`
matvec actually pays for, per operator call (`warpier_jvp_dual_argument_pruning_plan.md`).

Reproduces two findings from the 2026-08-24 investigation that started from a
user report -- `warpSPH/benchmarks/wave/bench_performance.py --schemes rk4
sdirk2_jfnk_jvp_1e-6 sdirk2_jfnk_fd_1e-6 --nxs 32 64 128 256 --steps 32` showed
`sdirk2_jfnk_jvp_1e-6`'s `msPerRhs` at ~4x `rk4`'s, against an expected ~2x
(one primal launch, one tangent-producing pass -- the minimum for an exact
forward-mode directional derivative):

1. **The combined geometry+value JVP path (see
   `warpier_jvp_dual_argument_pruning_plan.md`'s "Fix 1") is never reached by
   this specific benchmark.** `WaveSystemStatev3`
   (`warpSPH/src/warpSPH/systems/waveSystem.py`) tags `u`/`v` as
   `'position'`/`'velocity'` only to reuse integrator plumbing -- actual
   particle `positions` are a `constant()` field, never part of the
   JFNK-integrated state. Patches `warpSPHCore.operations.warpOperationJVP`
   with a counter and runs a few real `sdirk2_jfnk_jvp_1e-6` steps to confirm:
   every call takes the value-only path (0 calls ever have a live geometry
   tangent), which is why the Fix-1 kernel fusion measurably helps an isolated
   Gradient call but shows ~0% effect on this benchmark specifically.

2. **Even on that already-minimal value-only path, an isolated dual JVP call
   still costs ~3.7x one plain (non-dual) call of the same kernel** -- the
   correct fd-equivalent baseline (`jfnk.py`'s real `fd_matvec` calls `step()`
   exactly once per matvec, reusing the already-computed `G_y`; an earlier
   version of this script compared against *two* plain calls under a
   "fd-equivalent" label, which was wrong and overstated a mechanism fix
   below as a much bigger win than it actually was -- see
   `warpier_jvp_dual_argument_pruning_plan.md`'s "Correction" section). Of
   that 3.7x, CUDA kernel-name-level profiling shows ~2x is architecturally
   inherent (the value-only path genuinely launches the same primal kernel
   twice -- once for the primal, once relaunched on the tangent value,
   `operations.py`'s `warpOperationJVP` -- with identical per-launch GPU
   cost either way); the rest is CPU-side dispatch overhead, only partly
   explained: `aten::zeros_like`/`aten::empty_like` synthesis (~12 calls/call
   before `warpier_jvp_dual_argument_pruning_plan.md`'s Phase 1 pruning fix,
   0 after) plus a further `aten::to`/`_to_copy`/`copy_` chain (~77us/call)
   present only in the dual path and not yet traced to an exact call site.

Needs the sibling `warpSPH` repo checked out next to this one (this
workspace's convention: `dev/{warpSPHCore,warpSPH,warpSPHIntegrators}`) --
`warpSPH`/`warpSPHBootstrap` themselves resolve via `warpSPH`'s own editable
pip install in the `warp` conda env, but `benchmarks/common` is not an
installed package, so this script adds `warpSPH/benchmarks` to `sys.path`
relative to its own location.

    python scripts/spikes/spike_jvp_wave_case_overhead_profile.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_WARPSPH_BENCHMARKS = Path(__file__).resolve().parents[3] / "warpSPH" / "benchmarks"
if _WARPSPH_BENCHMARKS.is_dir():
    sys.path.insert(0, str(_WARPSPH_BENCHMARKS))

import warpSPHBootstrap  # noqa: E402  (must precede warpSPH imports)
warpSPHBootstrap.bootstrap()

import torch  # noqa: E402
import torch.autograd.forward_ad as fwAD  # noqa: E402
from torch.profiler import profile, ProfilerActivity  # noqa: E402

import warpSPHCore.operations as opsmod  # noqa: E402
from common.runner import buildWaveCase  # noqa: E402
from common.schemes import getScheme  # noqa: E402
from warpSPHCore import OperationProperties, WarpOperation, warpOperation  # noqa: E402
from warpSPHIntegrators import getIntegrator  # noqa: E402

DEVICE = "cuda:0"
NX = 128


def part1_geometry_tangent_liveness(ctx, system) -> None:
    print("=== Part 1: does sdirk2_jfnk_jvp_1e-6 ever hit a live geometry tangent? ===")
    orig_warpOperationJVP = opsmod.warpOperationJVP
    counts = {"geom_calls": 0, "value_only_calls": 0}

    def patched(*args, **kwargs):
        qs, rs = kwargs.get("queryTangentState"), kwargs.get("referenceTangentState")
        fields = []
        for s in (qs, rs):
            if s is not None:
                fields += [s.positions, s.supports, s.masses, s.densities]
        hasGeomTangent = any(f is not None for f in fields)
        if hasGeomTangent:
            counts["geom_calls"] += 1
        elif kwargs.get("tangentQueryValues") is not None or kwargs.get("tangentReferenceValues") is not None:
            counts["value_only_calls"] += 1
        return orig_warpOperationJVP(*args, **kwargs)

    opsmod.warpOperationJVP = patched
    try:
        scheme = getScheme("sdirk2_jfnk_jvp_1e-6")
        integrator = getIntegrator(scheme.integrationScheme)
        solver = scheme.makeSolver()
        state = system.initializeNewState()
        dt = ctx.config.dt
        kwargs = dict(f=ctx.stepFunction, dt=dt, config=ctx.config,
                      schemeConfig=ctx.schemeConfig, verbose=False, solver=solver)
        for _ in range(5):
            state = integrator.function(state=state, **kwargs).state
        counts["geom_calls"] = 0
        counts["value_only_calls"] = 0
        for _ in range(3):
            state = integrator.function(state=state, **kwargs).state
        print(f"  over 3 timed steps: {counts}")
        assert counts["geom_calls"] == 0, "expected 0 -- positions are constant() in this case"
        print("  confirmed: 0 calls ever had a live geometry tangent (positions are constant() here)")
    finally:
        opsmod.warpOperationJVP = orig_warpOperationJVP


def part2_isolated_dual_vs_plain(ctx, system) -> None:
    print("\n=== Part 2: isolated dual Laplacian(u,u) JVP call vs. one plain call ===")
    state = system.initializeNewState()
    u = state.state.u.clone()
    v_tangent = torch.randn_like(u)
    props = OperationProperties(
        operation=WarpOperation.Laplacian,
        kernel=ctx.schemeConfig.kernel,
        supportMode=ctx.schemeConfig.supportMode,
        laplacianMode=ctx.schemeConfig.laplacianMode,
        gradientMode=ctx.schemeConfig.gradientMode,
    )

    def call_plain(uval):
        return warpOperation(state.state, queryValues=uval, domain=system.domain,
                              adjacency=system.adjacency, operationProperties=props)

    def one_plain_call():
        return call_plain(u)

    def one_dual_call():
        with fwAD.dual_level():
            u_dual = fwAD.make_dual(u, v_tangent)
            result = call_plain(u_dual)
            return fwAD.unpack_dual(result)

    def bench(fn, n_warmup=200, n_iter=1000):
        # n_warmup=200 (not 50): this call runs right after Part 1's real DIRK/
        # JFNK steps above, and a too-short warmup measures residual GPU
        # clock-ramp/cache-state noise from that -- confirmed directly: at
        # n_warmup=50 in-process here, the plain baseline measured ~484us
        # (vs. this function's own steady-state ~161us in isolation), enough
        # noise to invert the qualitative finding (see
        # warpier_jvp_dual_argument_pruning_plan.md's "Correction" section).
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_iter * 1e6  # us/call

    t_plain = bench(one_plain_call)
    t_dual = bench(one_dual_call)
    print(f"  nx={NX}, particle count={u.shape[0]}")
    print(f"  one plain call  (fd-equivalent, no AD):           {t_plain:.2f} us")
    print(f"  one dual call   (real jvp path, forward+jvp):     {t_dual:.2f} us")
    print(f"  ratio (dual/plain): {t_dual / t_plain:.3f}x")

    print("\n  --- profiling the dual call ---")
    for _ in range(20):
        one_dual_call()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(200):
            one_dual_call()
        torch.cuda.synchronize()
    events = {e.key: e for e in prof.key_averages()}
    for name in ("aten::zeros_like", "aten::empty_like", "cudaStreamSynchronize", "StateAwareWarpFunction"):
        if name in events:
            e = events[name]
            print(f"  {name:28s} count={e.count:5d}  self_cpu_time_total={e.self_cpu_time_total/1000:.2f}ms")

    print("\n  --- profiling the plain baseline (for comparison) ---")
    for _ in range(20):
        one_plain_call()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(200):
            one_plain_call()
        torch.cuda.synchronize()
    events = {e.key: e for e in prof.key_averages()}
    for name in ("aten::zeros_like", "aten::empty_like", "cudaStreamSynchronize", "StateAwareWarpFunction"):
        if name in events:
            e = events[name]
            print(f"  {name:28s} count={e.count:5d}  self_cpu_time_total={e.self_cpu_time_total/1000:.2f}ms")
        else:
            print(f"  {name:28s} count=    0  (absent from the plain-call profile)")


if __name__ == "__main__":
    ctx, system, buildSeconds = buildWaveCase(NX, device=DEVICE)
    part1_geometry_tangent_liveness(ctx, system)
    # Part 1's real DIRK/JFNK steps leave measurable residual GPU-state noise
    # behind (confirmed directly: without this settle, Part 2's own 200-warmup
    # bench() still sometimes reports a plain-call baseline 2-3x its clean,
    # isolated-process steady state) -- a plain sync doesn't fully clear it,
    # so Part 2's ratio here is a noisier, upper-bound-leaning reading than
    # running it in its own process; the clean, repeated (3-run) reading is
    # ~3.6-3.7x, recorded in warpier_jvp_dual_argument_pruning_plan.md's
    # "Correction" section -- trust that number over a single run of this
    # combined script if they disagree.
    torch.cuda.synchronize()
    part2_isolated_dual_vs_plain(ctx, system)
