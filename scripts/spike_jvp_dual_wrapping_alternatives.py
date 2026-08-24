#!/usr/bin/env python3
"""JFNK-jvp overhead spike, part 2: two "wrap more fields as dual" directions
tested and ruled out, plus the PyTorch crash they'd have to route around
(`warpier_jvp_dual_argument_pruning_plan.md`).

`spike_jvp_wave_case_overhead_profile.py` shows an isolated dual JVP call
costs ~1.9x two plain calls, with the gap traced to torch's own
`Function.apply()` dispatch allocating a fresh zero tensor as the synthesized
tangent for every non-dual real-Tensor argument. Two superficially-plausible
fixes for that were tested here, both making things *worse*, not better:

1. **Cache a zero tangent for constant fields, wrap them dual too**
   (`fwAD.make_dual(t, cached_zero_tensor)`, tangent pre-allocated once,
   reused every call -- avoiding torch synthesizing a *fresh* one each time).
   Result: ~1.08x -- marginally worse than the baseline. Some synthesis
   persisted regardless (`aten::zeros_like` dropped, but not to 0), and the
   `make_dual`/`unpack_dual` bookkeeping for the newly-wrapped fields offset
   what savings there were.

2. **Wrap the whole state as dual with genuinely live (non-zero) tangents**
   -- motivated by: in a real coupled Lagrangian simulation,
   density->pressure->force->velocity->position coupling means nearly every
   field has *some* live tangent after the first step, so maybe the overhead
   mostly evaporates once nothing is "provably zero". Result: ~2x *worse*.
   Reason: this forces every operator call through the full combined
   geometry+value JVP path (`warpier_jvp_dual_argument_pruning_plan.md`'s
   "Fix 1") -- a real, physically-necessary second kernel pass (CRK/renorm-
   aware chain-rule terms for the position/support tangent) -- even for a
   matvec direction that doesn't actually need that field differentiated.
   Dual-wrapping doesn't make that compute cheaper; it just makes it happen
   unconditionally. This is the load-bearing finding against "wrap
   everything" as a family of fix: the problem is how MUCH gets
   differentiated per call, which dual-wrapping doesn't reduce.

3. **The crash a "wrap everything, including provably-zero fields" design
   would have to route around.** An all-zero-tangent dual tensor reaching a
   warp operator crashes with an internal PyTorch assertion --
   `jfnk.py`'s own `jvp_matvec` already works around this by skipping
   `make_dual` for any integrated field whose tangent slice is exactly zero
   for a given Krylov vector. Reproduced here in isolation to pin down the
   exact trigger: a single all-zero-tangent dual tensor crashes *on its own*,
   not just when mixed with a separate live-dual argument -- narrower framing
   than the inline comment in `jfnk.py` suggests, but the same underlying
   hazard, and the same required workaround (skip `make_dual` per-field when
   provably zero, exactly what's already there).

    python scripts/spike_jvp_dual_wrapping_alternatives.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_WARPSPH_BENCHMARKS = Path(__file__).resolve().parents[2] / "warpSPH" / "benchmarks"
if _WARPSPH_BENCHMARKS.is_dir():
    sys.path.insert(0, str(_WARPSPH_BENCHMARKS))

import warpSPHBootstrap  # noqa: E402
warpSPHBootstrap.bootstrap()

import torch  # noqa: E402
import torch.autograd.forward_ad as fwAD  # noqa: E402

from common.runner import buildWaveCase  # noqa: E402
from warpSPHCore import OperationProperties, WarpOperation, warpOperation  # noqa: E402
from warpSPHCore.dataTypes import ParticleState  # noqa: E402

DEVICE = "cuda:0"
NX = 128


def bench(fn, n_warmup=20, n_iter=200):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1e6  # us/call


def main() -> None:
    ctx, system, buildSeconds = buildWaveCase(NX, device=DEVICE)
    state = system.initializeNewState()

    u = state.state.u.clone()
    positions = state.state.positions
    supports = state.state.supports
    masses = state.state.masses
    densities = state.state.densities if state.state.densities is not None else torch.ones_like(u)

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

    def call_full_state(uval, posval, supval, massval, densval):
        st = ParticleState(positions=posval, supports=supval, masses=massval,
                            densities=densval, kinds=state.state.kinds)
        return warpOperation(st, queryValues=uval, domain=system.domain,
                              adjacency=system.adjacency, operationProperties=props)

    v_tangent = torch.randn_like(u)

    def baseline_one_dual():
        with fwAD.dual_level():
            u_dual = fwAD.make_dual(u, v_tangent)
            return fwAD.unpack_dual(call_plain(u_dual))

    t_baseline = bench(baseline_one_dual)
    print(f"nx={NX}, particle count={u.shape[0]}")
    print(f"baseline, only u dual (today's design):                  {t_baseline:.2f} us/call")

    # --- (1) cached-zero-tangent wrapping ---------------------------------
    zero_pos, zero_sup = torch.zeros_like(positions), torch.zeros_like(supports)
    zero_mass, zero_dens = torch.zeros_like(masses), torch.zeros_like(densities)

    def cached_zero_dual():
        with fwAD.dual_level():
            u_dual = fwAD.make_dual(u, v_tangent)
            pos_dual = fwAD.make_dual(positions, zero_pos)
            sup_dual = fwAD.make_dual(supports, zero_sup)
            mass_dual = fwAD.make_dual(masses, zero_mass)
            dens_dual = fwAD.make_dual(densities, zero_dens)
            return fwAD.unpack_dual(call_full_state(u_dual, pos_dual, sup_dual, mass_dual, dens_dual))

    t_cached_zero = bench(cached_zero_dual)
    print(f"(1) everything dual, cached ZERO tangents for constants: {t_cached_zero:.2f} us/call"
          f"  ratio={t_cached_zero / t_baseline:.3f}x")

    # --- (2) all-live-tangent wrapping -------------------------------------
    pos_t_live = torch.randn_like(positions)
    sup_t_live = torch.randn_like(supports) * 0.01
    # Query-side mass tangent has no derived JVP formula for value-having
    # operators (operations.py's own scope restriction) -- zero, not live.
    mass_t_live = torch.zeros_like(masses)
    dens_t_live = torch.randn_like(densities) * 0.01

    def all_live_dual():
        with fwAD.dual_level():
            u_dual = fwAD.make_dual(u, v_tangent)
            pos_dual = fwAD.make_dual(positions, pos_t_live)
            sup_dual = fwAD.make_dual(supports, sup_t_live)
            mass_dual = fwAD.make_dual(masses, mass_t_live)
            dens_dual = fwAD.make_dual(densities, dens_t_live)
            return fwAD.unpack_dual(call_full_state(u_dual, pos_dual, sup_dual, mass_dual, dens_dual))

    t_all_live = bench(all_live_dual)
    print(f"(2) ALL fields dual, ALL genuinely LIVE tangents:         {t_all_live:.2f} us/call"
          f"  ratio={t_all_live / t_baseline:.3f}x")

    # --- (3) crash repro -----------------------------------------------------
    print("\n--- (3) reproducing the all-zero-tangent-dual crash ---")

    def all_zero_u_alone():
        with fwAD.dual_level():
            u_dual = fwAD.make_dual(u, torch.zeros_like(u))
            return fwAD.unpack_dual(call_plain(u_dual))

    try:
        all_zero_u_alone()
        print("  NO crash: u alone with an all-zero tangent -- unexpected, re-check torch version")
    except RuntimeError as e:
        print(f"  CRASHED as expected: {type(e).__name__}: {e}")
        print("  (this is exactly what jfnk.py's own `if bool(tangent.abs().max() > 0):` guard"
              " in jvp_matvec already works around by skipping make_dual for exactly-zero slices)")


if __name__ == "__main__":
    main()
