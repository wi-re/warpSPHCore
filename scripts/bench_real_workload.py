#!/usr/bin/env python3
"""Step H: where does a *real* simulation's wall clock actually go?
(warpier_fields.md Step H.)

**Why this exists.** `bench_call_overhead.py` measures one operator call, on a
fixed particle count, against a neighbour structure built once and reused across
every timed iteration -- by construction it isolates exactly what Steps A-F
changed, and it confirmed a ~4-6x drop in that fixed dispatch cost. A real
70k-particle run then improved ~7% end to end (45 -> 42 min). Both numbers are
right; they answer different questions. This script answers the second one: of a
real solver step's wall clock, what fraction is the dispatch overhead Steps A-F
targeted, versus neighbour rebuilds, GPU kernel compute, integrator tensor churn,
and host-device synchronisation -- as a function of particle count.

**Workload.** The frontend's `dambreak` case configured as fully-periodic
Kolmogorov flow (the invocation the user runs for data generation), driven
through `warpSPH.runner.run` so the profile covers the real solver loop --
deltaSPH, RK2, adaptive dt -- and not a synthetic single-operator loop. Plotting
and I/O are off; `--store`/`--plot` would add their own (real, but separately
attributable) cost.

**Method.** `torch.profiler` with CPU+CUDA activities, stepped once per
simulation step through the case's `postStep` hook, so the recorded window is a
fixed number of *steady-state* steps after a warmup that absorbs warp's kernel
compilation and the allocator's growth. `WARPSPHCORE_PROFILING=1` is set before
importing warpSPHCore so its `record_function` regions are real (Step F gated
them off by default) -- that is what makes dispatch time attributable to
extract/convert/build_fn/allocate rather than appearing as anonymous Python.

Note the profiler's own overhead inflates absolute step time; every number here
is therefore reported as a *share* of the profiled window, with the unprofiled
per-step wall time measured separately (the runner's own CUDA-event timing) as
the ground truth for absolute cost.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------

The default scenario is the one the Step H audit was written against. Any other
frontend case, solver scheme, integrator or kernel can be driven instead --
**which matters, because the bottleneck is strongly scenario-dependent**: the
same code is dispatch-bound in 2D weakly-compressible flow, integrator-bound
under CRKSPH, and neighbour+integrator-bound in 3D. A conclusion drawn from one
scenario does not transfer to another; re-run rather than extrapolate.

    # the audit's own sweep, and the two comparisons that interpret it
    python scripts/bench_real_workload.py --nx 39 64 128 229 384
    python scripts/bench_real_workload.py --nx 39 64 128 229 384 --no-forcing
    python scripts/bench_real_workload.py --nx 39 128 229 384 --hatches --no-forcing

    # what cases are available, and the raw profiler event names behind the buckets
    python scripts/bench_real_workload.py --listCases
    python scripts/bench_real_workload.py --dump 128 --top 40

Scenario selection:

    --case NAME        frontend case by registered name (default: dambreak)
    --preset NAME      case-parameter preset: `kolmogorov` (default) or `none`.
                       Use `none` for any case other than dambreak -- the preset
                       sets dambreak-specific knobs.
    --param NAME=VALUE case knob, repeatable (lands in CaseSpec.params)
    --spec  NAME=VALUE CaseSpec field, repeatable: scheme, integrationScheme,
                       kernel, supportMode, n_h, adaptiveDt, dt, ...
    --no-forcing       drop the Kolmogorov forcing + noise IC (dambreak only)

Recipes that stress different paths, each verified to run:

    # Taylor-Green vortex: no obstacle, no forcing, different scheme family
    ... --case tgv --preset none

    # 4 integrator stages instead of 2: ~2x the operator calls, ~2.6x the
    # allocations, so this is the scenario for Section 7's integrator churn
    ... --spec integrationScheme=rungeKutta4

    # CRKSPH: exercises the CRK correction path (A/B/gradA/gradB) end to end
    ... --case sod2d --preset none --spec scheme=CRKSPH

    # 3D: higher neighbour count per particle, so the neighbour/compute balance
    # shifts substantially away from dispatch
    ... --case sod3d --preset none

    # heavier per-particle compute at fixed particle count (bigger neighbourhoods)
    ... --spec n_h=6.0

    # a different kernel, or a support scheme with a different symmetry cost
    ... --spec kernel=Wendland6 --spec supportMode=SuperSymmetric

Reading the output:

  * `step_ms` (unprofiled) is the only absolute number to trust. Everything in
    the HOST table is profiler-inflated -- compare shares, not milliseconds.
  * `gpu_share` = device-busy / step_ms. Low means host-bound (the GPU is idle
    waiting); as it approaches 100% the run is genuinely GPU-limited and further
    dispatch-level work cannot help.
  * `blocking_ms` is cross-cutting, not a bucket: it counts host stalls wherever
    they occur, so it deliberately overlaps the buckets rather than partitioning
    with them. When the run is GPU-bound, a large `blocking_ms` is correct
    behaviour (the host waiting on real work), not a pathology.
  * `op_calls`, `launches`, `syncs` and `allocs` per step are often more
    diagnostic than the timings when comparing two scenarios.
"""

from __future__ import annotations

import argparse
import os

# Must precede any warpSPHCore import: both of these are read at import time.
os.environ.setdefault("WARPSPHCORE_PROFILING", "1")
os.environ.setdefault("warpSPHCore_PRECISION", "float32")

import dataclasses  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
from typing import Dict, List, Optional  # noqa: E402

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile, schedule  # noqa: E402


# --------------------------------------------------------------------------
# The workload: the user's Kolmogorov invocation, minus plotting/IO.
# --------------------------------------------------------------------------

KOLMOGOROV_PARAMS = dict(
    obstacleType="circleMiddle",
    offsetX=0.0,
    W=2.0,
    fillRatio=1.0,
    fluidWidth=1.0,
    maxExtent=0.5,
    aoa=0.0,
    targetDt=0.0002,
    enableNoise=True,
    disableGravity=True,
    seed=2187561599,
    octaves=4,
    baseFrequency=2,
    fullyPeriodic=True,
    enableKolmogorovForcing=True,
    kolmogorovForcingWavenumber=2,
    noiseAmplitude=0.1,
)

#: Named case-parameter presets. "kolmogorov" is the invocation this audit was
#: built around (fully-periodic Kolmogorov flow on the dambreak case); "none"
#: leaves the case's own defaults alone, which is what any other case wants.
PRESETS = {"kolmogorov": KOLMOGOROV_PARAMS, "none": {}}


def loadCase(name: str):
    """Look a case up by the name it registered under (`--case`).

    Cases self-register on import and `warpSPH.cases` does not import its own
    submodules, so the registry is empty until they are pulled in.
    """
    import importlib

    from warpSPH.cases import CASE_MODULES
    from warpSPH.runner.case import getCase, listCases

    for module in CASE_MODULES:
        try:
            importlib.import_module(f"warpSPH.cases.{module}")
        except Exception:
            # A case that fails to import is not this benchmark's problem unless
            # it is the one being asked for, which getCase() will report.
            pass
    if name not in listCases():
        raise SystemExit(f"unknown case {name!r}; registered: {sorted(listCases())}")
    return getCase(name)


def runProfiled(nx: int, warmup: int, active: int, prof: Optional[object] = None,
                forcing: bool = True, caseName: str = "dambreak",
                preset: str = "kolmogorov", params: Optional[dict] = None,
                spec: Optional[dict] = None):
    """Run the case for warmup+active steps, stepping `prof` once per step.

    `forcing=False` drops the Kolmogorov forcing term (and the noise initial
    condition that comes with it), which is how the forcing's own cost gets
    separated from the solver's -- everything else about the case is unchanged.
    Only meaningful for a case that has those knobs.

    `params` are case knobs (they land in `CaseSpec.params`); `spec` are CaseSpec
    fields (`scheme`, `integrationScheme`, `kernel`, `dim`, ...). Both are applied
    on top of the preset, so either can override it.
    """
    baseCase = loadCase(caseName)
    from warpSPH.runner import run

    originalPostStep = baseCase.postStep

    def postStep(ctx, state, i):
        if originalPostStep is not None:
            originalPostStep(ctx, state, i)
        if prof is not None:
            prof.step()

    case = dataclasses.replace(baseCase, postStep=postStep)

    caseParams = dict(PRESETS.get(preset, {}))
    if not forcing:
        # Only applies to the Kolmogorov preset; harmless keys otherwise, but
        # dropping them keeps a non-Kolmogorov case's params untouched.
        if caseParams:
            caseParams.update(enableKolmogorovForcing=False, enableNoise=False)
    caseParams.update(params or {})
    params = caseParams

    overrides = dict(
        nx=nx,
        nSteps=warmup + active,
        params=params,
        plot=False,
        store=False,
        video=False,
        quiet=True,
        progress=False,
    )
    overrides.update(spec or {})

    allocBefore = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
    result = run(case, **overrides)
    # "allocation.all.allocated" is the cumulative *count* of allocator blocks
    # handed out (the plausible-looking "allocation.all.count" does not exist and
    # silently read as 0 until this was checked against memory_stats()'s real
    # keys). Setup allocates too, so this over-counts the loop slightly; what it
    # is for is comparing across scales and against Section 7's ~40-tensors-per-
    # step estimate, not an exact per-step figure.
    allocAfter = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
    result.allocationsPerStep = (allocAfter - allocBefore) / max(warmup + active, 1)
    return result


# --------------------------------------------------------------------------
# Bucketing.
#
# Host and device time have to be bucketed from *different* event sets, or the
# device numbers come out badly double-counted: a `record_function` region and
# each warp kernel launched inside it both report `self_device_time_total`, so
# the naive "sum it over key_averages()" gives ~3x the profiler's own
# "Self CUDA time total". Device time therefore comes only from
# DeviceType.CUDA events (the kernels and memcpys themselves) and host time only
# from the rest. Every needle below was read off a real --dump, not guessed.
# --------------------------------------------------------------------------

from torch.autograd import DeviceType  # noqa: E402

# Host-side attribution is by *innermost enclosing `record_function` region*,
# not by the event's own name, with the event's own name used only as a fallback
# when nothing encloses it.
#
# This matters more than it sounds. Bucketing `aten::copy_` by its own name put
# 18 of 20 ms/step at 54k particles into "integrator" and made the integrator
# look like the dominant cost -- while walking the parent chain showed 9 of those
# ms sitting inside `[warpSPH] - computeForcing` (the Kolmogorov forcing's
# host round-trip through a scipy interpolator) and only ~0.05 ms/step actually
# inside the `[Integration]` regions. Same total, opposite conclusion, and the
# wrong one would have promoted Section 7's buffer-pool work on false evidence.
REGION_RULES = [
    # The Kolmogorov forcing, broken out because it turned out to dominate: it
    # evaluates a scipy RegularGridInterpolator on the host every stage, so the
    # position array goes D->H and the result comes back H->D.
    ("forcing", ("computeForcing",)),
    ("neighbour", ("radiusSearch", "CompactHashMap", "Hashmap", "Verlet",
                   "Adjacency", "adjacency", "neighborList", "numNeighbors")),
    # What Steps A-F targeted: argument marshalling around each kernel launch.
    ("dispatch", ("extractStateInfo", "[ESI]", "SAWF.forward", "StateAwareWarpFunction",
                  "warpWrapper2", "[WW2]", "Warp Function", "Warp Kernel Launch",
                  "launch_kernel", "getCachedDummyTensor", "WarpFunctionWrapper")),
    # The integrator's own out-of-place arithmetic and state cloning
    # (warpier_fields.md Section 7).
    ("integrator", ("[Integration]",)),
    # Python and torch work inside the frontend's solver modules.
    ("solver", ("[warpSPH]", "warpSPH - Operation")),
]

# Fallback rules, for events with no enclosing region at all.
HOST_RULES = [
    ("sync", ("cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaMemcpy",
              "aten::item", "aten::_local_scalar_dense", "aten::nonzero")),
    ("dispatch", ("cudaLaunchKernel",)),
    ("neighbour", ("radiusSearch", "CompactHashMap", "Hashmap", "Verlet")),
    ("integrator", ("[Integration]",)),
    ("solver", ("[warpSPH]", "warpSPH - Operation", "WarpFunctionWrapper")),
    # The runner's own per-step work: diagnostics reductions, the NaN check, the
    # adaptive-dt computation.
    ("torch_ops", ("aten::", "cuLaunchKernel", "cuda")),
]

# Cross-cutting: host-side stalls, wherever they occur. Reported alongside the
# primary breakdown rather than as a bucket in it, since a blocking wait inside
# the forcing is both "forcing cost" and "a sync" and double-counting it into
# two primary buckets would make the shares meaningless.
BLOCKING_KEYS = ("cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaMemcpy",
                 "aten::item", "aten::_local_scalar_dense", "aten::nonzero")

DEVICE_RULES = [
    ("neighbour", ("hashCells", "countNeighbors", "collectNeighbors", "sort",
                   "Verlet", "numNeighbors", "prefixSum", "scan")),
    # Warp-generated operator kernels all end in _cuda_kernel_forward/backward.
    ("operator", ("_cuda_kernel_forward", "_cuda_kernel_backward", "_Kernel_")),
    ("integrator", ("elementwise_kernel", "Memcpy", "Memset", "reduce_kernel",
                    "fill_kernel", "index_elementwise")),
]


def _match(name: str, rules) -> str:
    for bucket, needles in rules:
        for needle in needles:
            if needle in name:
                return bucket
    return "other"


def hostBucket(event) -> str:
    """Bucket a host event by its innermost enclosing `record_function` region,
    falling back to its own name when nothing encloses it.

    A region's *own* self-time belongs to its own bucket, which is why its key is
    tested before the parent chain is walked: `[warpSPH] - computeForcing` is
    nested inside other `[warpSPH]` regions, so walking straight to the parent
    charged its 16 ms/step of scipy interpolation to the generic "solver" bucket
    and understated the forcing by a factor of two.
    """
    own = _match(event.key, REGION_RULES)
    if own != "other":
        return own

    parent = getattr(event, "cpu_parent", None)
    depth = 0
    while parent is not None and depth < 24:
        bucket = _match(parent.key, REGION_RULES)
        if bucket != "other":
            return bucket
        parent = getattr(parent, "cpu_parent", None)
        depth += 1
    return _match(event.key, HOST_RULES)


# Names whose call counts are worth reporting per step in their own right.
COUNTER_KEYS = {
    "cudaStreamSynchronize": "syncs",
    "cudaLaunchKernel": "launches",
    "StateAwareWarpFunction": "operator_calls",
    "aten::nonzero": "nonzero",
    "cudaMemcpyAsync": "memcpy_async",
}


def summarise(prof, steps: int) -> dict:
    """Per-step host-self and device-self milliseconds, grouped into buckets,
    plus per-step call counts for the events worth counting.

    Aggregated from the *raw* events rather than `key_averages()`, because
    `key_averages()` merges the CPU and CUDA aspects of one key into a single
    entry and labels the result `DeviceType.CUDA` -- so a `record_function`
    region like "SAWF.forward - launch" arrives carrying both its own host
    self-time and the device time of every kernel launched inside it, under one
    `device_type`. Bucketing off that gave a device total ~3.5x the profiler's
    own "Self CUDA time total" and GPU shares above 100%. The raw events expose
    `is_user_annotation`, which is the discriminator: device time is summed only
    over genuine device events (kernels, memcpys), while host self-time is
    summed over everything (self-time already excludes children, so regions and
    the ops inside them do not double-count).
    """
    host: Dict[str, float] = {}
    device: Dict[str, float] = {}
    counts: Dict[str, float] = {}
    blocking = 0.0

    for event in prof.events():
        selfCpu = float(getattr(event, "self_cpu_time_total", 0.0) or 0.0)
        if selfCpu:
            host[hostBucket(event)] = host.get(hostBucket(event), 0.0) + selfCpu
            if any(k in event.key for k in BLOCKING_KEYS):
                blocking += selfCpu

        isRealDevice = (event.device_type == DeviceType.CUDA
                        and not getattr(event, "is_user_annotation", False))
        if isRealDevice:
            selfDev = float(getattr(event, "self_device_time_total", 0.0) or 0.0)
            if selfDev:
                bucket = _match(event.key, DEVICE_RULES)
                device[bucket] = device.get(bucket, 0.0) + selfDev

        label = COUNTER_KEYS.get(event.key)
        if label is not None:
            counts[label] = counts.get(label, 0.0) + 1

    return dict(
        host={k: v / steps / 1000.0 for k, v in host.items()},
        device={k: v / steps / 1000.0 for k, v in device.items()},
        counts={k: v / steps for k, v in counts.items()},
        hostTotal=sum(host.values()) / steps / 1000.0,
        deviceTotal=sum(device.values()) / steps / 1000.0,
        blockingMs=blocking / steps / 1000.0,
    )


def dumpEvents(nx: int, warmup: int, active: int, top: int) -> None:
    """Print the profiler's own top events, so the classifier can be built from
    real names. Nothing here is load-bearing for the report -- it is the
    exploratory pass, kept because the next person will need it too."""
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    sched = schedule(wait=warmup, warmup=1, active=active, repeat=1)
    with profile(activities=activities, schedule=sched, record_shapes=False) as prof:
        runProfiled(nx, warmup + 1, active, prof)

    print(f"\n=== nx={nx}: top {top} by self CUDA time ===")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=top))
    print(f"\n=== nx={nx}: top {top} by self CPU time ===")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=top))


def stepTime(nx: int, warmup: int, active: int, **kwargs) -> dict:
    """Unprofiled per-step wall clock -- the runner's own CUDA-event timing, i.e.
    what a real run actually pays. No profiler attached, since profiling inflates
    host time substantially."""
    result = runProfiled(nx, warmup, active, prof=None, **kwargs)
    times = [row["stepTime_ms"] for row in result.trajectory
             if row.get("step", -1) >= warmup]
    return dict(
        particles=result.state.state.positions.shape[0],
        stepMs=statistics.median(times) if times else float("nan"),
        stepMsMin=min(times) if times else float("nan"),
        allocationsPerStep=result.allocationsPerStep,
    )


def measure(nx: int, warmup: int, active: int, **kwargs) -> dict:
    """One scale: unprofiled step time (ground truth) plus a profiled breakdown."""
    plain = stepTime(nx, warmup, active, **kwargs)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    sched = schedule(wait=warmup, warmup=1, active=active, repeat=1)
    with profile(activities=activities, schedule=sched, record_shapes=False) as prof:
        runProfiled(nx, warmup + 1, active, prof, **kwargs)
    summary = summarise(prof, active)

    return dict(nx=nx, **plain, **summary)


def parseAssignments(items: List[str]) -> dict:
    """`name=value` pairs off the command line, with values typed by inspection.

    Enum-valued CaseSpec fields (`kernel`, `scheme`, `integrationScheme`, ...) take
    their *names* as strings -- the runner resolves those itself, exactly as it does
    for the frontend's own CLI."""
    out = {}
    for item in items or ():
        if "=" not in item:
            raise SystemExit(f"expected name=value, got {item!r}")
        name, _, raw = item.partition("=")
        if raw.lower() in ("true", "false"):
            value = raw.lower() == "true"
        elif raw.lower() == "none":
            value = None
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        out[name] = value
    return out


HOST_BUCKETS = ["dispatch", "neighbour", "forcing", "integrator", "solver",
                "torch_ops", "sync", "other"]
DEVICE_BUCKETS = ["operator", "neighbour", "integrator", "other"]


def report(rows: List[dict]) -> None:
    print()
    print("Ground truth (no profiler attached; runner's own CUDA-event per-step timing):")
    print()
    print(f"{'particles':>10} {'step_ms':>9} {'us/particle':>12} {'allocs/step':>12} "
          f"{'op_calls':>9} {'launches':>9} {'syncs':>7}")
    print("-" * 76)
    for r in rows:
        c = r["counts"]
        print(f"{r['particles']:>10,} {r['stepMs']:>9.2f} "
              f"{r['stepMs']*1000/r['particles']:>12.3f} {r['allocationsPerStep']:>12.0f} "
              f"{c.get('operator_calls', 0):>9.0f} {c.get('launches', 0):>9.0f} "
              f"{c.get('syncs', 0):>7.0f}")

    print()
    print("HOST self-time per step, ms (profiled window; the profiler inflates these,")
    print("so read the shares, not the absolutes):")
    print()
    header = f"{'particles':>10} " + " ".join(f"{n[:9]:>10}" for n in HOST_BUCKETS) + f"{'total':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = " ".join(f"{r['host'].get(n, 0.0):>10.2f}" for n in HOST_BUCKETS)
        print(f"{r['particles']:>10,} {cells} {r['hostTotal']:>10.2f}")
    print()
    print(f"{'particles':>10} " + " ".join(f"{n[:9]+'%':>10}" for n in HOST_BUCKETS))
    print("-" * len(header))
    for r in rows:
        total = max(r["hostTotal"], 1e-9)
        cells = " ".join(f"{100*r['host'].get(n, 0.0)/total:>10.1f}" for n in HOST_BUCKETS)
        print(f"{r['particles']:>10,} {cells}")

    print()
    print("Cross-cutting (not a bucket above -- a blocking wait inside the forcing is both):")
    print(f"{'particles':>10} {'blocking_ms':>12} {'% of host':>10}")
    print("-" * 35)
    for r in rows:
        total = max(r["hostTotal"], 1e-9)
        print(f"{r['particles']:>10,} {r['blockingMs']:>12.2f} "
              f"{100*r['blockingMs']/total:>9.1f}%")

    print()
    print("DEVICE (GPU kernel) self-time per step, ms -- actual device-busy time:")
    print()
    header = f"{'particles':>10} " + " ".join(f"{n[:9]:>10}" for n in DEVICE_BUCKETS) + f"{'total':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = " ".join(f"{r['device'].get(n, 0.0):>10.2f}" for n in DEVICE_BUCKETS)
        print(f"{r['particles']:>10,} {cells} {r['deviceTotal']:>10.2f}")

    print()
    print("GPU-bound vs CPU-bound: device-busy time as a share of real (unprofiled) step time.")
    print("A low share means the step is dominated by host-side work, i.e. the GPU is idle waiting.")
    print()
    print(f"{'particles':>10} {'step_ms':>9} {'gpu_busy_ms':>12} {'gpu_share':>10}")
    print("-" * 45)
    for r in rows:
        print(f"{r['particles']:>10,} {r['stepMs']:>9.2f} {r['deviceTotal']:>12.2f} "
              f"{100*r['deviceTotal']/max(r['stepMs'], 1e-9):>9.1f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # Used by --hatches, which needs two interpreters because the escape hatches
    # are read at import time.
    parser.add_argument("--stepTimeOnly", type=int, default=None, metavar="NX",
                        help=argparse.SUPPRESS)
    parser.add_argument("--nx", type=int, nargs="+",
                        default=[39, 122, 229, 387],
                        help="lattice resolutions to sweep (particles ~ 1.34 * nx^2)")
    parser.add_argument("--warmup", type=int, default=12,
                        help="steps discarded before measuring (kernel compile, allocator growth)")
    parser.add_argument("--active", type=int, default=15, help="steps measured")
    parser.add_argument("--dump", type=int, default=None, metavar="NX",
                        help="exploratory mode: print the profiler's top events at this nx")
    parser.add_argument("--top", type=int, default=45, help="rows for --dump")
    parser.add_argument("--no-forcing", action="store_true",
                        help="drop the Kolmogorov forcing and the noise IC, to separate "
                             "the forcing's own cost from the solver's")
    parser.add_argument("--hatches", action="store_true",
                        help="in-situ before/after: per-step wall clock with every "
                             "Steps A-F escape hatch flipped off, no profiler attached")
    parser.add_argument("--case", default="dambreak",
                        help="frontend case to drive, by registered name (default: dambreak). "
                             "--listCases prints them.")
    parser.add_argument("--listCases", action="store_true",
                        help="print every registered case name and exit")
    parser.add_argument("--preset", default="kolmogorov", choices=sorted(PRESETS),
                        help="case-parameter preset (default: kolmogorov, the invocation "
                             "this audit was built around). Use 'none' for another case.")
    parser.add_argument("--param", action="append", metavar="NAME=VALUE",
                        help="case knob override, repeatable (lands in CaseSpec.params)")
    parser.add_argument("--spec", action="append", metavar="NAME=VALUE",
                        help="CaseSpec field override, repeatable: scheme, integrationScheme, "
                             "kernel, supportMode, dim, n_h, adaptiveDt, ...")
    args = parser.parse_args()

    if args.listCases:
        from warpSPH.cases import CASE_MODULES
        import importlib
        from warpSPH.runner.case import listCases
        for module in CASE_MODULES:
            try:
                importlib.import_module(f"warpSPH.cases.{module}")
            except Exception:
                pass
        print(" ".join(sorted(listCases())))
        return 0

    scenario = dict(
        caseName=args.case,
        preset=args.preset,
        params=parseAssignments(args.param),
        spec=parseAssignments(args.spec),
    )

    if not torch.cuda.is_available():
        print("CUDA required for this benchmark.", file=sys.stderr)
        return 1

    if args.dump is not None:
        dumpEvents(args.dump, args.warmup, args.active, args.top)
        return 0

    forcing = not args.no_forcing

    if args.stepTimeOnly is not None:
        r = stepTime(args.stepTimeOnly, args.warmup, args.active,
                     forcing=forcing, **scenario)
        print(f"STEPTIME {r['particles']} {r['stepMs']:.4f}")
        return 0

    if args.hatches:
        return runHatchComparison(args, scenario)
    rows: List[dict] = []
    for nx in args.nx:
        print(f"--- nx={nx} ---", file=sys.stderr)
        rows.append(measure(nx, args.warmup, args.active, forcing=forcing, **scenario))

    print(f"\nScenario: case={args.case} preset={args.preset} "
          f"forcing={'ON' if forcing else 'OFF'}"
          + (f" params={scenario['params']}" if scenario["params"] else "")
          + (f" spec={scenario['spec']}" if scenario["spec"] else ""))
    report(rows)
    return 0


# --------------------------------------------------------------------------
# In-situ before/after (Step H methodology item 4): what did Steps A-F actually
# buy on this workload, measured on the workload rather than inferred from
# bench_call_overhead.py?
#
# The escape hatches only reach Steps B-F's caching layers. Step F's StateBundle
# has no env-var hatch by design (there is no contract that would make grad-path
# struct sharing safe, so it is gated on requires_grad instead) -- so
# WARPSPHCORE_DISABLE_BUNDLE is added here purely as a benchmarking hatch, and
# this reports whether it exists rather than silently measuring less than it
# claims.
# --------------------------------------------------------------------------

HATCHES_OFF = {
    "WARPSPHCORE_DISABLE_FIELD_CACHE": "1",
    "WARPSPHCORE_FIELD_CACHE_GRAD": "0",
    "WARPSPHCORE_DISABLE_BUNDLE": "1",
}


def runHatchComparison(args, scenario: dict) -> int:
    import subprocess

    print("Per-step wall clock with Steps A-F's caching on (as landed) vs. off:")
    print("  " + ", ".join(f"{k}={v}" for k, v in HATCHES_OFF.items()))
    print("No profiler attached; each figure is a median over the measured steps.")
    print(f"Scenario: case={scenario['caseName']} preset={scenario['preset']} "
          f"forcing={'OFF' if args.no_forcing else 'ON'}")
    print()
    print(f"{'particles':>10} {'landed_ms':>10} {'hatched_ms':>11} {'delta':>8} {'speedup':>8}")
    print("-" * 52)

    passthrough = ["--case", args.case, "--preset", args.preset]
    for item in args.param or ():
        passthrough += ["--param", item]
    for item in args.spec or ():
        passthrough += ["--spec", item]
    if args.no_forcing:
        passthrough.append("--no-forcing")

    for nx in args.nx:
        out = []
        for hatched in (False, True):
            env = dict(os.environ)
            if hatched:
                env.update(HATCHES_OFF)
            else:
                for key in HATCHES_OFF:
                    env.pop(key, None)
            env["WARPSPHCORE_PROFILING"] = "0"
            proc = subprocess.run(
                [sys.executable, __file__, "--stepTimeOnly", str(nx),
                 "--warmup", str(args.warmup), "--active", str(args.active)]
                + passthrough,
                capture_output=True, text=True, env=env, timeout=3600,
            )
            line = [ln for ln in proc.stdout.splitlines() if ln.startswith("STEPTIME")]
            if not line:
                print(f"  nx={nx} hatched={hatched} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
                return 1
            _, particles, ms = line[0].split()
            out.append((int(particles), float(ms)))
        (particles, landed), (_, hatched_ms) = out
        print(f"{particles:>10,} {landed:>10.2f} {hatched_ms:>11.2f} "
              f"{hatched_ms-landed:>+8.2f} {hatched_ms/landed:>7.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
