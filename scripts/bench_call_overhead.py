#!/usr/bin/env python3
"""Per-stage call-overhead benchmark for warpSPHCore operator calls.

Deliverable for warpier_fields.md Step 0. Measures the fixed Python/marshalling
cost of a single ``warpOperation(...)`` call -- as opposed to
``scripts/operation_matrix.py`` (forward-value correctness) or the ``gradcheck``
skill (gradient correctness) -- across N, dimension, traversal mode, and
correction path, plus a grad-path variant.

Stage breakdown (matches warpier_fields.md Section 1.2):
    extract   -- extractStateInfo (state resolution, dummy-tensor lookups)
    convert   -- StateAwareWarpFunction.forward's torch->warp conversion loop
    build_fn  -- Warp struct assembly (build_fn closure)
    allocate  -- output allocation (allocateTorchWarp via launch_kernel)
    total     -- wall time for the whole warpOperation(...) call
    other     -- total - (extract + convert + build_fn + allocate); this is
                 wp.launch + torch.autograd.Function.apply overhead + the actual
                 GPU kernel dispatch.

Stage times are read from torch.profiler record_function labels already
present on the hot path (extractStateInfo's sub-stages, and the convert/
build_fn/allocate_output labels added alongside this script -- see
stateAwareWarpFunction.py and launcher.py). ``total`` is a plain wall-clock
measurement outside the profiler, so profiler overhead never contaminates it;
profiler-derived stage numbers are reported as a fraction of that total,
independent of the profiler's own instrumentation overhead.

Usage:
    python scripts/bench_call_overhead.py
    python scripts/bench_call_overhead.py --ns 2000 20000 200000 --dims 2
    python scripts/bench_call_overhead.py --grad --ns 20000
    python scripts/bench_call_overhead.py --record docs/regression/bench_call_overhead_baseline.md
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# --------------------------------------------------------------------------
# Lazy configuration, same idiom as operation_matrix.py: warpSPHCore bakes
# scalar_t into every @wp.kernel/@wp.func at first import, so precision must
# be set via warpSPHCore_PRECISION before any warpSPHCore submodule is
# touched, and every name below is populated by _configure().
# --------------------------------------------------------------------------
ParticleState = None
DomainDescription = None
OperationProperties = None
radiusSearchCompactHashMap = None
warpOperation = None
WarpOperation = None
KernelFunctions = None
SupportScheme = None
OperationDirection = None
generateNeighborTestData = None
n_h_to_nH = None


def _configure(precision: str) -> None:
    global ParticleState, DomainDescription, OperationProperties, radiusSearchCompactHashMap
    global warpOperation, WarpOperation, KernelFunctions, SupportScheme, OperationDirection
    global generateNeighborTestData, n_h_to_nH

    os.environ["warpSPHCore_PRECISION"] = precision

    import warp as wp
    wp.init()

    import warpSPHCore
    import warpSPHCore.util

    ParticleState = warpSPHCore.ParticleState
    DomainDescription = warpSPHCore.dataTypes.DomainDescription
    OperationProperties = warpSPHCore.OperationProperties
    radiusSearchCompactHashMap = warpSPHCore.radiusSearchCompactHashMap
    warpOperation = warpSPHCore.warpOperation
    WarpOperation = warpSPHCore.enumTypes.WarpOperation
    KernelFunctions = warpSPHCore.enumTypes.KernelFunctions
    SupportScheme = warpSPHCore.enumTypes.SupportScheme
    OperationDirection = warpSPHCore.enumTypes.OperationDirection
    generateNeighborTestData = warpSPHCore.util.generateNeighborTestData
    n_h_to_nH = warpSPHCore.n_h_to_nH


# --------------------------------------------------------------------------
# Scenario construction
# --------------------------------------------------------------------------

# Hard ceiling on actual particle count. generateNeighborTestData's first
# argument is particles-PER-AXIS, not a total count (total is roughly
# nx**dim) -- passing a total particle count straight through as nx has
# silently requested nx**dim particles before (e.g. N=20000 in 2D -> ~4e8
# particles -> tens of GB of VRAM and a hung system). This is a hard stop,
# not a warning: build_case must never allocate past this without the caller
# explicitly raising it.
_MAX_PARTICLES = 2_000_000


def build_case(n_target: int, dim: int, device: torch.device, dtype: torch.dtype,
                requires_grad: bool, n_h: float = 1.4):
    # Convert the desired total particle count into the per-axis count
    # generateNeighborTestData actually expects (total scales as nx**dim).
    nx = max(2, round(n_target ** (1.0 / dim)))
    estimated_total = nx ** dim
    if estimated_total > _MAX_PARTICLES:
        raise ValueError(
            f"build_case(n_target={n_target}, dim={dim}) would allocate an "
            f"estimated {estimated_total:,} particles (nx={nx} per axis), "
            f"over the {_MAX_PARTICLES:,} safety ceiling. Refusing."
        )
    target_neighbors = int(round(n_h_to_nH(n_h, dim)))
    positions, supports, _, domain, dx = generateNeighborTestData(
        nx, target_neighbors, dim, True, device
    )
    positions = positions.to(dtype=dtype)
    supports = supports.to(dtype=dtype)
    domain = DomainDescription(
        min=domain.min.to(dtype=dtype),
        max=domain.max.to(dtype=dtype),
        periodic=domain.periodic,
        dim=domain.dim,
    )
    masses = torch.full((positions.shape[0],), dx ** dim, device=device, dtype=dtype)
    kinds = torch.zeros(positions.shape[0], device=device, dtype=torch.int32)

    if requires_grad:
        positions = positions.detach().clone().requires_grad_(True)
        supports = supports.detach().clone().requires_grad_(True)
        masses = masses.detach().clone().requires_grad_(True)

    particles = ParticleState(
        positions=positions.contiguous(),
        supports=supports.contiguous(),
        masses=masses.contiguous(),
        densities=None,
        kinds=kinds.contiguous(),
    )
    return particles, domain


# --------------------------------------------------------------------------
# Profiler-based per-stage timing
# --------------------------------------------------------------------------

_STAGE_LABELS = {
    "extract":  "extractStateInfo [ESI]",
    "convert":  "SAWF.forward - convert",
    "build_fn": "SAWF.forward - build_fn",
    "allocate": "launch_kernel - allocate_output",
}


@dataclass
class StageTimes:
    total_us: float
    stage_us: dict = field(default_factory=dict)
    n_calls: int = 0

    def per_call(self) -> dict:
        out = {"total": self.total_us / self.n_calls}
        acct = 0.0
        for name, us in self.stage_us.items():
            v = us / self.n_calls
            out[name] = v
            acct += v
        out["other"] = out["total"] - acct
        return out


def _synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_bench(fn, device: torch.device, warmup: int, iters: int) -> StageTimes:
    for _ in range(warmup):
        fn()
    _synchronize(device)

    from torch.profiler import profile, ProfilerActivity
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    t0 = time.perf_counter()
    with profile(activities=activities, record_shapes=False) as prof:
        for _ in range(iters):
            fn()
        _synchronize(device)
    t1 = time.perf_counter()

    stage_us = {name: 0.0 for name in _STAGE_LABELS}
    for evt in prof.key_averages():
        for name, label in _STAGE_LABELS.items():
            if evt.key == label:
                # self_cpu_time_total excludes time attributed to nested
                # record_function children, so nested stages don't double count.
                stage_us[name] += evt.self_cpu_time_total

    return StageTimes(total_us=(t1 - t0) * 1e6, stage_us=stage_us, n_calls=iters)


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------

@dataclass
class Config:
    dim: int
    n: int
    traversal: str        # "grid" (prebuilt CompactHashMap, reused every call) -- see note below
    correction: str        # "none" (only "none" wired up here; CRK/renorm need extra state)
    grad: bool


def make_call(cfg: Config, device: torch.device, dtype: torch.dtype):
    # NOTE: "adjacency" traversal (a real neighbor-list AdjacencyListWarp, as
    # opposed to the compact-hash-grid CompactHashMap) has no construction
    # site anywhere in this repo today -- extractStateInfo/warpOperation
    # accept one, but nothing builds one. Only "grid" is exercised here.
    # The CompactHashMap is built once and reused across calls so the sweep
    # measures operator call overhead, not hash-map construction.
    particles, domain = build_case(cfg.n, cfg.dim, device, dtype, requires_grad=cfg.grad)
    actual_n = particles.positions.shape[0]
    adjacency = radiusSearchCompactHashMap(particles, domain, mode=SupportScheme.Gather)
    op_props = OperationProperties(
        kernel=KernelFunctions.Wendland2,
        operation=WarpOperation.Density,
        supportMode=SupportScheme.Gather,
        operationMode=OperationDirection.AllToAll,
    )

    def call():
        return warpOperation(particles, op_props, domain, adjacency=adjacency)

    return call, actual_n


def fmt(us: Optional[float]) -> str:
    if us is None:
        return "   --   "
    return f"{us:8.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ns", type=int, nargs="+", default=[2000, 20000, 200000])
    ap.add_argument("--dims", type=int, nargs="+", default=[2])
    ap.add_argument("--traversals", choices=["grid"], nargs="+", default=["grid"])
    ap.add_argument("--precision", choices=["float32", "float64"], default="float32")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--grad", action="store_true", help="also run the grad-path variant")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--record", type=str, default=None, help="write results as a markdown table to this path")
    args = ap.parse_args()

    _configure(args.precision)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    dtype = torch.float32 if args.precision == "float32" else torch.float64

    header = ["dim", "N", "traversal", "grad", "extract", "convert", "build_fn", "allocate", "other", "total"]
    rows = []
    print(f"{'dim':>4} {'N':>8} {'trav':>10} {'grad':>5} "
          f"{'extract':>8} {'convert':>8} {'build_fn':>8} {'allocate':>8} {'other':>8} {'total':>8}   (us/call)")

    grad_variants = [False, True] if args.grad else [False]

    for dim in args.dims:
        for traversal in args.traversals:
            for n in args.ns:
                for grad in grad_variants:
                    cfg = Config(dim=dim, n=n, traversal=traversal, correction="none", grad=grad)
                    try:
                        call, actual_n = make_call(cfg, device, dtype)
                        iters = args.iters if actual_n <= 20000 else max(20, args.iters // 10)
                        times = run_bench(call, device, args.warmup, iters)
                        per = times.per_call()
                        print(f"{dim:>4} {actual_n:>8} {traversal:>10} {str(grad):>5} "
                              f"{fmt(per['extract'])} {fmt(per['convert'])} {fmt(per['build_fn'])} "
                              f"{fmt(per['allocate'])} {fmt(per['other'])} {fmt(per['total'])}")
                        rows.append([dim, actual_n, traversal, grad, per['extract'], per['convert'],
                                     per['build_fn'], per['allocate'], per['other'], per['total']])
                    except Exception as e:
                        print(f"{dim:>4} {n:>8} {traversal:>10} {str(grad):>5}   ERROR: {e}")

    if args.record:
        os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
        with open(args.record, "w") as f:
            f.write(f"# bench_call_overhead baseline\n\n")
            f.write(f"precision={args.precision} device={args.device}\n\n")
            f.write("| " + " | ".join(header) + " |\n")
            f.write("|" + "---|" * len(header) + "\n")
            for r in rows:
                f.write("| " + " | ".join(f"{v:.1f}" if isinstance(v, float) else str(v) for v in r) + " |\n")
        print(f"\nWrote {args.record}")


if __name__ == "__main__":
    main()
