#!/usr/bin/env python3
"""Census of device->host readbacks per simulation step, by calling site.

Every `.item()`, `bool(tensor)`, `.tolist()` or `.numpy()` on a CUDA tensor drains
the CUDA queue: the host waits for every kernel already submitted. One buried in a
per-step code path costs real wall clock while looking free in the source, and a
run that is host-bound (which warpSPH is below ~150k particles -- see
`docs/regression/real_workload_bottleneck_audit.md`) pays for all of them.

**Why this exists alongside `bench_real_workload.py`'s `sync` bucket.** That bucket
attributes to the nearest enclosing `record_function` region, which cannot separate
a region's own cost from its callees'. It reported 15 readbacks/step in "the Verlet
validity check", which reads as the rebuild decision; the decision was 2.9/step and
the other 11.5 were `_minimum_image_delta` nested inside the same region. Acting on
the region number would have meant restructuring the rebuild logic instead of
deleting a per-axis `.item()`. So: when a bucket is about to become the target of
actual work, re-derive it here first.

Counting is exact (it wraps the tensor methods themselves), deterministic, and needs
no profiler. Sites are reported as `file:line (function)`, ready to act on.

    python scripts/count_host_syncs.py                     # default scenario
    python scripts/count_host_syncs.py --nx 229 --steps 10
    python scripts/count_host_syncs.py --case sod3d --preset none
    python scripts/count_host_syncs.py --filter warpSPHCore   # only this repo's sites
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import traceback
from pathlib import Path

# Keep the profiling hooks off: this measures the unmodified per-step behaviour, and
# record_function's own overhead is irrelevant to a readback count.
os.environ.setdefault("WARPSPHCORE_PROFILING", "0")
os.environ.setdefault("warpSPHCore_PRECISION", "float32")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import bench_real_workload as bench  # noqa: E402  -- reuses its scenario plumbing

SYNCING_OPS = ("item", "__bool__", "tolist", "numpy")


class ReadbackCensus:
    """Wraps the synchronizing tensor methods and records the calling site."""

    def __init__(self, pathFilter: str | None = None):
        self.pathFilter = pathFilter
        self.counts: collections.Counter = collections.Counter()
        self.enabled = False
        self._originals: dict = {}

    def _site(self) -> str:
        # The innermost frame that is neither torch internals nor this file: that is
        # the line a reader would have to go and change.
        for frame in reversed(traceback.extract_stack()[:-2]):
            if "/torch/" in frame.filename or frame.filename == __file__:
                continue
            short = frame.filename.split("/site-packages/")[-1]
            for marker in ("/src/", "/dev/"):
                if marker in short:
                    short = short.split(marker, 1)[1]
                    break
            return f"{short}:{frame.lineno} ({frame.name})"
        return "<unknown>"

    def __enter__(self):
        census = self

        def makeWrapper(name, original):
            def wrapper(self, *args, **kwargs):
                if census.enabled and self.is_cuda:
                    site = census._site()
                    if census.pathFilter is None or census.pathFilter in site:
                        census.counts[(name, site)] += 1
                return original(self, *args, **kwargs)
            return wrapper

        for name in SYNCING_OPS:
            original = getattr(torch.Tensor, name)
            self._originals[name] = original
            setattr(torch.Tensor, name, makeWrapper(name, original))
        return self

    def __exit__(self, *exc):
        for name, original in self._originals.items():
            setattr(torch.Tensor, name, original)
        return False


def countSteps():
    """Count executed steps by wrapping the runner's own per-step timer, so the
    per-step figures divide by what actually ran rather than what was requested (a
    diverged or time-limited run stops early)."""
    import warpSPH.runner.runner as runnerModule

    counter = {"steps": 0}
    originalTimer = runnerModule._Timer

    class CountingTimer(originalTimer):
        def __exit__(self, *exc):
            counter["steps"] += 1
            return originalTimer.__exit__(self, *exc)

    runnerModule._Timer = CountingTimer
    return counter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--nx", type=int, default=128, help="lattice resolution (default: 128)")
    parser.add_argument("--steps", type=int, default=5, help="steps to count over (default: 5)")
    parser.add_argument("--warmup", type=int, default=12,
                        help="steps run before counting starts is NOT supported -- setup and "
                             "warmup are counted too, then divided out; keep this modest")
    parser.add_argument("--filter", default=None, metavar="SUBSTRING",
                        help="only report sites whose path contains this (e.g. warpSPHCore)")
    parser.add_argument("--top", type=int, default=25, help="rows to print")
    parser.add_argument("--case", default="dambreak", help="frontend case (default: dambreak)")
    parser.add_argument("--preset", default="kolmogorov", choices=sorted(bench.PRESETS))
    parser.add_argument("--param", action="append", metavar="NAME=VALUE")
    parser.add_argument("--spec", action="append", metavar="NAME=VALUE")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required: a host readback only stalls when there is a device to wait for.",
              file=sys.stderr)
        return 1

    steps = countSteps()
    census = ReadbackCensus(args.filter)
    with census:
        census.enabled = True
        bench.runProfiled(
            args.nx, args.warmup, args.steps, prof=None,
            caseName=args.case, preset=args.preset,
            params=bench.parseAssignments(args.param),
            spec=bench.parseAssignments(args.spec),
        )
        census.enabled = False

    executed = max(steps["steps"], 1)
    total = sum(census.counts.values())
    print(f"\ncase={args.case} nx={args.nx}  steps executed: {executed}"
          + (f"  filter={args.filter!r}" if args.filter else ""))
    print(f"host readbacks: {total} total, {total/executed:.1f} per step\n")
    print(f"{'per step':>9}  {'op':<10} site")
    print("-" * 92)
    for (op, site), count in census.counts.most_common(args.top):
        print(f"{count/executed:>9.2f}  {op:<10} {site}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
