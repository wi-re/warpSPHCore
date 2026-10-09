#!/usr/bin/env python3
"""Frozen-particle PDE leg for the Phase 4/5 operators (MLS/RKPM, LABFM).

A thin wrapper over `run_frozen.run_one`: registers the `rkpm*` / `labfm*`
modes, runs each at the stencil size it needs (one `--target-neighbors` per
mode, unlike the stock leg), with a small CFL so classical RK4's O(dt^4)
time error stays below the O(h^k) spatial error of a k >= 4 operator, and
writes its OWN `REPORT_frozen_hiorder.md` / `results/frozen_hiorder_rows.csv`
(the stock `REPORT_frozen.md` is the frozen Phase 0 leg and is not touched).

Reports observed orders (L2 vs dx) and the stability signal (divergence).
"""

from __future__ import annotations

import os
os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import argparse
import csv
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
import warp as wp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_frozen as rf                                  # noqa: E402
from metrics import observed_order                       # noqa: E402
from rkpm_modes import register_labfm_modes, register_rkpm_modes  # noqa: E402

# (mode, stencil size)
PLAN = [("rkpm1", 40), ("rkpm2", 40), ("rkpm3", 60),
        ("labfm2", 25), ("labfm4", 40), ("labfm6", 60), ("labfm8", 80)]


def fit(rows, eq, mode):
    r = sorted((x.dx, x.error_l2) for x in rows
               if x.equation == eq and x.mode == mode and not x.diverged)
    if len(r) < 3:
        return "n/a"
    o = observed_order([a for a, _ in r], [b for _, b in r])
    return "exact" if o.exact else "sat" if o.saturated else f"{o.slope:.2f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--equations", nargs="+", default=["advection", "diffusion"])
    ap.add_argument("--ladder", nargs="+", type=int, default=[288, 572, 1152, 2304])
    ap.add_argument("--cfl", type=float, default=0.05)
    ap.add_argument("--cfl-diffusion", type=float, default=0.02)
    ap.add_argument("--t-end", type=float, default=0.5)
    ap.add_argument("--modes", nargs="+", default=[m for m, _ in PLAN])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)
    wp.init()
    torch.set_grad_enabled(False)
    register_rkpm_modes((1, 2, 3))
    register_labfm_modes((2, 4, 6, 8))

    rows, stencil = [], dict(PLAN)
    for eq in a.equations:
        for mode in a.modes:
            if eq == "diffusion" and mode == "rkpm1":
                continue                     # p = 1 has no Laplacian (None)
            args = SimpleNamespace(
                target_neighbors=stencil[mode], jitter=rf.DEFAULT_JITTER,
                seed=42, k=[1, 1], velocity=[1.0, 0.5], nu=0.05,
                t_end=a.t_end, cfl=a.cfl, cfl_diffusion=a.cfl_diffusion)
            for n in a.ladder:
                r = rf.run_one(n, eq, mode, args, a.device)
                rows.append(r)
                print(f"[{eq}/{mode} N_nbr={stencil[mode]}] N={r.N} steps={r.n_steps} "
                      f"err_l2={r.error_l2:.3e} diverged={r.diverged} "
                      f"wall={r.wall_s}s", flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    with (HERE / "results" / "frozen_hiorder_rows.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(rows[0])))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    L = ["# Frozen-particle PDE leg -- Phase 4/5 operators\n",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `run_frozen_hi.py` "
         f"(float64, 2-D periodic, jitter {rf.DEFAULT_JITTER}, RK4, "
         f"cfl {a.cfl} advection / {a.cfl_diffusion} diffusion, t_end "
         f"{a.t_end}, ladder {a.ladder}). Stock leg (standard / crk / renorm): "
         f"`REPORT_frozen.md`.\n",
         "| equation | mode | N neighbours | order (L2 vs dx) | L2 error @ coarsest -> finest | diverged |",
         "|---|---|---|---|---|---|"]
    for eq in a.equations:
        for mode in a.modes:
            if eq == "diffusion" and mode == "rkpm1":
                continue
            rs = sorted([x for x in rows if x.equation == eq and x.mode == mode],
                        key=lambda x: x.dx)
            e0, e1 = rs[-1].error_l2, rs[0].error_l2
            L.append(f"| {eq} | {mode} | {stencil[mode]} | {fit(rows, eq, mode)} | "
                     f"{e0:.2e} -> {e1:.2e} | "
                     f"{'yes' if any(x.diverged for x in rs) else 'no'} |")
    L.append("")
    (HERE / "REPORT_frozen_hiorder.md").write_text("\n".join(L))
    print("wrote REPORT_frozen_hiorder.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
