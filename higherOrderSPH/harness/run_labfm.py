#!/usr/bin/env python3
"""Phase 5 driver: LABFM order vs stencil size (King et al. 2020).

For each scheme order k and stencil size N (target neighbour count) the
resolution ladder (fixed h/dx, N_particles varied) is run on the periodic
smooth fields and the observed order of the gradient and Laplacian is
extracted -- the same question the paper's Fig. 9 asks ("which stencil size
gives order k?"), answered here with the unchanged Phase 0 error metrics.

Paper reference points (2-D, quoted from the abstract / Sec. 4.1): 4th order
with N ~ 25 nodes, 8th order with N ~ 60 (convergence shown at h/dr = 2.5,
N ~ 78), k <= 6 converging at h/dr = 2 (N ~ 50); critical stencil sizes for
the ABF moment matrix N_crit ~ {8, 21, 37, 57} for k = {2, 4, 6, 8}, versus
N_poly = {6, 15, 28, 45} for plain polynomial reconstruction. This driver
reports, per k, the smallest N that reaches order >= k - 0.5 (gradient) and
the moment-matrix conditioning at that N, next to those numbers.

Writes `results/labfm_rows.csv`, `REPORT_labfm.md`. Own outputs only: the
frozen baseline files are untouched.

    python run_labfm.py                 # k = 2 4 6 8, N = 25 40 60 80 120
    python run_labfm.py --orders 2 4 --stencils 25 40
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import run_baseline as rb            # sets float64 before warp is imported
import torch
import warp as wp

from warpSPHCore.enumTypes import KernelFunctions

from metrics import observed_order
from operators import CorrectionCache
from particle_sets import build_case
from report import Row
from rkpm_modes import get_operator, register_labfm_modes
from test_fields import smooth_periodic_fields

HERE = Path(__file__).resolve().parent
LADDER = (288, 572, 1152, 2304, 4608)
PAPER_N = {2: "—", 4: "~25", 6: "~50 (h/dr=2)", 8: "~60-78"}
PAPER_NCRIT = {2: 8, 4: 21, 6: 37, 8: 57}
PAPER_NPOLY = {2: 6, 4: 15, 6: 28, 8: 45}


def ladder_rows(kernel, k, nbrs, probes, fields_fn, device, jitter):
    rows = []
    cond = []
    mode = f"labfm{k}"
    for n in LADDER:
        case = build_case(n, 2, nbrs, jitter=jitter, seed=42, periodic=True,
                          device=device, kernel=kernel)
        cache = CorrectionCache(case)
        s = get_operator(case, cache, k, labfm=True).s
        cond.append((float(s.cond.median()), float(s.deficient.float().mean())))
        for f in fields_fn(case):
            for probe in probes:
                m = rb.measure(case, cache, f, probe, mode, "interior")
                if m:
                    rows.append(m["row"])
    return rows, cond


def fit_order(rows, probe, field, key="dx"):
    r = sorted([(getattr(x, key), x.linf) for x in rows
                if x.probe == probe and x.field == field])
    if len(r) < 3:
        return None
    o = observed_order([a for a, _ in r], [b for _, b in r])
    return o


def fmt(o):
    if o is None:
        return "n/a"
    if o.exact:
        return "exact"
    if o.saturated:
        return "sat"
    return f"{o.slope:.2f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--stencils", type=int, nargs="+",
                    default=[25, 40, 60, 80, 120])
    ap.add_argument("--jitter", type=float, default=0.3)
    ap.add_argument("--kernel", default="Wendland2")
    ap.add_argument("--field", default="sin_cos")
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wp.init()
    torch.set_grad_enabled(False)
    kernel = KernelFunctions[a.kernel]
    register_labfm_modes(tuple(a.orders))
    fields_fn = lambda case: [f for f in smooth_periodic_fields(2, case.box)
                              if f.name == a.field]

    all_rows: list[Row] = []
    table = []
    t0 = time.time()
    for k in a.orders:
        for nb in a.stencils:
            rows, cond = ladder_rows(kernel, k, nb,
                                     ("gradient", "laplacian"), fields_fn,
                                     device, a.jitter)
            all_rows += rows
            og = fit_order(rows, "gradient", a.field)
            ol = fit_order(rows, "laplacian", a.field)
            fine = [r.linf for r in rows if r.probe == "gradient"
                    and r.N == max(x.N for x in rows)]
            table.append(dict(k=k, nbrs=nb, grad=fmt(og), lap=fmt(ol),
                              grad_slope=(None if og is None or og.saturated or og.exact else og.slope),
                              lap_slope=(None if ol is None or ol.saturated or ol.exact else ol.slope),
                              grad_err_fine=(fine[0] if fine else float("nan")),
                              cond_med=cond[-1][0], deficient=cond[-1][1]))
            print(f"k={k} N={nb}: grad {table[-1]['grad']} lap {table[-1]['lap']} "
                  f"cond {cond[-1][0]:.2g} def {cond[-1][1]:.2g} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    out = Path(a.out)
    (out / "results").mkdir(exist_ok=True)
    with (out / "results" / "labfm_rows.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    L = ["# Phase 5 report -- LABFM order vs stencil size\n",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `run_labfm.py` "
         f"(float64, 2-D periodic, kernel {kernel.name}, jitter {a.jitter}, "
         f"field `{a.field}`, ladder N = {LADDER}). Operator: `rkpm.py` "
         f"`constant=False` (LABFM: kernel-weighted polynomial fit of f_j - f_i). "
         f"Orders are the log-log slope of the interior L-inf error vs dx at "
         f"fixed N (neighbours); `sat` = flat/floor-limited series.\n",
         "## Observed order, gradient (G) and Laplacian (L), by scheme order k "
         "and stencil size\n",
         "| k | N (nbrs) | gradient | Laplacian | grad err @ finest | "
         "cond (median) | rank-deficient |", "|---|---|---|---|---|---|---|"]
    for t in table:
        L.append(f"| {t['k']} | {t['nbrs']} | {t['grad']} | {t['lap']} | "
                 f"{t['grad_err_fine']:.2e} | {t['cond_med']:.2g} | "
                 f"{t['deficient']:.2g} |")
    L.append("")
    L.append("## Smallest stencil reaching gradient order >= k - 0.5\n")
    L.append("| k | measured N_min | paper N | paper N_crit (ABF) | N_poly |")
    L.append("|---|---|---|---|---|")
    for k in a.orders:
        ok = [t["nbrs"] for t in table if t["k"] == k
              and t["grad_slope"] is not None and t["grad_slope"] >= k - 0.5]
        L.append(f"| {k} | {min(ok) if ok else '> ' + str(max(a.stencils))} | "
                 f"{PAPER_N.get(k, '?')} | {PAPER_NCRIT.get(k, '?')} | "
                 f"{PAPER_NPOLY.get(k, '?')} |")
    L.append("")
    (out / "REPORT_labfm.md").write_text("\n".join(L))
    print(f"wrote {out / 'REPORT_labfm.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
