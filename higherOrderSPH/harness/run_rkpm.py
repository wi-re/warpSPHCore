#!/usr/bin/env python3
"""Phase 4 driver: the order-p MLS/RKPM operator through the Phase 0 harness.

Runs the *unchanged* static suites of `run_baseline.py` (patch, resolve-open,
resolve-periodic, smoothing) with the registered modes ``rkpm1..3`` next to
the ``crk`` / ``renorm`` reference columns, plus the Phase 4 additions:

* **p = 1 cross-check** -- ``rkpm1`` *values* must equal the ``crk`` values to
  round-off (the parent plan's check); the gradient is the diffuse
  derivative, which differs from CRK's full kernel gradient (see rkpm.py);
* **conditioning vs order** -- condition number of the (equilibrated) moment
  matrix vs p, jitter and neighbour count, interior vs boundary band, with
  the fraction of rank-deficient rows.

It writes its own `results/rkpm_rows.csv` / `results/cond_rkpm.csv` /
`REPORT_rkpm.md` and never touches the frozen baseline outputs, so it is not
a harness versioning event.

    python run_rkpm.py                       # full static matrix, 2-D
    python run_rkpm.py --suites patch        # one suite
    python run_rkpm.py --smoke               # CI-sized check, exit code
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import run_baseline as rb           # sets float64 before warp is imported
import torch
import warp as wp

from warpSPHCore.enumTypes import KernelFunctions

from particle_sets import build_case
from operators import CorrectionCache, run_probe
from report import Row, plot_error_curves, render_pivot, write_rows_csv
from rkpm_modes import get_operator, register_rkpm_modes
from test_fields import monomial_fields

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REFERENCE_MODES = ("crk", "renorm")


def suite_cond(kernel, dim, device, orders, out_csv: Path) -> list[dict]:
    """Moment-matrix conditioning vs order / jitter / neighbour count."""
    records = []
    for p in orders:
        for jitter in rb.COND_JITTERS:
            for tn in rb.SMOOTHING_NEIGHBORS:
                case = build_case(576, dim, tn, jitter=jitter, seed=42,
                                  periodic=False, device=device, kernel=kernel)
                s = get_operator(case, CorrectionCache(case), p).s
                for region, mask in (("interior", case.interior_mask),
                                     ("boundary", case.boundary_mask)):
                    if not bool(mask.any()):
                        continue
                    c = s.cond[mask]
                    records.append(dict(
                        dim=dim, order=p, jitter=jitter, target_neighbors=tn,
                        h_over_dx=case.h_over_dx, region=region,
                        n_basis=s.n_basis, cond_median=float(c.median()),
                        cond_p95=float(c.quantile(0.95)),
                        cond_max=float(c.max()),
                        deficient_frac=float(s.deficient[mask].float().mean())))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        for r in records:
            w.writerow({k: ("%.6g" % v if isinstance(v, float) else v)
                        for k, v in r.items()})
    return records


def crosscheck_p1(kernel, dim, device) -> dict:
    """rkpm1 vs crk on a jittered open set: values (must agree to round-off)
    and gradients (expected to differ at O(h) on a non-polynomial field)."""
    case = build_case(1152, dim, rb.DEFAULT_TARGET_NEIGHBORS,
                      jitter=rb.DEFAULT_JITTER, seed=42, periodic=False,
                      device=device, kernel=kernel)
    cache = CorrectionCache(case)
    x = case.positions
    f = torch.sin(2.0 * x[:, 0]) * torch.cos(1.5 * x[:, 1] if dim > 1 else x[:, 0])
    out = {}
    for probe in ("interpolate", "gradient"):
        a = run_probe(case, cache, f, probe, "rkpm1")
        b = run_probe(case, cache, f, probe, "crk")
        d = (a - b).abs()
        out[probe] = dict(
            interior_max=float(d[case.interior_mask].max()),
            boundary_max=float(d[case.boundary_mask].max()),
            scale=float(b.abs().max()))
    return out


def write_report(path: Path, patch_rows, orders, xattr, cond, cross, kernel,
                 dim) -> None:
    modes = REFERENCE_MODES + tuple(f"rkpm{p}" for p in (1, 2, 3))
    L: list[str] = []
    add = L.append
    add("# Phase 4 report -- order-p MLS/RKPM reproducing-kernel operator\n")
    add(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `run_rkpm.py` "
        f"(float64, {dim}-D, kernel {kernel.name}, jitter "
        f"{rb.DEFAULT_JITTER}, {rb.DEFAULT_TARGET_NEIGHBORS} target "
        f"neighbours). Operator: `rkpm.py` (pure-torch local polynomial fit, "
        f"diffuse derivatives). Reference columns `crk` / `renorm` from the "
        f"unchanged Phase 0 suites.\n")
    for region, title in (("interior", "interior"), ("boundary",
                                                     "boundary band")):
        add(f"## Monomial patch tests -- open domain, {title}\n")
        add("Max error; ~1e-15..1e-12 = exact reproduction. `rkpmP` should be "
            "exact through degree P for value / gradient and, for P >= 2, "
            "Laplacian / Hessian.\n")
        add(render_pivot([r for r in patch_rows if r.region == region],
                         row_keys=("field", "field_degree", "probe"),
                         col_key="mode", col_order=modes))
    for key, orows in orders.items():
        add(f"## Observed orders -- {key} (error vs {xattr[key]})\n")
        add(rb.render_orders(orows))
    add("## p = 1 cross-check against CRK\n")
    for probe, v in cross.items():
        add(f"- `{probe}`: max |rkpm1 - crk| interior {v['interior_max']:.2e}, "
            f"boundary {v['boundary_max']:.2e} (field scale {v['scale']:.2e})")
    add("")
    add("## Moment-matrix conditioning\n")
    hdr = ("order", "n_basis", "jitter", "target_neighbors", "region",
           "cond_median", "cond_p95", "cond_max", "deficient_frac")
    add("| " + " | ".join(hdr) + " |")
    add("|" + "|".join("---" for _ in hdr) + "|")
    for r in cond:
        add("| " + " | ".join(
            ("%.3g" % r[k] if isinstance(r[k], float) else str(r[k]))
            for k in hdr) + " |")
    add("")
    path.write_text("\n".join(L))


def run_smoke(device, kernel) -> bool:
    """CI-sized: exact degree-p reproduction (value/grad, interior+boundary)
    for p = 1, 2, 3 and the rkpm1 == crk value identity."""
    case = build_case(288, 2, rb.DEFAULT_TARGET_NEIGHBORS,
                      jitter=rb.DEFAULT_JITTER, seed=7, periodic=False,
                      device=device, kernel=kernel)
    cache = CorrectionCache(case)
    fields = {f.name: f for f in monomial_fields(2, degree_max=3)}
    ok = True
    for p in (1, 2, 3):
        for f in fields.values():
            if f.degree > p:
                continue
            for probe in ("interpolate", "gradient"):
                for region in ("interior", "boundary"):
                    e = rb.measure(case, cache, f, probe, f"rkpm{p}",
                                   region)["norms"]["linf"]
                    if not e < 1e-8:
                        print(f"FAIL rkpm{p} {probe} {f.name} {region}: {e:.2e}")
                        ok = False
    cross = crosscheck_p1(kernel, 2, device)
    if cross["interpolate"]["interior_max"] > 1e-10:
        print("FAIL rkpm1 != crk value:", cross)
        ok = False
    print("smoke", "ok" if ok else "FAILED")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dim", type=int, default=2, choices=(1, 2, 3))
    ap.add_argument("--kernel", default="Wendland2")
    ap.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--degree-max", type=int, default=4)
    ap.add_argument("--suites", default="patch,resolve-open,resolve-periodic,"
                                        "smoothing,conditioning")
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wp.init()
    torch.set_grad_enabled(False)
    kernel = KernelFunctions[a.kernel]
    names = register_rkpm_modes(tuple(a.orders))
    if a.smoke:
        return 0 if run_smoke(device, kernel) else 1

    args = SimpleNamespace(dim=a.dim, degree_max=a.degree_max,
                           modes=list(REFERENCE_MODES) + list(names))
    out = Path(a.out)
    suites = [s.strip() for s in a.suites.split(",") if s.strip()]
    all_rows: list[Row] = []
    patch_rows: list[Row] = []
    orders: dict = {}
    xattr: dict = {}
    cond: list[dict] = []
    for suite in suites:
        t0 = time.time()
        if suite == "conditioning":
            cond = suite_cond(kernel, a.dim, device, a.orders,
                              out / "results" / "cond_rkpm.csv")
            print(f"conditioning: {len(cond)} records {time.time()-t0:.1f}s",
                  flush=True)
            continue
        rows = {"patch": lambda: rb.suite_patch(kernel, args, device),
                "resolve-open": lambda: rb.suite_resolve(kernel, args, device, False),
                "resolve-periodic": lambda: rb.suite_resolve(kernel, args, device, True),
                "smoothing": lambda: rb.suite_smoothing(kernel, args, device),
                }[suite]()
        for r in rows:
            r.suite = suite
        all_rows += rows
        if suite == "patch":
            patch_rows += rows
        elif suite.startswith("resolve"):
            orders[suite] = rb.orders_for(rows, "dx")
            xattr[suite] = "dx"
            plot_error_curves(rows, out / "figures" / f"rkpm_{suite}.png",
                              xattr="dx", title=f"error vs dx -- {suite}")
        else:
            orders[suite] = rb.orders_for(rows, "h")
            xattr[suite] = "h"
            plot_error_curves(rows, out / "figures" / f"rkpm_{suite}.png",
                              xattr="h", title=f"error vs h -- {suite}")
        print(f"{suite}: {len(rows)} rows {time.time()-t0:.1f}s", flush=True)

    write_rows_csv(all_rows, out / "results" / "rkpm_rows.csv")
    cross = crosscheck_p1(kernel, a.dim, device)
    write_report(out / "REPORT_rkpm.md", patch_rows, orders, xattr, cond,
                 cross, kernel, a.dim)
    print(f"wrote {out / 'REPORT_rkpm.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
