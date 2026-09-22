#!/usr/bin/env python3
"""Baseline sweep driver for the convergence harness (first pass).

Runs the Phase 1 (standard SPH) and Phase 2 (CRKSPH) baselines -- plus
the renorm (Bonet-Lok-style covariance correction) reference column --
against the static test matrix of `PLAN.md`, and writes:

* `results/baseline_rows.csv`  -- every measurement, flat
* `results/cond_<kernel>.csv`  -- condition-number statistics
* `figures/*.png`              -- convergence curves
* `REPORT.md`                  -- the frozen "before" column

Precision is float64 (the env var must be set before importing
warpSPHCore -- hence at the very top of this file).

Suites (select with --suites):
  patch             open-domain monomial patch tests (degree 0..4)
  resolve-open      resolution sweep, open domain, h/dx fixed
  resolve-periodic  resolution sweep, periodic domain, periodic fields
  smoothing         N fixed, h varied (target_neighbors)
  conditioning      renorm condition numbers vs jitter / neighbor count

Usage:
    python run_baseline.py --smoke                 # CI gate matrix
    python run_baseline.py                          # full baseline
    python run_baseline.py --suites patch,resolve-open
    python run_baseline.py --kernels Wendland2 CubicSpline
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ["warpSPHCore_PRECISION"] = "float64"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import warp as wp

from warpSPHCore.enumTypes import KernelFunctions

from conditioning import cond_summary, condition_numbers, fallback_mask
from metrics import error_norms, observed_order
from operators import CorrectionCache, MODES, run_probe
from particle_sets import build_case
from report import (Row, plot_error_curves, render_pivot, write_rows_csv)
from test_fields import (
    monomial_fields,
    smooth_open_fields,
    smooth_periodic_fields,
    vector_field,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

DEFAULT_N_TARGET = 1152
DEFAULT_JITTER = 0.3
DEFAULT_TARGET_NEIGHBORS = 40

# (n, label) resolution ladder -- achieved N/dx are recorded, not assumed.
RESOLVE_N = (288, 572, 1152, 2304)
SMOOTHING_NEIGHBORS = (12, 20, 35, 60)
COND_JITTERS = (0.1, 0.3, 0.5)


# ---------------------------------------------------------------------------
# one measurement
# ---------------------------------------------------------------------------

def measure(case, cache, field, probe, mode, region) -> dict | None:
    if case.periodic:
        mask = None  # every particle is interior in a periodic domain
    else:
        mask = (case.interior_mask if region == "interior"
                else case.boundary_mask)
        if mask is None or not mask.any():
            return None
    values = field.value(case.positions)
    if probe == "interpolate":
        out = run_probe(case, cache, values, "interpolate", mode)
        analytic = values
    elif probe == "gradient":
        out = run_probe(case, cache, values, "gradient", mode)
        analytic = field.grad(case.positions)
    elif probe == "laplacian":
        if field.lap is None:
            return None
        out = run_probe(case, cache, values, "laplacian", mode)
        analytic = field.lap(case.positions)
    else:
        raise ValueError(probe)
    norms = error_norms(out, analytic, mask)
    return {"row": Row(
        dim=case.dim,
        kernel=case.kernel.name,
        mode=mode,
        periodic=case.periodic,
        jitter=case.jitter,
        N=case.N,
        dx=case.dx,
        h=case.h,
        h_over_dx=case.h_over_dx,
        target_neighbors=case.target_neighbors,
        probe=probe,
        field=field.name,
        field_degree=field.degree,
        region=region,
        **norms,
    ), "norms": norms}


def probe_fields(case, cache, fields, probes, modes, regions) -> list[Row]:
    rows = []
    for field in fields:
        for probe in probes:
            # Laplacian: monomials need degree >= 2 (else the analytic
            # target is identically zero); smooth fields (degree -1)
            # always have a nonzero, spatially varying target.
            if probe == "laplacian" and (field.lap is None
                                         or (field.degree >= 0
                                             and field.degree < 2)):
                continue
            for mode in modes:
                for region in regions:
                    m = measure(case, cache, field, probe, mode, region)
                    if m:
                        rows.append(m["row"])
    return rows


def case_regions(case) -> list[str]:
    return ["interior"] if case.periodic else ["interior", "boundary"]


# ---------------------------------------------------------------------------
# suites -- each returns its own rows so the report can keep them separate
# ---------------------------------------------------------------------------

def suite_patch(kernel, args, device) -> list[Row]:
    """Open-domain monomial patch tests: which degrees does each mode
    reproduce, interior vs boundary band."""
    case = build_case(DEFAULT_N_TARGET, args.dim, DEFAULT_TARGET_NEIGHBORS,
                      jitter=DEFAULT_JITTER, seed=42, periodic=False,
                      device=device, kernel=kernel)
    cache = CorrectionCache(case)
    fields = monomial_fields(args.dim, degree_max=args.degree_max)
    if args.dim >= 2:
        fields = fields + [vector_field(args.dim)]
    return probe_fields(case, cache, fields,
                       ["interpolate", "gradient", "laplacian"],
                       args.modes, case_regions(case))


def suite_resolve(kernel, args, device, periodic: bool) -> list[Row]:
    """Resolution sweep: h/dx fixed (target_neighbors), N varied."""
    rows: list[Row] = []
    for n in RESOLVE_N:
        case = build_case(n, args.dim, DEFAULT_TARGET_NEIGHBORS,
                          jitter=DEFAULT_JITTER, seed=42, periodic=periodic,
                          device=device, kernel=kernel)
        cache = CorrectionCache(case)
        # Monomials are not periodic: on a periodic domain the Difference
        # gradient is seam-contaminated (the field jumps by L*grad(f)
        # across the images), so the periodic sweep uses periodic smooth
        # fields only. The open sweep gets monomials + open smooth fields.
        if periodic:
            fields = smooth_periodic_fields(args.dim, case.box)
        else:
            fields = (monomial_fields(args.dim, degree_max=2)
                      + smooth_open_fields(args.dim, case.box))
        rows += probe_fields(case, cache, fields,
                            ["interpolate", "gradient", "laplacian"],
                            args.modes, case_regions(case))
    return rows


def suite_smoothing(kernel, args, device) -> list[Row]:
    """Smoothing sweep: N (and the lattice) fixed, h varied via
    target_neighbors."""
    rows: list[Row] = []
    for tn in SMOOTHING_NEIGHBORS:
        case = build_case(DEFAULT_N_TARGET, args.dim, tn,
                          jitter=DEFAULT_JITTER, seed=42, periodic=False,
                          device=device, kernel=kernel)
        cache = CorrectionCache(case)
        fields = monomial_fields(args.dim, degree_max=1)
        fields += smooth_open_fields(args.dim, case.box)
        rows += probe_fields(case, cache, fields,
                            ["interpolate", "gradient"],
                            args.modes, case_regions(case))
    return rows


def suite_conditioning(kernel, args, device, out_csv: Path) -> list[dict]:
    """Renorm condition numbers vs jitter and neighbor count."""
    records: list[dict] = []
    for jitter in COND_JITTERS:
        for tn in SMOOTHING_NEIGHBORS:
            case = build_case(576, args.dim, tn, jitter=jitter, seed=42,
                              periodic=False, device=device, kernel=kernel)
            cache = CorrectionCache(case)
            C, eigvals, _state = cache.renorm()
            cond = condition_numbers(eigvals)
            fb = fallback_mask(C)
            base = dict(dim=args.dim, kernel=kernel.name,
                        N=case.N, dx=case.dx, h=case.h,
                        h_over_dx=case.h_over_dx, target_neighbors=tn,
                        jitter=jitter)
            for region in case_regions(case):
                mask = (case.interior_mask if region == "interior"
                        else case.boundary_mask)
                stats = cond_summary(cond, mask)
                stats["fallback_frac"] = float(fb[mask].float().mean())
                records.append({**base, "region": region, **stats})
    if records:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        names = sorted({k for r in records for k in r})
        with out_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=names)
            writer.writeheader()
            for r in records:
                writer.writerow({k: ("%.10g" % v if isinstance(v, float)
                                     else str(v)) for k, v in r.items()})
    return records


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def orders_for(rows: list[Row], xattr: str) -> list[dict]:
    """Observed orders: group by (probe, field, mode, region), x = xattr."""
    groups: dict[tuple, list[Row]] = {}
    for r in rows:
        groups.setdefault((r.probe, r.field, r.mode, r.region), []).append(r)
    out = []
    for (probe, field, mode, region), rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: getattr(r, xattr))
        if len(rs) < 2:
            continue
        res = observed_order([getattr(r, xattr) for r in rs],
                             [r.linf for r in rs])
        out.append({"probe": probe, "field": field,
                    "field_degree": rs[0].field_degree, "mode": mode,
                    "region": region, "slope": res.slope, "r2": res.r_squared,
                    "saturated": res.saturated, "exact": res.exact})
    return out


def render_orders(orders: list[dict]) -> str:
    if not orders:
        return "*(no orders computed)*\n"
    header = ("probe", "field", "mode", "region", "slope", "r2", "saturated")
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for o in orders:
        if o.get("exact"):
            slope_cell, r2_cell, sat_cell = "exact", "exact", "no"
        elif o["saturated"]:
            slope_cell, r2_cell, sat_cell = "n/a", "n/a", "yes"
        else:
            slope_cell, r2_cell, sat_cell = (f"{o['slope']:.2f}",
                                             f"{o['r2']:.3f}", "no")
        cells = [o["probe"], o["field"], o["mode"], o["region"],
                 slope_cell, r2_cell, sat_cell]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(path: Path, patch_rows: list[Row],
                 orders: dict[str, list[dict]],
                 cond_records: list[dict], kernel_names: str,
                 xattr: dict[str, str]) -> None:
    lines: list[str] = []
    add = lines.append
    add("# Convergence harness -- baseline report (before column)\n")
    add(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `run_baseline.py` "
        f"(float64, kernels: {kernel_names}, jitter {DEFAULT_JITTER}, "
        f"default target neighbors {DEFAULT_TARGET_NEIGHBORS}, "
        f"patch N target {DEFAULT_N_TARGET}).\n")
    add("Frozen reference for the higher-order phases (parent plan "
        "`../../higher_order.md`): later phases diff against these numbers "
        "and do not re-baseline silently. A harness change is a "
        "versioning event that re-runs every phase.\n")

    add("## 1. Monomial patch tests -- open domain, interior\n")
    add("Max error per monomial / probe / mode. ~1e-15 (float64) = exact "
        "reproduction.\n")
    add(render_pivot(
        [r for r in patch_rows if r.region == "interior"],
        row_keys=("kernel", "field", "field_degree", "probe"),
        col_key="mode", col_order=MODES))

    add("## 2. Monomial patch tests -- open domain, boundary band\n")
    add("Same tests on particles within one support of an open wall "
        "(truncated kernel support).\n")
    add(render_pivot(
        [r for r in patch_rows if r.region == "boundary"],
        row_keys=("kernel", "field", "field_degree", "probe"),
        col_key="mode", col_order=MODES))

    sec = 3
    for label, orows in orders.items():
        x = xattr.get(label, "dx")
        add(f"## {sec}. Observed orders -- {label} (error vs {x})\n")
        add("`slope` = d log E / d log " + x + "; `saturated` = flat error "
            "series (floor/plateau, no order to read); `exact` = errors at "
            "machine precision (nothing to fit).\n")
        add(render_orders(orows))
        sat = [o for o in orows if o["saturated"]]
        if sat:
            add("Saturated series: " + ", ".join(
                f"{o['probe']}/{o['field']}/{o['mode']} ({o['region']})"
                for o in sat) + "\n")
        sec += 1

    add(f"## {sec}. Renorm condition numbers\n")
    sec += 1
    if cond_records:
        names = sorted({k for r in cond_records for k in r})
        lines.append("| " + " | ".join(str(h) for h in names) + " |")
        lines.append("|" + "|".join("---" for _ in names) + "|")
        for r in cond_records:
            lines.append("| " + " | ".join(
                ("%.4g" % r[k] if isinstance(r[k], float) else str(r[k]))
                for k in names) + " |")
        lines.append("")

    add(f"## {sec}. Findings\n")
    add("Curated interpretation of the tables above (regenerated reports "
        "do not carry the curation): see [`FINDINGS.md`](FINDINGS.md).\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# smoke (CI gate)
# ---------------------------------------------------------------------------

def run_smoke(device: str, kernel: KernelFunctions) -> dict:
    verdict: dict = {"checks": {}, "thresholds": {}}
    ok = True

    # synthetic order-extraction check (pure numpy, no GPU)
    xs = np.array([1.0, 0.5, 0.25, 0.125])
    res = observed_order(xs, 4.0 * xs ** 2)
    verdict["checks"]["synthetic_order"] = res.slope
    ok &= abs(res.slope - 2.0) < 1e-9

    case = build_case(288, 2, DEFAULT_TARGET_NEIGHBORS, jitter=DEFAULT_JITTER,
                      seed=7, periodic=False, device=device, kernel=kernel)
    cache = CorrectionCache(case)
    fields = {f.name: f for f in monomial_fields(2, degree_max=1)}
    f_const, f_lin = fields["1"], fields["x"]

    def linf(field, probe, mode, region):
        m = measure(case, cache, field, probe, mode, region)
        return m["norms"]["linf"] if m else float("nan")

    c1 = linf(f_const, "interpolate", "crk", "interior")
    c2 = linf(f_lin, "interpolate", "crk", "interior")
    c3 = linf(f_lin, "gradient", "renorm", "interior")
    c4 = linf(f_lin, "gradient", "standard", "interior")
    c5 = linf(f_const, "interpolate", "standard", "interior")
    c6 = linf(f_const, "interpolate", "standard", "boundary")
    _C, eigvals, _ = cache.renorm()
    cond_mean = float(condition_numbers(eigvals)[case.interior_mask].mean())

    checks = {
        "crk_interp_const_interior_linf": c1,          # ~1e-15
        "crk_interp_linear_interior_linf": c2,         # ~1e-15
        "renorm_grad_linear_interior_linf": c3,        # ~1e-15
        "standard_grad_linear_interior_linf": c4,      # ~1e-1 (NOT exact)
        "standard_interp_const_interior_linf": c5,     # ~1e-2
        "standard_interp_const_boundary_linf": c6,     # > interior
        "renorm_cond_interior_mean": cond_mean,        # O(1)
    }
    thresholds = {
        "crk_interp_const_interior_linf_max": 1e-10,
        "crk_interp_linear_interior_linf_max": 1e-10,
        "renorm_grad_linear_interior_linf_max": 1e-10,
        "standard_grad_linear_interior_linf_min": 1e-3,
        "standard_interp_const_interior_linf_min": 1e-4,
        "standard_interp_const_boundary_gt_interior": True,
        "renorm_cond_interior_mean_max": 10.0,
        "synthetic_order": 2.0,
    }
    verdict["checks"].update(checks)
    verdict["thresholds"] = thresholds
    ok &= c1 < 1e-10 and c2 < 1e-10 and c3 < 1e-10
    ok &= c4 > 1e-3 and c5 > 1e-4 and c6 > c5
    ok &= np.isfinite(cond_mean) and cond_mean < 10.0
    verdict["ok"] = bool(ok)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "smoke_verdict.json"
    out.write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))
    return verdict


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Convergence harness baseline driver")
    p.add_argument("--smoke", action="store_true",
                   help="run the minimal CI-gate matrix and exit")
    p.add_argument("--dim", type=int, default=2, choices=(1, 2, 3))
    p.add_argument("--kernels", nargs="+", default=["Wendland2"],
                   help="KernelFunctions member name(s)")
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    p.add_argument("--degree-max", type=int, default=4)
    p.add_argument("--suites",
                   default="patch,resolve-open,resolve-periodic,smoothing,"
                           "conditioning")
    p.add_argument("--out", default=str(HERE))
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wp.init()
    torch.set_grad_enabled(False)

    if args.smoke:
        return 0 if run_smoke(device, KernelFunctions[args.kernels[0]])["ok"] \
            else 1

    out_root = Path(args.out)
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    all_rows: list[Row] = []
    cond_records: list[dict] = []
    orders: dict[str, list[dict]] = {}
    xattr: dict[str, str] = {}
    suite_rows: dict[str, list[Row]] = {}

    for kernel_name in args.kernels:
        kernel = KernelFunctions[kernel_name]
        tag = kernel_name.lower()
        for suite in suites:
            t0 = time.time()
            if suite == "patch":
                rows = suite_patch(kernel, args, device)
            elif suite == "resolve-open":
                rows = suite_resolve(kernel, args, device, periodic=False)
            elif suite == "resolve-periodic":
                rows = suite_resolve(kernel, args, device, periodic=True)
            elif suite == "smoothing":
                rows = suite_smoothing(kernel, args, device)
            elif suite == "conditioning":
                records = suite_conditioning(
                    kernel, args, device,
                    out_root / "results" / f"cond_{tag}.csv")
                cond_records += records
                print(f"[{tag}] {suite}: {len(records)} records in "
                      f"{time.time() - t0:.1f}s", flush=True)
                continue
            else:
                raise SystemExit(f"unknown suite {suite!r}")
            key = f"{suite}/{tag}"
            for r in rows:
                r.suite = suite
            suite_rows[key] = rows
            all_rows += rows
            if suite in ("resolve-open", "resolve-periodic"):
                orders[key] = orders_for(rows, "dx")
                xattr[key] = "dx"
            elif suite == "smoothing":
                orders[key] = orders_for(rows, "h")
                xattr[key] = "h"
            print(f"[{tag}] {suite}: {len(rows)} rows in "
                  f"{time.time() - t0:.1f}s", flush=True)

    write_rows_csv(all_rows, out_root / "results" / "baseline_rows.csv")
    for key, rows in suite_rows.items():
        x = xattr.get(key, "dx")
        base = key.split("/")[0]
        if base.startswith("resolve"):
            plot_error_curves(
                rows, out_root / "figures" / f"convergence_{key}.png",
                xattr=x, title=f"error vs {x} -- {key}")
        elif base == "smoothing":
            plot_error_curves(
                rows, out_root / "figures" / f"convergence_smoothing_{key}.png",
                xattr=x, title=f"error vs {x} (N fixed) -- {key}")

    patch_rows = [r for key, rows in suite_rows.items()
                  if key.startswith("patch/") for r in rows]
    write_report(out_root / "REPORT.md", patch_rows=patch_rows,
                 orders=orders, cond_records=cond_records,
                 kernel_names=", ".join(args.kernels), xattr=xattr)
    print(f"wrote {out_root / 'REPORT.md'} and "
          f"{out_root / 'results' / 'baseline_rows.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
