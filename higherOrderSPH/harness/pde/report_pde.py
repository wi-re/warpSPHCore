#!/usr/bin/env python3
"""Report rendering + order fitting for the PDE benchmark (Pass 2).

Pure numpy/torch -- no warp, no warpSPH -- so the report can be regenerated
from the flat `results/pde_rows.csv` without importing warp/warpSPH (which
`run_pde.py` needs for the actual runs). `run_pde.py` imports the two
functions below; a standalone call (or the smoke path) can regenerate
`REPORT_pde.md` from the CSV via `load_rows` + `render_report`.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:      # harness dir: metrics
    sys.path.insert(0, str(_HERE.parent))
from metrics import (observed_order,         # noqa: E402
                     reference_corrected_order)


def _as_num(v):
    """Coerce a row value to float; missing / blank / non-numeric -> nan
    (older CSV rows predate newer metric columns and read back as blanks)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def observed_orders(rows: list[dict], xattr: str = "dx",
                    yattr: str = "error_l2") -> dict:
    """Group rows by case and fit the observed order of the `yattr` error
    column vs `xattr`. Cases with fewer than two finite points (e.g. because
    an older CSV predates the metric) are omitted from the result."""
    orders = {}
    by_case = {}
    for r in rows:
        # the reference-metric case's finest row is the reference itself
        # (error 0 by self-comparison) -- exclude it from the fit. The
        # exact-solution columns are real errors on every rung, including it.
        if r.get("is_reference") and yattr not in EXACT_METRICS:
            continue
        by_case.setdefault(r["case"], []).append(r)
    for case, rs in by_case.items():
        rs = sorted(rs, key=lambda r: r["N"])
        x = np.array([r[xattr] for r in rs])
        y = np.array([_as_num(r.get(yattr)) for r in rs])
        if len(rs) < 2 or np.any(np.isnan(y)):
            continue
        res = observed_order(x, y)
        orders[case] = {
            "slope": res.slope, "r_squared": res.r_squared,
            "saturated": res.saturated, "exact": res.exact,
            "pairwise": res.pairwise_slopes,
        }
    return orders


def reference_corrected_orders(rows: list[dict],
                               yattr: str = "error_l2") -> dict:
    """Finite-reference-corrected order per reference-metric case (cases
    with an `is_reference` row), fitting ``e = C (h^p - h_ref^p)`` -- see
    `metrics.reference_corrected_order`. Analytic cases are skipped (they
    have no reference bias); cases whose `yattr` column is blank are
    omitted."""
    out = {}
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case"], []).append(r)
    for case, rs in by_case.items():
        ref = [r for r in rs if r.get("is_reference")]
        if not ref:
            continue
        pts = sorted((r for r in rs if not r.get("is_reference")),
                     key=lambda r: r["N"])
        x = np.array([r["dx"] for r in pts])
        y = np.array([_as_num(r.get(yattr)) for r in pts])
        if len(pts) < 2 or np.any(np.isnan(y)):
            continue
        res = reference_corrected_order(x, y, float(ref[0]["dx"]))
        out[case] = {
            "slope": res.slope, "r_squared": res.r_squared,
            "saturated": res.saturated, "exact": res.exact,
            "pairwise": res.pairwise_slopes,
        }
    return out


# error columns scored against an exact solution (no reference row to drop)
EXACT_METRICS = ("error_l1_exact", "error_l2_exact")

# extra error columns rendered as their own order tables when present
ALT_METRICS = (("error_l1", "error L1 (area norm) vs dx"),
               ("error_l2_aligned",
                "error L2 after optimal 1D shock alignment vs dx"),
               ("error_l1_exact",
                "error L1 vs the exact solution, volume-weighted, vs dx"),
               ("error_l2_exact",
                "error L2 vs the exact solution, volume-weighted, vs dx"))


def _order_table(lines: list[str], orders: dict) -> None:
    lines += ["| case | slope | r^2 | saturated | exact |",
              "|---|---|---|---|---|"]
    for case, o in orders.items():
        slope = "exact" if o["exact"] else ("n/a" if o["saturated"]
                                            else f"{o['slope']:.2f}")
        r2 = "n/a" if o["saturated"] or o["exact"] else f"{o['r_squared']:.3f}"
        lines.append(f"| {case} | {slope} | {r2} | {o['saturated']} | "
                     f"{o['exact']} |")


def _fmt_opt(v) -> str:
    """Format an optional numeric column; blank when the row predates it."""
    x = _as_num(v)
    return "" if np.isnan(x) else f"{x:.3e}"


def render_report(rows: list[dict], orders: dict) -> str:
    lines = ["# PDE benchmark report (Pass 2)", "",
             "Multi-resolution convergence + conservation. float64, full t* "
             "overnight run.",
             "", "## Per-resolution rows", "",
             "Drift columns: |Δp| / |ΔL| are the absolute drifts of the "
             "momentum / angular-momentum vectors from the measured t=0 "
             "state (blank for rows run before 2026-09-26, when the initial "
             "values were zero-filled; L is about the origin, so "
             "only an invariant on open domains).",
             "", "| case | scheme | nx | N | dx | t | err L2 | mass drift | "
             "KE drift | TotE drift | \\|Δp\\| | \\|ΔL\\| | div | wall |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        tag = " (ref)" if r.get("is_reference") else ""
        lines.append(
            f"| {r['case']} | {r.get('scheme') or '?'} | {r['nx']}{tag} | "
            f"{r['N']} | {r['dx']:.4g} | "
            f"{r['t_final']:.3g} | {r.get('error_l2', float('nan')):.3e} | "
            f"{r['mass_drift']:+.2e} | {r['ke_drift']:+.2e} | "
            f"{r['total_energy_drift']:+.2e} | "
            f"{_fmt_opt(r.get('momentum_drift_abs'))} | "
            f"{_fmt_opt(r.get('angmom_drift_abs'))} | "
            f"{r['diverged']} | {r['wall_s']:.1f} |")
    lines += ["", "## Observed orders (error L2 vs dx)", ""]
    _order_table(lines, orders)
    for yattr, title in ALT_METRICS:
        alt = observed_orders(rows, yattr=yattr)
        if alt:
            lines += ["", f"## Observed orders ({title})", ""]
            _order_table(lines, alt)
    _corrected_section(lines, rows)
    return "\n".join(lines) + "\n"


def _corrected_section(lines: list[str], rows: list[dict]) -> None:
    """Plain vs finite-reference-corrected orders for the reference-metric
    cases. The plain fit over-reads a systematic error when the reference is
    only 1.5-2x finer than the last rung (true p = 1 fits ~1.4 / ~2.0 on the
    1D / 2D ladders). A fit on the search bound is rendered as ``≤ p`` /
    ``≥ p``; see `metrics.reference_corrected_order` for how to read it."""
    table = []
    for yattr, label in (("error_l2", "L2"),) + tuple(
            (a, a.replace("error_", "").replace("_", " ")) for a, _ in
            ALT_METRICS if a not in EXACT_METRICS):
        plain = observed_orders(rows, yattr=yattr)
        corr = reference_corrected_orders(rows, yattr=yattr)
        for case, c in corr.items():
            pl = plain.get(case)
            ptxt = ("n/a" if pl is None or pl["saturated"] or pl["exact"]
                    else f"{pl['slope']:.2f}")
            ctxt = ("exact" if c["exact"] else
                    ((f"≤ {c['slope']:.2f}" if c["slope"] < 1.0
                      else f"≥ {c['slope']:.2f}") if c["saturated"]
                     else f"{c['slope']:.2f}"))
            r2 = ("n/a" if c["exact"] or np.isnan(c["r_squared"])
                  else f"{c['r_squared']:.3f}")
            table.append(f"| {case} | {label} | {ptxt} | {ctxt} | {r2} |")
    if not table:
        return
    lines += ["", "## Finite-reference-corrected orders (reference-metric "
              "cases)", "",
              "Fit of e = C (h^p - h_ref^p) against the finest-ladder "
              "reference; assumes a systematic error (same sign structure "
              "at every level). A true first-order scheme reads ~1.4 "
              "(200/400/800 vs 1600) or ~2.0 (32/48/64 vs 96) in the plain "
              "fit. **Unreliable when the error changes sign between rungs**: "
              "validated against exact solutions it recovers Sod (L1 "
              "corrected 1.07 vs exact 0.98) but over-corrects Sedov "
              "(0.23 vs exact 1.03) -- prefer the exact-solution tables "
              "where they exist. `≤ 0.05` = fit on the lower search bound: with a low "
              "r^2 the model does not describe the data (use the plain fit); "
              "with a high r^2 the convergence is slower than any resolvable "
              "power law.", "",
              "| case | metric | plain | corrected | r^2 (corrected) |",
              "|---|---|---|---|---|"]
    lines += table


def _coerce(s: str):
    """Coerce a flat-CSV string back to bool/int/float/str. The writer emits
    floats as `.10g` strings and everything else verbatim, so the type is
    recoverable: True/False -> bool, then int, then float, else str."""
    if s in ("True", "False"):
        return s == "True"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def load_rows(csv_path: str | Path) -> list[dict]:
    """Load the flat `pde_rows.csv` back into row dicts (inverse of the
    writer in `run_pde.main`), with numeric/bool columns coerced to their
    original types."""
    rows = []
    with open(csv_path) as fh:
        for rec in csv.DictReader(fh):
            rows.append({k: _coerce(v) for k, v in rec.items()})
    return rows


def merge_rows(csv_path: str | Path, new_rows: list[dict],
               run_cases: list[str], order: list[str]) -> list[dict]:
    """Merge a (partial) run into the existing full-suite CSV: rows of the
    cases just run replace their old rows, every other case's rows are kept
    (a `--cases sod` run used to overwrite the whole file with Sod only).
    Output is ordered by `order` (the case registry), then by row order."""
    csv_path = Path(csv_path)
    old = load_rows(csv_path) if csv_path.exists() else []
    run = set(run_cases)
    merged = [r for r in old if r["case"] not in run] + list(new_rows)
    rank = {name: i for i, name in enumerate(order)}
    return sorted(merged, key=lambda r: rank.get(r["case"], len(rank)))
