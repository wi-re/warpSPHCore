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
from metrics import observed_order          # noqa: E402


def observed_orders(rows: list[dict], xattr: str = "dx") -> dict:
    """Group rows by case and fit the observed order of error vs resolution."""
    orders = {}
    by_case = {}
    for r in rows:
        # the reference-metric case's finest row is the reference itself
        # (error 0 by self-comparison) -- exclude it from the fit.
        if r.get("is_reference"):
            continue
        by_case.setdefault(r["case"], []).append(r)
    for case, rs in by_case.items():
        rs = sorted(rs, key=lambda r: r["N"])
        x = np.array([r[xattr] for r in rs])
        y = np.array([r.get("error_l2", float("nan")) for r in rs])
        if len(rs) < 2 or np.any(np.isnan(y)):
            continue
        res = observed_order(x, y)
        orders[case] = {
            "slope": res.slope, "r_squared": res.r_squared,
            "saturated": res.saturated, "exact": res.exact,
            "pairwise": res.pairwise_slopes,
        }
    return orders


def render_report(rows: list[dict], orders: dict) -> str:
    lines = ["# PDE benchmark report (Pass 2)", "",
             "Multi-resolution convergence + conservation. float64, full t* "
             "overnight run.",
             "", "## Per-resolution rows", "",
             "| case | nx | N | dx | t | err L2 | mass drift | KE drift | "
             "TotE drift | |p|f| |L|f| | div | wall |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        tag = " (ref)" if r.get("is_reference") else ""
        lines.append(
            f"| {r['case']} | {r['nx']}{tag} | {r['N']} | {r['dx']:.4g} | "
            f"{r['t_final']:.3g} | {r.get('error_l2', float('nan')):.3e} | "
            f"{r['mass_drift']:+.2e} | {r['ke_drift']:+.2e} | "
            f"{r['total_energy_drift']:+.2e} | "
            f"{r['momentum_norm_final']:.3e} | {r['angmom_norm_final']:.3e} | "
            f"{r['diverged']} | {r['wall_s']:.1f} |")
    lines += ["", "## Observed orders (error L2 vs dx)", "",
              "| case | slope | r^2 | saturated | exact |",
              "|---|---|---|---|---|"]
    for case, o in orders.items():
        slope = "exact" if o["exact"] else ("n/a" if o["saturated"]
                                            else f"{o['slope']:.2f}")
        r2 = "n/a" if o["saturated"] or o["exact"] else f"{o['r_squared']:.3f}"
        lines.append(f"| {case} | {slope} | {r2} | {o['saturated']} | "
                     f"{o['exact']} |")
    return "\n".join(lines) + "\n"


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
