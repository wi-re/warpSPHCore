#!/usr/bin/env python3
"""Pass-2 PDE benchmark driver: multi-resolution convergence + conservation.

Runs each registered `warpSPH` case at a 4-point resolution ladder (lean
budget), measures the field error (vs analytic for smooth closed-form cases,
vs a high-res reference for the rest) and the conservation drift
(mass / momentum / KE / angular momentum), and writes a standardized
`results/pde_rows.csv` + `REPORT_pde.md` reusing the static harness's
`metrics.observed_order` and `report` helpers so the output is directly
comparable to the Phase 1-3 static baseline.

float64 is required (float32's ~1e-7 floor caps the observable order), so the
precision env var is set *before* importing warpSPH, mirroring run_baseline.py.

Usage:
    python run_pde.py --cases tgv            # one case
    python run_pde.py                        # all registered cases
    python run_pde.py --cases tgv --smoke    # CI gate (lowest resolution)
"""

from __future__ import annotations

import os
os.environ["warpSPHCore_PRECISION"] = "float64"   # must precede warpSPH import

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # harness dir: metrics, report, ...
sys.path.insert(0, str(HERE))          # pde dir: pde_cases, conservation

import warp as wp
wp.init()

from warpSPH.runner import CaseSpec, run            # noqa: E402

from conservation import (ConservedQuantities, conserved,
                          drift)                     # noqa: E402
from field_error import grid_l2_error                # noqa: E402
from pde_cases import CASES, load_case_entry        # noqa: E402
from metrics import error_norms, observed_order     # noqa: E402

RESULTS = HERE / "results"
REPORT = HERE / "REPORT_pde.md"


def build_spec(case, entry, nx: int, device: str) -> CaseSpec:
    spec = CaseSpec(caseName=entry.name, scheme=case.scheme,
                    params=dict(case.params))
    spec = spec.merged(**case.defaults)
    return spec.merged(
        nx=nx, tLimit=entry.t_star, nSteps=None,
        plot=False, show=False, store=False, video=False,
        progress=False, quiet=True, device=device,
    )


def run_one(case, entry, nx: int, device: str) -> dict:
    spec = build_spec(case, entry, nx, device)
    t0 = time.perf_counter()
    res = run(case, spec)
    wall = time.perf_counter() - t0

    state = res.state.state if hasattr(res.state, "state") else res.state
    t_final = float(getattr(res.state, "t", entry.t_star))
    # internalEnergies is None for incompressible states -> conserved() takes
    # that as IE = 0 (so total energy == KE there).
    final_c = conserved(state.masses, state.velocities, state.positions,
                        getattr(state, "internalEnergies", None))

    # Initial conserved quantities from the t=0 trajectory entry (scalars) +
    # the final mass (mass is exactly conserved, masses are never modified).
    # The t=0 entry carries the initial KE (all cases) and, for compressible
    # cases, the initial thermal (internal) energy. Momentum / angular
    # momentum start ~0 for the symmetric ICs, so the *final* norms are the
    # spurious-growth signal (a relative drift would divide by ~0). This
    # avoids a second (wasted) system build for the initial state.
    traj0 = res.trajectory[0] if res.trajectory else {}
    ke_0 = float(traj0.get("kineticEnergy", final_c.kinetic_energy))
    ie_0 = float(traj0.get("thermalEnergy", 0.0))
    init_c = ConservedQuantities(
        mass=final_c.mass,
        momentum=torch.zeros_like(final_c.momentum),
        momentum_norm=0.0,
        kinetic_energy=ke_0,
        angular_momentum=torch.zeros(1, dtype=final_c.momentum.dtype),
        angular_momentum_norm=0.0,
        internal_energy=ie_0,
    )
    drifts = drift(init_c, final_c)
    drifts["mass_final"] = final_c.mass

    row = {
        "case": entry.name, "nx": nx,
        "N": int(state.positions.shape[0]),
        "dim": entry.dim,
        "dx": float(spec.L / nx),
        "n_h": float(spec.n_h),
        "t_final": t_final,
        "n_steps": int(res.nSteps),
        "wall_s": round(wall, 2),
        "diverged": bool(res.diverged),
    }
    row.update(drifts)

    # field error vs analytic (smooth closed-form cases); the reference metric
    # is computed in main after the whole ladder has run (it needs the finest
    # run's state as the reference).
    if entry.metric == "analytic" and entry.analytic is not None:
        analytic = entry.analytic(state.positions, t_final, case.params)
        sim = getattr(state, entry.field)
        row["error_l2"] = error_norms(sim, analytic)["l2"]
        row["error_linf"] = error_norms(sim, analytic)["linf"]

    return row, state


def apply_reference_metric(rows: list[dict], states: dict, entry, case) -> None:
    """Fill `error_l2` for a reference-metric case in place.

    The finest ladder run is the reference; each coarser run's field is
    compared to it on a common grid (mass-weighted cell averages). The finest
    row is the reference itself (error 0) and is excluded from the order fit
    downstream by keeping only the 3 coarser points.
    """
    ladder = sorted(states)
    ref_nx = max(ladder)
    ref_state = states[ref_nx]
    L = float(case.defaults["L"])
    periodic = bool(case.defaults.get("periodic", False))
    # A grid several times finer than the reference makes the CIC projection
    # smooth and well-resolved, so the L2 error tracks the field difference.
    n_grid = 4 * ref_nx
    by_nx = {r["nx"]: r for r in rows}
    for nx in ladder:
        if nx == ref_nx:
            by_nx[nx]["error_l2"] = 0.0
            by_nx[nx]["error_linf"] = 0.0
            by_nx[nx]["is_reference"] = True   # excluded from the order fit
            continue
        st = states[nx]
        err = grid_l2_error(
            getattr(st, entry.field), st.masses, st.positions,
            getattr(ref_state, entry.field), ref_state.masses, ref_state.positions,
            L=L, dim=entry.dim, periodic=periodic, n_grid=n_grid,
        )
        by_nx[nx]["error_l2"] = err
        by_nx[nx]["error_linf"] = err


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
             "TotE drift | |p|f| |L|f| | wall |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", nargs="+", default=list(CASES),
                   choices=list(CASES))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--smoke", action="store_true",
                   help="run only the lowest resolution of each case (CI gate)")
    args = p.parse_args(argv)

    all_rows = []
    for name in args.cases:
        entry = CASES[name]
        case = load_case_entry(entry)
        ladder = [entry.nx_ladder[0]] if args.smoke else entry.nx_ladder
        rows = []
        states = {}
        for nx in ladder:
            print(f"[{name}] nx={nx} ...", flush=True)
            row, state = run_one(case, entry, nx, args.device)
            rows.append(row)
            states[nx] = state
            print(f"    N={row['N']} err_l2={row.get('error_l2', float('nan')):.3e} "
                  f"mass_drift={row['mass_drift']:+.2e} wall={row['wall_s']:.1f}s",
                  flush=True)
        # reference metric needs the finest run as the reference, so it is
        # computed only once the whole ladder has run (and needs >=2 points).
        if entry.metric == "reference" and len(ladder) >= 2:
            apply_reference_metric(rows, states, entry, case)
            for r in rows:
                tag = " (ref)" if r.get("is_reference") else ""
                print(f"    [{name}]{tag} nx={r['nx']} err_l2="
                      f"{r.get('error_l2', float('nan')):.3e}", flush=True)
        del states                       # free GPU tensors before the next case
        all_rows.extend(rows)

    orders = observed_orders(all_rows)
    REPORT.write_text(render_report(all_rows, orders))
    RESULTS.mkdir(parents=True, exist_ok=True)
    # For the PDE suite we write a plain flat CSV of the row dicts
    # (self-describing). The union of keys is used because analytic and
    # reference rows carry different optional columns.
    if all_rows:
        cols = []
        for r in all_rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        with (RESULTS / "pde_rows.csv").open("w") as fh:
            import csv as _csv
            w = _csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in all_rows:
                w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v)
                            for k, v in r.items()})
    print(f"wrote {REPORT} and {RESULTS / 'pde_rows.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
