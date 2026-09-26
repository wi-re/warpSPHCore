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
import dataclasses
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # harness dir: metrics, report, ...
sys.path.insert(0, str(HERE))          # pde dir: pde_cases, conservation

import warp as wp
wp.init()

from warpSPH.runner import CaseSpec, run            # noqa: E402

from conservation import conserved, drift          # noqa: E402
from field_error import (aligned_1d_l2_error, grid_error_p,  # noqa: E402
                         grid_l2_error, particle_error_norms)
from pde_cases import CASES, load_case_entry        # noqa: E402
from metrics import error_norms                      # noqa: E402
from report_pde import (merge_rows, observed_orders,  # noqa: E402
                        render_report)

RESULTS = HERE / "results"
REPORT = HERE / "REPORT_pde.md"


def build_spec(case, entry, nx: int, device: str) -> CaseSpec:
    spec = CaseSpec(caseName=entry.name, scheme=entry.scheme or case.scheme,
                    params={**dict(case.params), **entry.params})
    spec = spec.merged(**case.defaults)
    if entry.spec:
        spec = spec.merged(**entry.spec)
    return spec.merged(
        nx=nx, tLimit=entry.t_star, nSteps=None,
        plot=False, show=False, store=False, video=False,
        progress=False, quiet=True, device=device,
    )


def _capture_initial(case, box: dict):
    """Return a copy of `case` whose `extraData` hook also records the
    conserved quantities of the state it is first handed.

    The runner calls `extraData(ctx, runningState)` exactly once on the t=0
    state (after `initialConditions` and `initializeNewState`, before the
    first step); with store/plot off it is not called again. Piggy-backing on
    it gives the true initial momentum / angular momentum / mass without a
    second system build and without a frontend change. The case's own hook
    (if any) still runs and its return value is passed through.
    """
    inner = case.extraData

    def hook(ctx, system):
        if "init" not in box:
            # same unwrap as the final state in `run_one`
            st = system.state if hasattr(system, "state") else system
            box["init"] = conserved(st.masses, st.velocities, st.positions,
                                    getattr(st, "internalEnergies", None))
        return inner(ctx, system) if inner is not None else {}

    return dataclasses.replace(case, extraData=hook)


def run_one(case, entry, nx: int, device: str) -> dict:
    spec = build_spec(case, entry, nx, device)
    box: dict = {}
    t0 = time.perf_counter()
    res = run(_capture_initial(case, box), spec)
    wall = time.perf_counter() - t0

    state = res.state.state if hasattr(res.state, "state") else res.state
    t_final = float(getattr(res.state, "t", entry.t_star))
    # internalEnergies is None for incompressible states -> conserved() takes
    # that as IE = 0 (so total energy == KE there).
    final_c = conserved(state.masses, state.velocities, state.positions,
                        getattr(state, "internalEnergies", None))

    # Initial conserved quantities measured on the actual t=0 state (captured
    # by `_capture_initial`). Before 2026-09-26 the driver zero-filled the
    # initial momentum / angular momentum and copied the final mass, so the
    # old `momentum_norm_final` / `angmom_norm_final` columns were the
    # carried-through initial values (KH |p| ~ 0.23, Gresho |L| ~ 0.06), not
    # drift, and `mass_drift` was 0 by construction.
    init_c = box["init"]
    drifts = drift(init_c, final_c)
    drifts["mass_final"] = final_c.mass

    row = {
        "case": entry.name, "nx": nx,
        "N": int(state.positions.shape[0]),
        "dim": entry.dim,
        "dx": float(spec.L / nx),
        "n_h": float(spec.n_h),
        "scheme": entry.scheme or case.scheme,
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
    # exact solution of a reference case (Sod / Sedov), evaluated at this
    # run's own t_final: no finite-reference bias, no t_final mismatch, and
    # the finest rung gets a real error too.
    if entry.exact is not None:
        exact = entry.exact(state.positions, t_final,
                            {**dict(case.params), **entry.params})
        ex = particle_error_norms(getattr(state, entry.field), exact,
                                  state.masses / state.densities)
        row["error_l1_exact"] = ex["l1"]
        row["error_l2_exact"] = ex["l2"]

    return row, state


def apply_reference_metric(rows: list[dict], states: dict, entry, case) -> None:
    """Fill the reference-metric error columns for a case, in place.

    The finest ladder run is the reference; each coarser run's field is
    compared to it on a common grid (mass-weighted cell averages). The finest
    row is the reference itself (errors 0) and is excluded from the order fit
    downstream by keeping only the 3 coarser points.

    Columns: `error_l2` (and its `error_linf` stand-in, as before),
    `error_l1` (the shock-friendly area norm), and -- for 1D cases, where a
    translation is a well-defined alignment -- `error_l2_aligned` plus
    `align_shift` (the reference translation that best aligns the shock; a
    resolution-dependent shock position is the main L2 error source for
    shock cases, so the aligned metric isolates the shape error).
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
            by_nx[nx]["error_l1"] = 0.0
            by_nx[nx]["is_reference"] = True   # excluded from the order fit
            if entry.dim == 1:
                by_nx[nx]["error_l2_aligned"] = 0.0
                by_nx[nx]["align_shift"] = 0.0
            continue
        st = states[nx]
        err = grid_l2_error(
            getattr(st, entry.field), st.masses, st.positions,
            getattr(ref_state, entry.field), ref_state.masses, ref_state.positions,
            L=L, dim=entry.dim, periodic=periodic, n_grid=n_grid,
        )
        by_nx[nx]["error_l2"] = err
        by_nx[nx]["error_linf"] = err
        by_nx[nx]["error_l1"] = grid_error_p(
            getattr(st, entry.field), st.masses, st.positions,
            getattr(ref_state, entry.field), ref_state.masses, ref_state.positions,
            L=L, dim=entry.dim, periodic=periodic, n_grid=n_grid, p=1,
        )
        if entry.dim == 1:
            # 4x the coarse dx bounds the shock-position drift to search.
            aerr, shift = aligned_1d_l2_error(
                getattr(st, entry.field), st.masses, st.positions,
                getattr(ref_state, entry.field), ref_state.masses,
                ref_state.positions,
                L=L, periodic=periodic, n_grid=n_grid,
                max_shift=4.0 * float(by_nx[nx]["dx"]),
            )
            by_nx[nx]["error_l2_aligned"] = aerr
            by_nx[nx]["align_shift"] = shift


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

    # Smoke runs (the CI gate) write their own artifacts so they never
    # clobber the full-suite results file.
    report = HERE / ("REPORT_pde_smoke.md" if args.smoke else "REPORT_pde.md")
    csv_name = "pde_rows_smoke.csv" if args.smoke else "pde_rows.csv"
    if not args.smoke:
        all_rows = merge_rows(RESULTS / csv_name, all_rows, args.cases,
                              order=list(CASES))
    orders = observed_orders(all_rows)
    report.write_text(render_report(all_rows, orders))
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
        with (RESULTS / csv_name).open("w") as fh:
            import csv as _csv
            w = _csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in all_rows:
                w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v)
                            for k, v in r.items()})
    print(f"wrote {report} and {RESULTS / csv_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
