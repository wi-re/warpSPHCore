#!/usr/bin/env python3
"""CRKSPH limiter-constant sweep over the frontend's compressible cases.

Runs every case under CRKSPH with the limiter constants the frontend
currently hard-codes, (eta_crit, eta_fold) = (1/3, 0.2), and with the
paper-consistent constants derived per case by `derive_crk_limiter.py`
(eta_crit = dx/H, eta_fold = 0.2 dx/H, 'spacing' convention), and reports
per case:

* common: total-energy drift, final KE change, `ke_rebound` (KE rise above
  its running minimum / KE(0); flagged for the steady cases), divergence;
* case-specific accuracy where a reference exists -- Sod: the ripple probe
  (`run_sod_ripple.ripple_metrics`); Sedov / Noh: density vs the exact
  solution; Gresho / Yee: velocity vs the exact steady field; linearWave:
  velocity vs analytic; hydrostatic: spurious velocity (exact v = 0);
* for every case: how far the setting moves the final state -- per-particle
  (Lagrangian) relative L1 density difference and RMS position difference
  (in units of dx) between the two runs (same IC, same ordering).

Budget-sized resolutions / end times (the goal is a like-for-like A/B, not
the cases' production settings); see `SWEEP`.

Usage:
    python run_crk_limiter_sweep.py                  # all cases
    python run_crk_limiter_sweep.py --cases sod gresho
"""

from __future__ import annotations

import os
os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
RESULTS = HERE / "results"

import derive_crk_limiter as D      # noqa: E402

# case -> spec overrides for the A/B (on top of derive_crk_limiter.CASES),
#         whether KE must be non-increasing (unforced steady / decaying flow)
SWEEP = {
    "sod":             (dict(nx=800, tLimit=0.15), False),
    "sedov":           (dict(nx=400), False),
    "noh":             (dict(nx=200), False),
    "kidder":          (dict(nx=100), False),
    "woodwardColella": (dict(nx=1000), False),
    "linearWave":      (dict(nx=200), False),
    "gresho":          (dict(nx=64, tLimit=3.0), True),
    "yeeVortex":       (dict(tLimit=2.0), True),
    "kelvinHelmholtz": (dict(nx=64, tLimit=2.0), False),
    "hydrostatic":     (dict(nx=64, tLimit=2.0), False),
    "shearingNoh":     (dict(nx=64, tLimit=0.6), False),
    "triplePoint":     (dict(nx=64, tLimit=2.0), False),
    "rayleighTaylor":  (dict(nx=64, tLimit=2.0), False),
    "sod2d":           (dict(nx=100, tLimit=0.15), False),
}
CURRENT = D.CURRENT


# ---------------------------------------------------------------------------
# case-specific accuracy metrics
# ---------------------------------------------------------------------------

def _vw_l1(field, exact, st):
    from field_error import particle_error_norms
    return particle_error_norms(field, exact, st.masses / st.densities)


def accuracy(name: str, st, t: float, params: dict, res) -> dict:
    """Case-specific error vs a reference, where one exists."""
    import pde_cases as PC
    x = st.positions
    if name == "sod":
        import run_sod_ripple as R
        m, _ = R.ripple_metrics(st, t, params)
        return m
    if name == "sedov":
        n = _vw_l1(st.densities, PC.sedov_exact_density(x, t, params), st)
        return {"l1_rho_exact": n["l1"], "l2_rho_exact": n["l2"]}
    if name == "noh":
        gamma, rho0, vs = (float(params[k]) for k in ("gamma", "rho0", "v_s"))
        rho_s = rho0 * ((gamma + 1) / (gamma - 1)) ** 1
        ex = torch.where(x[:, 0].abs() < vs * t, torch.full_like(x[:, 0], rho_s),
                         torch.full_like(x[:, 0], rho0))
        n = _vw_l1(st.densities, ex, st)
        return {"l1_rho_exact": n["l1"], "l2_rho_exact": n["l2"]}
    if name == "kidder":
        # exact self-similar Kidder solution (frontend caseUtils), evaluated
        # over the interior -- the outer bands are driven from it each step
        from warpSPH.caseUtils.compressible.kidder.bc import kidderDensity
        ctx = res.ctx
        ex = torch.as_tensor(kidderDensity(t, x, ctx.schemeConfig, ctx.scratch["solution"]),
                             dtype=x.dtype, device=x.device).reshape(-1)
        n = _vw_l1(st.densities, ex, st)
        return {"l1_rho_exact": n["l1"], "l2_rho_exact": n["l2"]}
    if name == "linearWave":
        from metrics import error_norms
        ex = PC.linearWave_analytic(x, t, params)
        return {"l2_v_analytic": error_norms(st.velocities, ex)["l2"]}
    if name == "gresho":
        n = _vw_l1(st.velocities, PC.gresho_exact_velocity(x, t, params), st)
        return {"l1_v_exact": n["l1"], "l2_v_exact": n["l2"]}
    if name == "yeeVortex":
        beta, xc, yc = (float(params[k]) for k in ("beta", "xc", "yc"))
        r2 = (x ** 2).sum(-1)
        term = beta / (2 * math.pi) * torch.exp((1 - r2) / 2)
        ex = torch.stack([term * (-(x[:, 1]) + yc), term * (x[:, 0] - xc)], -1)
        # the steady vortex lives in the interior; the buffer rings are driven
        core = r2.sqrt() < 4.0
        n = _vw_l1(st.velocities[core], ex[core], _Sub(st, core))
        return {"l1_v_exact_core": n["l1"], "l2_v_exact_core": n["l2"]}
    if name == "hydrostatic":
        v = st.velocities.norm(dim=-1)
        return {"max_speed": float(v.max()), "rms_speed": float(v.pow(2).mean().sqrt())}
    return {}


class _Sub:
    """Minimal masked view (masses / densities) for particle_error_norms."""
    def __init__(self, st, mask):
        self.masses = st.masses[mask]
        self.densities = st.densities[mask]


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def run_case(name: str, eta: tuple[float, float], device: str):
    from warpSPH.runner import run
    from conservation import ke_rebound
    case = D.load_case(name)
    inner = case.configureScheme

    def cfg(ctx):
        if inner is not None:
            inner(ctx)
        ctx.schemeConfig.crkViscosityParams.eta_crit = float(eta[0])
        ctx.schemeConfig.crkViscosityParams.eta_fold = float(eta[1])

    case2 = dataclasses.replace(case, configureScheme=cfg)
    over, _ = SWEEP[name]
    spec = D.build_spec(name, case2, nSteps=None, plot=False, show=False,
                        store=False, video=False, progress=False, quiet=True,
                        device=device, **over)
    t0 = time.perf_counter()
    res = run(case2, spec)
    wall = time.perf_counter() - t0
    st = res.state.state if hasattr(res.state, "state") else res.state
    t = float(res.state.t)
    ke = res.series("kineticEnergy")
    E = res.series("totalEnergy")
    out = dict(case=name, eta_crit=eta[0], eta_fold=eta[1], t_final=t,
               n_steps=int(res.nSteps), wall_s=round(wall, 1),
               diverged=bool(getattr(res, "diverged", False)),
               tot_e_drift=float((E[-1] - E[0]) / E[0]) if len(E) and E[0] else float("nan"),
               ke_final_rel=float(ke[-1] / ke[0] - 1) if len(ke) and ke[0] else float("nan"),
               ke_rebound=ke_rebound(ke) if len(ke) else float("nan"))
    try:
        params = {**dict(case.params), **dict(getattr(getattr(res, "ctx", None), "spec", spec).params or {})}
        out.update(accuracy(name, st, t, params, res))
    except Exception as e:
        out["accuracy_error"] = f"{type(e).__name__}: {str(e)[:100]}"
    snap = dict(rho=st.densities.detach().clone(), x=st.positions.detach().clone(),
                dx=float(((st.masses / st.densities) ** (1.0 / st.positions.shape[1])).median()))
    return out, snap


def compare(a: dict, b: dict) -> dict:
    """Lagrangian difference between two final states of the same IC."""
    if a["rho"].shape != b["rho"].shape:
        return {"diff_rho_rel_l1": float("nan"), "diff_x_rms_dx": float("nan")}
    drho = (a["rho"] - b["rho"]).abs().mean() / a["rho"].abs().mean()
    dxr = (a["x"] - b["x"]).norm(dim=-1).pow(2).mean().sqrt() / a["dx"]
    return {"diff_rho_rel_l1": float(drho), "diff_x_rms_dx": float(dxr)}


# the metric that carries the verdict for each case (lower is better unless
# noted); None = no reference -> judged on stability / energy / effect size
PRIMARY = {
    "sod": ["l1_rho", "plateau_u_std", "plateau_u_maxdev", "tail_u_overshoot",
            "tv_excess_rho", "entropy_err"],
    "sedov": ["l1_rho_exact", "l2_rho_exact"],
    "noh": ["l1_rho_exact", "l2_rho_exact"],
    "kidder": ["l1_rho_exact", "l2_rho_exact"],
    "linearWave": ["l2_v_analytic", "ke_final_rel"],
    "gresho": ["l1_v_exact", "ke_final_rel", "ke_rebound"],
    "yeeVortex": ["l1_v_exact_core", "ke_final_rel", "ke_rebound"],
    "hydrostatic": ["max_speed", "rms_speed"],
    "kelvinHelmholtz": ["ke_final_rel"],
}


def render(rows: list[dict]) -> str:
    """Markdown summary: one block per case, current vs derived."""
    by = {}
    for r in rows:
        by.setdefault(r["case"], {})[r.get("setting")] = r
    lines = ["# CRKSPH limiter constants: current (1/3, 0.2) vs derived "
             "(dx/H, 0.2 dx/H)", "",
             "`run_crk_limiter_sweep.py`; derived constants from "
             "`derive_crk_limiter.py`. ratio = derived / current for error "
             "metrics (< 1 = derived better).", "",
             "| case | metric | current | derived | ratio |",
             "|---|---|---|---|---|"]
    for case, d in by.items():
        c, v = d.get("current", {}), d.get("derived", {})
        if "error" in c or "error" in v:
            lines.append(f"| {case} | ERROR | {c.get('error', '')[:60]} | "
                         f"{v.get('error', '')[:60]} | |")
            continue
        keys = PRIMARY.get(case, []) + ["tot_e_drift", "ke_rebound"]
        seen = set()
        for k in keys:
            if k in seen or k not in c:
                continue
            seen.add(k)
            a, b = c[k], v.get(k, float("nan"))
            try:
                ratio = f"{b / a:.3g}" if a not in (0, None) and not math.isnan(a) else ""
            except TypeError:
                ratio = ""
            lines.append(f"| {case} | {k} | {a:.4g} | {b:.4g} | {ratio} |")
        diff = d.get("diff", {})
        if diff:
            lines.append(f"| {case} | effect: rel. L1 Δρ / RMS Δx (dx) | | "
                         f"{diff.get('diff_rho_rel_l1', float('nan')):.3g} / "
                         f"{diff.get('diff_x_rms_dx', float('nan')):.3g} | |")
        lines.append(f"| {case} | diverged (cur / der) | {c.get('diverged')} | "
                     f"{v.get('diverged')} | |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    import warp as wp
    wp.init()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", nargs="+", default=list(SWEEP), choices=list(SWEEP))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--derivation", default=str(RESULTS / "crk_limiter_derivation.json"))
    p.add_argument("--out", default=str(RESULTS / "crk_limiter_sweep.json"))
    p.add_argument("--report-only", action="store_true",
                   help="render REPORT from an existing --out JSON and exit")
    args = p.parse_args(argv)
    report = Path(args.out).with_name("REPORT_crk_limiter.md")
    if args.report_only:
        report.write_text(render(json.loads(Path(args.out).read_text())))
        print(report.read_text())
        return 0

    derived = {}
    if Path(args.derivation).exists():
        for r in json.loads(Path(args.derivation).read_text()):
            if "derived_spacing" in r:
                derived[r["case"]] = tuple(r["derived_spacing"])
    rows = []
    for name in args.cases:
        eta_d = derived.get(name)
        if eta_d is None:
            eta_d = tuple(D.measure(name, args.device)["derived_spacing"])
        pair = {}
        for label, eta in (("current", CURRENT), ("derived", eta_d)):
            try:
                r, snap = run_case(name, eta, args.device)
            except Exception as e:
                r, snap = dict(case=name, eta_crit=eta[0], eta_fold=eta[1],
                               error=f"{type(e).__name__}: {str(e)[:150]}"), None
            r["setting"] = label
            pair[label] = snap
            rows.append(r)
            print(json.dumps(r), flush=True)
        if pair.get("current") is not None and pair.get("derived") is not None:
            d = compare(pair["current"], pair["derived"])
            d.update(case=name, setting="diff")
            rows.append(d)
            print(json.dumps(d), flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2))
    report.write_text(render(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
