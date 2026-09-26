#!/usr/bin/env python3
"""CRK Sod ripple probe -- run alongside every CRKSPH limiter / viscosity /
pair-force change (the CRK limiter is delicate: tweaks that cure one case
easily add ripple elsewhere), and at the end Sod should match Frontiere et
al. 2017, Fig. 4 as closely as possible.

Setup = the paper's Sod (their Sec. 4.3.1): gamma = 5/3, (rho, P) =
(1, 1) | (0.25, 0.1795), equal-mass particles (400, 100) per tube, t = 0.15.
The frontend case is two mirrored tubes in a periodic [-1, 1] box; nx=800
gives exactly (400, 100) per tube. CRKSPH runs with symmetric support
(KernelMeanSymmetric -- its energy conservation needs it) and no viscosity
switch.

Metrics (single-tube coordinate s = |x|, interface at s = 0.5; tube
velocity u = v * sign(x); exact solution from the frontend's Riemann solver):

* `l1_rho`, `l1_u`, `l1_P` -- volume-weighted L1 vs exact;
* `plateau_u_std`, `plateau_u_maxdev` -- velocity ringing in the post-shock
  plateau between contact and shock (exact u is constant there; the paper's
  "post-shock ringing");
* `tail_u_overshoot` -- max (u - u_plateau) at the rarefaction's trailing
  edge (the paper notes a slight CRKSPH overshoot there);
* `entropy_err` -- mean |A - A_exact| / A_exact, A = P / rho^gamma, in the
  shocked region (the paper's "superior entropy evolution");
* `tv_excess_rho` -- total variation of rho(s) minus the exact profile's
  (spurious oscillation measure, 0 for a monotone profile);
* `shock_width` -- particles across the 10-90 % density rise at the shock;
* `tot_e_drift` -- total-energy drift (must stay at round-off).

Writes `results/sod_ripple_<label>.json` and a Fig.-4-style plot
`figures/sod_ripple_<label>.png` (rho, P, u, entropy vs s, exact overlaid).

Usage:
    python run_sod_ripple.py --label baseline
    python run_sod_ripple.py --scheme CompSPH --support Gather --label std
"""

from __future__ import annotations

import os
os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"


def _exact(t: float, params: dict, npts: int = 200001):
    from warpSPH.caseUtils.compressible.sod.sodSolution import solve
    left = (float(params["left_pressure"]), float(params["left_rho"]),
            float(params["left_velocity"]))
    right = (float(params["right_pressure"]), float(params["right_rho"]),
             float(params["right_velocity"]))
    return solve(left_state=left, right_state=right, geometry=(0.0, 1.0, 0.5),
                  t=float(t), gamma=float(params["gamma"]), npts=npts)


def ripple_metrics(state, t: float, params: dict) -> tuple[dict, dict]:
    """Ripple / accuracy metrics of a mirrored-tube Sod state (see module
    docstring). Returns (metrics, arrays-for-plotting)."""
    gamma = float(params["gamma"])
    x = state.positions[:, 0].detach().cpu().double().numpy()
    v = state.velocities[:, 0].detach().cpu().double().numpy()
    rho = state.densities.detach().cpu().double().numpy()
    m = state.masses.detach().cpu().double().numpy()
    uint = state.internalEnergies.detach().cpu().double().numpy()
    P = (gamma - 1.0) * rho * uint
    s = np.abs(x)
    u = v * np.sign(x)
    order = np.argsort(s)
    s, u, rho, P, m = s[order], u[order], rho[order], P[order], m[order]
    V = m / rho

    pos, reg, val = _exact(t, params)
    xs = np.asarray(val["x"])
    rho_e = np.interp(s, xs, val["rho"])
    u_e = np.interp(s, xs, val["u"])
    P_e = np.interp(s, xs, val["p"])
    l1 = lambda a, b: float((V * np.abs(a - b)).sum() / V.sum())

    x_foot = pos["Foot of Rarefaction"]
    x_c = pos["Contact Discontinuity"]
    x_s = pos["Shock"]
    P4, rho4, u4 = (float(q) for q in reg["Region 4"])
    A4 = P4 / rho4 ** gamma
    dpost = float(np.median(V[(s > x_c) & (s < x_s)]))       # plateau spacing
    plateau = (s > x_c + 4 * dpost) & (s < x_s - 4 * dpost)
    tail = (s > x_foot - 4 * dpost) & (s < x_c - 2 * dpost)
    A = P / rho ** gamma

    tv = lambda a: float(np.abs(np.diff(a)).sum())
    near = (s > 0.2) & (s < 0.85)            # the whole wave pattern
    fine = (xs > 0.2) & (xs < 0.85)
    tv_excess = tv(rho[near]) - tv(np.asarray(val["rho"])[fine])

    rho5 = float(params["right_rho"])
    lo, hi = rho5 + 0.1 * (rho4 - rho5), rho5 + 0.9 * (rho4 - rho5)
    band = (s > x_c) & (s < x_s + 10 * dpost) & (rho > lo) & (rho < hi)

    metrics = dict(
        l1_rho=l1(rho, rho_e), l1_u=l1(u, u_e), l1_P=l1(P, P_e),
        plateau_u_std=float(np.std(u[plateau] - u4)),
        plateau_u_maxdev=float(np.max(np.abs(u[plateau] - u4))),
        tail_u_overshoot=float(np.max(u[tail] - u4)) if tail.any() else float("nan"),
        entropy_err=float(np.mean(np.abs(A[plateau] - A4)) / A4),
        tv_excess_rho=tv_excess,
        shock_width=int(band.sum()),
        n_plateau=int(plateau.sum()),
    )
    arrays = dict(s=s, rho=rho, u=u, P=P, A=A, xs=xs, val=val, A4=A4,
                  gamma=gamma)
    return metrics, arrays


def plot(arrays: dict, label: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = arrays
    val, xs, g = a["val"], a["xs"], a["gamma"]
    A_e = np.asarray(val["p"]) / np.asarray(val["rho"]) ** g
    fig, ax = plt.subplots(4, 1, figsize=(6, 10), sharex=True)
    for axi, sim, ex, name in ((ax[0], a["rho"], val["rho"], "density"),
                               (ax[1], a["P"], val["p"], "pressure"),
                               (ax[2], a["u"], val["u"], "velocity"),
                               (ax[3], a["A"], A_e, "entropy P/rho^g")):
        axi.plot(xs, ex, "r-", lw=1, label="exact")
        axi.plot(a["s"], sim, "k.", ms=2, label=label)
        axi.set_ylabel(name)
    ax[0].legend(loc="upper right", fontsize=8)
    ax[3].set_xlim(0.2, 0.85)
    ax[3].set_xlabel("s = |x| (interface at 0.5)")
    fig.suptitle(f"Sod t=0.15 -- {label}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run(scheme: str = "CRKSPH", support: str = "KernelMeanSymmetric",
        nx: int = 800, t_end: float = 0.15, device: str = "cuda:0",
        case=None) -> tuple[dict, dict]:
    """Run the Sod case and return (metrics, arrays). `case` lets an
    experiment pass a modified Case (e.g. a wrapped configureScheme)."""
    from warpSPH.runner import CaseSpec, run as run_case
    from warpSPH.cases.sod import sodCase
    case = case or sodCase
    spec = CaseSpec(caseName="sod", scheme=scheme,
                    params=dict(case.params)).merged(**case.defaults)
    spec = spec.merged(nx=nx, tLimit=t_end, nSteps=None, supportMode=support,
                       plot=False, show=False, store=False, video=False,
                       progress=False, quiet=True, device=device)
    res = run_case(case, spec)
    st = res.state.state if hasattr(res.state, "state") else res.state
    metrics, arrays = ripple_metrics(st, float(res.state.t), dict(case.params))
    E = res.series("totalEnergy")
    metrics["tot_e_drift"] = float((E[-1] - E[0]) / E[0]) if len(E) else float("nan")
    metrics["t_final"] = float(res.state.t)
    metrics["n_steps"] = int(res.nSteps)
    return metrics, arrays


def main(argv=None):
    import warp as wp
    wp.init()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scheme", default="CRKSPH")
    p.add_argument("--support", default="KernelMeanSymmetric")
    p.add_argument("--nx", type=int, default=800)
    p.add_argument("--label", default="baseline")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args(argv)
    metrics, arrays = run(args.scheme, args.support, args.nx, device=args.device)
    metrics.update(scheme=args.scheme, support=args.support, nx=args.nx,
                   label=args.label)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"sod_ripple_{args.label}.json").write_text(json.dumps(metrics, indent=2))
    plot(arrays, f"{args.scheme} {args.label}", FIGURES / f"sod_ripple_{args.label}.png")
    print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
