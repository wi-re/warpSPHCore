#!/usr/bin/env python3
"""Frozen-particle linear PDE leg of the convergence harness.

Integrates a linear PDE in time on a *fixed* (periodic, jittered) particle
set with the harness's own operator probes, so the spatial operator is the
only error source -- no pressure solve, no particle shifting, no moving
particles, no scheme-specific dissipation. Two equations:

* ``advection``  u_t + a . grad u = 0      (uses the `gradient` probe)
* ``diffusion``  u_t = nu lap u            (uses the `laplacian` probe)

both from a commensurate sinusoid ``u0 = sin(kappa . x)`` with the exact
solutions ``u0(x - a t)`` and ``exp(-nu |kappa|^2 t) u0(x)``.

What this adds over the static probes (`run_baseline.py`): error
*accumulation* and **stability** in time. A non-conservative / non-symmetric
high-order operator (Phase 4 MLS, Phase 5 LABFM) can have spectrum with a
positive real part, which a static patch test never sees and which blows up
here (the hyperviscosity question of `king_lind_2020_labfm`). The Pass-2 PDE
suite cannot answer this: its orders are capped at ~2 by floors unrelated to
the operator (`higher_order.md`, Verification pass finding 6).

Time integration is classical RK4 at ``dt = cfl * dx / |a|`` (advection) or
``dt = cfl * dx^2 / nu`` (diffusion), clipped to land exactly on `t_end`.
The RK4 error is O(dt^4); at the default CFL it sits well below an O(h^2)
spatial error, but a Phase 5 operator of order >= 4 needs a smaller `--cfl`
(the rows record dt so this is checkable). A run whose max |u| exceeds
`BLOWUP` x the initial amplitude is recorded as ``diverged`` (the stability
signal) rather than fitted.

Any mode works, including external ones registered via
`operators.register_mode` -- that is how Phase 4/5 operators enter this leg.

Usage:
    python run_frozen.py                              # both equations, all modes
    python run_frozen.py --equations advection --modes standard crk
    python run_frozen.py --smoke                      # CI gate (tiny)
"""

from __future__ import annotations

import os
os.environ.setdefault("warpSPHCore_PRECISION", "float64")  # before warpSPHCore

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RESULTS = HERE / "results"
REPORT = HERE / "REPORT_frozen.md"

BLOWUP = 1e3             # |u| growth factor that counts as a divergence
DEFAULT_LADDER = (576, 1152, 2304, 4608, 9216)
DEFAULT_TARGET_NEIGHBORS = 40
DEFAULT_JITTER = 0.3


# ---------------------------------------------------------------------------
# pure pieces (torch only; unit-tested without warp)
# ---------------------------------------------------------------------------

def wavevector(box, k=(1, 1)) -> torch.Tensor:
    """Commensurate wavevector kappa = 2 pi (k_d / L_d) for a periodic box."""
    return torch.tensor([2.0 * math.pi * kd / float(Ld)
                         for kd, Ld in zip(k, box)], dtype=torch.float64)


def exact_solution(equation: str, x: torch.Tensor, t: float,
                   kappa: torch.Tensor, a: torch.Tensor | None = None,
                   nu: float | None = None) -> torch.Tensor:
    """Exact solution at time `t` for the sinusoidal initial condition."""
    kappa = kappa.to(x)
    if equation == "advection":
        return torch.sin((x - a.to(x) * t) @ kappa)
    if equation == "diffusion":
        return math.exp(-nu * float(kappa @ kappa) * t) * torch.sin(x @ kappa)
    raise ValueError(equation)


def rk4(rhs, u0: torch.Tensor, t_end: float, dt_max: float,
        blowup: float = BLOWUP) -> tuple[torch.Tensor, int, bool]:
    """Classical RK4 from 0 to `t_end` with the largest uniform step
    <= `dt_max` that lands exactly on `t_end`. Stops early (diverged=True)
    when max|u| exceeds `blowup` x max|u0| or turns non-finite."""
    n = max(1, math.ceil(t_end / dt_max - 1e-12))
    dt = t_end / n
    u = u0.clone()
    amp0 = float(u0.abs().max())
    for step in range(n):
        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2)
        k4 = rhs(u + dt * k3)
        u = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        amp = float(u.abs().max())
        if not math.isfinite(amp) or amp > blowup * amp0:
            return u, step + 1, True
    return u, n, False


@dataclass
class FrozenRow:
    equation: str
    mode: str
    N: int
    dim: int
    dx: float
    h: float
    h_over_dx: float
    jitter: float
    dt: float
    n_steps: int
    t_end: float
    diverged: bool
    error_l2: float
    error_linf: float
    amp_final: float        # max|u| at t_end (vs exact amplitude below)
    amp_exact: float
    wall_s: float


# ---------------------------------------------------------------------------
# driver (needs warp / warpSPHCore)
# ---------------------------------------------------------------------------

def run_one(n: int, equation: str, mode: str, args, device: str) -> FrozenRow:
    from particle_sets import build_case
    from operators import CorrectionCache, run_probe

    case = build_case(n, 2, args.target_neighbors, jitter=args.jitter,
                      seed=args.seed, periodic=True, device=device)
    cache = CorrectionCache(case)
    x = case.positions
    kappa = wavevector(case.box, k=tuple(args.k)).to(x)
    a = torch.tensor(args.velocity, dtype=x.dtype, device=x.device)

    if equation == "advection":
        def rhs(u):
            g = run_probe(case, cache, u.contiguous(), "gradient", mode)
            return -(g @ a)
        dt_max = args.cfl * case.dx / float(a.norm())
    else:
        def rhs(u):
            return args.nu * run_probe(case, cache, u.contiguous(),
                                       "laplacian", mode)
        dt_max = args.cfl_diffusion * case.dx ** 2 / args.nu

    u0 = exact_solution(equation, x, 0.0, kappa, a, args.nu)
    t0 = time.perf_counter()
    u, steps, diverged = rk4(rhs, u0, args.t_end, dt_max)
    if x.is_cuda:
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    ue = exact_solution(equation, x, args.t_end, kappa, a, args.nu)
    err = (u - ue)
    return FrozenRow(
        equation=equation, mode=mode, N=case.N, dim=case.dim, dx=case.dx,
        h=case.h, h_over_dx=case.h_over_dx, jitter=case.jitter,
        dt=args.t_end / max(1, math.ceil(args.t_end / dt_max - 1e-12)),
        n_steps=steps, t_end=args.t_end, diverged=diverged,
        error_l2=float(err.pow(2).mean().sqrt()) if not diverged else float("nan"),
        error_linf=float(err.abs().max()) if not diverged else float("nan"),
        amp_final=float(u.abs().max()), amp_exact=float(ue.abs().max()),
        wall_s=round(wall, 2),
    )


def render(rows: list[FrozenRow], args) -> str:
    from metrics import observed_order
    lines = ["# Frozen-particle linear PDE leg", "",
             f"Periodic 2D densest-packing lattice, jitter {args.jitter}, "
             f"target neighbors {args.target_neighbors}, Wendland2, float64, "
             f"RK4; u0 = sin(kappa . x), k = {tuple(args.k)}, "
             f"a = {tuple(args.velocity)}, nu = {args.nu}, "
             f"t_end = {args.t_end}. Spatial operator = the only error "
             "source (`run_frozen.py`).", "",
             "## Rows", "",
             "| equation | mode | N | dx | dt | steps | diverged | err L2 | "
             "err Linf | max|u| / exact |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ratio = r.amp_final / r.amp_exact if r.amp_exact else float("nan")
        lines.append(
            f"| {r.equation} | {r.mode} | {r.N} | {r.dx:.4g} | {r.dt:.3g} | "
            f"{r.n_steps} | {r.diverged} | {r.error_l2:.3e} | "
            f"{r.error_linf:.3e} | {ratio:.4f} |")
    lines += ["", "## Observed orders (err L2 vs dx)", "",
              "| equation | mode | slope | r^2 | pairwise | note |",
              "|---|---|---|---|---|---|"]
    groups: dict = {}
    for r in rows:
        groups.setdefault((r.equation, r.mode), []).append(r)
    for (eq, mode), rs in groups.items():
        rs = sorted(rs, key=lambda r: r.N)
        if any(r.diverged for r in rs):
            lines.append(f"| {eq} | {mode} | n/a | n/a | | **diverged** at "
                         f"N = {', '.join(str(r.N) for r in rs if r.diverged)} |")
            continue
        if len(rs) < 2:
            continue
        o = observed_order([r.dx for r in rs], [r.error_l2 for r in rs])
        slope = "n/a" if o.saturated or o.exact else f"{o.slope:.2f}"
        r2 = "n/a" if o.saturated or o.exact else f"{o.r_squared:.3f}"
        note = "saturated" if o.saturated else ("exact" if o.exact else "")
        pw = ", ".join(f"{p:.2f}" for p in o.pairwise_slopes)
        lines.append(f"| {eq} | {mode} | {slope} | {r2} | {pw} | {note} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    import warp as wp
    wp.init()
    from operators import MODES
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--equations", nargs="+", default=["advection", "diffusion"],
                   choices=["advection", "diffusion"])
    p.add_argument("--modes", nargs="+", default=list(MODES))
    p.add_argument("--ladder", nargs="+", type=int, default=list(DEFAULT_LADDER))
    p.add_argument("--target-neighbors", type=int, default=DEFAULT_TARGET_NEIGHBORS)
    p.add_argument("--jitter", type=float, default=DEFAULT_JITTER)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", nargs=2, type=int, default=[1, 1])
    p.add_argument("--velocity", nargs=2, type=float, default=[1.0, 0.5])
    p.add_argument("--nu", type=float, default=0.05)
    p.add_argument("--t-end", type=float, default=0.5)
    p.add_argument("--cfl", type=float, default=0.2)
    p.add_argument("--cfl-diffusion", type=float, default=0.1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--smoke", action="store_true",
                   help="tiny CI run: 2-point ladder, t_end 0.05, writes "
                        "*_smoke artifacts")
    args = p.parse_args(argv)
    if args.smoke:
        args.ladder = [288, 576]
        args.t_end = 0.05

    rows = []
    for eq in args.equations:
        for mode in args.modes:
            for n in args.ladder:
                r = run_one(n, eq, mode, args, args.device)
                rows.append(r)
                print(f"[{eq}/{mode}] N={r.N} dx={r.dx:.4g} steps={r.n_steps} "
                      f"err_l2={r.error_l2:.3e} diverged={r.diverged} "
                      f"wall={r.wall_s}s", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    import csv
    with (RESULTS / f"frozen_rows{suffix}.csv").open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(rows[0])))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    report = HERE / f"REPORT_frozen{suffix}.md"
    report.write_text(render(rows, args))
    verdict = {"ok": all(math.isfinite(r.error_l2) or r.diverged for r in rows),
               "n_rows": len(rows),
               "diverged": [f"{r.equation}/{r.mode}/{r.N}" for r in rows if r.diverged]}
    (RESULTS / f"frozen_verdict{suffix}.json").write_text(json.dumps(verdict, indent=2))
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
