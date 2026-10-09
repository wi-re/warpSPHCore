#!/usr/bin/env python3
"""Phase 6/7 groundwork: smooth-region order of the MLS interface-state
reconstruction (`RKPMOperator.interface_states`).

Left/right states of a smooth periodic field at every pair midpoint, from the
local polynomial fits of order p (MLS, ``mls<p>``) and the LABFM variant
(``labfm<k>``), on the jittered periodic set; error vs the exact midpoint
value, and the L/R jump (what a Riemann solver would see as a spurious
discontinuity in smooth flow). Pure torch + the harness case builder; no
solver, no TENO/WENO selection (those need the Riemann-SPH scheme choice).

    python run_interface.py
"""

from __future__ import annotations

import os
os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys
import time
from pathlib import Path

import torch
import warp as wp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from metrics import observed_order                    # noqa: E402
from particle_sets import build_case                  # noqa: E402
from rkpm import RKPMOperator, build_system           # noqa: E402

LADDER = (288, 572, 1152, 2304)
CONFIG = [("mls1", 1, False, 40), ("mls2", 2, False, 40), ("mls3", 3, False, 60),
          ("labfm2", 2, True, 25), ("labfm4", 4, True, 40)]


def field_and_midpoint(case):
    box = torch.as_tensor(case.box, dtype=torch.float64, device=case.positions.device)
    k = 2 * torch.pi / box
    f = lambda x: torch.sin(k[0] * x[:, 0]) * torch.cos(k[1] * x[:, 1])
    return f


def run(device):
    rows = []
    for name, p, lab, nb in CONFIG:
        for n in LADDER:
            case = build_case(n, 2, nb, jitter=0.3, seed=42, periodic=True, device=device)
            pos = case.positions
            vol = (case.particles.masses / case.particles.densities).to(pos.dtype)
            box = torch.as_tensor(case.box, dtype=pos.dtype, device=pos.device)
            op = RKPMOperator(build_system(pos, vol, case.h, p,
                                           kernel=case.kernel.name.lower(),
                                           box=box, constant=not lab))
            f = field_and_midpoint(case)
            i, j, fl, fr = op.interface_states(f(pos))
            d = pos[j] - pos[i]
            d = d - box * torch.round(d / box)
            mid = pos[i] + 0.5 * d
            ex = f(mid)
            rows.append(dict(scheme=name, N=case.N, dx=case.dx,
                             err=float(torch.maximum((fl - ex).abs(), (fr - ex).abs()).max()),
                             jump=float((fl - fr).abs().max()),
                             deficient=float(op.s.deficient.float().mean())))
            print(rows[-1], flush=True)
    return rows


def main() -> int:
    wp.init()
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = run(device)
    L = ["# Interface-state reconstruction -- smooth-region order\n",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `run_interface.py` "
         f"(float64, 2-D periodic jitter 0.3, ladder {LADDER}). Error = max over "
         "all pairs of |f_L - f(mid)|, |f_R - f(mid)|; jump = max |f_L - f_R|.\n",
         "| scheme | N nbrs | order(err) | order(jump) | err @ finest | jump @ finest |",
         "|---|---|---|---|---|---|"]
    nb = {c[0]: c[3] for c in CONFIG}
    for name, *_ in CONFIG:
        rs = sorted([r for r in rows if r["scheme"] == name], key=lambda r: r["dx"])
        oe = observed_order([r["dx"] for r in rs], [r["err"] for r in rs])
        oj = observed_order([r["dx"] for r in rs], [r["jump"] for r in rs])
        fmt = lambda o: "sat" if o.saturated else f"{o.slope:.2f}"
        L.append(f"| {name} | {nb[name]} | {fmt(oe)} | {fmt(oj)} | "
                 f"{rs[0]['err']:.2e} | {rs[0]['jump']:.2e} |")
    L.append("")
    (HERE / "REPORT_interface.md").write_text("\n".join(L))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
