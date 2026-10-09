#!/usr/bin/env python3
"""Phase 6/7 static evidence: TENO vs WENO vs plain MLS interface states.

On the harness's jittered periodic hexagonal lattice, reconstruct left / right
states at all pair midpoints (`reconstruct.py`) and measure

* **smooth-region order** -- error vs the exact midpoint value of a smooth
  periodic field (ladder of particle counts);
* **non-oscillatory behaviour** -- overshoot of a unit step (max excursion
  outside [0, 1]);
* **jump capture / false jumps** -- for pairs straddling the discontinuity the
  mean |f_L - f_R| (1 = fully resolved), and for pairs on the same side the
  max |f_L - f_R| (should be 0: any value is a spurious Riemann problem);
* **smooth background next to a jump** -- max error of ``0.5 sin(2 pi (x+y))``
  superposed on the step, over pairs >= 2 spacings from the jump, next to the
  same error without the step (how far the discontinuity's influence reaches:
  a dissipation proxy; the true head-to-head dissipation comparison needs the
  Riemann-SPH solver, see higher_order.md Phase 6).

Plain MLS (`mlsP`) is the unlimited reference. Matched formal order: TENO O4
(central degree 3) vs WENO M=3 vs mls3; TENO O5/O6 vs WENO M=4/5.

    python run_reconstruct.py
"""

from __future__ import annotations

import os
os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import math
import sys
import time
from pathlib import Path

import torch
import warp as wp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from metrics import observed_order                                     # noqa: E402
from particle_sets import build_case                                   # noqa: E402
from reconstruct import teno_reconstructor, weno_reconstructor          # noqa: E402
from rkpm import RKPMOperator, build_system                            # noqa: E402

LADDER = (288, 1152, 4608)       # particle counts, 2-D hex
SCHEMES = ["mls3", "teno-O4", "weno-M3", "mls4", "teno-O5", "weno-M4",
           "mls5", "teno-O6", "weno-M5"]


def make(scheme, case, vol, box):
    pos = case.positions
    kind, par = scheme.split("-") if "-" in scheme else (scheme[:3], scheme[3:])
    if kind == "mls":
        p = int(par)
        # R = 2.5*(p+1)/4 sqrt(V): roughly the TENO central radii
        sv = float(vol.mean()) ** 0.5
        op = RKPMOperator(build_system(pos, vol, {3: 2.5, 4: 3.2, 5: 4.0}[p] * sv, p,
                                       kernel="wendland2", box=box))
        class _R:                       # same interface as Reconstructor
            def interface_states(self, f, radius_factor=1.6):
                i, j, fl, fr = op.interface_states(f)
                s = op.s
                pair = s.i_idx != s.j_idx
                d = (s.xi[pair] * s.h)
                keep = torch.linalg.norm(d, dim=-1) < radius_factor * sv
                return i[keep], j[keep], fl[keep], fr[keep], 0.5 * d[keep]
        return _R()
    if kind == "teno":
        return teno_reconstructor(pos, vol, box, par)
    return weno_reconstructor(pos, vol, box, int(par[1:]) if par.startswith("M") else int(par))


def midpoint(pos, i, d):
    return pos[i] + d


def run(device):
    out = {}
    for scheme in SCHEMES:
        errs, dxs = [], []
        for n in LADDER:
            case = build_case(n, 2, 40, jitter=0.3, seed=42, periodic=True, device=device)
            pos = case.positions
            box = torch.as_tensor(case.box, dtype=pos.dtype, device=pos.device)
            vol = torch.full((case.N,), case.cell_vol, dtype=pos.dtype, device=pos.device)
            R = make(scheme, case, vol, box)
            kx = 2 * math.pi / float(box[0]); ky = 2 * math.pi / float(box[1])
            smooth = lambda x: torch.sin(kx * x[:, 0]) * torch.cos(ky * x[:, 1])
            i, j, fl, fr, d = R.interface_states(smooth(pos))
            ex = smooth(pos[i] + d)
            errs.append(float(torch.maximum((fl - ex).abs(), (fr - ex).abs()).max()))
            dxs.append(case.dx)
        # step tests at the finest rung
        sdist = lambda x: (kx * x[:, 0] + 0.3) / math.pi          # jump at integers
        step = lambda x: (torch.sin(kx * x[:, 0] + 0.3) > 0).double()
        f = step(pos)
        i, j, fl, fr, d = R.interface_states(f)
        over = float(torch.maximum(torch.maximum(fl.max() - 1, -fl.min()),
                                   torch.maximum(fr.max() - 1, -fr.min())))
        straddle = f[i] != f[j]
        jump_capture = float((fl - fr).abs()[straddle].mean())
        false_jump = float((fl - fr).abs()[~straddle].max())
        # smooth background next to the jump
        bg = lambda x: 0.5 * torch.sin(kx * x[:, 0] + ky * x[:, 1])
        g = bg(pos) + f
        i, j, gl, gr, d = R.interface_states(g)
        mid = pos[i] + d
        far = (lambda x: (sdist(x) - torch.round(sdist(x))).abs() * float(box[0]) / 2 > 2 * case.dx)
        m = far(pos[i]) & far(pos[j])
        e_with = float(torch.maximum((gl - (bg(mid) + step(mid))).abs(),
                                     (gr - (bg(mid) + step(mid))).abs())[m].max())
        i, j, gl0, gr0, d = R.interface_states(bg(pos))
        mid0 = pos[i] + d
        e_no = float(torch.maximum((gl0 - bg(mid0)).abs(), (gr0 - bg(mid0)).abs()).max())
        o = observed_order(dxs, errs)
        out[scheme] = dict(order=("sat" if o.saturated else f"{o.slope:.2f}"),
                           err=errs[-1], overshoot=over, capture=jump_capture,
                           false_jump=false_jump, e_with=e_with, e_no=e_no)
        print(scheme, out[scheme], flush=True)
    return out


def main() -> int:
    wp.init()
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = run(device)
    L = ["# Phase 6/7 static evidence -- TENO vs WENO vs plain MLS interface states\n",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `run_reconstruct.py` (float64, 2-D hex "
         f"lattice, jitter 0.3, periodic, particle counts {LADDER}). Matched formal order per "
         "block of three rows. See the module docstring of `run_reconstruct.py` for the "
         "metric definitions; **no Riemann solver is involved** -- these are reconstruction "
         "states only.\n",
         "| scheme | smooth order | smooth err @ finest | step overshoot | jump capture | "
         "false jump (same side) | smooth-background error >= 2 dx from the jump | same, no jump present |",
         "|---|---|---|---|---|---|---|---|"]
    for k in SCHEMES:
        r = res[k]
        L.append(f"| {k} | {r['order']} | {r['err']:.2e} | {r['overshoot']:.2e} | "
                 f"{r['capture']:.3f} | {r['false_jump']:.2e} | {r['e_with']:.2e} | {r['e_no']:.2e} |")
    L.append("")
    (HERE / "REPORT_reconstruct.md").write_text("\n".join(L))
    print("wrote REPORT_reconstruct.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
