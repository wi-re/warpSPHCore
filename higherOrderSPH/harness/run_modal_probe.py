#!/usr/bin/env python3
"""Modal decomposition of a static operator error (Laplacian by default).

The static probes report the *pointwise* error of an operator on a smooth
field; the frozen-particle PDE leg (`run_frozen.py`) reports what a time
integration actually accumulates. For the Brookshaw Laplacian the two
disagree (static: does not converge; frozen diffusion: order ~2 for CRK).
This probe explains the gap by splitting the error
``e = L_h u - lap u`` of a single Fourier mode ``u = sin(kappa . x)`` on a
periodic jittered set into

* the **modal** part -- the projection onto the modes the exact dynamics can
  reach (``a u + b cos(kappa . x)``). ``a / |kappa|^2`` is the relative
  error in the effective diffusivity: it is what a diffusion solve sees;
* the **residual** -- everything else: particle-scale (jitter) noise, which
  a dissipative PDE damps at rate ~ nu / dx^2, so its net effect on the
  solution is O(dx) even when its pointwise size grows.

Result (2026-09-26, `FINDINGS.md` section 3): the residual is essentially
all of the pointwise error and grows like 1/dx in every mode; the modal part
converges at ~2 for CRK / renorm (~0.9, decaying, for standard) -- matching
the frozen diffusion orders. A new Laplacian (Phase 4/5) should be judged on
both: the modal error for dissipative use, the residual for non-dissipative
use (source terms, Poisson right-hand sides).

Usage:
    python run_modal_probe.py                       # Laplacian, all modes
    python run_modal_probe.py --probe gradient      # modal error of d/dx
"""

from __future__ import annotations

import os
os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RESULTS = HERE / "results"

DEFAULT_LADDER = (576, 1152, 2304, 4608, 9216)


def modal_split(e: torch.Tensor, basis: list[torch.Tensor]) -> tuple:
    """Least-squares split of `e` into span(basis) + residual (basis assumed
    mutually orthogonal over the particle set, as sin/cos of one mode are to
    O(jitter) on a periodic set). Returns (coefficients, residual)."""
    coefs = [float((e * b).sum() / (b * b).sum()) for b in basis]
    r = e.clone()
    for c, b in zip(coefs, basis):
        r = r - c * b
    return coefs, r


def main(argv=None):
    import warp as wp
    wp.init()
    from metrics import observed_order
    from operators import MODES, CorrectionCache, run_probe
    from particle_sets import build_case
    from run_frozen import wavevector

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe", default="laplacian", choices=["laplacian", "gradient"])
    p.add_argument("--modes", nargs="+", default=list(MODES))
    p.add_argument("--ladder", nargs="+", type=int, default=list(DEFAULT_LADDER))
    p.add_argument("--jitter", type=float, default=0.3)
    p.add_argument("--target-neighbors", type=int, default=40)
    p.add_argument("--k", nargs=2, type=int, default=[1, 1])
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args(argv)

    out = {m: [] for m in args.modes}
    for n in args.ladder:
        case = build_case(n, 2, args.target_neighbors, jitter=args.jitter,
                          seed=42, periodic=True, device=args.device)
        cache = CorrectionCache(case)
        x = case.positions
        kappa = wavevector(case.box, tuple(args.k)).to(x)
        s, c = torch.sin(x @ kappa), torch.cos(x @ kappa)
        k2 = float(kappa @ kappa)
        for mode in args.modes:
            val = run_probe(case, cache, s.contiguous(), args.probe, mode)
            if args.probe == "laplacian":
                e = val - (-k2) * s
                (a, b), r = modal_split(e, [s, c])
                modal_rel = abs(a) / k2           # relative diffusivity error
            else:
                # d/dx_0 of sin(kappa.x) = kappa_0 cos(kappa.x): project
                # the x-component error onto cos (phase-speed error) and sin
                e = val[:, 0] - float(kappa[0]) * c
                (a, b), r = modal_split(e, [c, s])
                modal_rel = abs(a) / abs(float(kappa[0]))
            out[mode].append(dict(N=case.N, dx=case.dx,
                                  total_rms=float(e.pow(2).mean().sqrt()),
                                  modal_rel=modal_rel,
                                  residual_rms=float(r.pow(2).mean().sqrt())))

    summary = {}
    for mode, rows in out.items():
        dx = [r["dx"] for r in rows]
        summary[mode] = {}
        print(f"\n[{args.probe}/{mode}]  dx  total_rms  modal_rel  residual_rms")
        for r in rows:
            print(f"  {r['dx']:.4f}  {r['total_rms']:.3e}  {r['modal_rel']:.3e}"
                  f"  {r['residual_rms']:.3e}")
        for key in ("total_rms", "modal_rel", "residual_rms"):
            o = observed_order(dx, [r[key] for r in rows])
            summary[mode][key] = None if o.saturated else round(o.slope, 3)
            print(f"  order {key}: {summary[mode][key]}  "
                  f"pairwise {np.round(o.pairwise_slopes, 2).tolist()}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"modal_probe_{args.probe}.json").write_text(
        json.dumps({"args": vars(args), "rows": out, "orders": summary},
                   indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
