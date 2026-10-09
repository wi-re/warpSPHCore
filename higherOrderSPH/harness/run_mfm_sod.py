#!/usr/bin/env python3
"""MFM / MFV on a periodic double Sod tube (1-D), scored against the exact
Riemann solution; conservation to round-off is checked on the way.

    python run_mfm_sod.py --n 200 400 800
"""
from __future__ import annotations

import os
os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from warpSPHCore import DomainDescription                       # noqa: E402
from warpSPHCore.enumTypes import KernelFunctions               # noqa: E402

import exact_riemann as ex                                      # noqa: E402
import mfm_driver as drv                                        # noqa: E402

G = 1.4


def run(n, mode, closure, order, nngb, tend, device, kernel):
    dt_ = torch.float64
    x = (torch.arange(n, dtype=dt_, device=device) + 0.5) / n
    high = (x >= 0.25) & (x < 0.75)
    rho = torch.where(high, 1.0, 0.125).to(dt_)
    P = torch.where(high, 1.0, 0.1).to(dt_)
    vel = torch.zeros(n, 1, dtype=dt_, device=device)
    mass = rho / n
    dom = DomainDescription(torch.zeros(1, dtype=dt_, device=device), torch.ones(1, dtype=dt_, device=device),
                            torch.ones(1, dtype=torch.bool, device=device), 1)
    h = torch.full((n,), nngb / (2.0 * n), dtype=dt_, device=device)
    st = drv.init_state(x[:, None], mass, rho, vel, P, h, G)
    Q0 = st.Q.sum(0)
    steps = 0
    while st.t < tend - 1e-12:
        st, g, dt = drv.step(st, dom, kernel, G, nngb, mode=mode, closure=closure, order=order, tmax=tend)
        steps += 1
    g = drv.geometry(st, dom, kernel, closure)
    r, v, p = drv.primitives(st.Q, g.volume, G)
    return st, g, (r, v, p), (st.Q.sum(0) - Q0).abs().max().item(), steps


def score(st, prim, tend):
    r, v, p = [t.cpu().numpy() for t in prim]
    x = st.pos[:, 0].cpu().numpy()
    xr = np.where(x >= 0.5, x, 1.0 - x)
    s = ex.sample((xr - 0.75) / tend, 1.0, 0.0, 1.0, 0.125, 0.0, 0.1, G)
    sign = np.where(x >= 0.5, 1.0, -1.0)
    exr, exv, exp_ = s[:, 0], sign * s[:, 1], s[:, 2]
    n = len(x)
    return (np.abs(r - exr).mean(), np.abs(v[:, 0] - exv).mean(), np.abs(p - exp_).mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[200, 400, 800])
    ap.add_argument("--tend", type=float, default=0.12)
    ap.add_argument("--nngb", type=float, default=7.0)
    ap.add_argument("--modes", nargs="+", default=["MFM", "MFV"])
    a = ap.parse_args(argv)
    wp.init()
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kernel = KernelFunctions.Wendland2
    for mode in a.modes:
        for order in (1, 2):
            for n in a.n:
                t0 = time.time()
                st, g, prim, cons, steps = run(n, mode, "project", order, a.nngb, a.tend, device, kernel)
                e = score(st, prim, a.tend)
                print(f"{mode} order {order} n={n:5d}: L1 rho {e[0]:.4e} v {e[1]:.4e} P {e[2]:.4e}  "
                      f"|dQ| {cons:.1e}  steps {steps}  ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
