#!/usr/bin/env python3
"""Smooth-flow convergence of the core MFM / MFV backend: a small-amplitude
right-going sound wave, one period, periodic box, Lagrangian particles.

Exact solution (linear, amplitude 1e-4 so the nonlinear error is ~1e-8): the
initial state returns after T = L / c. Scored by the relative L2 error of the
density perturbation and of v_x. 1-D uses an equidistant lattice, 2-D the
jittered hexagonal lattice of the harness (`build_case`), so the effect of
particle disorder and of the face-vector closure (`--closure none|project`) is
visible.

    python run_mfm_smooth.py --dim 1 --n 50 100 200 400
    python run_mfm_smooth.py --dim 2 --n 288 1152 4608 --jitter 0.3 --closure none project
"""
from __future__ import annotations

import os
os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import warp as wp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from warpSPHCore import DomainDescription                       # noqa: E402
from warpSPHCore.enumTypes import KernelFunctions               # noqa: E402

import mfm_driver as drv                                        # noqa: E402

G = 1.4
EPS = 1e-4
KERNEL = KernelFunctions.Wendland2


def setup(dim, n, jitter, nngb, device):
    dt_ = torch.float64
    if dim == 1:
        x = ((torch.arange(n, dtype=dt_, device=device) + 0.5) / n)[:, None]
        L = torch.ones(1, dtype=dt_, device=device)
        dom = DomainDescription(torch.zeros(1, dtype=dt_, device=device), L,
                                torch.ones(1, dtype=torch.bool, device=device), 1)
        V0 = torch.full((n,), 1.0 / n, dtype=dt_, device=device)
    else:
        from particle_sets import build_case
        case = build_case(n, 2, int(nngb), jitter=jitter, seed=42, periodic=True, device=device,
                          kernel=KERNEL)
        x = case.particles.positions.to(dt_)
        dom = case.domain
        L = torch.as_tensor(case.box, dtype=dt_, device=device)
        V0 = torch.full((x.shape[0],), float(case.cell_vol), dtype=dt_, device=device)
    N = x.shape[0]
    h = drv.supports_from_volume(V0, nngb, dim)
    return x, dom, L, V0, h


def run(dim, n, mode, closure, order, jitter, nngb, device):
    x, dom, L, V0, h = setup(dim, n, jitter, nngb, device)
    N = x.shape[0]
    k = 2.0 * math.pi / L[0]
    c = math.sqrt(G)
    ph = torch.sin(k * x[:, 0])
    # masses from the *effective* volumes so that rho_i = m_i / V_i matches the sinusoid exactly
    st0 = drv.MFMState(pos=x, mass=torch.ones(N, dtype=x.dtype, device=x.device),
                       Q=torch.zeros(N, 2 + dim, dtype=x.dtype, device=x.device), h=h)
    for _ in range(4):                      # fixed point of h <-> V so that the initial state is consistent
        g0 = drv.geometry(st0, dom, KERNEL, closure)
        st0.h = drv.supports_from_volume(g0.volume, nngb, dim)
    h = st0.h
    rho = 1.0 + EPS * ph
    mass = rho * g0.volume
    vel = torch.zeros(N, dim, dtype=x.dtype, device=x.device)
    vel[:, 0] = EPS * c * ph
    P = 1.0 + G * EPS * ph
    st = drv.init_state(x, mass, rho, vel, P, h, G)
    T = float(L[0]) / c
    steps = 0
    while st.t < T - 1e-12:
        st, g, dt = drv.step(st, dom, KERNEL, G, nngb, mode=mode, closure=closure, order=order, tmax=T)
        steps += 1
    g = drv.geometry(st, dom, KERNEL, closure)
    r, v, p = drv.primitives(st.Q, g.volume, G)
    # compare at the particle positions after one period (the exact solution is the initial one)
    ph1 = torch.sin(k * st.pos[:, 0])
    dr = r - 1.0 - EPS * ph1
    er = ((dr - dr.mean()) ** 2).mean().sqrt() / (EPS * (ph1 ** 2).mean().sqrt())      # mean offset removed (a 1e-6 volume bias)
    ev = ((v[:, 0] - EPS * c * ph1) ** 2).mean().sqrt() / (EPS * c * (ph1 ** 2).mean().sqrt())
    return float(er), float(ev), steps, (st.Q.sum(0) - drv.conserved(mass, rho, vel, P, G).sum(0)).abs().max().item()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=1)
    ap.add_argument("--n", type=int, nargs="+", default=[50, 100, 200, 400])
    ap.add_argument("--modes", nargs="+", default=["MFM", "MFV"])
    ap.add_argument("--closure", nargs="+", default=["project"])
    ap.add_argument("--orders", type=int, nargs="+", default=[2])
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--nngb", type=float, default=None)
    a = ap.parse_args(argv)
    nngb = a.nngb if a.nngb is not None else (7.0 if a.dim == 1 else 28.0)
    wp.init()
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for mode in a.modes:
        for closure in a.closure:
            for order in a.orders:
                prev = None
                for n in a.n:
                    t0 = time.time()
                    er, ev, steps, cons = run(a.dim, n, mode, closure, order, a.jitter, nngb, device)
                    rate = "" if prev is None else (
                        f"  order rho {math.log(prev[0] / er) / math.log(math.sqrt(n / prev[1]) if a.dim == 2 else n / prev[1]):.2f}")
                    prev = (er, n)
                    print(f"{mode} {closure:7s} order {order} jitter {a.jitter} n={n:5d}: err rho {er:.3e} v {ev:.3e}"
                          f"  |dQ| {cons:.1e} steps {steps}{rate} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
