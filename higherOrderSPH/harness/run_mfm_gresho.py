#!/usr/bin/env python3
"""Gresho-Chan vortex with the core MFM / MFV backend (reference stepper).

The vortex is a steady solution, so the volume-weighted L1 error of |v - v_exact|
(Springel 2010; Frontiere 2017) after `t = 3` is pure scheme error. The Phase 1
anomaly this addresses: CRKSPH's pair pressure force injects kinetic energy
(`ke_rebound`); a Riemann-flux meshless scheme should not -- the number to read
is the KE growth next to the L1 error. Same IC as `warpSPH.caseUtils.compressible.
greshoVortex` (rho = 1, gamma = 5/3, centred in the periodic unit box, standard
piecewise profile), harness hexagonal lattice, optional jitter.

    python run_mfm_gresho.py --n 4096 --closure none project
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

from warpSPHCore.enumTypes import KernelFunctions               # noqa: E402

import mfm_driver as drv                                        # noqa: E402
from particle_sets import build_case                            # noqa: E402

G = 5.0 / 3.0
P_OUTER = 3 + 4 * math.log(2)


def profile(r):
    vphi = torch.where(r < 0.2, 5 * r, torch.where(r < 0.4, 2 - 5 * r, torch.zeros_like(r)))
    rs = r.clamp(min=1e-30)
    P = torch.where(r < 0.2, 12.5 * r ** 2 + 5,
                    torch.where(r < 0.4, 12.5 * r ** 2 - 20 * r + 4 * torch.log(5 * rs) + 9,
                                torch.full_like(r, P_OUTER)))
    return vphi, P


def vortex_velocity(x, centre, box):
    d = x - centre
    d = d - box * torch.round(d / box)
    r = torch.linalg.norm(d, dim=-1)
    vphi, _ = profile(r)
    safe = torch.where(r > 0, r, torch.ones_like(r))
    return torch.stack([-vphi * d[:, 1] / safe, vphi * d[:, 0] / safe], -1), r


def run(n, mode, closure, order, jitter, nngb, kernel, tend, device, cfl, log):
    case = build_case(n, 2, int(nngb), jitter=jitter, seed=42, periodic=True, device=device, kernel=kernel)
    dt_ = torch.float64
    x = case.particles.positions.to(dt_)
    dom = case.domain
    box = torch.as_tensor(case.box, dtype=dt_, device=device)
    centre = 0.5 * box
    V0 = torch.full((x.shape[0],), float(case.cell_vol), dtype=dt_, device=device)
    h = drv.supports_from_volume(V0, nngb, 2)
    v, r = vortex_velocity(x, centre, box)
    _, P = profile(r)
    N = x.shape[0]
    st0 = drv.MFMState(pos=x, mass=torch.ones(N, dtype=dt_, device=device),
                       Q=torch.zeros(N, 4, dtype=dt_, device=device), h=h)
    for _ in range(4):
        g0 = drv.geometry(st0, dom, kernel, closure)
        st0.h = drv.supports_from_volume(g0.volume, nngb, 2)
    rho = torch.ones(N, dtype=dt_, device=device)
    st = drv.init_state(x, rho * g0.volume, rho, v, P, st0.h, G)
    ke0 = float((0.5 * st.Q[:, 1:3].pow(2).sum(-1) / st.Q[:, 0]).sum())
    kes, steps, t0 = [], 0, time.time()
    while st.t < tend - 1e-12:
        st, g, dt = drv.step(st, dom, kernel, G, nngb, cfl=cfl, mode=mode, closure=closure, order=order, tmax=tend)
        steps += 1
        if steps % 20 == 0:
            kes.append(float((0.5 * st.Q[:, 1:3].pow(2).sum(-1) / st.Q[:, 0]).sum()) / ke0)
            if log and steps % 200 == 0:
                print(f"   t {st.t:.2f} KE/KE0 {kes[-1]:.4f} steps {steps}", flush=True)
    g = drv.geometry(st, dom, kernel, closure)
    rho_, vel, P_ = drv.primitives(st.Q, g.volume, G)
    vex, _ = vortex_velocity(st.pos, centre, box)
    w = g.volume
    l1 = float((w * (vel - vex).norm(dim=-1)).sum() / w.sum())
    ke1 = float((0.5 * st.Q[:, 1:3].pow(2).sum(-1) / st.Q[:, 0]).sum()) / ke0
    rebound = max(kes) - min(kes[kes.index(min(kes)):] or [min(kes)]) if kes else 0.0
    cons = (st.Q.sum(0) - drv.conserved(rho * g0.volume, rho, v, P, G).sum(0)).abs().max().item()
    return l1, ke1, max(kes) if kes else ke1, steps, cons, time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[4096])
    ap.add_argument("--modes", nargs="+", default=["MFM", "MFV"])
    ap.add_argument("--closure", nargs="+", default=["project"])
    ap.add_argument("--orders", type=int, nargs="+", default=[2])
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--nngb", type=float, default=30.0)
    ap.add_argument("--kernel", default="Wendland4")
    ap.add_argument("--tend", type=float, default=3.0)
    ap.add_argument("--cfl", type=float, default=0.2)
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--backend", default="torch", choices=["torch", "warp"])
    a = ap.parse_args(argv)
    drv.BACKEND = a.backend
    wp.init()
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kernel = KernelFunctions[a.kernel]
    for n in a.n:
        for mode in a.modes:
            for closure in a.closure:
                for order in a.orders:
                    l1, ke1, kemax, steps, cons, sec = run(n, mode, closure, order, a.jitter, a.nngb, kernel, a.tend,
                                                          device, a.cfl, a.log)
                    print(f"{mode} {closure:7s} order {order} N={n} jitter {a.jitter} {a.kernel} nngb {a.nngb}: "
                          f"L1(v) {l1:.4f}  KE/KE0 final {ke1:.4f} max {kemax:.4f}  |dQ| {cons:.1e} steps {steps} ({sec:.0f}s)",
                          flush=True)


if __name__ == "__main__":
    main()
