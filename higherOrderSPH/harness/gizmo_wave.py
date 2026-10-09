#!/usr/bin/env python3
"""Matched small-amplitude sound wave: the core MFM / MFV backend (with and without the face closure) against GIZMO.

Same particles (square lattice in the periodic unit box, optional jitter), masses from the core's effective
volumes (rho_i = 1 + eps sin(2 pi x) to round-off), Wendland C2 (GIZMO `KERNEL_FUNCTION=6`), gamma = 1.4,
eps = 1e-4, one period `T = 1/c`. Error: rms of (rho - rho_exact) with the mean offset removed and of
(v_x - v_exact) over the particles at the final positions, relative to the amplitude (`run_mfm_smooth.py`).
The case that decides whether the paper's face vector really stalls on a lattice or whether the core has a
convention difference: GIZMO's `A_ij` is the paper's (plus its guards), without a closure.

    python gizmo_wave.py --gizmo-dir <scratch GIZMO build> --work <dir> --n1d 16 32 64 --modes MFM MFV
"""
from __future__ import annotations

import os
os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import argparse
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from warpSPHCore.enumTypes import KernelFunctions               # noqa: E402

import mfm_driver as drv                                        # noqa: E402
from gizmo_gresho import (BOX, PARAMS, build_gizmo, domain, gizmo_volumes, lattice, write_ics)  # noqa: E402

GAMMA = 1.4
EPS = 1e-4
KERNEL = KernelFunctions.Wendland2
CONFIG = ["BOX_PERIODIC", "BOX_SPATIAL_DIMENSION=2", "SELFGRAVITY_OFF", "EOS_GAMMA=(1.4)", "KERNEL_FUNCTION=6",
          "INPUT_IN_DOUBLEPRECISION", "OUTPUT_IN_DOUBLEPRECISION", "DEVELOPER_MODE"]
K = 2.0 * math.pi / BOX
C = math.sqrt(GAMMA)
T = BOX / C


def initial_state(x, nngb, device, closure):
    dom = domain(device)
    N = x.shape[0]
    h = drv.supports_from_volume(torch.full((N,), BOX ** 2 / N, dtype=torch.float64, device=device), nngb, 2)
    st0 = drv.MFMState(pos=x, mass=torch.ones(N, dtype=x.dtype, device=device),
                       Q=torch.zeros(N, 4, dtype=x.dtype, device=device), h=h)
    for _ in range(4):
        g0 = drv.geometry(st0, dom, KERNEL, closure)
        st0.h = drv.supports_from_volume(g0.volume, nngb, 2)
    ph = torch.sin(K * x[:, 0])
    rho = 1.0 + EPS * ph
    vel = torch.zeros(N, 2, dtype=x.dtype, device=device)
    vel[:, 0] = EPS * C * ph
    P = 1.0 + GAMMA * EPS * ph
    return dom, vel, P, rho, rho * g0.volume, st0.h


STATIC = False      # eps = 0: a uniform state; errors are the absolute rms density deviation and rms v / c


def errors(x_final, rho, vx):
    if STATIC:
        return float(np.std(np.asarray(rho))), float(np.sqrt(np.mean(np.asarray(vx) ** 2)) / C)
    ph = np.sin(K * np.asarray(x_final)[:, 0])
    amp = EPS * math.sqrt(np.mean(ph ** 2))
    dr = np.asarray(rho) - 1.0 - EPS * ph
    er = math.sqrt(np.mean((dr - dr.mean()) ** 2)) / amp
    ev = math.sqrt(np.mean((np.asarray(vx) - EPS * C * ph) ** 2)) / (C * amp)
    return er, ev


def run_mine(x_np, mode, closure, nngb, device, cfl):
    drv.BACKEND = "warp"
    x = torch.tensor(x_np, dtype=torch.float64, device=device)
    dom, vel, P, rho, mass, h = initial_state(x, nngb, device, closure)
    st = drv.init_state(x, mass, rho, vel, P, h, GAMMA)
    t0, steps = time.time(), 0
    while st.t < T - 1e-12:
        st, g, dt = drv.step(st, dom, KERNEL, GAMMA, nngb, cfl=cfl, mode=mode, closure=closure, order=2, tmax=T)
        steps += 1
    g = drv.geometry(st, dom, KERNEL, closure)
    r, v, _ = drv.primitives(st.Q, g.volume, GAMMA)
    er, ev = errors(st.pos.cpu().numpy(), r.cpu().numpy(), v[:, 0].cpu().numpy())
    return er, ev, steps, time.time() - t0


def run_gizmo(exe, work, x_np, mass, vel, P, rho, nngb, cfl, ranks):
    work.mkdir(parents=True, exist_ok=True)
    out = work / "output"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    write_ics(work / "gresho_ics.hdf5", x_np, vel, P, mass, gamma=GAMMA, rho0=rho)
    (work / "gresho.params").write_text(PARAMS.format(tend=T, nngb=nngb, cfl=cfl))
    t0 = time.time()
    with open(work / "gizmo.out", "w") as fo, open(work / "gizmo.err", "w") as fe:
        rc = subprocess.run(["mpirun", "--oversubscribe", "-np", str(ranks), str(exe), "gresho.params"], cwd=work,
                            stdout=fo, stderr=fe).returncode
    snaps = sorted(out.glob("snapshot_*.hdf5"))
    if rc != 0 or not snaps:
        raise RuntimeError(f"GIZMO failed (rc {rc}); see {work}/gizmo.err")
    import h5py
    with h5py.File(snaps[-1], "r") as f:
        pos = f["PartType0/Coordinates"][:, :2]
        v = f["PartType0/Velocities"][:, 0]
        rho_g = f["PartType0/Density"][:]
        m = f["PartType0/Masses"][:]
        hk = f["PartType0/KernelMaxRadius"][:]
        t = float(f["Header"].attrs["Time"])
    # GIZMO's density is the mass-weighted kernel sum, whose particle noise on a disordered set is a few per cent --
    # not comparable to the core's rho = m / V_eff. The like-for-like density: m / V_eff on GIZMO's positions and h
    from warpSPHCore.mfm import MeshlessWarp
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w = MeshlessWarp.build(torch.tensor(pos, dtype=torch.float64, device=dev), torch.tensor(hk, dtype=torch.float64, device=dev),
                           domain(dev), KERNEL, closure="none")
    rho_v = m / w.volume.cpu().numpy()
    er, ev = errors(pos, rho_v, v)
    er_g, _ = errors(pos, rho_g, v)
    return er, ev, t, time.time() - t0, er_g


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gizmo-dir", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--n1d", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--modes", nargs="+", default=["MFM", "MFV"])
    ap.add_argument("--closures", nargs="+", default=["project", "none"])
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--nngb", type=float, default=28.0)
    ap.add_argument("--cfl", type=float, default=0.2)
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--h-iters", type=int, default=0, help="driver h <-> V fixed-point iterations per step (0: one step stale)")
    ap.add_argument("--static", action="store_true", help="eps = 0: uniform pressure on the (jittered) lattice; report rms rho deviation and rms v / c")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--extra-config", nargs="*", default=[])
    a = ap.parse_args(argv)
    global STATIC, EPS
    STATIC = a.static
    if a.static:
        EPS = 0.0
    drv.H_ITERS = a.h_iters
    wp.init()
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gdir, work = Path(a.gizmo_dir), Path(a.work)
    exes = {}
    for mode in a.modes:
        exe = gdir / f"GIZMO_{mode}_wave"
        if a.build or not exe.exists():
            exe = build_gizmo(gdir, mode, a.extra_config, config=CONFIG, tag="_wave")
        exes[mode] = exe
    for mode in a.modes:
        prev = {}
        for n1d in a.n1d:
            x = lattice(n1d, a.jitter, 7)
            xt = torch.tensor(x, dtype=torch.float64, device=device)
            dom, vel, P, rho, mass, h = initial_state(xt, a.nngb, device, "project")
            rows = []
            for closure in a.closures:
                er, ev, steps, sec = run_mine(x, mode, closure, a.nngb, device, a.cfl)
                rows.append((f"core closure={closure:7s}", er, ev, f"steps {steps} ({sec:.0f}s)"))
            wdir = work / f"{mode}_{n1d}_j{a.jitter}"
            VG = gizmo_volumes(exes[mode], wdir, x, mass.cpu().numpy(), vel.cpu().numpy(), P.cpu().numpy(), GAMMA, a.nngb, a.ranks)
            er, ev, t, sec, er_g = run_gizmo(exes[mode], wdir, x, rho.cpu().numpy() * VG,
                                       vel.cpu().numpy(), P.cpu().numpy(), rho.cpu().numpy(), a.nngb, a.cfl, a.ranks)
            rows.append(("GIZMO               ", er, ev, f"t={t:.4f} ({sec:.0f}s)  [rho from its own Density: {er_g:.3e}]"))
            print(f"{mode} N={n1d}^2 jitter {a.jitter} nngb {a.nngb:g} cfl {a.cfl}", flush=True)
            for name, er, ev, extra in rows:
                p = prev.get(name)
                rate = "" if p is None else f"  order rho {math.log(p / er) / math.log(n1d / prevn):.2f}"
                prev[name] = er
                print(f"   {name}: err rho {er:.3e}  v {ev:.3e}{rate}  {extra}", flush=True)
            prevn = n1d


if __name__ == "__main__":
    main()
