#!/usr/bin/env python3
"""Phase 6/7 solver-light comparison: linear advection with MLS / TENO / WENO
interface states and upwind fluxes, frozen jittered particles.

The Riemann-SPH coupling is the frontend's (Godunov SPH); what the Phase 6/7
head-to-head needs *from the reconstruction* is how it behaves inside a time
loop: numerical dissipation of smooth waves, dispersion, and the overshoot /
smearing of a discontinuity. For linear advection ``u_t + a . grad u = 0`` the
exact Riemann solution is the upwind state, so the whole "solver" is: pick the
left or right interface state by the sign of ``a . (r_j - r_i)``.

Pairwise update (the Godunov-SPH form, gradient-corrected):

    du_i/dt = - sum_j 2 (u*_ij - u_i)  a . G_ij ,     G_ij = [Minv_i w_ij P_ij]_grad / h

``G_ij`` is the order-``g_order`` LABFM gradient weight (``sum_j G_ij (u_j - u_i)``
is the order-k LABFM gradient), so with the exact interface value
``u*_ij = u(r_ij)`` the update is consistent to ``O(h^k)`` on the jittered set and
the *reconstruction* is what differs between the rows of the table, not the
discretisation of the divergence. ``u*`` of the first-order row is the upwind
particle value (no reconstruction). RK4 in time, ``dt = cfl dx / |a|``; particles
are frozen (this is the dissipation test, not a Lagrangian-motion test).

Fields (periodic box, ``a = (1, 0)``, one period): smooth waves
``sin(2 pi m x / Lx) cos(2 pi y / Ly)`` (relative L2 error, amplitude ratio) and
a top hat (overshoot, undershoot, L1 error).

    python run_advection.py                    # N = 288 1152 4608
    python run_advection.py --ladder 288 1152 --schemes teno4 weno3

Writes `results/advection_rows.csv` and `REPORT_advection.md` (own outputs only).
"""

from __future__ import annotations

import os
os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch
import warp as wp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from metrics import observed_order                              # noqa: E402
from particle_sets import build_case                            # noqa: E402
from rkpm import build_system                                  # noqa: E402

from warpSPHCore.polyfit import PolyFit, Reconstructor         # noqa: E402

SCHEMES = ("first", "mls3", "mls5", "teno4", "teno6", "weno3", "weno5")
FIELDS = ("wave1", "wave3", "tophat")


def make_reconstruction(scheme, case):
    """``f(u, i, j, d) -> (u_L, u_R)`` for a scheme name."""
    P, dom, K = case.particles, case.domain, case.kernel
    if scheme == "first":
        return None
    if scheme.startswith("mls"):
        deg = int(scheme[3:])
        rec = Reconstructor.teno(P, dom, K, {3: "O4", 4: "O5", 5: "O6"}[deg])
        fit = rec.fits[0]                                       # the unlimited central MLS fit
        h = rec.interfaceSupports

        def central(u, i, j, d):
            oi, oj = PolyFit.interfaceOffsets(d, h[i], h[j])
            c = fit.coefficients(u)
            return fit.evaluate(u, i, oi, c), fit.evaluate(u, j, oj, c)
        return central, rec
    if scheme.startswith("teno"):
        rec = Reconstructor.teno(P, dom, K, f"O{int(scheme[4:])}")
    else:
        rec = Reconstructor.weno(P, dom, K, degree=int(scheme[4:]))
    return rec.interfaceStates, rec


class Advection:
    def __init__(self, case, scheme, g_order, a=(1.0, 0.0)):
        self.case, self.scheme = case, scheme
        pos = case.particles.positions
        N = pos.shape[0]
        dev, dt = pos.device, pos.dtype
        vol = (case.particles.masses / case.particles.densities).to(dt)
        box = torch.as_tensor(case.box, dtype=dt, device=dev)
        sysG = build_system(pos, vol, case.h, g_order, kernel=case.kernel.name.lower(),
                            box=box, constant=False)
        iy = {tuple(e): k for k, e in enumerate(sysG.exps.tolist())}
        rows = [iy[(1, 0)], iy[(0, 1)]]
        cvec = torch.einsum("pab,pb->pa", sysG.Minv[sysG.i_idx], sysG.wP)
        G = cvec[:, rows] / case.h                               # (P, 2): G_ij for ordered pairs
        keep = sysG.i_idx != sysG.j_idx
        key = sysG.i_idx[keep] * N + sysG.j_idx[keep]
        order = torch.argsort(key)
        self.key, self.G = key[order], G[keep][order]
        self.a = torch.tensor(a, dtype=dt, device=dev)
        recon = make_reconstruction(scheme, case)
        if recon is None:
            self.recon, self.rec = None, None
            from warpSPHCore.polyfit import interfacePairs
            self.i, self.j, self.d = interfacePairs(case.particles, case.domain)
        else:
            self.recon, self.rec = recon
            self.i, self.j, self.d = self.rec.pairs()
        self.N = N
        self.aG_ij = self._lookup(self.i, self.j) @ self.a
        self.aG_ji = self._lookup(self.j, self.i) @ self.a
        self.upwind_i = (self.d @ self.a) > 0                    # information flows i -> j

    def _lookup(self, i, j):
        q = i * self.N + j
        pos = torch.searchsorted(self.key, q)
        pos = pos.clamp(max=self.key.shape[0] - 1)
        assert bool((self.key[pos] == q).all()), "pair sets of the gradient weights and of the reconstruction differ"
        return self.G[pos]

    def rhs(self, u):
        i, j = self.i, self.j
        if self.recon is None:
            ustar = torch.where(self.upwind_i, u[i], u[j])
        else:
            uL, uR = self.recon(u, i, j, self.d)
            ustar = torch.where(self.upwind_i, uL, uR)
        du = torch.zeros_like(u)
        du.index_add_(0, i, -2.0 * (ustar - u[i]) * self.aG_ij)
        du.index_add_(0, j, -2.0 * (ustar - u[j]) * self.aG_ji)
        return du

    def run(self, u, T, dt):
        steps = int(math.ceil(T / dt))
        dt = T / steps
        for _ in range(steps):
            k1 = self.rhs(u)
            k2 = self.rhs(u + 0.5 * dt * k1)
            k3 = self.rhs(u + 0.5 * dt * k2)
            k4 = self.rhs(u + dt * k3)
            u = u + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            if not bool(torch.isfinite(u).all()) or float(u.abs().max()) > 1e3:
                return None, steps
        return u, steps


def initial(case, name):
    x = case.particles.positions
    Lx, Ly = float(case.box[0]), float(case.box[1])
    if name.startswith("wave"):
        m = int(name[4:])
        return torch.sin(2 * math.pi * m * x[:, 0] / Lx) * torch.cos(2 * math.pi * x[:, 1] / Ly)
    xr = (x[:, 0] / Lx) % 1.0
    return ((xr > 0.25) & (xr < 0.75)).to(x.dtype)


def measure(name, u, u0, vol):
    if u is None:
        return dict(unstable=1)
    w = vol / vol.sum()
    out = dict(unstable=0)
    if name.startswith("wave"):
        out["l2"] = float(torch.sqrt((w * (u - u0) ** 2).sum() / (w * u0 ** 2).sum()))
        out["amp"] = float((w * u * u0).sum() / (w * u0 ** 2).sum())
    else:
        out["over"] = float(u.max() - 1.0)
        out["under"] = float(-u.min())
        out["l1"] = float((w * (u - u0).abs()).sum())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", type=int, nargs="+", default=[288, 1152, 4608])
    ap.add_argument("--schemes", nargs="+", default=list(SCHEMES))
    ap.add_argument("--fields", nargs="+", default=list(FIELDS))
    ap.add_argument("--nbrs", type=int, default=60)
    ap.add_argument("--g-order", type=int, default=5)
    ap.add_argument("--jitter", type=float, default=0.3)
    ap.add_argument("--cfl", type=float, default=0.2)
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wp.init()
    torch.set_grad_enabled(False)
    rows = []
    t0 = time.time()
    for n in a.ladder:
        case = build_case(n, 2, a.nbrs, jitter=a.jitter, seed=42, periodic=True, device=device)
        vol = (case.particles.masses / case.particles.densities).to(torch.float64)
        T = float(case.box[0])
        for scheme in a.schemes:
            try:
                adv = Advection(case, scheme, a.g_order)
            except AssertionError as e:                          # noqa: PERF203
                print(f"N={case.N} {scheme}: {e}")
                continue
            for name in a.fields:
                u0 = initial(case, name)
                u, steps = adv.run(u0.clone(), T, a.cfl * case.dx)
                m = measure(name, u, u0, vol)
                rows.append(dict(N=case.N, dx=case.dx, scheme=scheme, field=name, steps=steps, **m))
                print(f"N={case.N} {scheme:6s} {name:7s} "
                      + " ".join(f"{k}={v:.3g}" for k, v in m.items())
                      + f" ({time.time() - t0:.0f}s)", flush=True)

    out = Path(a.out)
    (out / "results").mkdir(exist_ok=True)
    keys = ["N", "dx", "scheme", "field", "steps", "unstable", "l2", "amp", "over", "under", "l1"]
    with (out / "results" / "advection_rows.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in keys})
    write_report(out, rows, a)
    return 0


def _get(rows, scheme, field, N, key):
    for r in rows:
        if r["scheme"] == scheme and r["field"] == field and r["N"] == N:
            return None if r.get("unstable") else r.get(key)
    return None


def write_report(out, rows, a):
    Ns = sorted({r["N"] for r in rows})
    schemes = [s for s in SCHEMES if any(r["scheme"] == s for r in rows)]
    L = ["# Phase 6/7 report -- advection with reconstructed interface states\n",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `run_advection.py` (float64, 2-D periodic, "
         f"jitter {a.jitter}, ~{a.nbrs} neighbours, divergence weights: LABFM order {a.g_order}, RK4, "
         f"CFL {a.cfl}, frozen particles, one period at a = (1, 0)). Upwind flux; the first-order row "
         f"uses the upwind particle value. Ladder N = {tuple(Ns)}.\n"]
    for f in [x for x in FIELDS if any(r["field"] == x for r in rows)]:
        L.append(f"## {f}\n")
        if f.startswith("wave"):
            L.append("| scheme | " + " | ".join(f"L2 err N={n}" for n in Ns) + " | order | "
                     + " | ".join(f"amplitude N={n}" for n in Ns) + " |")
            L.append("|---|" + "---|" * (2 * len(Ns) + 1))
            for s in schemes:
                e = [_get(rows, s, f, n, "l2") for n in Ns]
                am = [_get(rows, s, f, n, "amp") for n in Ns]
                if all(v is not None and v > 0 for v in e) and len(e) >= 3:
                    dxs = [math.sqrt(1.0 / n) for n in Ns]
                    o = observed_order(dxs, e)
                    ord_s = "sat" if o.saturated else f"{o.slope:.2f}"
                else:
                    ord_s = "unstable" if any(v is None for v in e) else "n/a"
                L.append(f"| {s} | " + " | ".join("unstable" if v is None else f"{v:.2e}" for v in e)
                         + f" | {ord_s} | " + " | ".join("-" if v is None else f"{v:.4f}" for v in am) + " |")
        else:
            L.append("| scheme | " + " | ".join(f"overshoot N={n}" for n in Ns) + " | "
                     + " | ".join(f"undershoot N={n}" for n in Ns) + " | "
                     + " | ".join(f"L1 err N={n}" for n in Ns) + " |")
            L.append("|---|" + "---|" * (3 * len(Ns)))
            for s in schemes:
                cells = []
                for key in ("over", "under", "l1"):
                    cells += ["unstable" if _get(rows, s, f, n, key) is None else f"{_get(rows, s, f, n, key):.3e}"
                              for n in Ns]
                L.append(f"| {s} | " + " | ".join(cells) + " |")
        L.append("")
    (out / "REPORT_advection.md").write_text("\n".join(L))
    print(f"wrote {out / 'REPORT_advection.md'}")


if __name__ == "__main__":
    sys.exit(main())
