#!/usr/bin/env python3
"""Refit the D&A (2012) eq. 19 constants in 1D, 2D and 3D.

The paper fits `eps = eps_100 (N_H/100)^(-alpha)` (eq. 19) against the
eq. 18 self-term correction `rho_hat_corr = rho_hat - eps m W(0, H)` in
3D only, for the three Wendland kernels. This script does the same fit
on the *densest lattice* of each dimension -- 1D uniform, 2D hexagonal,
3D FCC (all from `warpSPHCore.sampling.lattice.sampleDensestLattice`,
the same source fig03's FCC config uses) -- for the full
Wendland + B-spline set, so the library feature
(`warpSPHCore/util/densityCorrection.py`) can ship a
`(dim, kernel) -> (eps_100, alpha, validity window)` table that is
parameterisation-invariant (built entirely from the shipped kernels).

Procedure (per dimension, per kernel):
  * lattice at mean density rho = 1 (mass = box volume / count),
    translated so a lattice point sits at the origin (translational
    invariance: the origin's sum is the exact lattice sum);
  * for each N_H on the same log grid as fig03 (20..800, 34 points):
    H = (N_H m / V_d)^(1/d), V_1, V_2, V_3 = 2, pi, 4 pi/3;
    rho_hat = m sum_j W(|x_j|, H) including the self term (shipped
    kernel, W(r, H) = C_d f(r/H)/H^d);
    implied eps(N_H) = (rho_hat - 1) / (m W(0, H)) (eq. 18 solved for
    eps, W(0, H) at the CODE support = the paper's H -- common.W0);
  * least-squares fit of log|eps| vs log(N_H/100) over a window; the
    B-splines UNDER-estimate, so their implied eps is NEGATIVE (the
    eq. 18 machinery and the paper's constant-in-h Lagrangian argument
    are linear in eps, so a negative eps -- adding back a fraction of
    the self-term -- is well defined and removes the under-estimate).

Self-test: the 3D path must reproduce the paper's Wendland constants
within x1.5 (fig03's own check #4 threshold) and the Phase-4 FCC refit
within x1.02 (this run is the same computation, so it is a
transcription guard, not a new measurement).

Outputs:
  * printed table (fit constants, log-residual and corrected-curve
    band over the fit window, per fit window tried)
  * results/eps_constants_multidim.json  (the machine-readable table
    the library constants are transcribed from)
  * results/eps_constants_multidim.txt   (the printed table)

Run:  python scripts/eps_constants_multidim.py [--kernel K[,K...]]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from common import C_d, W0, check, shape
from stability import parse_kernel_arg
from warpSPHCore.sampling.lattice import sampleDensestLattice

_HERE = Path(__file__).resolve().parent
RESDIR = _HERE.parent / "results"

# The user-requested set: the three Wendland + the five B-splines.
ORDER = [
    "cubic_b4", "quartic_b5", "quintic_b6", "b7", "b8",
    "wendland_C2", "wendland_C4", "wendland_C6",
]
BSPLINES = {"cubic_b4", "quartic_b5", "quintic_b6", "b7", "b8"}

NH_GRID = np.logspace(math.log10(20.0), math.log10(800.0), 34)  # as in fig03
V = {1: 2.0, 2: math.pi, 3: 4.0 * math.pi / 3.0}
#: Fit windows to report; [40, 400] is the paper's 3D window.
WINDOWS = [(40.0, 400.0), (40.0, 800.0), (20.0, 800.0)]
#: Lattice sizes: large enough that the N_H = 800 support (the widest
#: sum) stays clear of the box half-width in every dimension.
LATTICE_N = {1: 2048, 2: 4096, 3: 4000}


def H_of_NH(N_H: float, mass: float, dim: int) -> float:
    """Support radius H from N_H = V_d H^d (rho/m) at rho = 1."""
    return (N_H * mass / V[dim]) ** (1.0 / dim)


def originDistances(dim: int) -> tuple[np.ndarray, float, np.ndarray]:
    """(r_j sorted, mass, box) from the densest lattice at rho = 1,
    translated so a lattice point sits at the origin."""
    lat = sampleDensestLattice(LATTICE_N[dim], 1.0, dim)
    box = lat.box
    pos = (lat.positions - lat.positions[0]) % box  # lattice point at origin
    d = pos - box * np.round(pos / box)  # minimum image
    r = np.sqrt((d ** 2).sum(-1))
    mass = float(np.prod(box)) / lat.count
    order = np.argsort(r)
    return r[order], mass, box


def fitWindow(eps: np.ndarray, NH: np.ndarray, lo: float, hi: float
              ) -> tuple[float, float]:
    """Least-squares `eps = eps_100 (N_H/100)^(-alpha)` in log-log over
    [lo, hi] (|eps| for the sign; the sign is restored on eps_100).
    Returns (eps_100, alpha); the residual and corrected-curve band are
    the caller's (they need the window mask and rho_hat)."""
    m = (NH >= lo) & (NH <= hi)
    slope, log_eps100 = np.polyfit(np.log(NH[m] / 100.0),
                                   np.log(np.abs(eps[m])), 1)
    sign = 1.0 if eps[m].min() > 0 else -1.0
    return sign * float(math.exp(log_eps100)), float(-slope)


def rawDensity(r_sorted: np.ndarray, mass: float, dim: int,
               name: str, N_H: float) -> float:
    """rho_hat at the origin for one (kernel, N_H) on the lattice."""
    H = H_of_NH(N_H, mass, dim)
    rr = r_sorted[r_sorted < H]
    f = shape(rr / H, dim, name)
    return float(C_d(dim, name) * mass * f.sum() / H ** dim)


def ratiosAndEps(r_sorted: np.ndarray, mass: float, dim: int,
                 name: str) -> tuple[np.ndarray, np.ndarray]:
    """(rho_hat, implied eps) per N_H for one kernel on the lattice."""
    rho = np.array([rawDensity(r_sorted, mass, dim, name, N_H)
                    for N_H in NH_GRID])
    eps = np.empty_like(NH_GRID)
    for i, N_H in enumerate(NH_GRID):
        eps[i] = (rho[i] - 1.0) / (mass * W0(name, dim, H_of_NH(N_H, mass, dim)))
    return rho, eps


def windowBand(r_sorted: np.ndarray, mass: float, dim: int, name: str,
               eps100: float, alpha: float, lo: float, hi: float,
               npts: int = 41) -> float:
    """max |corrected - 1| over a DENSE sweep of the window, evaluated
    exactly as the LIBRARY evaluates it (design note section 4): the
    runtime N_H is estimated from the RAW estimate,
    N_H,est = V_d H^d rho_hat / m (= the exact N_H times rho_hat on
    this lattice), NOT the exact N_H -- at the window edges the bias is
    a few percent, so the two conventions differ by O(alpha * bias) in
    eps. Also dense because the 34-point fit grid misses window edges
    (e.g. N_H = 40.0 is not a grid point; the first grid point in
    [40, 400] is 43.76), where the bias is largest."""
    worst = 0.0
    for N_H in np.linspace(lo, hi, npts):
        H = H_of_NH(N_H, mass, dim)
        rho_hat = rawDensity(r_sorted, mass, dim, name, N_H)
        n_h_est = V[dim] * H ** dim * rho_hat / mass
        eps = eps100 * (n_h_est / 100.0) ** (-alpha)
        worst = max(worst, abs(rho_hat - eps * mass * W0(name, dim, H) - 1.0))
    return float(worst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", action="append", default=None,
                    help="kernel(s) to fit, repeatable and/or comma-separated; "
                         f"known: {ORDER} (default: all eight)")
    args = ap.parse_args()
    kernels = parse_kernel_arg(args.kernel, default=ORDER)

    lats = {}
    for dim in (1, 2, 3):
        r, mass, box = originDistances(dim)
        lats[dim] = (r, mass, box)
        half = 0.5 * float(box.min())
        Hmax = H_of_NH(NH_GRID.max(), mass, dim)
        print(f"dim {dim}: N = {lats[dim][0].size:5d}  box = "
              + " x ".join(f"{b:.6f}" for b in box)
              + f"  mass = {mass:.6e}  H(N_H=800) = {Hmax:.5f} "
              f"(< half-box {half:.5f}: {Hmax < half})")
        assert Hmax < half, "lattice too small for the widest sum"

    table = {}
    for dim in (1, 2, 3):
        r, mass, box = lats[dim]
        print(f"\n=== dim {dim} ===")
        data = {n: ratiosAndEps(r, mass, dim, n) for n in kernels}
        for n in kernels:
            rho, eps = data[n]
            entry = {"eps_sign": "positive" if eps.min() > 0 else "negative",
                     "fits": {}}
            for lo, hi in WINDOWS:
                eps100, alpha = fitWindow(eps, NH_GRID, lo, hi)
                m = (NH_GRID >= lo) & (NH_GRID <= hi)
                fitted = eps100 * (NH_GRID[m] / 100.0) ** (-alpha)
                resid = float(np.max(np.abs(np.log10(np.abs(fitted / eps[m])))))
                H = np.array([H_of_NH(x, mass, dim) for x in NH_GRID[m]])
                corr = (rho[m] - fitted * mass
                        * np.array([W0(n, dim, h) for h in H]))
                band = float(np.max(np.abs(corr - 1.0)))
                raw = float(np.max(np.abs(rho[m] - 1.0)))
                fit = {
                    "eps_100": eps100, "alpha": alpha,
                    "max_log10_resid": resid, "max_corr_dev": band,
                    "max_raw_dev": raw}
                if (lo, hi) == (40.0, 400.0):
                    fit["window_band_dense"] = windowBand(
                        r, mass, dim, n, eps100, alpha, lo, hi)
                entry["fits"][f"{lo:.0f}-{hi:.0f}"] = fit
            table[(dim, n)] = entry
            f40 = entry["fits"]["40-400"]
            print(f"  {n:<14} {entry['eps_sign']:<8} 40-400: "
                  f"eps_100 = {f40['eps_100']:.5f}  alpha = {f40['alpha']:.3f}"
                  f"  log-resid = {f40['max_log10_resid']:.2e}"
                  f"  raw band = {f40['max_raw_dev']:.2e}"
                  f"  corr band (grid/dense) = {f40['max_corr_dev']:.2e}/"
                  f"{f40['window_band_dense']:.2e}")

    # Self-test: 3D vs the paper's Wendland constants (fig03 threshold x1.5)
    # and vs fig03 check #4's own output (the IDENTICAL computation: same
    # FCC lattice/mass, W0, grid and 40-400 window -- a transcription
    # guard, so x1.005).
    from common import load_reference
    ref = load_reference()
    fig03_3d = {"wendland_C2": (0.02949, 0.999),
                "wendland_C4": (0.01361, 1.633),
                "wendland_C6": (0.01131, 2.218)}
    print("\nself-test (3D, window 40-400):")
    for n in [k for k in kernels if k in fig03_3d]:
        e = table[(3, n)]["fits"]["40-400"]
        p = ref["density_correction"][n]
        ok_p = max(e["eps_100"] / p["eps_100"], p["eps_100"] / e["eps_100"]) <= 1.5
        ok_a = max(e["alpha"] / p["alpha"], p["alpha"] / e["alpha"]) <= 1.5
        f = fig03_3d[n]
        ok_f = (max(e["eps_100"] / f[0], f[0] / e["eps_100"]) <= 1.005
                and max(e["alpha"] / f[1], f[1] / e["alpha"]) <= 1.005)
        check(f"{n} vs paper x1.5 / fig03-3D x1.005",
              (ok_p and ok_a, ok_f), (True, True))
        assert ok_p and ok_a and ok_f

    # Dump.
    RESDIR.mkdir(exist_ok=True)
    out = {f"{dim}d_{name}": table[(dim, name)] for dim, name in table}
    (RESDIR / "eps_constants_multidim.json").write_text(json.dumps(
        out, indent=2))
    txt = json.dumps(out, indent=2)
    (RESDIR / "eps_constants_multidim.txt").write_text(txt + "\n")
    print(f"\nsaved {RESDIR / 'eps_constants_multidim.json'} (+ .txt)")


if __name__ == "__main__":
    main()
