#!/usr/bin/env python3
"""Figure 4/5 of D&A (2012) — linear-stability contours of conservative SPH
for densest-sphere packing and P ∝ rho^{5/3}.

Replicates Figs 4-5: contours of omega^2_k / c^2 k^2 (top: the longitudinal /
sound mode, the eigenvector most aligned with k) and omega^2_perp2 / c^2 k^2
(bottom: the smallest transverse mode) over wave number |k| and smoothing scale,
both scaled to the nearest-neighbour distance d_nn.  Left column k // (1,1,1),
right column k // (1,1,0).  The P matrix is the EXACT linearisation
(``StabilityOracle.exact_p_matrix``), validated against the ground-truth real-FD
Jacobian of the actual force (phase5_stability_log.md).

Contour convention (paper caption): the red region is omega^2 <= 0 (the pairing
instability); the green contours hug the continuum (omega^2/c^2k^2 = 1, cyan);
the blue contours are the small-omega^2 (log) region.  We plot the field on a
symmetric log color scale with the omega^2 = 0 (red) and omega^2/c^2k^2 = 1
(cyan) contours drawn explicitly, so the long-wavelength longitudinal dip is
visible (see the log: the cubic's omega^2_parallel < 0 at |k|d_nn ~ 0.3-0.6 is
a real feature of the exact P, present for all N_H).

The y-axis is N_H (the code's neighbour number = the paper's support-volume
count); the paper also labels it h·d_nn (= N_H / kernelScale^3 · ...).  The
Gaussian (16-sigma truncation) is shown at N_H = N_h * 8^3.

Usage:
  python fig04_fig05_stability_contours.py                 # default kernels
  python fig04_fig05_stability_contours.py --kernel cubic_b4
  python fig04_fig05_stability_contours.py --kernel cubic_b4,gaussian
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common
import stability as st

_HERE = Path(__file__).resolve().parent
FIGDIR = _HERE.parent / "figures"

# the (|k|d_nn, N_H) grid for the contours
NH_GRID = np.logspace(math.log10(30.0), math.log10(500.0), 14)
KDN_GRID = np.linspace(0.1, 5.0, 26)
DIRS = {
    "111": np.array([1, 1, 1.0]) / np.sqrt(3),
    "110": np.array([1, 1.0, 0.0]) / np.sqrt(2),
}


def omega2_fields(name: str, lat: st.Lattice, kdd: np.ndarray,
                  nh_grid, kdn_grid):
    """(lon, tr): the longitudinal / smallest-transverse omega^2 / (c^2 k^2),
    each of shape (len(kdn_grid), len(nh_grid)), for one (kernel, direction)."""
    dnn = lat.dnn
    lon = np.full((len(kdn_grid), len(nh_grid)), np.nan)
    tr = np.full((len(kdn_grid), len(nh_grid)), np.nan)
    for i, NH in enumerate(nh_grid):
        o = st.StabilityOracle(name, lat, float(NH))
        c2k2 = o.c2 * (kdn_grid / dnn) ** 2          # c^2 k^2 per kdn
        for j, kdn in enumerate(kdn_grid):
            k = kdn * kdd / dnn
            P = o.exact_p_matrix(k)
            w, v = np.linalg.eigh(P)
            kd = k / np.linalg.norm(k)
            ov = np.array([abs(v[:, a] @ kd) for a in range(3)])
            iL = int(np.argmax(ov))                   # longitudinal
            lon[j, i] = w[iL] / c2k2[j]
            itr = int(np.argmin(w))                   # smallest (transverse)
            tr[j, i] = w[itr] / c2k2[j]
        print(f"    N_H = {NH:7.1f}  done")
    return lon, tr


def plot_kernel(name: str, lon111, tr111, lon110, tr110) -> None:
    c = st.COLORS[name]
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 11})
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)

    def panel(a, lon, tr, which, dname, kdd):
        field = (lon if which == "lon" else tr).T          # (len(NH), len(KDN))
        a.pcolormesh(KDN_GRID, NH_GRID, field, shading="auto",
                     cmap="RdBu_r", vmin=-1.0, vmax=2.0)
        a.contour(KDN_GRID, NH_GRID, field, levels=[0.0], colors="red",
                  linewidths=1.6)
        a.contour(KDN_GRID, NH_GRID, field, levels=[1.0], colors="cyan",
                  linewidths=1.0)
        a.contour(KDN_GRID, NH_GRID, field,
                  levels=[0.95, 0.99, 1.01, 1.05], colors="green",
                  linewidths=0.7, alpha=0.7)
        a.set_xscale("linear")
        a.set_yscale("log")
        a.set_title(f"k // {dname}  ({which})", fontsize=10)

    panel(ax[0, 0], lon111, tr111, "lon", "111", DIRS["111"])
    panel(ax[0, 1], lon110, tr110, "lon", "110", DIRS["110"])
    panel(ax[1, 0], lon111, tr111, "tr", "111", DIRS["111"])
    panel(ax[1, 1], lon110, tr110, "tr", "110", DIRS["110"])

    ax[0, 0].set_ylabel(r"$N_H$")
    ax[1, 0].set_ylabel(r"$N_H$")
    ax[0, 0].set_xlabel("")
    ax[0, 1].set_xlabel("")
    ax[1, 0].set_xlabel(r"$|k|\, d_{nn}$")
    ax[1, 1].set_xlabel(r"$|k|\, d_{nn}$")

    fig.suptitle(f"D&A (2012) Figs 4/5 — linear stability, "
                 f"{st.LABELS[name]}  "
                 r"($\omega^2_k / c^2 k^2$ top, $\omega^2_{\perp 2}/c^2 k^2$ "
                 r"bottom; red = unstable, cyan $\omega^2/c^2k^2 = 1$)")

    FIGDIR.mkdir(exist_ok=True)
    out = FIGDIR / f"fig04_fig05_stability_contours_{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out} (+ .pdf)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", action="append", default=None,
                    help="kernel(s), repeatable and/or comma-separated; "
                         f"known: {st.STABILITY_ORDER} "
                         "(default: cubic_b4 + gaussian, the paper's Figs 4-5)")
    ap.add_argument("--nh-max", type=float, default=500.0,
                    help="max N_H for the grid (default 500)")
    args = ap.parse_args()
    kernels = st.parse_kernel_arg(args.kernel,
                                  default=["cubic_b4", "gaussian"])

    global NH_GRID
    NH_GRID = np.logspace(math.log10(30.0), math.log10(args.nh_max), 14)

    common.init()
    lat = st.Lattice(N=4000, L=1.0)

    for name in kernels:
        print(f"\n=== {st.LABELS[name]}  (N_H {NH_GRID[0]:.0f}.."
              f"{NH_GRID[-1]:.0f}, |k|d_nn {KDN_GRID[0]:.1f}.."
              f"{KDN_GRID[-1]:.1f}) ===")
        print("  k // 111:")
        lon111, tr111 = omega2_fields(name, lat, DIRS["111"], NH_GRID, KDN_GRID)
        print("  k // 110:")
        lon110, tr110 = omega2_fields(name, lat, DIRS["110"], NH_GRID, KDN_GRID)
        plot_kernel(name, lon111, tr111, lon110, tr110)


if __name__ == "__main__":
    main()
