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

The y-axis is h/d_nn (the paper's resolution scale h = H/kernelScale, in
units of the nearest-neighbour spacing), LINEAR scale, bounds [0.9, 3] —
matching the paper's own axis; a secondary axis on the right shows the
corresponding N_H (the code's neighbour number = the paper's support-volume
count; kernel-dependent, since h/d_nn = (H/d_nn)/kernelScale but N_H is a
function of H alone). The x-axis is |k|d_nn on a LOG scale, matching the
paper. The Gaussian (16-sigma truncation) is shown at N_H = N_h * 8^3.

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

# the |k|d_nn grid (log-spaced, matching the paper's log x-axis) and the
# h/d_nn bounds (matching the paper's y-axis; converted to a kernel-specific
# N_H grid in main(), since h = H/kernelScale is kernel-dependent).
KDN_MIN, KDN_MAX = 0.1, 6.0
KDN_GRID = np.logspace(math.log10(KDN_MIN), math.log10(KDN_MAX), 26)
HDN_MIN, HDN_MAX = 0.9, 3.0
DIRS = {
    "111": np.array([1, 1, 1.0]) / np.sqrt(3),
    "110": np.array([1, 1.0, 0.0]) / np.sqrt(2),
}


def hdn_grid_to_NH(lat: st.Lattice, kernel_scale: float, hdn_grid) -> np.ndarray:
    """h/d_nn values -> N_H, for one kernel (h = H/kernel_scale)."""
    H = np.asarray(hdn_grid) * kernel_scale * lat.dnn
    return np.array([lat.NH_of_H(float(h)) for h in H])


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


def plot_kernel(name: str, lat: st.Lattice, hdn_grid: np.ndarray,
                nh_grid: np.ndarray, lon111, tr111, lon110, tr110) -> None:
    kernel_scale = common.kernel_scale(3, name)
    dnn = lat.dnn

    # vectorized (scalar-in/scalar-out, array-in/array-out) so matplotlib's
    # secondary_yaxis internals get back exactly the shape they passed in
    def hdn_to_NH(hdn):
        H = np.asarray(hdn, dtype=float) * kernel_scale * dnn
        return (4.0 * math.pi / 3.0) * H ** 3 / lat.m

    def NH_to_hdn(NH):
        H = (np.asarray(NH, dtype=float) * lat.m / (4.0 * math.pi / 3.0)) ** (1.0 / 3.0)
        return H / (kernel_scale * dnn)

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 11})
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)

    def panel(a, lon, tr, which, dname, kdd, right_axis):
        field = (lon if which == "lon" else tr).T          # (len(hdn), len(KDN))
        # centred on 1 (RdBu_r: low->blue, high->red) so the background is
        # pale near the continuum value and only genuinely large deviations
        # (incl. all omega^2<=0, which is <= vmin) read as strongly coloured
        a.pcolormesh(KDN_GRID, hdn_grid, field, shading="auto",
                     cmap="RdBu_r", vmin=0.0, vmax=2.0)
        a.contour(KDN_GRID, hdn_grid, field, levels=[0.0], colors="red",
                  linewidths=1.6)
        a.contour(KDN_GRID, hdn_grid, field, levels=[1.0], colors="cyan",
                  linewidths=1.0)
        a.contour(KDN_GRID, hdn_grid, field,
                  levels=[0.95, 0.99, 1.01, 1.05], colors="green",
                  linewidths=0.7, alpha=0.7)
        a.set_xscale("log")
        a.set_xlim(KDN_MIN, KDN_MAX)
        a.set_xticks([0.1, 0.5, 1, 5])
        a.set_xticks([], minor=True)
        a.set_xticklabels(["0.1", "0.5", "1", "5"])
        a.set_yscale("linear")
        a.set_ylim(HDN_MIN, HDN_MAX)
        a.set_title(f"k // {dname}  ({which})", fontsize=10)
        if right_axis:
            sec = a.secondary_yaxis("right", functions=(hdn_to_NH, NH_to_hdn))
            sec.set_ylabel(r"$N_H$")

    panel(ax[0, 0], lon111, tr111, "lon", "111", DIRS["111"], False)
    panel(ax[0, 1], lon110, tr110, "lon", "110", DIRS["110"], True)
    panel(ax[1, 0], lon111, tr111, "tr", "111", DIRS["111"], False)
    panel(ax[1, 1], lon110, tr110, "tr", "110", DIRS["110"], True)

    ax[0, 0].set_ylabel(r"$h/d_{nn}$")
    ax[1, 0].set_ylabel(r"$h/d_{nn}$")
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
    ap.add_argument("--hdn-min", type=float, default=HDN_MIN,
                    help="min h/d_nn for the grid (default 0.9, paper's bound)")
    ap.add_argument("--hdn-max", type=float, default=HDN_MAX,
                    help="max h/d_nn for the grid (default 3.0, paper's bound)")
    ap.add_argument("--n-hdn", type=int, default=14,
                    help="number of h/d_nn rows (default 14)")
    args = ap.parse_args()
    kernels = st.parse_kernel_arg(args.kernel,
                                  default=["cubic_b4", "gaussian"])

    common.init()
    hdn_grid = np.linspace(args.hdn_min, args.hdn_max, args.n_hdn)
    # N_H(H) only depends on H via rho/m = N/L^3 = const (rho=1 fixed), so a
    # small reference lattice is enough to convert the h/d_nn bound to N_H
    # regardless of which N the real (per-kernel) lattice ends up using.
    lat_ref = st.Lattice(N=4000, L=1.0)

    for name in kernels:
        kernel_scale = common.kernel_scale(3, name)
        nh_max = lat_ref.NH_of_H(args.hdn_max * kernel_scale * lat_ref.dnn)
        lat = st.lattice_for_NH(nh_max)
        nh_grid = hdn_grid_to_NH(lat, kernel_scale, hdn_grid)
        print(f"\n=== {st.LABELS[name]}  (N={lat.N}, h/d_nn {hdn_grid[0]:.2f}.."
              f"{hdn_grid[-1]:.2f} -> N_H {nh_grid[0]:.1f}..{nh_grid[-1]:.1f}, "
              f"|k|d_nn {KDN_GRID[0]:.2f}..{KDN_GRID[-1]:.2f}) ===")
        print("  k // 111:")
        lon111, tr111 = omega2_fields(name, lat, DIRS["111"], nh_grid, KDN_GRID)
        print("  k // 110:")
        lon110, tr110 = omega2_fields(name, lat, DIRS["110"], nh_grid, KDN_GRID)
        plot_kernel(name, lat, hdn_grid, nh_grid, lon111, tr111, lon110, tr110)


if __name__ == "__main__":
    main()
