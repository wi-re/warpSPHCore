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

Resolution (closeout item 5): the grid is `--grid ROWSXKDN` (default
14x26, the original figure resolution; the high-res sweep is 60x200).
The ROWS rows are LOG-spaced in N_H over the kernel's N_H range
corresponding to h/d_nn in [hdn-min, hdn-max] (N_H ~ hdn^3, so log-N_H
spacing resolves the small-N_H instability islands; the original 14
linear-hdn rows were approximately this), and the KDN columns are
log-spaced over [0.1, 6.0] |k|d_nn.

Each kernel's fields are cached to results/stability_<kernel>.npz
(gitignored) and loaded if present, so an interrupted sweep resumes and
the figures regenerate without recompute; --recompute forces a
recompute.

Usage:
  python fig04_fig05_stability_contours.py                 # all ten kernels
  python fig04_fig05_stability_contours.py --kernel cubic_b4
  python fig04_fig05_stability_contours.py --kernel cubic_b4,gaussian
  python fig04_fig05_stability_contours.py --grid 60x200   # the high-res sweep
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
RESDIR = _HERE.parent / "results"

# the |k|d_nn range (log-spaced, matching the paper's log x-axis) and the
# h/d_nn bounds (matching the paper's y-axis; the rows are converted to a
# kernel-specific N_H grid in main(), since h = H/kernelScale is
# kernel-dependent).
KDN_MIN, KDN_MAX = 0.1, 6.0
HDN_MIN, HDN_MAX = 0.9, 3.0
DIRS = {
    "111": np.array([1, 1, 1.0]) / np.sqrt(3),
    "110": np.array([1, 1.0, 0.0]) / np.sqrt(2),
}


def kdn_grid(n_kdn: int) -> np.ndarray:
    return np.logspace(math.log10(KDN_MIN), math.log10(KDN_MAX), n_kdn)


def NH_grid(log_nh_min: float, log_nh_max: float, n_rows: int) -> np.ndarray:
    """LOG-spaced N_H rows (the plan's 'N_H (log)' resolution axis)."""
    return np.logspace(log_nh_min, log_nh_max, n_rows)


def NH_to_hdn(lat: st.Lattice, kernel_scale: float, nh_grid) -> np.ndarray:
    """N_H -> h/d_nn = H/(kernelScale d_nn), for one kernel."""
    return np.array([lat.H_of_NH(float(nh)) / (kernel_scale * lat.dnn)
                     for nh in nh_grid])


def cache_path(name: str) -> Path:
    return RESDIR / f"stability_{name}.npz"


def load_cache(name: str, n_rows: int, n_kdn: int, hdn_min: float,
               hdn_max: float):
    """The cached (kernel, grid) fields, or None if absent / stale (a
    different grid) so an interrupted sweep resumes but a grid change
    recomputes."""
    p = cache_path(name)
    if not p.exists():
        return None
    d = np.load(p)
    if (int(d["n_rows"]) != n_rows or int(d["n_kdn"]) != n_kdn
            or abs(float(d["kdn_min"]) - KDN_MIN) > 1e-12
            or abs(float(d["kdn_max"]) - KDN_MAX) > 1e-12
            or abs(float(d["hdn_min"]) - hdn_min) > 1e-12
            or abs(float(d["hdn_max"]) - hdn_max) > 1e-12):
        return None
    return d


def save_cache(name: str, n_rows: int, n_kdn: int, lat: st.Lattice,
               kernel_scale: float, hdn_min: float, hdn_max: float,
               nh_grid, hdn_grid, kdn, lon111, tr111, lon110, tr110) -> None:
    RESDIR.mkdir(exist_ok=True)
    np.savez_compressed(
        cache_path(name),
        name=name, N=lat.N, dnn=lat.dnn, kernel_scale=kernel_scale,
        hdn_min=hdn_min, hdn_max=hdn_max, kdn_min=KDN_MIN, kdn_max=KDN_MAX,
        n_rows=n_rows, n_kdn=n_kdn,
        nh_grid=nh_grid, hdn_grid=hdn_grid, kdn_grid=kdn,
        lon111=lon111, tr111=tr111, lon110=lon110, tr110=tr110)
    print(f"cached {cache_path(name)}")


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
                nh_grid: np.ndarray, kdn_grid: np.ndarray,
                lon111, tr111, lon110, tr110) -> None:
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
        field = (lon if which == "lon" else tr).T          # (len(hdn), len(kdn))
        # centred on 1 (RdBu_r: low->blue, high->red) so the background is
        # pale near the continuum value and only genuinely large deviations
        # (incl. all omega^2<=0, which is <= vmin) read as strongly coloured
        a.pcolormesh(kdn_grid, hdn_grid, field, shading="auto",
                     cmap="RdBu_r", vmin=0.0, vmax=2.0)
        a.contour(kdn_grid, hdn_grid, field, levels=[0.0], colors="red",
                  linewidths=1.6)
        a.contour(kdn_grid, hdn_grid, field, levels=[1.0], colors="cyan",
                  linewidths=1.0)
        a.contour(kdn_grid, hdn_grid, field,
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
                         "(default: all ten)")
    ap.add_argument("--grid", type=str, default="14x26",
                    help="ROWSxKDN resolution: log-N_H rows x log-|k|d_nn "
                         "columns (default 14x26, the original figures; "
                         "the closeout item-5 sweep is 60x200)")
    ap.add_argument("--hdn-min", type=float, default=HDN_MIN,
                    help="min h/d_nn for the grid (default 0.9, paper's bound)")
    ap.add_argument("--hdn-max", type=float, default=HDN_MAX,
                    help="max h/d_nn for the grid (default 3.0, paper's bound)")
    ap.add_argument("--recompute", action="store_true",
                    help="recompute instead of loading the results/ cache")
    args = ap.parse_args()
    n_rows, n_kdn = (int(v) for v in args.grid.lower().split("x"))
    kernels = st.parse_kernel_arg(args.kernel)   # default: the full set

    common.init()
    kdn = kdn_grid(n_kdn)
    # N_H(H) only depends on H via rho/m = N/L^3 = const (rho=1 fixed), so a
    # small reference lattice is enough to convert the h/d_nn bounds to the
    # N_H range regardless of which N the real (per-kernel) lattice ends up
    # using.
    lat_ref = st.Lattice(N=4000, L=1.0)

    for name in kernels:
        kernel_scale = common.kernel_scale(3, name)
        nh_min = lat_ref.NH_of_H(args.hdn_min * kernel_scale * lat_ref.dnn)
        nh_max = lat_ref.NH_of_H(args.hdn_max * kernel_scale * lat_ref.dnn)
        cached = (None if args.recompute
                  else load_cache(name, n_rows, n_kdn,
                                  args.hdn_min, args.hdn_max))
        if cached is not None:
            nh_grid = cached["nh_grid"]
            hdn_grid = cached["hdn_grid"]
            lon111, tr111 = cached["lon111"], cached["tr111"]
            lon110, tr110 = cached["lon110"], cached["tr110"]
            lat = st.Lattice(N=int(cached["N"]), L=1.0)
            print(f"\n=== {st.LABELS[name]}  (loaded cache {cache_path(name)}) ===")
        else:
            lat = st.lattice_for_NH(nh_max)
            nh_grid = NH_grid(math.log10(nh_min), math.log10(nh_max), n_rows)
            hdn_grid = NH_to_hdn(lat, kernel_scale, nh_grid)
            print(f"\n=== {st.LABELS[name]}  (N={lat.N}, "
                  f"N_H {nh_grid[0]:.1f}..{nh_grid[-1]:.1f} log-spaced "
                  f"({n_rows} rows), |k|d_nn {kdn[0]:.2f}..{kdn[-1]:.2f} "
                  f"({n_kdn} columns)) ===")
            print("  k // 111:")
            lon111, tr111 = omega2_fields(name, lat, DIRS["111"], nh_grid, kdn)
            print("  k // 110:")
            lon110, tr110 = omega2_fields(name, lat, DIRS["110"], nh_grid, kdn)
            save_cache(name, n_rows, n_kdn, lat, kernel_scale,
                       args.hdn_min, args.hdn_max, nh_grid, hdn_grid, kdn,
                       lon111, tr111, lon110, tr110)
        plot_kernel(name, lat, hdn_grid, nh_grid, kdn,
                    lon111, tr111, lon110, tr110)


if __name__ == "__main__":
    main()
