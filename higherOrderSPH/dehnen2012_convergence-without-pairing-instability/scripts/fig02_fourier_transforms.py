#!/usr/bin/env python3
"""Figure 2 of D&A (2012) — Fourier transforms at a common h = 2sigma.

Replicates Fig. 2: "Fourier transforms W(k)b for the Gaussian, the HOCT4
and the kernels of Table 1 scaled to the same common scale h = 2sigma.
Negative values are plotted with broken curves."

With kappa-hat = |k| h (h = 2sigma = 1), the 3D FT (eq. 14) depends only
on kappa = H|k| = kappa-hat * (H/h):

    wbar(kappa-hat) = 4 pi C3 / kappa  int_0^1 f(r) r sin(kappa r) dr,
    wbar(0) = 1.

The ft_kernels FT functions take kappa = H|k|, so every evaluation here
passes kappa = kappa-hat * scale_3.

Primary curve: numerical FT, composite Simpson in r (n = 20001; the
Simpson error at kappa-hat <= 12 is ~1e-14, far below the plot scale).
Cross-check: exact closed form for the 7 piecewise-polynomial kernels,
derived from the piecewise definitions (supersedes the paper's eq. 15,
whose PDF transcription is garbled -- see PLAN.md findings log). The
Gaussian (truncated at 16 sigma) is numerical only.

Checks (assert, exit non-zero on failure):
  * wbar(0) = 1 for all 8 kernels (normalisation)
  * closed form vs numerical: max|dwbar| < 1e-8 on the plot grid
  * eq. 17 at small wave numbers: wbar = 1 - kappa-hat^2/8 + O(k^4) for
    ALL kernels (they "all overlap at small wave numbers"): fit the
    numerical FT on [0.05, 0.5], |a2 + 1/8| < 1e-4
  * non-negativity: min wbar >= -1e-10 on [0, 12] for Wendland C2/C4/C6,
    HOCT4, Gaussian (pairing-stable per the paper)
  * B-splines "oscillate about zero": >= 3 zeros of wbar in [0, 30]
    (closed form, bisection-refined); first zeros recorded -- they set
    the pairing criterion kappa_0 > kappa_Nyquist

Output: figures/fig02_fourier_transforms.{png,pdf} (gitignored).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import load_reference
from ft_kernels import (ft3d_closed, ft3d_numeric, kernel_shapes,
                        moment2)
from fig01_kernel_shapes import COLORS, LABELS, ORDER

_HERE = Path(__file__).resolve().parent
FIGDIR = _HERE.parent / "figures"

KMAX_PLOT = 12.0   # covers the first zero of every B-spline
NK_PLOT = 12001    # d(kappa-hat) = 1e-3
N_SIMPSON = 20001  # Simpson points in r (error ~1e-14 at these k)
BSPLINES = ["cubic_b4", "quartic_b5", "quintic_b6"]
NONNEG = ["wendland_C2", "wendland_C4", "wendland_C6", "hoct4", "gaussian"]


def wbar0(name: str, d: dict) -> float:
    """wbar(0) = 4 pi C3 int_0^1 f q^2 dq (exact / Simpson)."""
    if d["pieces"] is not None:
        return 4.0 * math.pi * d["C3"] * moment2(d["pieces"])
    q = np.linspace(0.0, 1.0, 200001)
    return 4.0 * math.pi * d["C3"] * np.trapezoid(
        np.atleast_1d(d["shape"](q)) * q * q, q)


def find_zeros(d: dict, xg: np.ndarray, w: np.ndarray,
               floor: float = 1e-13, side: int = 25) -> list[float]:
    """Real zeros (sign changes) of w (closed form on a kappa-hat grid),
    bisection-refined on the closed form. Returns zeros in kappa-hat
    units.

    A candidate crossing only counts if the values `side` grid points
    (0.05 in kappa-hat) to its left and right are of opposite sign and
    both exceed `floor` in magnitude. This rejects:
    * tangential near-zeros, where wbar touches ~0 from one side only
      (e.g. b5 at kappa-hat ~ 7.78, 15.56, 23.34, |wbar| ~ 1e-11: the
      closed form's ~1e-16 rounding flips the sign across the touch);
    * sign flips inside the closed form's ~1e-15 noise floor far out,
      where the decaying FT has dropped below round-off.
    Verified against the high-resolution numerical FT (2026-09-18).
    """
    H = d["scale3"]
    s = np.sign(w)
    out = []
    for i in np.where(s[1:] != s[:-1])[0]:
        if i - side < 0 or i + side + 1 > len(w):
            continue
        wl, wh = w[i - side], w[i + side]
        if wl * wh > 0.0:
            continue  # tangential near-zero: same sign on both sides
        if min(abs(wl), abs(wh)) < floor:
            continue  # below the closed form's noise floor
        # bisect on the closed form, starting from the wide verified
        # bracket (the scan-grid bracket can straddle the ~1e-16 noise
        # floor for very flat crossings)
        a, fa = xg[i - side], wl
        b = xg[i + side]
        for _ in range(80):
            m = 0.5 * (a + b)
            fm = float(ft3d_closed(d["pieces"], d["C3"],
                                   np.array([m * H]))[0])
            if fa * fm <= 0.0:
                b = m
            else:
                a, fa = m, fm
        out.append(0.5 * (a + b))
    # candidates from one crossing all bisect to the same point
    dgrid = xg[1] - xg[0]
    merged = []
    for z in out:
        if merged and z - merged[-1] < 2 * side * dgrid:
            continue
        merged.append(z)
    return merged


def checks(ks: dict, num: dict, x: np.ndarray) -> dict:
    info: dict = {}

    # -- wbar(0) = 1 -------------------------------------------------------
    for n in ORDER:
        w0 = wbar0(n, ks[n])
        assert abs(w0 - 1.0) < 1e-10, f"{n}: wbar(0) = {w0:.12f}"
        info.setdefault(n, {})["w0"] = w0

    # -- closed form vs numerical on the plot grid -------------------------
    for n in ORDER:
        d = ks[n]
        if d["pieces"] is None:
            continue
        cl = ft3d_closed(d["pieces"], d["C3"], x[1:] * d["scale3"])
        dmax = float(np.max(np.abs(cl - num[n][1:])))
        assert dmax < 1e-8, f"{n}: closed vs numeric max|dwbar| = {dmax:.3e}"
        info[n]["clnum"] = dmax

    # -- eq. 17: wbar = 1 - kappa^2/8 + O(kappa^4) at small kappa ----------
    kt = np.linspace(0.05, 0.5, 46)
    for n in ORDER:
        d = ks[n]
        wt = ft3d_numeric(d["shape"], d["C3"], kt * d["scale3"], n=N_SIMPSON)
        a4, a2, a0 = np.polyfit(kt ** 2, wt - 1.0, 2)
        assert abs(a2 + 1.0 / 8.0) < 1e-4, \
            f"{n}: eq. 17 slope a2 = {a2:.8f} (expect -1/8)"
        info[n]["a2"], info[n]["a4"] = float(a2), float(a4)

    # -- non-negativity (Wendland, HOCT4, Gaussian) -------------------------
    for n in NONNEG:
        mn = float(num[n].min())
        assert mn >= -1e-10, f"{n}: min wbar on [0,12] = {mn:.3e}"
        info[n]["min12"] = mn

    # -- B-splines oscillate about zero -------------------------------------
    zg = np.linspace(0.0, 30.0, 15001)  # d = 2e-3, kappa-hat units
    for n in BSPLINES:
        d = ks[n]
        w = ft3d_closed(d["pieces"], d["C3"], zg[1:] * d["scale3"])
        zeros = find_zeros(d, zg[1:], w)
        assert len(zeros) >= 3, f"{n}: only {len(zeros)} zeros in [0,30]"
        info[n]["zeros30"] = zeros
    return info


def plot_signed(ax, x: np.ndarray, y: np.ndarray, color: str, lw: float = 1.6):
    """Solid where y >= 0, broken where y < 0 (the paper's convention)."""
    xs, ys = list(x), list(y)
    s = np.sign(y)
    for i in np.where((s[1:] != s[:-1]) & (s[1:] != 0) & (s[:-1] != 0))[0]:
        xc = x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
        xs.insert(int(i) + 1, float(xc))
        ys.insert(int(i) + 1, 0.0)
    xs, ys = np.asarray(xs), np.asarray(ys)
    s2 = np.sign(ys)
    start = 0
    for i in range(1, len(xs)):
        if s2[i] != s2[i - 1]:
            _seg(ax, xs[start:i + 1], ys[start:i + 1], color, lw, s2[start])
            start = i
    _seg(ax, xs[start:], ys[start:], color, lw, s2[start])


def _seg(ax, x, y, color, lw, sign):
    if len(x) < 2:
        return
    ax.plot(x, y, color=color, lw=lw,
            ls=(0, (4, 2)) if sign < 0 else "-")


def plot(ks: dict, num: dict, info: dict) -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
    })
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.linspace(0.0, KMAX_PLOT, NK_PLOT)
    for n in ORDER:
        plot_signed(ax, x, num[n], COLORS[n])

    # first zeros of the B-splines (set the pairing criterion)
    ytop = 1.09
    for n in BSPLINES:
        z1 = info[n]["zeros30"][0]
        ax.axvline(z1, color=COLORS[n], ls=":", lw=0.9, alpha=0.6)
        ax.text(z1, ytop, f"z$_1$ = {z1:.3f}", ha="center", fontsize=8,
                color=COLORS[n])

    ax.axhline(0.0, color="0.8", lw=0.7)
    ax.set_xlim(0.0, KMAX_PLOT)
    ax.set_ylim(-0.2, ytop)
    ax.set_xlabel(r"$|k|\,h$   ($h = 2\sigma$)")
    ax.set_ylabel(r"$\bar W(k)$")
    ax.set_title("D&A (2012) Fig. 2 — Fourier transforms at common "
                 "h = 2σ (ν = 3)")
    ax.legend(handles=[plt.Line2D([], [], color=COLORS[n], lw=1.6,
                                  label=LABELS[n]) for n in ORDER],
              loc="upper right")

    FIGDIR.mkdir(exist_ok=True)
    fig.savefig(FIGDIR / "fig02_fourier_transforms.png", dpi=150)
    fig.savefig(FIGDIR / "fig02_fourier_transforms.pdf")
    plt.close(fig)
    print(f"saved {FIGDIR / 'fig02_fourier_transforms.png'} (+ .pdf)")


def main() -> None:
    ref = load_reference()
    ks = kernel_shapes(ref)

    x = np.linspace(0.0, KMAX_PLOT, NK_PLOT)  # kappa-hat = |k|h
    num: dict = {}
    for n in ORDER:
        d = ks[n]
        w = ft3d_numeric(d["shape"], d["C3"], x[1:] * d["scale3"],
                         n=N_SIMPSON)
        num[n] = np.concatenate([[1.0], w])

    info = checks(ks, num, x)

    print("kappa-hat = |k|h, h = 2σ = 1;  primary = numerical FT (eq. 14)")
    print(f"{'kernel':<14}{'w̄(0)':>10}{'a2 (fit)':>12}{'a4 (fit)':>12}"
          f"{'min [0,12]':>12}{'#zeros[0,30]':>13}")
    for n in ORDER:
        i = info[n]
        mn = f"{i['min12']:+.4f}" if "min12" in i else "   (osc.)"
        nz = len(i["zeros30"]) if "zeros30" in i else "—"
        print(f"{n:<14}{i['w0']:>10.7f}{i['a2']:>12.7f}{i['a4']:>12.6e}"
              f"{mn:>12}{nz:>13}")

    print("\nB-spline zeros in [0,30] (kappa-hat; z1 also in H units):")
    for n in BSPLINES:
        zs = info[n]["zeros30"]
        zstr = "  ".join(f"{z:.4f}" for z in zs)
        print(f"  {n:<14} {zstr}")
        print(f"  {'':<14} z1 = {zs[0]:.6f}   "
              f"(kappa = H|k|: {zs[0] * ks[n]['scale3']:.4f})")

    print("\nclosed-form vs numerical max|dwbar| on the plot grid:")
    for n in ORDER:
        if "clnum" in info[n]:
            print(f"  {n:<14} {info[n]['clnum']:.3e}")

    plot(ks, num, info)


if __name__ == "__main__":
    main()
