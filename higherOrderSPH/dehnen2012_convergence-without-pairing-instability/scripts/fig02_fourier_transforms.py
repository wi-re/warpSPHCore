#!/usr/bin/env python3
"""Figure 2 of D&A (2012) — Fourier transforms at a common h = 2sigma.

Replicates Fig. 2: "Fourier transforms W(k)b for the Gaussian, the HOCT4
and the kernels of Table 1 scaled to the same common scale h = 2sigma.
Negative values are plotted with broken curves."

The figure set is extended with b7/b8 (same B-spline family as b4-b6,
shipped as enum B7/B8; NOT part of the paper's Fig. 2 -- added
2026-09-18 on user request).

As in the paper, |W̄| is plotted on a log axis (1e-6 to 1) over
|k|σ ∈ [0, 3π] (σ = h/2; equivalently κ̂ = |k|h ∈ [0, 6π] — the
cubic spline's first zero sits just over π on the paper's axis);
broken (dashed) segments mark where W̄ < 0.

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
  * non-negativity: min wbar >= -1e-10 on the plot grid [0, 6π] for
    Wendland C2/C4/C6, HOCT4, Gaussian (pairing-stable); b8 uses the
    audit's 1e-6 practical floor (its FT has a tiny lobe, min -8.47e-7
    at kappa-hat 13.11, exit zero 14.30, below that floor)
  * B-splines (b4-b6 + b7) "oscillate about zero": >= 3 zeros of wbar
    in [0, 40] (closed form, bisection-refined); first zeros recorded
    -- they set the pairing criterion kappa_0 > kappa_Nyquist

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

# The paper's Fig. 2 x-axis is |k|sigma, sigma = h/2: in kappa-hat =
# |k|h units the data range is [0, 2 * KMAX_PLOT] = [0, 6 pi], which
# contains the first zero of EVERY B-spline (3.44, 4.29, 5.56 in
# |k|sigma units).
KMAX_PLOT = 3.0 * math.pi  # plot x-range, in |k|sigma units
NK_PLOT = 12001            # d(|k|sigma) ~ 1.6e-3
YFLOOR = 1e-6              # the paper's log-axis floor
N_SIMPSON = 20001  # Simpson points in r (error ~1e-14 at these k)
# b7/b8 join 2026-09-18 (user request): b7 oscillates (tiny negative
# lobes -> "barely pairing-unstable"); b8 is classified non-negative
# like the audit's ft_expected -- its FT does have a tiny lobe (min
# -8.47e-7 at kappa-hat 13.11, exit zero 14.30, entry zero below the
# resolvable ~1e-13 floor) but below the audit's 1e-6 practical floor.
BSPLINES = ["cubic_b4", "quartic_b5", "quintic_b6", "b7"]
NONNEG = ["wendland_C2", "wendland_C4", "wendland_C6", "b8", "hoct4",
          "gaussian"]
# non-negativity floor per kernel; b8 uses the audit's 1e-6 practical
# noise floor (scripts/kernels/kernel_audit.py _FT_NOISE_FLOOR)
NONNEG_FLOOR = {"b8": -1e-6}


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

    A grid sign-flip at i only counts if the window [i-side, i+side]
    (0.1 in kappa-hat) contains a true excursion: a value below -floor
    AND a value above +floor. This rejects:
    * tangential near-zeros, where wbar touches ~0 from one side only
      (e.g. b5 at kappa-hat ~ 7.78, 15.56, 23.34, |wbar| ~ 1e-11: the
      closed form's ~1e-16 rounding flips the sign across the touch);
    * sign flips inside the closed form's ~1e-15 noise floor far out,
      where the decaying FT has dropped below round-off.
    The bisection bracket is [window edge, window extremum of the
    opposite sign], so it starts from verified opposite signs even for
    very flat crossings (b7's zero at 32.347 sits at the edge of a
    ~1e-9 lobe). Verified against the high-resolution numerical FT
    (2026-09-18): b4-b6 reproduce the Phase-2 zeros, and b7's zeros
    13.330 / 22.917 / 32.347 (first-lobe min -3.17e-6 at 14.43) replace
    the stale Phase-1 audit estimates 21.97 / 22.02 / 31.45.
    """
    H = d["scale3"]
    s = np.sign(w)
    out = []
    for i in np.where(s[1:] != s[:-1])[0]:
        lo = max(i - side, 0)
        hi = min(i + side + 1, len(w))
        wlo, whi = float(w[lo:hi].min()), float(w[lo:hi].max())
        if wlo > -floor or whi < floor:
            continue  # no true excursion: tangential / noise floor
        a, fa = xg[lo], w[lo]
        if fa < 0.0:
            j = lo + int(np.argmax(w[lo:hi]))
        else:
            j = lo + int(np.argmin(w[lo:hi]))
        b, fb = xg[j], w[j]
        for _ in range(80):
            m = 0.5 * (a + b)
            fm = float(ft3d_closed(d["pieces"], d["C3"],
                                   np.array([m * H]))[0])
            if fa * fm <= 0.0:
                b, fb = m, fm
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

    # -- non-negativity (Wendland, b8, HOCT4, Gaussian) ---------------------
    for n in NONNEG:
        mn = float(num[n].min())
        lim = NONNEG_FLOOR.get(n, -1e-10)
        assert mn >= lim, f"{n}: min wbar on [0,6π] = {mn:.3e}"
        info[n]["min12"] = mn

    # -- B-splines oscillate about zero -------------------------------------
    # [0, 40] in kappa-hat: b7's third zero (32.347) needs the extra
    # range.
    zg = np.linspace(0.0, 40.0, 20001)  # d = 2e-3, kappa-hat units
    for n in BSPLINES:
        d = ks[n]
        w = ft3d_closed(d["pieces"], d["C3"], zg[1:] * d["scale3"])
        zeros = find_zeros(d, zg[1:], w)
        assert len(zeros) >= 3, f"{n}: only {len(zeros)} zeros in [0,40]"
        info[n]["zeros40"] = zeros
    return info


def plot_signed(ax, x: np.ndarray, y: np.ndarray, color: str,
                floor: float, lw: float = 1.6):
    """Plot |y| (for a log axis): solid where y >= 0, broken where y < 0
    (the paper's convention for negative values), clipped at `floor`."""
    ys = np.abs(y)
    s = np.sign(y)
    xs, ysv, ss = list(x), list(ys), list(s)
    for i in np.where((s[1:] != s[:-1]) & (s[1:] != 0) & (s[:-1] != 0))[0]:
        xc = x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
        xs.insert(int(i) + 1, float(xc))
        ysv.insert(int(i) + 1, floor)
        ss.insert(int(i) + 1, s[i + 1])  # crossing joins the new sign
    xs, ysv, ss = np.asarray(xs), np.asarray(ysv), np.asarray(ss)
    start = 0
    for i in range(1, len(xs)):
        if ss[i] != ss[i - 1]:
            _seg(ax, xs[start:i + 1], ysv[start:i + 1], color, lw, ss[start])
            start = i
    _seg(ax, xs[start:], ysv[start:], color, lw, ss[start])


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
    x = 0.5 * np.linspace(0.0, 2.0 * KMAX_PLOT, NK_PLOT)  # |k|sigma
    for n in ORDER:
        plot_signed(ax, x, num[n], COLORS[n], YFLOOR)

    # first zeros of the B-splines (set the pairing criterion); stored
    # in kappa-hat = |k|h units, plotted halved in |k|sigma units.
    # Staggered label heights avoid overlap (3.44, 4.29, 5.56); b7's
    # first zero (10.98) lies beyond the 3-pi axis and is skipped.
    for iy, n in enumerate(BSPLINES):
        z1 = 0.5 * info[n]["zeros40"][0]
        if z1 > KMAX_PLOT:
            continue
        ax.axvline(z1, color=COLORS[n], ls=":", lw=0.9, alpha=0.6)
        ax.text(z1, [0.55, 0.30, 0.10, 0.55][iy % 4], f"z$_1$ = {z1:.3f}",
                ha="center", fontsize=8, color=COLORS[n],
                bbox=dict(facecolor="white", alpha=0.7,
                          edgecolor="none", pad=0.8))

    ax.set_yscale("log")
    ax.set_xlim(0.0, KMAX_PLOT)
    ax.set_ylim(YFLOOR, 1.0)
    ax.set_xticks([0.0, math.pi, 2.0 * math.pi, 3.0 * math.pi])
    ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$", r"$3\pi$"])
    ax.set_xlabel(r"$|k|\,\sigma$   ($\sigma = h/2$)")
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

    xh = np.linspace(0.0, 2.0 * KMAX_PLOT, NK_PLOT)  # kappa-hat = |k|h
    num: dict = {}
    for n in ORDER:
        d = ks[n]
        w = ft3d_numeric(d["shape"], d["C3"], xh[1:] * d["scale3"],
                         n=N_SIMPSON)
        num[n] = np.concatenate([[1.0], w])

    info = checks(ks, num, xh)

    print("kappa-hat = |k|h, h = 2σ = 1;  primary = numerical FT (eq. 14)")
    print(f"{'kernel':<14}{'w̄(0)':>10}{'a2 (fit)':>12}{'a4 (fit)':>12}"
          f"{'min [0,6π]':>12}{'#zeros[0,40]':>13}")
    for n in ORDER:
        i = info[n]
        mn = f"{i['min12']:+.4f}" if "min12" in i else "   (osc.)"
        nz = len(i["zeros40"]) if "zeros40" in i else "—"
        print(f"{n:<14}{i['w0']:>10.7f}{i['a2']:>12.7f}{i['a4']:>12.6e}"
              f"{mn:>12}{nz:>13}")

    print("\nB-spline zeros in [0,40] (kappa-hat = |k|h units; "
          "the figure's axis is |k|sigma = kappa-hat/2):")
    for n in BSPLINES:
        zs = info[n]["zeros40"]
        zstr = "  ".join(f"{z:.4f}" for z in zs)
        print(f"  {n:<14} {zstr}")
        print(f"  {'':<14} z1 = {zs[0]/2:.6f} |k|σ   "
              f"(kappa = H|k|: {zs[0] * ks[n]['scale3']:.4f})")

    print("\nclosed-form vs numerical max|dwbar| on the plot grid:")
    for n in ORDER:
        if "clnum" in info[n]:
            print(f"  {n:<14} {info[n]['clnum']:.3e}")

    plot(ks, num, info)


if __name__ == "__main__":
    main()
