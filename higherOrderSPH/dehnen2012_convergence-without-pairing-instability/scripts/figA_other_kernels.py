#!/usr/bin/env python3
"""Supplementary figure A — the library kernels OUTSIDE the D&A (2012) set.

Figs 1-2 replicate the paper's kernel set (Table 1 + HOCT4 + Gaussian,
plus the b7/b8 family extension). This script runs the REMAINING kernels
in src/warpSPHCore/kernels/kernelFunctions/ through the same test
battery, to separate the genuine density kernels from the special-purpose
ones:

  poly6, spiky            candidate general density kernels
  adhesion, cohesion      surface-tension force kernels
  viscosity               viscosity (Laplacian) kernel

Per-kernel battery (3D; f(q) and C_d from the SHIPPED warp functions,
q = r/H; no re-transcription):
  * normalisation   4 pi C3 int_0^1 f q^2 dq   (= 1 for a density kernel;
                          N vs N/4 convergence check -- adhesion has a
                          fractional-power corner at q = 0.5 and 1)
  * non-negativity  min f on [0, 1]
  * centre          f(0), f'(0)   (C^2 needs f'(0) = 0 and no singularity)
  * support end     f'(1), f''(1) (C^2 needs both 0)
  * 3D FT (eq. 14)  wbar(0) = normalisation, min wbar and first zero on
                          |k|sigma in [0, 3 pi] (negative lobe =
                          pairing-unstable, the D&A criterion)

Common scale: each kernel is scaled to ITS OWN h = 2 sigma, i.e.
H_i/h = 1/(2 sqrt(sigma2/H2)) from the shape's normalised moments
(the D&A Table-1 convention; C3-independent, so it stays meaningful
for the non-normalised special-purpose kernels). NOTE: the code ships
kernelScale = 1.0 for all five (h_code = H), which does NOT follow
that convention (e.g. poly6's shape gives H/h = 1.6583) -- recorded
in the PLAN.md findings log, not changed here (src/ decisions are
Phase 8).

The viscosity kernel is singular at the origin (f ~ 0.5/q); the FT and
moment integrands are finite there (f(q) q -> 0.5, f(q) q^m -> 0) and
the q = 0 endpoint is taken as that limit.

The top W(r) panel caps the axis at W = 1 (WCAP): the viscosity kernel
is singular (W ~ 1/r) and would otherwise dominate the axis (its value
at the first grid point is ~3.5e2).

Checks (assert, exit non-zero on failure):
  * moment integrals converged (N vs N/4, relative 1e-7)
  * poly6, spiky, viscosity: |normalisation - 1| < 1e-8 (their shipped
    C_d IS a density normalisation)
Expected FAILURES are reported, not asserted: the adhesion kernel is
not normalised, cohesion is negative at the centre, the viscosity
kernel is singular at the centre, spiky is not C^2 at the centre.

Output: figures/figA_other_kernels.{png,pdf} (gitignored).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import C_d, init, shape, simpson
from fig02_fourier_transforms import plot_signed
from stability import parse_kernel_arg

_HERE = Path(__file__).resolve().parent
FIGDIR = _HERE.parent / "figures"

ORDER = ["poly6", "spiky", "adhesion", "cohesion", "viscosity"]

LABELS = {
    "poly6": "poly6",
    "spiky": "spiky",
    "adhesion": "adhesion",
    "cohesion": "cohesion",
    "viscosity": "viscosity",
}

# Not in any paper figure (our extension) -- distinct tab10 colours.
COLORS = {
    "poly6": "#1f77b4",
    "spiky": "#ff7f0e",
    "adhesion": "#2ca02c",
    "cohesion": "#d62728",
    "viscosity": "#9467bd",
}

# kernels whose shipped C_d is a density normalisation (W integrates to 1)
NORMALISED = ["poly6", "spiky", "viscosity"]

KMAX_SIG = 3.0 * math.pi  # |k|sigma axis (same as fig02)
NK = 12001
YFLOOR = 1e-6             # log-axis floor (same as fig02)
N_FT = 20001              # Simpson points in q for the FT
N_MOM = 200001            # Simpson points in q for the moments
R0 = 0.02                 # top-panel start for the singular viscosity curve
WCAP = 1.0                # top-panel W(r) cap (viscosity is singular, W ~ 1/r)


def _simpson_weights(n: int) -> np.ndarray:
    w = np.full(n, 2.0)
    w[0] = w[-1] = 1.0
    w[1:-1:2] = 4.0
    return w


def moment(f, order: int, n: int = N_MOM) -> float:
    """int_0^1 f(q) q^order dq; the q -> 0 endpoint is taken as a limit
    (f(q) q^order -> 0 for the viscosity kernel, order >= 1)."""
    q = np.linspace(0.0, 1.0, n)
    fq = np.empty_like(q)
    m = q > 0.0  # f(0) is inf for the viscosity kernel; skip the multiply
    fq[m] = np.atleast_1d(f(q[m])) * q[m]**order
    fq[0] = float(np.atleast_1d(f(np.array([1e-6])))[0]) * 1e-6**order
    return simpson(fq, q)


def kernel_data(name: str) -> dict:
    """C3, H/h (shape-derived, D&A convention), f(q) from the shipped code.

    sigma2/H2 uses the normalised second moment (1/3) int f q^4 / int f q^2
    (independent of C3): equal to the paper's eq. 8 (1/nu int |x|^2 W)
    for the normalised kernels, but the only meaningful definition for
    the non-normalised special-purpose ones (adhesion, cohesion).
    """
    C3 = C_d(3, name)

    def f(r):
        return shape(np.atleast_1d(r), 3, name)

    s2 = moment(f, 4) / moment(f, 2) / 3.0
    scale = 1.0 / (2.0 * math.sqrt(s2))  # H/h

    return dict(C3=C3, scale3=scale, shape=f)


def W(d: dict, r: np.ndarray) -> np.ndarray:
    """W(r) = C3 f(r/H)/H^3, r in units of h (h = 2sigma = 1)."""
    r = np.asarray(r, float)
    H = d["scale3"]
    q = r / H
    out = np.zeros_like(r)
    m = q <= 1.0
    out[m] = d["C3"] / H**3 * np.atleast_1d(
        d["shape"](np.clip(q[m], 0.0, 1.0)))
    return out


def ft3d(f, C3: float, k: np.ndarray, n: int = N_FT) -> np.ndarray:
    """eq. 14, wbar(kappa) = 4 pi C3 / kappa int_0^1 f q sin(kappa q) dq.

    The q = 0 endpoint is finite for every kernel here (f(q) q -> 0.5
    for the singular viscosity kernel), so the endpoint value never
    affects the sum (sin(0) = 0) and is set to the limit.
    """
    k = np.atleast_1d(np.asarray(k, float))
    q = np.linspace(0.0, 1.0, n)
    h = 1.0 / (n - 1)
    wts = _simpson_weights(n) * h / 3.0
    fq = np.empty_like(q)
    m = q > 0.0  # f(0) is inf for the viscosity kernel; skip the multiply
    fq[m] = np.atleast_1d(f(q[m])) * q[m]
    fq[0] = float(np.atleast_1d(f(np.array([1e-6])))[0]) * 1e-6
    out = np.empty_like(k)
    for i0 in range(0, len(k), 128):
        kk = k[i0:i0 + 128]
        S = np.sin(np.outer(kk, q))
        out[i0:i0 + 128] = 4.0 * math.pi * C3 * \
            np.sum(S * fq * wts, axis=1) / kk
    return out


def checks(ks: dict, kernels: list | None = None) -> dict:
    kernels = kernels or ORDER
    info: dict = {}
    for name in kernels:
        d = ks[name]
        f = d["shape"]
        i = info[name] = {}
        i["C3"] = d["C3"]
        i["scale3"] = d["scale3"]

        # -- moments (with N vs N/4 convergence) ----------------------------
        # adhesion's shape has q^(1/4) corners at q = 0.5 and q = 1,
        # which limit Simpson convergence to ~1e-6 (relative); the other
        # kernels are smooth (converge to ~1e-13).
        m2 = moment(f, 2)
        m2h = moment(f, 2, n=N_MOM // 4 + 1)  # even number of intervals
        conv = abs(m2 - m2h) / max(abs(m2), 1e-300)
        ctol = 1e-5 if name == "adhesion" else 1e-7
        assert conv < ctol, f"{name}: moment not converged ({conv:.1e})"
        i["norm"] = 4.0 * math.pi * d["C3"] * m2
        if name in NORMALISED:
            assert abs(i["norm"] - 1.0) < 1e-8, \
                f"{name}: normalisation = {i['norm']:.12f}"

        # -- shape diagnostics ------------------------------------------------
        q = np.linspace(0.0, 1.0, 20001)
        i["minf"] = float(np.atleast_1d(f(q)).min())
        i["f0"] = float(np.atleast_1d(f(np.array([0.0])))[0])
        if math.isfinite(i["f0"]):
            i["fp0"] = (float(f(np.array([1e-4]))[0]) - i["f0"]) / 1e-4
        else:
            i["fp0"] = float("inf")
        # support-end smoothness (C2 needs f'(1) = f''(1) = 0). The
        # finite differences are O(h) for the C2 kernels (h = 5e-4 ->
        # <= 0.025) vs O(1) for spiky (6) / adhesion (~1e5).
        h1 = 5e-4
        f1, fm1, fm2 = (float(f(np.array([v]))[0])
                        for v in (1.0, 1.0 - h1, 1.0 - 2.0 * h1))
        i["fp1"] = (f1 - fm1) / h1
        i["fpp1"] = (f1 - 2.0 * fm1 + fm2) / h1**2

        # -- FT ----------------------------------------------------------------
        xh = np.linspace(0.0, 2.0 * KMAX_SIG, NK)  # kappa-hat = |k|h
        w = ft3d(f, d["C3"], xh[1:] * d["scale3"])
        i["wbar"] = np.concatenate([[i["norm"]], w])  # wbar(0) = norm
        i["minw"] = float(w.min())
        i["firstzero_sig"] = None
        s = np.sign(w)
        flip = np.where((s[1:] != s[:-1]) & (np.abs(w[1:]) > 1e-13)
                        & (np.abs(w[:-1]) > 1e-13))[0]
        if flip.size:
            i["firstzero_sig"] = float(xh[flip[0] + 1]) / 2.0
    return info


def verdict(name: str, i: dict) -> tuple[bool, str]:
    """Is this usable as a general density kernel? + reasons."""
    reasons = []
    ok = True
    if abs(i["norm"] - 1.0) > 1e-6:
        ok = False
        reasons.append(f"not normalised (4πC₃∫fq² = {i['norm']:.4f})")
    if i["minf"] < -1e-12:
        ok = False
        reasons.append(f"negative (min f = {i['minf']:.3e})")
    if not math.isfinite(i["f0"]):
        ok = False
        reasons.append("singular at the centre (f(0) = ∞)")
    elif abs(i["fp0"]) > 1e-2:
        ok = False
        reasons.append(f"not differentiable at the centre (f'(0) = {i['fp0']:.2f})")
    if abs(i["fp1"]) > 1e-2 or abs(i["fpp1"]) > 1e-1:
        ok = False
        reasons.append(f"not C² at the support (f'(1) = {i['fp1']:.1e}, "
                       f"f''(1) = {i['fpp1']:.1e})")
    return ok, "; ".join(reasons)


def pairing_note(i: dict) -> str:
    if i["minw"] < -1e-10:
        z = (f", first zero {i['firstzero_sig']:.3f}"
             if i["firstzero_sig"] else "")
        return f"pairing-unstable (min w̄ = {i['minw']:.3e}{z} |k|σ)"
    return f"pairing-stable (min w̄ = {i['minw']:+.2e})"


def plot(ks: dict, info: dict, kernels: list | None = None) -> None:
    kernels = kernels or ORDER
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
    })

    # --- top: shapes ---------------------------------------------------------
    # The viscosity kernel is singular (W ~ 1/r): plot it from R0 and
    # cap the axis at WCAP so the singularity does not dominate the
    # panel (every regular kernel peaks at <= 0.73, spiky).
    Hmax = max(ks[n]["scale3"] for n in kernels)
    r_top = np.linspace(0.0, Hmax * 1.12, 3301)
    curves = {}
    for n in kernels:
        r, w = r_top, W(ks[n], r_top)
        if n == "viscosity":  # singular at r = 0: start off the origin
            m = r >= R0
            curves[n] = (r[m], w[m])
        else:
            curves[n] = (r, w)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.5, 9),
        gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.12},
    )

    for n in kernels:
        r, w = curves[n]
        ax1.plot(r, w, color=COLORS[n], lw=1.6, label=LABELS[n])

    # support arrows: vertical, pointing down at r = H (same convention
    # as fig01)
    for n in kernels:
        H = ks[n]["scale3"]
        ax1.annotate(
            "", xy=(H, 0.0), xytext=(H, 0.96 * WCAP),
            arrowprops=dict(arrowstyle="-|>", color=COLORS[n],
                            lw=1.0, alpha=0.85),
        )
    if "viscosity" in kernels:
        ax1.text(0.02 * Hmax, 0.92 * WCAP,
                 "viscosity singular at r = 0 (leaves the panel at W = 1)",
                 fontsize=8, color=COLORS["viscosity"])

    ax1.set_xlim(0.0, Hmax * 1.12)
    ax1.set_ylim(-0.12 * WCAP, WCAP)
    ax1.set_ylabel("W(r)")
    ax1.set_title("Kernels outside the D&A (2012) set (ν = 3) — "
                  "shapes + FT at common h = 2σ")
    ax1.legend(loc="upper right")

    # --- bottom: FT, log |wbar| over |k|sigma in [0, 3 pi] ------------------
    x = 0.5 * np.linspace(0.0, 2.0 * KMAX_SIG, NK)  # |k|sigma
    for n in kernels:
        plot_signed(ax2, x, info[n]["wbar"], COLORS[n], YFLOOR)

    ytop = max(1.0, max(info[n]["wbar"][0] for n in kernels)) * 1.1
    ax2.set_yscale("log")
    ax2.set_xlim(0.0, KMAX_SIG)
    ax2.set_ylim(YFLOOR, ytop)
    ax2.set_xticks([0.0, math.pi, 2.0 * math.pi, 3.0 * math.pi])
    ax2.set_xticklabels(["0", r"$\pi$", r"$2\pi$", r"$3\pi$"])
    ax2.set_xlabel(r"$|k|\,\sigma$   ($\sigma = h/2$)")
    ax2.set_ylabel(r"$|\bar W(k)|$")
    ax2.legend(handles=[plt.Line2D([], [], color=COLORS[n], lw=1.6,
                                   label=LABELS[n]) for n in kernels],
               loc="center right")

    FIGDIR.mkdir(exist_ok=True)
    fig.savefig(FIGDIR / "figA_other_kernels.png", dpi=150)
    fig.savefig(FIGDIR / "figA_other_kernels.pdf")
    plt.close(fig)
    print(f"saved {FIGDIR / 'figA_other_kernels.png'} (+ .pdf)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", action="append", default=None,
                    help="kernel(s) to plot, repeatable and/or comma-separated; "
                         f"known: {ORDER} (default: all five)")
    args = ap.parse_args()
    kernels = parse_kernel_arg(args.kernel, default=ORDER, valid=ORDER)

    init()
    ks = {n: kernel_data(n) for n in kernels}
    info = checks(ks, kernels)

    print("kernels outside the D&A set, scaled to their own h = 2sigma "
          "(3D, shipped warp functions):")
    print(f"{'kernel':<12}{'C₃':>11}{'H/h':>10}{'norm':>11}{'min f':>11}"
          f"{'f(0)':>10}{'f\'(1)':>10}{'min w̄':>11}{'z1 |k|σ':>10}")
    for n in kernels:
        i = info[n]
        f0 = "∞" if not math.isfinite(i["f0"]) else f"{i['f0']:.4f}"
        fp1 = f"{i['fp1']:.2e}"
        z = f"{i['firstzero_sig']:.3f}" if i["firstzero_sig"] else "—"
        print(f"{n:<12}{i['C3']:>11.6f}{i['scale3']:>10.6f}"
              f"{i['norm']:>11.6f}{i['minf']:>11.3e}{f0:>10}{fp1:>10}"
              f"{i['minw']:>11.3e}{z:>10}")

    print("\nverdict (general density kernel? / pairing stability):")
    for n in kernels:
        i = info[n]
        ok, why = verdict(n, i)
        print(f"  {n:<12} {'YES' if ok else 'no':<4} "
              f"{(why + '; ' if why else '') + pairing_note(i)}")

    plot(ks, info, kernels)


if __name__ == "__main__":
    main()
