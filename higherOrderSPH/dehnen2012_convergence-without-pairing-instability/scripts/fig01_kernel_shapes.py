#!/usr/bin/env python3
"""Figure 1 of D&A (2012) — kernel shapes at a common resolution scale.

Replicates Fig. 1: "Kernels of Table 1, the Gaussian, and the HOCT4 kernel
of Read et al. (2010) scaled to a common resolution scale of h = 2sigma for
3D (top: linear plot, arrows indicating |x| = H; bottom: logarithmic
plot)."

Scaling: h_paper = 2sigma = 1 (the paper's resolution scale, eq. 8); each
kernel's support H_i = scale_3,i from Table 1, so with r in units of h

    W_i(r) = C3_i f_i(r / H_i) / H_i^3.

Shapes: the audited warp functions (common.shape, float64 on CPU) for the
six Table-1 kernels; HOCT4 and the Gaussian use the verified definitions
from data/da2012_reference.yaml (they enter the core library in Phase 3).

Checks (assert, exit non-zero on failure):
  * W_i(r) >= 0 on [0, 8.4]
  * compact kernels: W_i(r) = 0 exactly for r >= H_i; the Gaussian is
    truncated at H = 16 sigma (W(H) ~ 8e-16, zero beyond)
  * normalisation 4 pi C3 int_0^1 f q^2 dq = 1 (exact polynomial
    integration for the 7 piecewise kernels, Simpson for the Gaussian)

Output: figures/fig01_kernel_shapes.{png,pdf} (gitignored).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import load_reference
from ft_kernels import kernel_shapes, moment2

_HERE = Path(__file__).resolve().parent
FIGDIR = _HERE.parent / "figures"

# D&A Table 1 lists b4-b6; b7/b8 (shipped as enum B7/B8) are the same
# B-spline family and join the figure set 2026-09-18 (user request).
ORDER = [
    "cubic_b4", "quartic_b5", "quintic_b6", "b7", "b8",
    "wendland_C2", "wendland_C4", "wendland_C6",
    "hoct4", "gaussian",
]

LABELS = {
    "cubic_b4": "cubic spline ($b_4$)",
    "quartic_b5": "quartic spline ($b_5$)",
    "quintic_b6": "quintic spline ($b_6$)",
    "b7": "spline $b_7$",
    "b8": "spline $b_8$",
    "wendland_C2": "Wendland C$^2$",
    "wendland_C4": "Wendland C$^4$",
    "wendland_C6": "Wendland C$^6$",
    "hoct4": "HOCT4 (Read 2010)",
    "gaussian": "Gaussian",
}

# HOCT4 is orange in the paper (Fig. 3 caption: "the HOCT4 kernel of Read
# et al. (2010, orange)", colour coding shared with Figs 1-2). b7/b8 are
# not in the paper's figure (our extension) -- remaining tab10 colours.
COLORS = {
    "cubic_b4": "#1f77b4",
    "quartic_b5": "#17becf",
    "quintic_b6": "#9467bd",
    "b7": "#e377c2",
    "b8": "#bcbd22",
    "wendland_C2": "#2ca02c",
    "wendland_C4": "#d62728",
    "wendland_C6": "#8c564b",
    "hoct4": "#ff7f0e",
    "gaussian": "#7f7f7f",
}

CLIP = 1e-6  # log-panel floor (the paper's Fig. 1 bottom axis)


def W(d: dict, r: np.ndarray) -> np.ndarray:
    """W_i(r) = C3 f(r/H)/H^3, r in units of h (h = 2sigma = 1)."""
    r = np.asarray(r, float)
    H = d["scale3"]
    q = r / H
    out = np.zeros_like(r)
    m = q <= 1.0
    out[m] = d["C3"] / H ** 3 * np.atleast_1d(d["shape"](np.clip(q[m], 0.0, 1.0)))
    return out


def check(ks: dict) -> None:
    r = np.linspace(0.0, 8.4, 4201)
    for name in ORDER:
        d = ks[name]
        w = W(d, r)
        H = d["scale3"]

        assert w.min() >= -1e-15, f"{name}: W takes negative values"

        if name != "gaussian":
            beyond = w[r >= H]
            assert np.max(np.abs(beyond)) == 0.0, \
                f"{name}: W != 0 beyond its support H = {H}"
        else:
            assert np.max(np.abs(w[r > H])) == 0.0, \
                "gaussian: not truncated beyond H = 16 sigma"
            wH = float(W(d, np.array([H]))[0])  # exact at r = H
            assert wH < 1e-13, f"gaussian: W(H) = {wH:.3e} too large"

        if d["pieces"] is not None:
            norm = 4.0 * math.pi * d["C3"] * moment2(d["pieces"])
            # HOCT4's C3 (read2010's N_3d) is rounded to 10 dp in the
            # paper; exact value from the piecewise form is
            # 6.515049930564806 (deviation 5e-12 in the normalisation).
            tol = 1e-10 if name == "hoct4" else 1e-12
        else:
            q = np.linspace(0.0, 1.0, 200001)
            norm = 4.0 * math.pi * d["C3"] * np.trapezoid(
                np.atleast_1d(d["shape"](q)) * q * q, q)
            tol = 1e-10
        assert abs(norm - 1.0) < tol, \
            f"{name}: normalisation 4 pi C3 int f q^2 dq = {norm:.12f}"

    print("checks: non-negativity, support, normalisation -- all pass")


def plot(ks: dict) -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
    })

    # --- top: linear -------------------------------------------------------
    r_top = np.linspace(0.0, 2.7, 2701)
    Ws = {n: W(ks[n], r_top) for n in ORDER}
    Wmax = max(w.max() for w in Ws.values())

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.5, 9),
        gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.12},
    )

    for n in ORDER:
        ax1.plot(r_top, Ws[n], color=COLORS[n], lw=1.6, label=LABELS[n])

    # support arrows: vertical, pointing down at r = H (the kernel's
    # zero), caption: "arrows indicating |x| = H"; the Gaussian has no
    # arrow (truncated, not compact) -- its truncation is marked in the
    # log panel.
    compact = [n for n in ORDER if n != "gaussian"]
    for n in compact:
        H = ks[n]["scale3"]
        ax1.annotate(
            "", xy=(H, 0.0), xytext=(H, 1.06 * Wmax),
            arrowprops=dict(arrowstyle="-|>", color=COLORS[n],
                            lw=1.0, alpha=0.85),
        )

    ax1.set_ylim(0.0, 1.06 * Wmax)
    ax1.set_ylabel("W(r)")
    ax1.set_title("D&A (2012) Fig. 1 — kernels at common h = 2σ (ν = 3)")
    ax1.legend(loc="upper right")

    # --- bottom: logarithmic ----------------------------------------------
    # r range ends just past where the (truncated) Gaussian crosses the
    # 1e-6 floor (r ~ 2.6); its 16-sigma truncation (r = H = 8,
    # W ~ 1.7e-56) is far below the visible floor and is noted in text.
    r_bot = np.linspace(0.0, 4.2, 2101)
    for n in ORDER:
        w = W(ks[n], r_bot)
        m = w > CLIP
        ax2.plot(r_bot[m], np.log10(np.maximum(w[m], CLIP)),
                 color=COLORS[n], lw=1.6)
    ax2.text(4.15, math.log10(CLIP) + 0.3,
             "Gaussian truncated at 16σ (r = 8, off panel)",
             ha="right", fontsize=8, color=COLORS["gaussian"])

    ax2.set_xlim(0.0, 4.2)
    ax2.set_ylim(math.log10(CLIP), math.log10(Wmax) + 0.2)
    ax2.set_xlabel("|x| / h   (h = 2σ)")
    ax2.set_ylabel("log₁₀ W(r)")

    FIGDIR.mkdir(exist_ok=True)
    fig.savefig(FIGDIR / "fig01_kernel_shapes.png", dpi=150)
    fig.savefig(FIGDIR / "fig01_kernel_shapes.pdf")
    plt.close(fig)
    print(f"saved {FIGDIR / 'fig01_kernel_shapes.png'} (+ .pdf)")


def main() -> None:
    ref = load_reference()
    ks = kernel_shapes(ref)

    print(f"common scale: h = 2σ = 1 (paper units); support H = scale₃:")
    print(f"{'kernel':<14}{'C₃':>12}{'H/h':>12}{'W(0)':>12}")
    for n in ORDER:
        d = ks[n]
        W0 = d["C3"] / d["scale3"] ** 3 * float(
            np.atleast_1d(d["shape"](np.array([0.0])))[0])
        supp = f"{d['scale3']:.6f}" if n != "gaussian" else "8 (trunc.)"
        print(f"{n:<14}{d['C3']:>12.6f}{supp:>12}{W0:>12.6f}")

    check(ks)
    plot(ks)


if __name__ == "__main__":
    main()
