#!/usr/bin/env python3
"""Figure 3 of D&A (2012) — density estimate vs neighbour number N_H.

Replicates Fig. 3: "SPH density estimate (1) obtained from particles in
three-dimensional densest-sphere packing (solid), glass (squares), or
pairing (crosses), plotted against neighbour number N_H for the kernels
of Table 1 and the HOCT4 kernel (colour coding as in Figs. 1&2). For the
Wendland kernels, the corrected density estimate (18) is shown with
dashed curves and triangles for densest-sphere packing, and a glass,
respectively."

Extension beyond the paper: b7, b8 and the Gaussian join the ten
shipped kernels (the paper plots Table 1 + HOCT4 and notes the Gaussian
is not shown).

Setup (particle_configs.py, all at mean density rho = 1):
  FCC    4000 particles (p = 10 FCC cells), box 1^3
  glass  4096 particles (16^3 lattice start), box 1^3, relaxed with the
         warpSPH delta-shift components (cached; see particle_configs)
  paired 8000 particles, box 2^(1/3) -- the section-5.1.1 fully-paired
         distribution (each FCC point -> two coincident particles,
         spacing x 2^(1/3))

The x-axis is N_H = (4 pi/3) H^3 (rho/m) per configuration (its own
mass). rho_hat_i = sum_j m W(|x_i - x_j|, H) INCLUDING the self term
(the paper's self-contribution discussion, eq. 18); the plotted value
is mean_i rho_hat_i / rho.

Checks (assert, exit non-zero on failure):
  * FCC translational invariance: every particle's estimate identical
  * section-5.1.1 identity: paired(H) == FCC(H / 2^(1/3)) for every
    kernel (exact for spherically symmetric kernels with finite W(0))
  * section-5.1.1 pairing criterion: "pairing occurs if
    rho_hat(N_H/f) < rho_hat(N_H) for some 1 < f <= 2 ... for the
    B-spline kernels rho_hat(N_H) always has a minimum and hence
    satisfies our condition, while this never occurs for the Wendland
    or HOCT4 kernels." Grid version: the B-spline FCC curves are
    non-monotone (an adjacent rise, grid ratio 1.117 < 2) with an
    interior minimum below 1; the Wendland/HOCT4/Gaussian curves are
    monotone decreasing and stay above 1
  * eq. 19 fit to the FCC over-estimation of the three Wendland
    kernels, eps = eps_100 (N_H/100)^(-alpha): within a factor <= 1.5
    of the paper's (0.0294, 0.977) / (0.01342, 1.579) / (0.0116, 2.236)
  * eq. 18 correction with the PAPER's constants: within 5% of 1 over
    40 <= N_H <= 400 (FCC and glass, all three Wendland kernels)

Also printed and saved to results/: the Table 2 N_H bookkeeping -- the
code's packingRatio is the paper's h (rho/m)^(1/3) itself, so the
code's N_H = (4 pi/3) (packing * kernelScale)^3 reproduces the paper's
Table 2 N_H exactly, except the deliberate deviations (cubic x 1.0175,
quintic / B7 / B8 x 1.1425; PLAN findings log).

Output: figures/fig03_density_estimation.{png,pdf} (gitignored).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (C_d, W0, check, kernel_scale, load_reference, packing_ratio,
                    shape as common_shape)
from particle_configs import ParticleConfig, fccConfig, glassConfig, pairedConfig

_HERE = Path(__file__).resolve().parent
FIGDIR = _HERE.parent / "figures"
RESDIR = _HERE.parent / "results"

# Same kernel set and colour coding as fig01/fig02 (Figs 1-2 of the
# paper; b7/b8/gaussian are our extension, same colours there).
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
BSPLINES = {"cubic_b4", "quartic_b5", "quintic_b6", "b7", "b8"}
WENDLANDS = ("wendland_C2", "wendland_C4", "wendland_C6")

NH_GRID = np.logspace(math.log10(20.0), math.log10(500.0), 30)
V3 = 4.0 * math.pi / 3.0
FIT_RANGE = (40.0, 400.0)  # eq. 19 fit / eq. 18 check range


def H_of_NH(N_H: float, mass: float) -> float:
    """Support radius H from N_H = (4 pi/3) H^3 (rho/m) at rho = 1."""
    return (N_H * mass / V3) ** (1.0 / 3.0)


def pairList(cfg: ParticleConfig, H: float, onlyParticle0: bool = False):
    """Minimum-image pairs with r < H as (i, r); includes the self pair
    (r = 0) -- the density estimate (eq. 1) includes the self term."""
    pos, L = cfg.positions, cfg.box
    N = 1 if onlyParticle0 else cfg.n
    pos0 = pos[:N]
    ii, rr = [], []
    for s in range(0, N, 1024):
        d = pos0[s:s + 1024][:, None, :] - pos[None, :, :]
        d -= L * np.round(d / L)
        r = np.sqrt((d ** 2).sum(-1))
        msk = r < H
        idx = np.nonzero(msk)
        ii.append(np.arange(s, s + 1024)[idx[0]])
        rr.append(r[idx])
    return np.concatenate(ii), np.concatenate(rr)


def ratios(cfg: ParticleConfig, H: float, names: list,
           onlyParticle0: bool = False) -> dict:
    """mean_i rho_hat_i / rho for each kernel (rho = 1 by construction).

    rho_hat_i = sum_j m W(x_i - x_j, H) with the SHIPPED kernel
    W(r, H) = C_d f(r/H) / H^3 (f = common.shape, unit-support shape)."""
    ii, rr = pairList(cfg, H, onlyParticle0)
    n = 1 if onlyParticle0 else cfg.n
    out = {}
    for name in names:
        f = common_shape(rr / H, 3, name)
        out[name] = float(C_d(3, name) * cfg.mass * np.bincount(
            ii, weights=f, minlength=n).mean() / H**3)
    return out


def fccRatios(fcc: ParticleConfig, H: float, names: list) -> dict:
    """FCC: every particle is equivalent (translational invariance), so
    particle 0 gives the exact lattice sum."""
    return ratios(fcc, H, names, onlyParticle0=True)


def allRatios(fcc: ParticleConfig, glass: ParticleConfig,
              paired: ParticleConfig, names: list) -> dict:
    """{name: {key: (N_H, ratio)}} with keys 'fcc', 'glass', 'paired'
    and 'fcc_half' = the FCC lattice sum at H / 2^(1/3) (the reference
    for the section-5.1.1 identity check)."""
    data = {n: {} for n in names}
    for k, N_H in enumerate(NH_GRID):
        if k == 0 or (k + 1) % 5 == 0:
            print(f"  N_H = {N_H:8.1f}  (H = {H_of_NH(N_H, fcc.mass):.5f})")
        for cfg, key in ((fcc, "fcc"), (glass, "glass"), (paired, "paired")):
            H = H_of_NH(N_H, cfg.mass)
            r = ratios(cfg, H, names, onlyParticle0=(cfg is fcc))
            for n in names:
                data[n].setdefault(key, ([], []))
                data[n][key][0].append(N_H)
                data[n][key][1].append(r[n])
        H_half = H_of_NH(N_H, fcc.mass) / 2.0**(1.0 / 3.0)
        r_half = fccRatios(fcc, H_half, names)
        for n in names:
            data[n].setdefault("fcc_half", ([], []))
            data[n]["fcc_half"][0].append(N_H)
            data[n]["fcc_half"][1].append(r_half[n])
    return {n: {k: (np.array(v[0]), np.array(v[1]))
                for k, v in d.items()}
            for n, d in data.items()}


def checks(data: dict, fcc: ParticleConfig, glass: ParticleConfig,
           paired: ParticleConfig, ref: dict) -> None:
    print("checks:")

    # 1. FCC translational invariance (particle 0 vs every particle).
    H = H_of_NH(100.0, fcc.mass)
    ii, rr = pairList(fcc, H)
    f = common_shape(rr / H, 3, "cubic_b4")
    rho_hat = (np.bincount(ii, weights=f, minlength=fcc.n) * C_d(3, "cubic_b4")
               * fcc.mass / H**3)
    spread = float(rho_hat.max() - rho_hat.min())
    check("FCC: all particles identical (cubic, N_H = 100)", spread, 0.0,
          atol=1e-12)
    assert spread < 1e-12, f"FCC not translation invariant: spread {spread}"

    # 2. section-5.1.1 identity: paired(H) == FCC(H / 2^(1/3)).
    worst = 0.0
    for n in ORDER:
        worst = max(worst, float(np.max(np.abs(
            data[n]["paired"][1] - data[n]["fcc_half"][1]))))
    check("5.1.1 identity: paired(H) == FCC(H/2^(1/3)) all kernels",
          worst, 0.0, atol=1e-10)
    assert worst < 1e-10, f"paired != FCC(H/2^(1/3)): worst {worst}"

    # 3. section-5.1.1 pairing criterion. The paper: "pairing occurs if
    # rho_hat(N_H/f) < rho_hat(N_H) for some 1 < f <= 2. From Fig. 3 we
    # see that for the B-spline kernels rho_hat(N_H) always has a minimum
    # and hence satisfies our condition, while this never occurs for the
    # Wendland or HOCT4 kernels." On the grid, "has a minimum" means the
    # curve is non-monotone (an adjacent rise, grid ratio 1.117 < 2) with
    # an interior minimum below 1; "never occurs" means the curve is
    # monotone decreasing, staying above 1.
    print("  5.1.1 criterion: B-splines have a minimum (the curve rises "
          "somewhere); Wendland/HOCT4/Gaussian never do:")
    for n in ORDER:
        NH, r = data[n]["fcc"]
        i_min = int(np.argmin(r))
        rises = int(np.sum(r[1:] > r[:-1]))
        interior = 0 < i_min < len(r) - 1
        if n in BSPLINES:
            ok = interior and r[i_min] < 1.0 and rises > 0
        else:
            ok = rises == 0 and r.min() >= 1.0
        print(f"    {'PASS' if ok else 'FAIL'}  {n:<14} "
              f"min = {r[i_min]:.6f} @ N_H = {NH[i_min]:7.1f}  "
              f"adjacent rises = {rises}")
        assert ok, f"5.1.1 criterion wrong for {n}"

    # 4. eq. 19 fit (FCC over-estimation, Wendland kernels).
    print("  eq. 19 fit  eps = eps_100 (N_H/100)^(-alpha)  "
          f"(range {FIT_RANGE[0]:.0f}..{FIT_RANGE[1]:.0f}):")
    for n in WENDLANDS:
        NH, r = data[n]["fcc"]
        H = np.array([H_of_NH(x, fcc.mass) for x in NH])
        # W(0, H) is the physical kernel central value, independent of
        # whether the kernel is parameterised by the support H or by the
        # paper's h = H/kernelScale; the paper's eps constants are fitted
        # against it (probe: implied eps(100) within ~5% of the paper).
        w0 = np.array([W0(n, 3, h) for h in H])
        eps = (r - 1.0) / (fcc.mass * w0)
        m = (NH >= FIT_RANGE[0]) & (NH <= FIT_RANGE[1])
        slope, log_eps100 = np.polyfit(np.log(NH[m] / 100.0), np.log(eps[m]), 1)
        alpha, eps100 = -slope, math.exp(log_eps100)
        paper = ref["density_correction"][n]
        ratio_eps = max(eps100 / paper["eps_100"],
                        paper["eps_100"] / eps100)
        ratio_alpha = max(alpha / paper["alpha"], paper["alpha"] / alpha)
        ok = ratio_eps <= 1.5 and ratio_alpha <= 1.5
        print(f"    {'PASS' if ok else 'FAIL'}  {n:<14} "
              f"eps_100 = {eps100:.5f} (paper {paper['eps_100']}, "
              f"x{ratio_eps:.2f})   alpha = {alpha:.3f} "
              f"(paper {paper['alpha']}, x{ratio_alpha:.2f})")
        assert ok, f"eq. 19 fit off by more than x1.5 for {n}"

    # 5. eq. 18 correction with the paper's constants.
    print("  eq. 18 correction (paper eps) within 5% of 1 "
          f"over {FIT_RANGE[0]:.0f} <= N_H <= {FIT_RANGE[1]:.0f}:")
    for n in WENDLANDS:
        c = ref["density_correction"][n]
        for cfg, key in ((fcc, "fcc"), (glass, "glass")):
            NH, r = data[n][key]
            H = np.array([H_of_NH(x, cfg.mass) for x in NH])
            w0 = np.array([W0(n, 3, h) for h in H])
            eps = c["eps_100"] * (NH / 100.0) ** (-c["alpha"])
            corr = r - eps * cfg.mass * w0
            m = (NH >= FIT_RANGE[0]) & (NH <= FIT_RANGE[1])
            dev = float(np.max(np.abs(corr[m] - 1.0)))
            ok = dev < 0.05
            print(f"    {'PASS' if ok else 'FAIL'}  {n:<14} {key:<6} "
                  f"max|corr - 1| = {dev:.4f}")
            assert ok, f"eq. 18 correction off by more than 5% " \
                f"({n}, {key})"


def table2(ref: dict) -> str:
    """The paper's Table 2 N_H per kernel vs the code's N_H
    (= (4 pi/3) (packingRatio * kernelScale)^3 = sphKernelN_H)."""
    lines = [
        "Table 2 N_H bookkeeping (3D, densest-sphere packing, "
        "n = sqrt(2) d_nn^-3)",
        "",
        "The code's packingRatio IS the paper's h (rho/m)^(1/3), so the",
        "code's N_H reproduces the paper's exactly except for the",
        "deliberate deviations (PLAN findings log):",
        "  cubic   x 1.0175  (Price 2012: 57.9 neighbours in 3D)",
        "  quintic x 1.1425  (CRKSPH alignment)",
        "  B7/B8   x 1.1425  (inherited from the quintic packing; no",
        "                  paper row -- not in the paper's Table 2)",
        "",
        f"{'kernel':<14}{'paper N_H':>11}{'paper h(rho/m)^(1/3)':>22}"
        f"{'code packing':>15}{'code N_H':>11}{'code/paper':>12}",
    ]
    keys = [("cubic_b4", "cubic"), ("quartic_b5", "quartic"),
            ("quintic_b6", "quintic"), ("wendland_C2", "wendland_C2"),
            ("wendland_C4", "wendland_C4"), ("wendland_C6", "wendland_C6"),
            ("hoct4", "hoct4")]
    worst = 0.0
    for name, key in keys:
        pk = packing_ratio(name)
        code = V3 * pk**3 * kernel_scale(3, name)**3
        for row in ref["table2"][key]:
            pNH, _, _, h_rho13 = row
            # self-consistency of the YAML transcription: the paper's
            # h (rho/m)^(1/3) must reproduce its N_H via H/h.
            from_h = V3 * (h_rho13 * kernel_scale(3, name))**3
            worst = max(worst, abs(from_h - pNH) / pNH)
            lines.append(f"{name:<14}{pNH:>11.0f}{h_rho13:>22.3f}"
                         f"{pk:>15.4f}{code:>11.1f}{code / pNH:>12.4f}")
    for name in ("b7", "b8"):
        pk = packing_ratio(name)
        code = V3 * pk**3 * kernel_scale(3, name)**3
        lines.append(f"{name:<14}{'--':>11}{'--':>22}{pk:>15.4f}"
                     f"{code:>11.1f}{'(no paper row)':>12}")
    # The Gaussian is tabulated by N_h (no finite support); with the
    # 16-sigma truncation N_H = N_h * 8^3.
    pk = packing_ratio("gaussian")
    code = V3 * (pk * 8.0)**3
    for row in ref["table2"]["gaussian"]:
        Nh = row["N_h"]
        lines.append(f"{'gaussian':<14}{Nh * 8**3:>11.0f}{row['h_rho13']:>22.3f}"
                     f"{pk:>15.4f}{code:>11.0f}{code / (Nh * 8**3):>12.4f}"
                     f"   (paper N_h = {Nh})")
    lines += ["",
              f"YAML self-consistency: max |N_H(h(rho/m)^(1/3)) - N_H| / N_H "
              f"= {worst:.2e}"]
    return "\n".join(lines)


def plot(data: dict, ref: dict, fcc: ParticleConfig,
         glass: ParticleConfig) -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 8,
        "legend.framealpha": 0.9,
    })
    fig, ax = plt.subplots(figsize=(8.5, 6))

    for n in ORDER:
        NHf, rf = data[n]["fcc"]
        NHg, rg = data[n]["glass"]
        NHp, rp = data[n]["paired"]
        c = COLORS[n]
        ax.plot(NHp, rp, color=c, lw=0.7, alpha=0.55, marker="x",
                ms=4, mfc=c, mec="none")
        ax.plot(NHg, rg, color=c, lw=0.7, alpha=0.55, marker="s",
                ms=3.5, mfc=c, mec="none")
        ax.plot(NHf, rf, color=c, lw=1.6, label=LABELS[n])
        if n in WENDLANDS:
            cc = ref["density_correction"][n]
            for cfg, key, marker in ((fcc, "fcc", "^"),
                                     (glass, "glass", "s")):
                NH, r = data[n][key]
                H = np.array([H_of_NH(x, cfg.mass) for x in NH])
                w0 = np.array([W0(n, 3, h) for h in H])
                eps = cc["eps_100"] * (NH / 100.0) ** (-cc["alpha"])
                ax.plot(NH, r - eps * cfg.mass * w0, color=c, lw=1.2,
                        ls="--", marker=marker, ms=3.5, mfc="none",
                        mec=c, mew=0.8)

    ax.axhline(1.0, color="k", lw=0.6, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("neighbour number $N_H$")
    ax.set_ylabel(r"$\hat\rho / \rho$")
    ax.set_title("D&A (2012) Fig. 3 — density estimate vs $N_H$ (3D, "
                 r"$\rho = 1$)")
    # The 16-sigma-truncated Gaussian over-estimates by 54x at N_H = 20
    # (2.2x at N_H = 500): every point lies above this window, so it is
    # off-axis everywhere -- exactly why the paper does not show it.
    ax.set_ylim(0.98, 1.65)
    leg1 = ax.legend(loc="lower left", ncol=2)
    ax.add_artist(leg1)  # keep it when the style legend is added below

    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([], [], color="0.4", lw=1.6,
               label="densest-sphere packing (FCC)"),
        Line2D([], [], color="0.4", lw=0.7, alpha=0.55, marker="s",
               ms=3.5, mfc="0.4", mec="none", label="glass"),
        Line2D([], [], color="0.4", lw=0.7, alpha=0.55, marker="x",
               ms=4, mfc="0.4", mec="none", label="pairing "
               r"($\S 5.1.1$)"),
        Line2D([], [], color="0.4", lw=1.2, ls="--", label=r"corrected "
               r"$\hat\rho_{corr} = \hat\rho - \epsilon\, m W(0,h)$ "
               r"(eq. 18, paper $\epsilon$)"),
    ]
    ax.legend(handles=style_handles, loc="upper right", fontsize=8)

    # Caption (the paper's glass configurations are not specified; the
    # proxy and the off-axis Gaussian are documented here, per PLAN).
    fig.text(0.01, 0.01,
             "glass = delta-shift-relaxed proxy (warpSPH components, "
             "jittered 16$^3$ lattice, seed 42); pairing = $\\S 5.1.1$ "
             "fully-paired FCC (each point $\\to$ two coincident). "
             "Gaussian (16$\\sigma$-truncated) lies above the axis for "
             "all $N_H$ (54$\\times$ at $N_H = 20$); the paper does not "
             "show it. B-splines are pairing-unstable beyond "
             "$N_H \\sim 55/67/190$ (cubic/quartic/quintic, $\\S 4.1$).",
             fontsize=6.5, color="0.3")

    FIGDIR.mkdir(exist_ok=True)
    fig.savefig(FIGDIR / "fig03_density_estimation.png", dpi=150)
    fig.savefig(FIGDIR / "fig03_density_estimation.pdf")
    plt.close(fig)
    print(f"saved {FIGDIR / 'fig03_density_estimation.png'} (+ .pdf)")


def main() -> None:
    ref = load_reference()

    print("building particle configurations ...")
    fcc = fccConfig()
    paired = pairedConfig(fcc)
    glass = glassConfig(
        cachePath=_HERE.parent.parent.parent / ".tmp" /
        "glass_N4096_L1.0_seed42.npz")
    for c in (fcc, glass, paired):
        print(f"  {c.name:<8} N = {c.n:6d}  box = {c.box:.6f}  "
              f"rho = {c.density:.9f}")
    assert abs(glass.density - 1.0) < 1e-12
    assert abs(paired.density - 1.0) < 1e-12

    print("evaluating density estimates (30 x N_H grid, 10 kernels, "
          "3 configurations) ...")
    data = allRatios(fcc, glass, paired, ORDER)

    checks(data, fcc, glass, paired, ref)

    print()
    tbl = table2(ref)
    print(tbl)
    RESDIR.mkdir(exist_ok=True)
    (RESDIR / "table2_nH_bookkeeping.txt").write_text(tbl + "\n")
    print(f"saved {RESDIR / 'table2_nH_bookkeeping.txt'}")

    plot(data, ref, fcc, glass)


if __name__ == "__main__":
    main()
