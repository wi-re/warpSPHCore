# Sedov 1D blast wave — shock metrics (L1 + alignment)

Context: the Pass-2 PDE suite's reference metric for the 1D Sedov case is the
raw L2 density error vs the finest-ladder run (nx=1600), which gave a
non-monotonic ladder (0.333 → 0.417 → 0.173, order 0.47, r² 0.51 — nx=400
*worse* than nx=200). This note records the shock-metric follow-up: the L1
area norm and the optimally shift-aligned L2 (both added to the reference
metric for 1D cases in `field_error.py` / `run_pde.py`), and the diagnosis of
the non-monotonicity.

## Setup

- 1D periodic Sedov–Taylor blast (L=2, γ=5/3, ρ0=1, E0=1, `hat` init),
  CRKSPH/B7, run to `goalRadius=0.8`. The runs stop at different times per
  resolution (t_final = 0.5552 / 0.5555 / 0.5556 / 0.5556) because the
  stopping rule is resolution-dependent.
- Metric: CIC mass-weighted projection onto a common 4·ref-nx grid
  (`field_error.py`). `error_l1` = mean |Δρ| over the cells both sets fill;
  `error_l2_aligned` = L2 after optimally translating the *reference* profile
  against the coarse one (sub-cell, periodic, linearly interpolated; two-stage
  scan + parabolic refinement; ±4 dx_coarse search range). The CIC projection
  is gappy — each particle fills 2 of the ~4–8 cells between neighbours, empty
  cells hold 0 — so both profiles are first densified (periodic linear
  interpolation over the gaps); shifting the zero-padded profile directly
  interpolates across zeros and inverts the error landscape.

## Results (reference = nx=1600)

| nx | dx | err L2 | err L1 | err L2 aligned | align shift | shift / dx_coarse |
|---|---|---|---|---|---|---|
| 200 | 0.01 | 3.33e-1 | 1.58e-1 | 2.66e-1 | +4.2e-3 | +0.42 |
| 400 | 0.005 | 4.17e-1 | 1.39e-1 | 1.92e-1 | +2.0e-3 | +0.41 |
| 800 | 0.0025 | 1.73e-1 | 4.44e-2 | 9.45e-2 | −1.2e-4 | −0.05 |

Observed orders: L2 0.47 (r² 0.51, non-monotonic); L1 **0.92** (r² 0.82);
aligned L2 **0.75** (r² 0.96, monotonic).

## Findings

1. **The shock is positioned correctly.** The optimal alignment shifts are
   sub-cell (+0.42, +0.41, −0.05 of the *coarse* dx) and sign-convergent —
   coarse shocks lag the reference, the finest-probed lead it slightly — and
   sit far inside the ±4 dx search range (no boundary saturation). Misalignment
   is not the error source.
2. **The L2 non-monotonicity survives alignment** (0.266 → 0.192 → 0.094 still
   has 400 > 200), so it is resolution-dependent shock *structure* (numerical
   dissipation), not position. The L1-down/L2-up pattern on 200→400 (L1
   0.158→0.139, L2 0.333→0.417) says the nx=400 error is more *concentrated*
   — a localized overshoot-type feature near the shock with less total area
   but a taller peak — not merely wider.
3. **L1 (area) is the robust shock metric**: order 0.92 (r² 0.82), consistent
   with the first-order accuracy a discontinuity limits shock-capturing to
   (shock width O(dx)). The 200→400 leg is nearly flat (pairwise 0.18); the
   400→800 leg is 1.65.
4. **Aligned L2 (order 0.75, r² 0.96) is the cleanest L2-family diagnostic** —
   monotonic with a tight fit — but it is still dissipation-limited.
5. **Implication:** judge shock cases on L1 (or aligned L2), not raw L2. The
   suite's raw-L2 "0.47 order" for Sedov was a metric artifact on top of the
   genuine first-order shock limit.

## Driver / tests

- `run_pde.py --cases sedov` — the reference metric now also fills
  `error_l1`, `error_l2_aligned`, `align_shift` (1D cases) in
  `results/pde_rows.csv`; `report_pde.py` renders an order table per
  alternate metric when the columns are present (blank for pre-metric rows).
- Pure-CPU unit tests: `tests/convergence/test_pde.py` — L1-vs-L2 scaling of a
  shifted step (O(δ) vs O(√δ)), aligned-shift recovery on smooth and *gappy*
  CIC projections, 2D rejection, and blank-column handling in the order fits.
