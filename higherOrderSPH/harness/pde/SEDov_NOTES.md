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
- **Ensemble note (matters for this case):** `error_l2` / `error_l1` are
  evaluated over the *common cells* only (cells both projections fill — i.e.
  where the coarse particles sit), while `error_l2_aligned` is over *all*
  cells of the densified profile. The two ensembles disagree materially
  (finding 2), so every number below states its ensemble.

## Results (reference = nx=1600)

| nx | dx | err L2 | err L1 | err L2 aligned | align shift | shift / dx_coarse |
|---|---|---|---|---|---|---|
| 200 | 0.01 | 3.33e-1 | 1.58e-1 | 2.66e-1 | +4.2e-3 | +0.42 |
| 400 | 0.005 | 4.17e-1 | 1.39e-1 | 1.92e-1 | +2.0e-3 | +0.41 |
| 800 | 0.0025 | 1.73e-1 | 4.44e-2 | 9.45e-2 | −1.2e-4 | −0.05 |

Observed orders: L2 0.47 (r² 0.51, non-monotonic); L1 **0.92** (r² 0.82);
aligned L2 **0.75** (r² 0.96, monotonic).

All-cells raw L2 (densified profile, no shift — not a CSV column):
2.68e-1 / 1.93e-1 / 9.48e-2 — **monotonic**, order ≈ 0.75. The
non-monotonicity is confined to the common-cells ensemble (finding 2).

## Findings

1. **The shock is positioned correctly.** The optimal alignment shifts are
   sub-cell (+0.42, +0.41, −0.05 of the *coarse* dx) and sign-convergent —
   coarse shocks lag the reference, the finest-probed lead it slightly — and
   sit far inside the ±4 dx search range (no boundary saturation). Alignment
   changes the all-cells L2 by < 0.7% at every resolution; misalignment is not
   the error source.
2. **The L2 non-monotonicity (0.333 → 0.417 → 0.173) is a common-cells
   ensemble artifact on top of a real effect.** The raw L2 is evaluated only
   on the cells both projections fill — where the coarse particles sit. Those
   cluster in the compressed post-shock shell: 29–33% of the common cells fall
   inside the 10%-wide shock band at every resolution (3× the uniform rate).
   Inside that band the mean e² of the *common* cells RISES 200→400
   (0.317 → 0.523) because the coarse shock sharpens toward the reference's
   2–3-cell drop (finding 3): wherever the reference has already dropped to
   1.0, the coarse profile is still near its 3.9–4.0 plateau, so the local
   error peak grows (2.89 → 2.93) and narrows. The band's 32% weight of the
   common cells outweighs the ×4.3 collapse of the error outside the band, so
   the common-cell L2² rises 200→400. On the full grid the band is only 10%
   of the weight, so the same physics gives a monotonic L2 (0.268 → 0.193 →
   0.095).
3. **The error is a width-mismatch spike, not an overshoot.** The reference
   shock is a 2–3-cell drop: post-shock plateau peaking at 4.004 right at the
   front (x ≈ 0.800), falling to the undisturbed 1.0 by x ≈ 0.805; behind it a
   broad compression ramp (2.88 at x = 0.74 → 4.0 at the front); the interior
   (|x| < 0.6) is clean (ρ → 0.03 at the centre). The coarse profiles peak
   LOWER and WIDER (max 3.875 / 3.950 / 3.903 at 200/400/800, spanning ~12/7/4
   cells vs ~2–3; no overshoot above the reference max at any resolution).
   Peak |err|: 2.89 / 2.93 / 2.26 ≈ the post-shock jump of 3.0; peakiness
   (RMS/mean) rises 2.77 → 3.60 → 4.30. Budget (±0.05 band around both
   shocks, all cells): the shock carries 43 / 45 / 37% of the L1; the
   ramp/contact 8–10%; the rest 47–55% — the contact is not an issue.
4. **L1 (area) is the robust shock metric**: order 0.92 (r² 0.82, common
   cells; ≈ 1.05 all-cells), consistent with the first-order accuracy a
   discontinuity limits shock-capturing to (shock width O(dx)). The 200→400
   leg is nearly flat on the common cells (pairwise 0.18); the 400→800 leg is
   1.65.
5. **Aligned L2 (order 0.75, r² 0.96) is the cleanest L2-family diagnostic** —
   monotonic with a tight fit; the all-cells raw L2 is equally monotonic
   (alignment is a < 0.7% no-op at sub-cell shifts) — but both are
   dissipation-limited (the width-mismatch spike).
6. **Implication:** judge shock cases on L1 (or the full-grid L2 / aligned
   L2), not the common-cells raw L2. The suite's raw-L2 "0.47 order" for
   Sedov was a sampling artifact on top of the genuine first-order shock
   limit.
7. **Exact-solution check (2026-09-26) — confirms finding 4.** Scoring
   every rung against the frontend's exact self-similar solution
   (`SedovSolution`, at each run's own t_final; `error_l1_exact` /
   `error_l2_exact`, volume-weighted, 4-point fit incl. nx=1600) gives
   **L1 order 1.03 (r² 0.993)** and **L2 0.59 (r² 0.987)** — exactly the
   first-order shock-capturing signature (L1 ~ dx, L2 ~ √dx). (Pre-CRK-fix
   run; see `higher_order.md` Phase 2 — to be re-read after the re-run.)
   A same-day caveat here had claimed the opposite: fitting the
   finite-reference model e = C(h^p − h_ref^p) to the reference-metric L1
   gave p ≈ 0.23, "below first order". **That was wrong** — the model
   assumes the coarse-vs-reference error keeps one sign structure across
   rungs, but here the shock-position error changes sign between rungs
   (alignment shifts +0.42, +0.41, −0.05 dx, finding 1), so the correction
   over-corrects. For Sod, where the assumption holds, the corrected fit
   (1.07) does land near the exact-solution order (0.98). Rule: prefer the
   exact metric; treat the bias-corrected fit as unreliable unless validated
   against an exact solution for that case.

## Profile inspection (2026-09-25)

The findings above came from inspecting the run states directly (the harness
driver discards them). The ladder was re-run with a scratch driver
(`.tmp/sedov_state_dump.py` → `.tmp/sedov_states/sedov_nx{200,400,800,
1600}.pt`) and the analysis is pure-CPU on the dump
(`.tmp/sedov_error_profiles.py` → `.tmp/sedov_error_profiles.png` +
`.tmp/sedov_errprof_{200,400,800}.csv`; probes `.tmp/allcells_l2.py`,
`.tmp/common_cells_bias.py`, `.tmp/check_800.py`).

**Reproducibility check:** the runs are bit-reproducible — the committed
metric functions, run on the dumped states, reproduce the CSV exactly
(L2 0.333057 / 0.417460 / 0.172644; aligned 0.266370 / 0.192173 / 0.094480;
shifts +0.004218 / +0.002030 / −0.000123).

## Driver / tests

- `run_pde.py --cases sedov` — the reference metric now also fills
  `error_l1`, `error_l2_aligned`, `align_shift` (1D cases) in
  `results/pde_rows.csv`; `report_pde.py` renders an order table per
  alternate metric when the columns are present (blank for pre-metric rows).
- Pure-CPU unit tests: `tests/convergence/test_pde.py` — L1-vs-L2 scaling of a
  shifted step (O(δ) vs O(√δ)), aligned-shift recovery on smooth and *gappy*
  CIC projections, 2D rejection, and blank-column handling in the order fits.
