# Baseline findings (first pass 2026-09-22; Phase 3 `renormVal` 2026-09-23; CRK gradient fix 2026-09-26)

Curated interpretation of [`REPORT.md`](REPORT.md) — the frozen "before"
column for the higher-order phases (parent plan `../../higher_order.md`).
Wendland2, float64, jitter 0.3, default target neighbors 40
(h/Δx ≈ 3.57 in 2D at N = 1152). Re-run the driver to regenerate the
tables; update this file when the baseline changes (a harness change is a
versioning event).

**Versioning event (2026-09-23):** added the `renormVal` mode — the full
Phase 3 operator (Bonet–Lok corrected *gradient* + Randles–Libersky *value*
renormalization `f̂/S`). This is purely **additive**: the three pre-existing
modes (`standard`/`crk`/`renorm`) are byte-identical to the 2026-09-22
baseline (verified by diffing the CSV). The new column and its findings
below are the only change.

**Versioning event (2026-09-26): core CRK gradient bug fixed.**
`src/warpSPHCore/crk/kernel.py`'s `correctGradientCRK` contracted
`gradB` (stored [derivative, component]) on the wrong axis —
`matmul(transpose(gradB), x_ij)` instead of `matmul(gradB, x_ij)`, a
transpose introduced by the 2026-08-11 adjoint fix (5ca1882; the loop it
replaced was correct). `gradB` is symmetric on a regular lattice, so only
jittered / boundary particles saw it. The "intrinsic flat ~1e-3 CRK
linear-gradient residual" this file used to report was that bug. Re-running
the sweep changed **only** the `crk` × `gradient` rows (134 of 1536 rows;
every other mode/probe byte-identical). Numbers below are post-fix; the
pre-fix values are quoted where they mattered.

## 1. Reproduction (monomial patch tests, open domain)

**Interior** (max error, N = 1152):

| capability | standard | crk | renorm | renormVal |
|---|---|---|---|---|
| interpolate constant | 5.5e-2 | **7e-16 (exact)** | 5.5e-2 | **0 (exact)** |
| interpolate linear | 4-6e-2 | **7e-16 (exact)** | 4-6e-2 | 2-2.5e-3 |
| interpolate degree ≥ 2 | 2-3e-2 | 5e-5 – 4e-3 (not exact) | = standard | 2-8e-3 |
| gradient linear (scalar) | 1.0e-1 | **7e-16 (exact)** | **7e-16 (exact)** | **7e-16 (exact)** |
| gradient linear (vector Mx) | 1.6e-1 | **2e-15 (exact)** | **2e-15 (exact)** | **2e-15 (exact)** |
| gradient degree ≥ 2 | 9e-2 – 2.7e-1 | 3e-3 – 3.6e-2 | 3e-3 – 2.7e-2 | = renorm |
| laplacian (quadratic) | 5 – 12 | 0.5 – 1.4 | = standard | = standard |

- **renormVal** (the full Phase 3 operator: Bonet–Lok *gradient* +
  Randles–Libersky *value* renormalization `f̂/S`, `S` = the 0th kernel
  moment) is the new column. Its *gradient* and *laplacian* are identical
  to `renorm` (only the value is additionally corrected). Its *value* is
  the result: dividing the uncorrected interpolant by `S` makes constant
  reproduction exact (0) and cuts the value error ~10-18× vs standard at
  every degree (linear 2-2.5e-3, quartic 3.5-8.2e-3). It does **not** make
  linear value reproduction exact — only the 0th moment is corrected, not
  the 1st. That is the precise difference from CRK, which enforces both
  moments and is machine-exact to degree 1.
- **CRK** enforces 0th/1st moments of the corrected kernel: value
  reproduction of constants and linears is machine-exact, and this is the
  only mode that does so. Its gradient (the analytic derivative of the
  corrected kernel, incl. ∇A / ∇B) is then machine-exact for linears too —
  the only mode exact in **both** value and gradient at degree 1. It does
  *not* reproduce degree ≥ 2 (as expected from its 2-moment construction).
- *(Pre-fix, 2026-09-22 – 09-26: the CRK linear gradient showed a flat
  7e-4 – 1.5e-3 residual, reported here as "intrinsic". It was the
  `gradB` transpose bug — see the versioning note at the top.)*
- **renorm** (the Phase 3 Bonet–Lok machinery, already in the core) is
  machine-exact for linear *gradients* but applies no value correction —
  its interpolation column is identical to standard.
- **standard SPH** reproduces nothing non-trivially: constant
  interpolation carries a ~5e-2 bias at h/Δx ≈ 3.57, linear gradients
  ~1e-1.

## 2. Boundary band (within one support of an open wall)

- **CRK value reproduction stays machine-exact at the boundary** (const
  and linear interpolation ~7e-16, vs 2.3e-1 for standard) — the
  apparent-volume correction absorbs the truncated kernel support.
- **renormVal also makes boundary constant interpolation exact** (0, vs
  2.3e-1 standard) and cuts boundary linear value error to 1.4-1.6e-2
  (~14×) — the per-particle `f̂/S` renormalization absorbs the truncated
  support just like CRK's apparent volume does, but it only fixes the 0th
  moment, so its boundary *linear* value error (1.4-1.6e-2) is ~2 orders
  above CRK's (7e-16).
- **renorm / renormVal linear gradients stay machine-exact in the band**
  (6.7e-15) where standard degrades to 0.6-1.2.
- **CRK linear gradients are machine-exact in the band too** (1.4e-15 –
  6.3e-15; pre-fix 0.12 – 0.25 — the transpose bug hit the boundary
  hardest, where `gradB` is most asymmetric). CRK is also now the best
  mode for degree ≥ 2 gradients in the band (e.g. x² 4.2e-2 vs renorm
  4.7e-2, xy 1.3e-2 vs 2.1e-2 — better at every degree-2..4 monomial; the
  interior is the other way round, renorm ~2× better; pre-fix CRK was 2-10×
  *worse* than renorm
  there, 0.21 – 0.55).
- Standard interpolation error is ~4× the interior value at the band;
  gradient errors 5-15×.
- **Laplacian at the boundary is bad for every mode** (40-270) and CRK is
  *worse* than standard there (the apparent-volume correction does not
  help the Brookshaw laplacian). Boundary laplacian handling is an open
  problem for the higher-order phases.

## 3. Convergence (decoupled refinement: h/Δx fixed, N varied)

Key reading rule for the orders tables: **at constant h/Δx, polynomial
interpolation error does not converge** — the 0th-moment lattice-sum bias
(ΣW − 1 ≈ 5e-2) is a property of lattice × kernel × (h/Δx) and does not
shrink as dx → 0. That is why the constant/linear/quadratic
*interpolation* series read `saturated` in the resolve sweeps; it is
expected behavior, not a harness defect. The smooth-field series are the
meaningful order signal.

Interior, smooth fields (error vs dx):

| probe | standard | crk | renorm | renormVal |
|---|---|---|---|---|
| gradient | ~0.95 (O(h)) | 1.6 - 1.7 | **1.7 - 1.86** | = renorm |
| interpolate | ~0.95, floors at ΣW−1 bias | **1.84 - 1.95 (true O(h²))** | ~0.95, same floor | **1.84 - 1.91 (O(h²))** |

- CRK's moment correction buys a full order on smooth interpolation
  (removes the ΣW−1 floor); renorm leaves it in place.
- **renormVal's value renormalization buys the same full order as CRK** on
  smooth interpolation (1.84-1.91, matching CRK's 1.84-1.95): dividing by
  the 0th moment `S` removes the ΣW−1 floor exactly as CRK's `A` factor
  does. So on *smooth* values, renormVal and CRK are both O(h²); they
  differ on *polynomial* values — CRK also fixes the 1st moment, so it is
  machine-exact for linears, whereas renormVal's linear value error
  saturates (flat ~2e-3 in the decoupled sweep, per the §3 reading rule:
  the residual 1st-moment error is O(1) in dx at fixed h/Δx). renormVal's
  *gradient* is unchanged from renorm (Bonet–Lok).
- On smooth *gradients* renorm/renormVal are marginally best (1.7-1.86 vs
  CRK 1.6-1.7); standard is O(h). (Interior smooth-field CRK orders were
  unaffected by the 2026-09-26 fix at two decimals; boundary sin_cos
  0.70 → 0.76.)
- **The Brookshaw laplacian does not converge**: for monomial fields the
  observed slopes are negative in all three modes (interior ~
  −1.2…−1.7, boundary ~ −1.0…−1.1) — the error *grows* as dx → 0 at
  fixed h/Δx; smooth-field series show the same trend (slope ~
  −0.8…−1.4, or flat within a factor of 2). Even at the finest
  resolution the error is O(1) relative to the analytic target (patch:
  quadratics ~ 7 against a target of 2). This is the largest baseline
  defect *pointwise* — but see the modal decomposition below before
  treating it as the primary motivation for the higher-order phases.
- **The non-converging Brookshaw error is particle-scale noise; its
  consistent part converges (2026-09-26, `run_modal_probe.py`).** Splitting
  the error of L_h on one Fourier mode (periodic, jitter 0.3, N 576→9216)
  into the projection onto that mode (the effective-diffusivity error — what
  a diffusion solve sees) and the residual:

  | mode | pointwise total | modal (a/κ²) | residual |
  |---|---|---|---|
  | standard | −0.93 | 0.88 (→ 0.45) | −0.96 |
  | crk | grows (saturated fit) | **2.00** | −0.76 |
  | renorm | −0.95 | **2.10** | −0.97 |

  The residual is essentially all of the pointwise error and grows ~1/dx
  (jitter noise); a dissipative PDE damps it at rate ~ν/dx², so its net
  effect is O(dx). That is why the frozen-particle diffusion leg
  (`REPORT_frozen.md`) converges at 2.00 (crk) / 1.72 (renorm) / 0.88 →
  0.47 (standard) — the modal orders predict the PDE orders. CRK's
  residual is ~9× smaller than the other modes'. The gradient decomposes
  the same way, except that the corrected gradients' residual also
  converges (~1.0): modal 1.98 (crk/renorm) predicts frozen advection
  1.93 / 1.98. **Consequence for Phases 4/5:** judge a new Laplacian on
  both numbers — the modal error for dissipative use, the pointwise
  residual for non-dissipative use (source terms, Poisson right-hand
  sides), where Brookshaw's noise does not get filtered.
- CRK linear gradients are exact here too (the "flat residual" of the
  first pass was the transpose bug, §1).
- The periodic sweep (smooth periodic fields only) agrees with the open
  sweep: gradient 0.95 / 1.39-1.69 / 1.70-1.83, interpolate 0.85-0.98 /
  1.84-1.97 / 0.85-0.98.

**Do not use monomial fields on periodic domains.** They are not
periodic, so the Difference gradient is seam-contaminated (the field
jumps by L·∇f across the images): the monomial gradient/laplacian series
in a periodic sweep show *negative* slopes (~ −1…−2) with errors growing
as O(L)·h⁻¹. The driver therefore restricts the periodic sweep to
periodic smooth fields; monomial patch tests run on the open domain.

## 4. Smoothing sweep (N fixed, h varied via target neighbors)

Error direction inverts vs the resolve sweep: larger h smooths
discretization noise but increases truncation bias.

- standard, gradient of linear: error grows ∝ h² (slope −2.0) — the
  textbook O(h²) linear-gradient error of the Difference scheme.
- renorm, gradient of gaussian: ~1.0 (error decreases with h — the
  covariance correction damps the discretization component).
- CRK, interpolation of smooth fields: 2.0 (interior) to 3.0 (boundary).
- standard, interpolation of a constant: interior error grows with h
  (−1.18), boundary error decreases (+1.47) — bias vs truncation
  competing exactly as expected.

## 5. Renorm conditioning

- Interior covariance matrices are well conditioned: mean κ ≈ 1.01-1.34
  across the jitter × neighbor grid; median ≈ 1.0.
- The boundary band is worse: mean κ ≈ 1.28-1.75, max 2.3-4.7 (worst at
  jitter 0.5 with 12 target neighbors).
- The shipped identity fallback (`num_nbrs < dim+2`) triggers on ≤ 0.7 %
  of boundary-band rows, only at jitter ≥ 0.3 with 12 target neighbors.
- Condition numbers rise with jitter and fall with neighbor count —
  the expected trade for the higher-order corrections built on this
  matrix.

## 6. Method notes

- **float64 is required** for these numbers (`warpSPHCore_PRECISION=
  float64`, set by the driver before import). float32 floors at ~1e-7
  and caps observable order.
- Decoupled refinement at *constant* h/Δx is what makes polynomial
  interpolation saturate (§3); the smoothing sweep isolates h with the
  lattice fixed.
- Jitter 0.3 is the default; the conditioning grid covers 0.1/0.3/0.5.
  The hex/FCC lattice without jitter would be exactly isotropic to
  second order and would hide the O(1) lattice-sum bias that CRK
  corrects — the jittered case is the honest one.
- **`renormVal` is a derived column, not a new `src/` operator.** The
  core's `Interpolate` only supports CRK value correction, so the
  Randles–Libersky value renormalization is composed from two standard
  `Interpolate` calls in `operators.py`: the raw interpolant `f̂` and the
  0th kernel moment `S = Interpolate(ones)` (cached per case), returning
  `f̂/S`. The gradient probe reuses the `renorm` `L` matrix; the laplacian
  is uncorrected (there is no Bonet–Lok Laplacian in the core).
