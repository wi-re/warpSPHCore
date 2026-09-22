# Baseline findings (first pass, 2026-09-22)

Curated interpretation of [`REPORT.md`](REPORT.md) — the frozen "before"
column for the higher-order phases (parent plan `../../higher_order.md`).
Wendland2, float64, jitter 0.3, default target neighbors 40
(h/Δx ≈ 3.57 in 2D at N = 1152). Re-run the driver to regenerate the
tables; update this file when the baseline changes (a harness change is a
versioning event).

## 1. Reproduction (monomial patch tests, open domain)

**Interior** (max error, N = 1152):

| capability | standard | crk | renorm |
|---|---|---|---|
| interpolate constant | 5.5e-2 | **7e-16 (exact)** | 5.5e-2 |
| interpolate linear | 4-6e-2 | **7e-16 (exact)** | 4-6e-2 |
| interpolate degree ≥ 2 | 2-3e-2 | 5e-5 – 4e-3 (not exact) | = standard |
| gradient linear (scalar) | 1.0e-1 | 7e-4 – 1.5e-3 (flat) | **7e-16 (exact)** |
| gradient linear (vector Mx) | 1.6e-1 | 1.5e-3 (flat) | **2e-15 (exact)** |
| gradient degree ≥ 2 | 9e-2 – 2.7e-1 | 3e-3 – 3.6e-2 | 3e-3 – 2.7e-2 |
| laplacian (quadratic) | 5 – 12 | 0.5 – 1.4 | = standard |

- **CRK** enforces 0th/1st moments of the corrected kernel: value
  reproduction of constants and linears is machine-exact, and this is the
  only mode that does so. It does *not* reproduce degree ≥ 2 (as expected
  from its 2-moment construction).
- **CRK linear-gradient residual ~1e-3 is intrinsic and flat**: it does
  not decrease with resolution (see §3) — the core's CRK gradient
  operator is not strictly linear-consistent. Documented, not a wiring
  bug; worth a cross-check against D&A's reported CRK behavior when the
  replication effort extends to gradient convergence.
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
- **renorm linear gradients stay machine-exact in the band** (6.7e-15)
  where standard degrades to 0.6-1.2.
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

| probe | standard | crk | renorm |
|---|---|---|---|
| gradient | ~0.95 (O(h)) | 1.6 - 1.7 | **1.7 - 1.86** |
| interpolate | ~0.95, floors at ΣW−1 bias | **1.84 - 1.95 (true O(h²))** | ~0.95, same floor |

- CRK's moment correction buys a full order on smooth interpolation
  (removes the ΣW−1 floor); renorm leaves it in place.
- On smooth *gradients* renorm is marginally best (1.7-1.86 vs CRK
  1.6-1.7); standard is O(h).
- **The Brookshaw laplacian does not converge**: for monomial fields the
  observed slopes are negative in all three modes (interior ~
  −1.2…−1.7, boundary ~ −1.0…−1.1) — the error *grows* as dx → 0 at
  fixed h/Δx; smooth-field series show the same trend (slope ~
  −0.8…−1.4, or flat within a factor of 2). Even at the finest
  resolution the error is O(1) relative to the analytic target (patch:
  quadratics ~ 7 against a target of 2). This is the largest baseline
  defect and the primary motivation for the higher-order phases.
- CRK linear-gradient residual is flat here too (§1).
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
