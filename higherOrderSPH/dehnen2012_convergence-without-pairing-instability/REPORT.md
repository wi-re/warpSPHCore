# REPORT — Dehnen & Aly 2012 replication

**Paper:** *Improving convergence in SPH simulations without pairing
instability*, Dehnen & Aly 2012, MNRAS 425(2), 1068 (arXiv:1204.2471).
**Status:** Phases 0–6 of `PLAN.md` complete (2026-09-17 … 2026-09-20);
closeout items 1–7 complete (2026-09-20, `closeout_plan.md`). Phase 7
(dynamic tests, Figs 7–13) remains — separate effort, §8.
**Re-read first:** `paper_notes.md` (notation map incl. the h/H clash,
Tables 1–2, equations, figure inventory), `PLAN.md` (phase status +
findings log), `phase5_stability_log.md` (stability work, entries a–n),
`closeout_plan.md` (closeout items + progress log).

---

## 1. Verdict (TL;DR)

The `warpSPHCore` kernels used by the paper are **exactly the paper's
kernels** — functional form, normalisation, and scale convention verified
to machine precision by the Phase-1 audit battery (10/10 kernels, CI
gated) — and they **behave as the paper claims**: the Fourier
non-negativity / pairing-stability structure, the density-estimation
bias ordering and the eq.-18/19 self-term correction, and the
linear-stability boundaries (cubic ≲ 55, quartic ≈ 67, quintic ≈ 190,
Wendland stable, HOCT4 island near 150) all reproduce. Every
discrepancy encountered was rooted in **our** implementation (three
independent bugs in the Phase-5 P-matrix, one float32-constant class,
one wrong kernel identity/scale) or in **paper presentation choices**
(the Fig.-2 axis, the hidden long-λ region, the Sod IC), not in the
kernels — see the discrepancy table, §3.

Success-criteria scorecard (`PLAN.md`):

| # | Criterion | Status |
|---|---|---|
| 1 | Paper kernels match to machine precision (or documented mismatch) | **MET** — zero known deviations; the one historical deviation (B7) resolved 2026-09-18 (§5.2) |
| 2 | Static evaluations (Figs 1–6, Tables 1–2) reproduced quantitatively | **MET** — §2 |
| 3 | Dynamic evaluations (Figs 7–13) | **PENDING** — Phase 7, §8 (Phase-6 prerequisite delivered and validated, §6) |
| 4 | Gaussian + HOCT4 on-boarded via the pipeline; pipeline green over every spec'd kernel | **MET** — Phase 3; 10/10 audited, 14/14 CI |
| 5 | `warpSPH` C&D 2010 switch + R&H 2012 conductivity as validated first-class options | **MET** — Phase 6, §6 |
| 6 | Replication code in this folder, imports from `warpSPHCore`, re-runnable | **MET** — §9 |

**Global limitation (applies to every figure below):** the model has no
image input, so no pixel-level comparison against the paper's rendered
figures was possible. The PNGs/PDFs in `figures/` are generated for the
user's visual check; quantitative claims rest on the paper's *numbers*
(Tables 1–2, quoted positions, tick labels, stated tolerances) and on
independent analytic cross-checks (closed forms, ground-truth FD
Jacobians, exact Riemann solutions).

---

## 2. Per-figure comparison

### Fig. 1 — kernel shapes (`fig01_kernel_shapes.py`)
All Table-1 kernels + Gaussian + HOCT4 (8 curves, ν = 3, common
h = 2σ scaling, linear + log panels, support arrows, paper's 1e-6 log
floor). Checks per kernel: non-negativity, compact support,
normalisation (exact piecewise + Simpson). **Matches the paper's
qualitative content** (supports, central values, small-scale emphasis);
the curves are built *entirely* from the shipped audited warp functions
(no re-transcriptions). b7/b8 added as an extension (same B-spline
family, not in the paper's Fig. 1; user-checked against the PDF).

### Fig. 2 — Fourier transforms (`fig02_fourier_transforms.py`)
Numerical 3D FT (eq. 14, Simpson n = 20001) as primary; a derived
piecewise-polynomial closed form as cross-check (the paper's eq. 15
transcription is garbled beyond repair — §3, row 7). Verified:
- `w̄(0) = 1` to ≤ 5e-12 for all 8 kernels;
- closed-form vs numerical ≤ 1e-11 on the plotted grid (far below the
  1e-3 acceptance bar);
- eq.-17 small-k Taylor slope a₂ = −1/8 ± 1e-4 for **all** kernels —
  the paper's "all curves overlap at small k" statement holds;
- non-negativity on [0, 3π] for Wendland C²/C⁴/C⁶, HOCT4, Gaussian
  (min w̄ ≥ −1e-10, the numerical floor);
- B-spline sign changes with first zeros (pairing criterion κ₀):
  cubic 3.4414 |k|σ (just over π, as the paper places it), quartic
  5.5641, quintic 4.2881, b7 6.6648; b8 carries a sub-noise-floor
  negative lobe (min −8.5e-7, below the audit's 1e-6 practical floor —
  documented in `kernel_specs.yaml`).
**Axis correction (user-verified from the PDF):** the paper's x-axis is
**|k|σ (σ = h/2), not |k|h**; the replication plots |k|σ ∈ [0, 3π] with
log |w̄| and dashed segments where w̄ < 0.

### figA — supplementary: the five non-D&A library kernels
(`figA_other_kernels.py`; the five CG-specialty force/Laplacian kernels
run through the same battery). Verdicts — general density kernel? /
pairing-stable?:
- **poly6** — yes, but pairing-unstable (min w̄ = −1.19e-2, first zero
  2.467 |k|σ);
- **spiky** — no (f′(0) = −3, not C² at the centre), yet pairing-stable
  (min w̄ = +2.3e-4);
- **adhesion** — no (4πC₃∫fq² = 0.0158, f′(1) diverges — surface-tension
  kernel);
- **cohesion** — no (norm 0.2351, negative at the centre);
- **viscosity** — no (singular at the centre — Laplacian kernel).
Only poly6 is usable as a general density kernel; none of the five is
both pairing-stable and C² at the centre. **Decision (user, 2026-09-19):**
the five are excluded from the audit pipeline and from any D&A-convention
re-scaling — they are specialty kernels, not density kernels (§7.3).

### Fig. 3 — density estimation (`fig03_density_estimation.py`)
34-point log N_H grid 20–800, all ten kernels, three configurations
(FCC; delta-shift-relaxed glass proxy — the paper's glass configurations
are not fully specified, documented in the caption; §5.1.1 fully-paired).
Paper's axis bounds (x 20–800 log, y 0.99–1.01):
- **Bias ordering matches the paper:** cubic under-estimates in its
  accessible range (0.9936–0.9966 on [31, 55], crosses 1 at ~75),
  Wendland/HOCT4/Gaussian over-estimate, HOCT4 worst at low N_H
  (1.54× at 20), 16σ-Gaussian 54× at N_H = 20 (off-axis for all N_H —
  the paper omits it too).
- **§5.1.1 "crosses" identity:** the paired configuration's curve is the
  FCC curve shifted by factor 2 in N_H — verified to 4.3e-14 for all
  ten kernels; the paper's criterion reduces to "does ρ̂(N_H) have an
  interior minimum" (B-splines: yes, below 1; Wendland/HOCT4/Gaussian:
  no).
- **Eq.-19 ε fit** (FCC over-estimation, 40 ≤ N_H ≤ 400) vs the
  paper's (ε₁₀₀, α): C² (0.02949, 0.999) vs (0.0294, 0.977); C⁴
  (0.01361, 1.633) vs (0.01342, 1.579); C⁶ (0.01131, 2.218) vs
  (0.0116, 2.236) — **every constant within ×1.03** (bar: ×1.5).
- **Eq.-18 correction** with the paper's ε: within **0.10–2.40 %** of 1
  over 40 ≤ N_H ≤ 400 (FCC and glass, all three Wendland kernels).

### Tables 1–2
- **Table 1** — every row (function, C, σ²/H², H/h) verified by the
  Phase-1 audit: form < 1e-13, independent numerical normalisation,
  moments, derivatives vs AD, support boundary, FT sign structure.
  The one identity problem — the code's "B7" — is resolved (§5.2).
- **Table 2** — N_H bookkeeping cross-check
  (`results/table2_nH_bookkeeping.txt`): quartic/W2/W4/W6/HOCT4
  reproduce the paper's N_H exactly (×1.0000 ± 0.0002); the code's
  *deliberate* packing deviations (in-code comments) give cubic
  ×1.0175 → 57.9 (Price 2012 alignment), quintic ×1.1425 → 268.4
  (CRKSPH alignment), b7/b8 ×1.1425 → 333.1/402.3 (no paper rows),
  Gaussian N_h = 10 → 5126 (paper 5120, 3-decimal rounding).

### Figs. 4–5 — linear stability contours
(`fig04_fig05_stability_contours.py`, `check_stability_boundaries.py`)
ω²_∥/c²k² and ω²_⊥/c²k² over (|k|d_nn, h/d_nn 0.9–3.0), k ∝ (1,1,1) and
(1,1,0), from the **exact** real-lattice-sum P matrix (validated
against the ground-truth FD Jacobian of the actual force — §3, rows
2–4). Final resolution **60 N_H (log) × 200 |k|d_nn (log)** per
kernel, per-kernel npz cache in `results/` (two-mode fields:
longitudinal = eigenmode most aligned with k; transverse = smallest
eigenvalue — both ω²/c²k²).

**Methodological finding (critical):** the stability fields contain
**two distinct ω²<0 regions** that must not be conflated:
- **Longitudinal (pairing) mode** — the small region (0.2–8 % of the
  field); this is the paper's pairing instability and the correct
  "accessible N_H" acceptance metric.
- **Transverse (shear) mode** — ω²<0 over **37–80 % of the field for
  EVERY kernel**, a generic SPH pathology (no shear stiffness in
  standard SPH), present from the smallest N_H. It is not a
  kernel-quality metric and is reported separately in the figures and
  the boundary check.

**Longitudinal (pairing) onsets at 60×200** vs the Phase-5 acceptance
boundaries (paper's accessible-N_H statements):

| Kernel | Onset N_H | Main island | Paper expectation | Verdict |
|---|---|---|---|---|
| cubic b₄ | 62 (4-pt kdn-edge artifact at 30) | N_H 62–973, \|k\|d_nn 2.2–6.0 | ≲ 55 (gradual) | ✓ (grid tolerance) |
| quartic b₅ | 66 | 66–129 at \|k\| 5.1–6.0; 285–1316 at 2.4–6.0 | ≈ 67 | ✓✓ |
| quintic b₆ | 225 | 225–1693, \|k\| 2.6–6.0 | ≈ 190 (+ island near 100) | ~ onset; **island near 100 NOT present** (§3, row 14) |
| Wendland C² | clean (3-pt edge artifact 35–40, \|k\| = 6.0) | — | island near 40 | ✓ as a trace only |
| b₇ | 193 (h/d_nn 1.35) | 193–356 at \|k\| 5.2–6.0; 789–2102 | (not stated) | high-h region only, clean below h/d_nn 1.2 |
| b₈ | 549 (h/d_nn 1.80) | 549–2539, \|k\| 3.0–6.0 | (not stated) | high-h region only, clean below h/d_nn 1.2 |
| Wendland C⁴ | **clean (0.00 %)** | — | (not stated) | ✓ |
| Wendland C⁶ | **clean (0.00 %)** | — | "stable to 700" | ✓ (clean to 2351) |
| HOCT4 | 114 | **N_H 114–185 (centre ≈ 145)**, \|k\| 5.6–6.0 | island near 150 | ✓ confirmed |

The b7/b8 onsets are the paper's Ŵ(H|k|) < 0 pairing region pushed to
large N_H by the higher kernel order (monotone trend: 62 < 193 < 549 <
W4/W6 clean); both are clean in the practical h/d_nn ≲ 1.2 regime. The
paper's "Wendland stable to N_H = 700" is confirmed with margin (W2
clean to 1161, W4 to 1722, W6 to 2351); the HOCT4 island at 114–185
leaves the paper's Table-2 choice HOCT4 @ 442 clean, as the paper uses
it. **The cubic long-λ dip (|k|d_nn 0.3–0.6, N_H 40–100) — the
Phase-5's only open discrepancy — is EMPTY at this resolution (0
longitudinal unstable points): resolved as the phase-reference bug,
§3 row 3.**

**Gaussian excluded from the high-res sweep** (user decision,
2026-09-19): it is by far the largest (N_H to ~82 000 over h/d_nn
0.9–3.0) and the oracle's *build* is still dense O(n1·n2) memory —
measured peak RSS 3.1 GB (N_H 2211) → 23.8 GB (N_H 7000), OOM above;
the full range would need ~1.3 TB. It is also not practical in a real
simulation. Its stability was covered at Phase-5 resolution (stable at
N_H = 100/300, matching the paper) and in Fig. 6. A future
sparse/chunked oracle build would be required to extend the sweep.

### Fig. 6 — sound speed (`fig06_sound_speed.py`, `fig06_paper_config.py`)
c_SPH/c = ω_∥/|k| vs log |k|d_nn (0.2–7.0, **200 points**), N_H cuts
50/100/200/400, k ∝ (1,1,1), λ = 8h verticals — plus a companion
script reproducing the paper's *actual* Fig.-6 layout (three k-direction
panels, all ten Table-2 kernel-N_H rows overlaid, Gaussian N_h = 10/20
omitted — §3 row 15). After the Phase-5 bug fixes: curves start within
~1–7 % of 1 (the paper's tick labels read 0.982–1.025 at the left edge),
flat just above 1 for |k|d_nn < 1, a broad dip to a minimum around
|k|d_nn ~ 4, a slight rebound, and convergence near |k|d_nn ~ 6–7, with
the **same kernel ordering as the paper (cubic N_H = 42 highest, HOCT4
lowest)**. Acceptance: cubic few-% error ✓, quartic (N_H = 60) < 1 %
✓, Wendland improving with N_H ✓, λ = 8h ≲ 1 % ✓.

---

## 3. Discrepancy log (findings → root cause → resolution)

Consolidated from the `PLAN.md` findings log, `phase5_stability_log.md`,
`closeout_plan.md`, and the warpSPH Phase-6 log.

| # | Finding | Root cause | Status |
|---|---|---|---|
| 1 | "Three-way discrepancy" in the small-k longitudinal limit (oracle ≈ 39 vs paper eq.-23 ≈ 0.44 vs derived ≈ −0.16 ×c̄²) | The **complex-step force oracle is invalid for γ = 5/3** — ρ^γ is not holomorphic; the complex step returns the real part of the wrong analytic continuation. The "agreement" between the old oracle and the FD check was only on the real part | **RESOLVED** (phase-5 entry d) — deliverable is the exact real-lattice-sum P matrix |
| 2 | `exact_p_matrix` phase evaluated at raw box-wrapped coordinates → translation non-invariance (ω² changed 2–3× under particle relabelling), spurious long-λ "instability" for the cubic, sawtooth fig06 | The plane-wave phase must use the minimum-image displacement relative to the reference particle, not the raw stored array coordinate (~7/8 of particle 0's neighbours are close only via periodic wraparound) | **RESOLVED** (entry i) — translation invariance restored exactly (bit-identical eigenvalues across 6 reference particles) |
| 3 | **Cubic long-λ instability** (the Phase-5 open item): ω²_∥ < 0 at |k|d_nn ≈ 0.3–0.6, all N_H 40–100, contradicting the paper's "accessible for N_H ≲ 55" | **Same bug as row 2** (the FD ground truth used to validate the buggy phase was itself self-consistent with it — a false negative, caught only after a second, independent bug forced a re-check) | **RESOLVED 2026-09-20** — the 60×200 sweep's dip box (|k|d_nn 0.3–0.6, N_H 40–100) is EMPTY (0 points); not physics |
| 4 | Spurious extra factor K and wrong sign in the P matrix's density-response term (M_rho) | A mis-copied chain-rule step double-counted K (K·B̄ = K²ρ₀^(γ−2)) with the sign flipped; the old "FD agreement" did not shrink with the FD step — it was never truncation error | **RESOLVED** (entry k) — fixed sign and K removal required *simultaneously*; re-validated to true O(h²) FD truncation (0.0001–0.2) |
| 5 | `sphKernelN_H` did not compile under a float64 build; the warp tracer evaluates Python-float BinOps in traced `@wp.func` as **float32 constants** (silent ~1e-8 corruption) | `scalar_t(np.pi * 3.0/4.0)`-style constants trace as float32 in float64 builds; int32×float64 muls have no overload | **RESOLVED** — `kernels/properties.py` (Phase 1) and `util/support.py` (closeout item 2, 2026-09-19: module-level constants + int cast; float64 probe now exact to 0.0 rel; the fix also revealed `volumeToSupport_warp` did not compile in *any* build — the bug was latent behind a compile error) |
| 6 | Code "B7" is not the classical b₇ | The code shape is exactly the D&A eq.-11 family member **b₈** (order 8, degree 7 — the code name is by degree, the paper indexes by order), and its `kernelScale` copied the quintic row (off 13–15 %) | **RESOLVED 2026-09-18** (user-directed) — shape-derived scales adopted; former B7 renamed **B8** (enum value kept, stored configs unaffected); the genuine classical b₇ derived and added as new **B7** (§5.2) |
| 7 | Paper eq. 15 (B-spline closed-form 3D FT) garbled in the PDF text layer and mathematically wrong as printed (poles where the true FT is smooth) | PDF transcription artefact | **RESOLVED** — superseded by a derived piecewise-polynomial closed form (cross-check ≤ 1e-11 vs numerical) |
| 8 | D&A's Sod IC is not the `warpSPH` `sodND` reference problem | The paper's quoted discontinuity positions (contact ≈ 0.17, shock ≈ 0.378 at t = 0.2) match the **R&H 2012 problem** (1,1,0)→(0.125,0.1,0) (exact: contact 0.168239, shock 0.368895), not (1,1,0)→(0.25,0.1795,0) (0.122843/0.315505) | **RESOLVED (working IC)** — the R&H 2012 problem adopted; Phase-6 runs reproduce the exact positions within SPH smearing; final confirmation against the Fig.-11 overlay is a Phase-7 item |
| 9 | Paper Fig. 2's x-axis | It is |k|σ (σ = h/2), not |k|h (user-verified: the cubic's first zero sits just over π) | **RESOLVED** — replication axis corrected |
| 10 | `warpSPH` `CullenDehnen2010.py` sign note ("the signs here should have been wrong, double check!") | Resolved from C&D 2010 App. B eq. B11: D(div v)/Dt = div(dv/dt) − tr(V²) — the **minus** (active path) is correct; the dead `tr(A+V²)` branch was wrong and removed | **RESOLVED 2026-09-19** (Phase 6) |
| 11 | `warpSPH` `sampleOptimal` broken as shipped (ParticleSet vs ParticleState crash; raw unscaled delta-shift → divergence/clumping; lattice start overwritten with random) | Three bugs + two latent ones (unimported `ParticleSet`, ignored `kernel` arg) | **RESOLVED 2026-09-19** (closeout item 3, warpSPH `0bcfe28`) — verified per-particle to 4.4e-15 vs an unmodified CPU run of the original recipe; a `seed` kwarg added |
| 12 | Fig.-3 "glass" configurations not specified in the paper | — | **DOCUMENTED** — delta-shift-relaxed glass proxy (jittered lattice, warpSPH components; density std 0.30 %, nn/dx 0.86) used and noted in the figure caption |
| 13 | Timing figures (12–13) machine-dependent (ALICE vs this box) | — | **SCOPED** — relative scaling only, Phase 7 |
| 14 | The paper's small-N_H instability islands — quintic ≈ 100, Wendland C² ≈ 40 (stated to exist in the *linear* analysis, "but did not trigger in the §4.1 runs") | Not reproduced at 60×200: the quintic has **no** longitudinal island near 100; the W2 "island near 40" survives only as a 3-point trace at the \|k\|d_nn = 6.0 grid edge (N_H 35–40) | **OPEN (minor)** — either below the paper's grid resolution/conventions or a presentation choice; the paper's own dynamic tests did not trigger them either. Documented here; no action |
| 15 | Gaussian N_h = 10/20 rows (N_H = 5120/10240) infeasible for the contour sweep | The oracle's *build* materializes the dense (n1, n2) grid: ~5e8 elements ≈ 40 GB at N_H = 10240 | **DOCUMENTED** — rows omitted from the high-res sweep (user decision; §2 Figs 4–5); a sparse/chunked build is the future fix |

---

## 4. What the replication establishes about the kernels

1. **Pairing stability tracks kernel order/smoothness, as the paper
   claims.** Monotone longitudinal onsets: cubic 62 < b7 193 < b8 549 <
   W4/W6 clean (to 1722/2351); the B-spline instability is the
   Ŵ(H|k|) < 0 region at high |k|, pushed to large N_H by higher order.
   The Wendland family (C²/C⁴/C⁶) has **no** longitudinal instability
   anywhere in h/d_nn 0.9–3.0 — the paper's core claim.
2. **The "accessible N_H" numbers are the linear onsets of the
   longitudinal mode** (cubic ≲ 55 → measured 62; quartic 67 → 66;
   quintic 190 → 225) — consistent within grid tolerance, with the
   transverse mode irrelevant to kernel acceptance.
3. **The density-estimate correction (eqs. 18–19) works as published**
   (fits within ×1.03; corrected band 0.10–2.40 %) and is now a
   first-class warpSPHCore feature (§5.3).
4. **The h = 2σ convention is the right resolution scale** (eq. 17
   overlap at small k for all kernels only under common-h = 2σ
   scaling).

---

## 5. Implications for `warpSPHCore` kernel defaults

### 5.1 Wendland2 + its N_H
Wendland C² is the default-kernel candidate (paper + this replication:
pairing-stable, few-% sound speed, correctable density bias). The
paper's Table-2 choice (W2 @ N_H = 100, h/d_nn = 1.325) sits clean in
the stability field. The built ε correction (§5.3) removes W2's
low-N_H over-estimation (0.27–1.0 % corrected band over the closed
40–400 window, 3D).

### 5.2 B7 → B8 rename + classical B7
The former code B7 (= family b₈, pairing-stable, 3D FT non-negative up
to the 1e-6 floor) is now **B8** with shape-derived scales
2.449490/2.481044/2.513123 (the old scale copied the quintic row, off
13–15 %). The genuine classical b₇ (order 7, degree 6; C = 823543/92160,
5764801/113149π, 5764801/61440π; scales 2.291288/2.325170/2.359700;
3D FT min −3.2e-6, first zero κ̂ ≈ 21.96) is shipped as the new **B7** —
*barely* pairing-unstable by the Fourier criterion, and its linear
onset (193, high-h region only) confirms it is not a replacement for
the B-splines' conventional roles. Stored configs keep resolving to the
same shape (enum values preserved).

### 5.3 The eq.-18 ε self-term correction (closeout item 7, BUILT)
`warpSPHCore/util/densityCorrection.py` (pure-torch post-process inside
`computeDensities`; the warp kernel is untouched; off = bit-exact):
multi-dim refit (8 kernels × 3 dims, densest lattice, library runtime
convention ε(N_H,est)) shipped **six Wendland 2D/3D entries** (window
40–400); 3D agrees with the paper within ×1.04. Not shipped: 1D (raw
estimate already ≤ 1e-6 of exact; implied ε crosses zero — not a power
law) and the B-splines 2D/3D (oscillating lattice bias — the corrected
band is *worse* than raw for 8 of 10 entries; quartic-3D fits a negative
α). warpSPH frontend: `DensityCorrection` config sibling to
`calibrateNormalization` (a *different* correction — lattice
normalisation, not the self-term fraction; both default off),
`--densityCorrection` case flag. Verified: warpSPHCore 46/46, warpSPH
45/45 (CPU f32, real operator + Verlet list). Design note:
`renorm_eps_design_note.md` (status BUILT).

### 5.4 Phase-3 kernel additions
`Gaussian` (truncated at 16σ, the paper's own stability convention —
any truncation invalidates FT non-negativity, paper footnote 10;
kernelScale = 8.0) and `HOCT4` (Read et al. 2010 eqs. 46–51, verified:
N₃ = 6.5150499306 vs the paper's 6.52; σ = 0.228343 H exact) shipped
via the onboarding pipeline; both pass the full audit (perturbation
spot-checks caught deliberate breaks). Figs 1–2 are reproducible
entirely from the shipped audited set.

---

## 6. Frontend deliverables in `warpSPH` (Phase 6)

Ran entirely in `warpSPH` (no warpSPHCore changes). Commits: `dcd0423`
(C&D + tests), `652ed28` (R&H + wiring), `3cf40f5`/`128d82b` (log),
plus `0bcfe28` (sampleOptimal) and `3de37e7` (ε correction config).

- **C&D 2010 switch** — equation-by-equation verified against the
  paper (R eq. 17, ξ eq. 18 with β_ξ = 2, target α eq. 13–14,
  v_sig eq. 15, relaxation eq. 16, l = 0.05); sign question resolved
  (§3 row 10); wired into the **Monaghan** scheme (the switch's
  physical target; compSPH/CRKSPH paths unchanged); Gresho–Chan vortex
  control: C&D-on dissipates *less* than the full-viscosity baseline
  (KE 0.093272 vs 0.093243) — no shear-driven dissipation.
- **R&H 2012 SPHS** (`modules/shockCapturing/ReadHayfield2012.py`) —
  switch eq. 21 (with the h²|∇(∇·v)| denominator), Balsara eq. 32,
  relaxation eqs. 22–25, entropy dissipation eqs. 33–35 (the negative
  K_ij = dW/dr makes it a diffusion), from the full paper transcription
  (supersedes D&A §4.3).
- **Sod validation** (D&A/R&H IC, Monaghan, exact Riemann overlay,
  contact-spike metric = max deviation in a ±0.07 window around the
  contact):
  - **1D @ nx = 400** (clean): R&H suppresses vs NoneSwitch on every
    metric — P +7.15 % vs +8.01 %, e +7.54 % vs +7.68 %, A +11.13 % vs
    +12.09 %.
  - **2D @ nx = 400:** e +6.28 % vs +7.10 %, A +9.55 % vs +10.57 %
    (suppressed); P +5.44 % vs +5.06 % (0.38 pp wash — the earlier
    coarse-res "R&H raises P" was a single-particle-max smearing
    artifact that trends to zero with nx: +5.5 → −4.0 → +1.21 → +0.38
    pp over nx = 40→400).
  - **3D @ nx = 100** (clean PASS): P +7.97 % vs +9.16 %, e +5.36 % vs
    +8.07 %, A +5.39 % vs +10.80 % — all suppressed.
  - C&D (no entropy term) raises or matches the P spike in every
    dimension and never suppresses e/A — confirming the division of
    labour (switch = no shear dissipation; R&H entropy term = contact
    suppression). All switches reproduce the exact head/foot/contact/
    shock within SPH smearing.
- **`sampleOptimal` fixed** (§3 row 11); **ε-correction config**
  (§5.3); **density hook unified** in `computeDensities` (optional
  `supportMode`; monaghan passes SuperSymmetric — zero behaviour
  change, bit-exact vs the old direct calls).

---

## 7. Closeout `src/` decisions

### 7.1 `util/support.py` float32-BinOp constants (item 2)
The dim-3 branch of `volumeToSupport_warp` evaluated
`scalar_t(np.pi * 3.0/4.0)` / `scalar_t(1.0/3.0)` as float32 constants
even in float64 builds (same bug class as the B7 knot) — corrupting the
(4π/3) volume factor at ~1e-8. Fixed with module-level Python-float
constants + an explicit `scalar_t(targetNeighbors)` cast. The fix also
exposed that the function did not compile in *any* build (no
int32×float64 mul overload) — the bug was latent behind that. Verified
exact (0.0 rel) in float64, ~7.5e-8 (f32 round-off) in float32; full
suite 451 passed / 1 skipped.

### 7.2 `sampleOptimal` (item 3, warpSPH)
See §3 row 11 / §6.

### 7.3 The five CG-specialty kernels (decision, user 2026-09-19)
poly6/spiky/adhesion/cohesion/viscosity are force/Laplacian/surface-tension
kernels, **not** general-purpose density kernels → excluded from the
audit pipeline and from any D&A-convention re-scaling. No code change;
figA records their properties for reference.

---

## 8. Remaining work — Phase 7 (dynamic tests, Figs 7–13)

Not started; separate effort. The Phase-6 prerequisite (C&D + R&H) is
delivered and validated (§6). Open items carried into Phase 7:
- **7a** pairing relaxation (§4.1, Figs 7–8): 32 000 particles, FCC +
  1D-Gaussian offsets, evolve to glass; acceptance: cubic pairs
  gradually beyond N_H ≈ 55, quartic > 67, quintic > 190, Wendland
  clean to 700.
- **7b** Gresho–Chan vortex 3D (Figs 9–10, 13): the existing 2D CRKSPH
  case needs a 3D variant; N₁D = 51…406 (6.7×10⁷ at 406 — possible
  documented axial-slab reduction).
- **7c** Sod 3D glass-like ICs (Fig. 11): cross-validate against the
  `warpSPH` reference solver; confirm the R&H-2012 working IC against
  the Fig.-11 exact-solution overlay (§3 row 8).
- **7d** cost (Figs 12–13): relative scaling only (machine ≠ ALICE).
- **Hardware:** the 96 GB GPU is available only when the local LLM is
  not running — schedule large runs accordingly; memory is not the
  constraint at the paper's sizes, runtime is.

---

## 9. Reproducibility

- **Scripts** (this folder, `scripts/`): one deterministic script per
  figure/table, all with `--kernel` (shipped names, repeatable/
  comma-separated, default the full set). `common.py` pins the notation
  (code h = paper H; h_paper = h_code/kernelScale) and evaluates
  kernels via the *shipped* host-callable warp functions.
- **Stability cache:** `results/stability_<kernel>.npz` (two-mode
  fields, gitignored) — regenerate fig04/05/figures from cache with no
  recompute; `check_stability_boundaries.py` reprocesses the cached
  fields with different thresholds/island criteria.
- **Audit pipeline:** `warpSPHCore/scripts/kernels/` (specs + battery +
  onboarding checklist), CI-gated via `tests/kernels/`.
- **Environment:** conda env `warp`; cap BLAS threads
  (`OMP_NUM_THREADS=2`) for all numerics (shared box); Phases 1–6 ran
  on CPU. Scratch files in the repo's `.tmp/` (gitignored).
- **Reference data:** `data/da2012_reference.yaml` (Tables 1–2, eq.-19
  constants, Gresho–Chan IC, Sod Riemann data, full R&H 2012 SPHS
  transcription).
