# Replication plan — Dehnen & Aly 2012

Goal: replicate the results and evaluations of
*Improving convergence in SPH simulations without pairing instability*
(Dehnen & Aly 2012, MNRAS 425, 1068; arXiv:1204.2471) to establish that the
`warpSPHCore` kernels are (a) exactly the paper's kernels and (b) working as
the paper claims — the foundation for the higher-order kernel work in
`../../higher_order.md`.

This effort has grown beyond a pure replication (scope added 2026-09-17):

- **Gaussian and HOCT4 become first-class kernels in `warpSPHCore`** (they
  are the paper's comparison kernels and useful in their own right).
- **A permanent, streamlined kernel-onboarding + audit pipeline** at
  `warpSPHCore` repo level: adding a kernel (B7-style fixes, future general
  Wendland ψ_ℓk, anything) is a checklist run against a machine-readable
  spec, gated by CI.
- **The `warpSPH` frontend gets a proper Cullen & Dehnen (2010)
  viscosity switch and Read & Hayfield (2012) artificial conductivity**
  built and validated — a prerequisite for the paper's dynamic tests,
  which is a good investment in its own right (the current compressible
  scheme is Monaghan-style with rudimentary switch support).

Read `paper_notes.md` first: it contains the notation map (including the
h/H clash between paper and code), the transcribed Tables 1–2, the key
equations, and a figure-by-figure inventory.

## Success criteria

1. Every kernel in the code that appears in the paper (Wendland2/4/6,
   Cubic/Quartic/Quintic Spline) matches the paper's functional form,
   normalisation, and scale convention to machine precision — or any
   mismatch is documented in the findings log below with a root cause.
2. The paper's static evaluations (kernel shapes, Fourier transforms,
   density estimates, linear stability, sound speed — Figs 1–6, Tables
   1–2) are reproduced quantitatively.
3. The paper's dynamic evaluations (pairing relaxation, Gresho–Chan
   vortex, Sod shock — Figs 7–13) are reproduced qualitatively and, where
   the paper gives numbers, quantitatively — cross-validated against the
   `warpSPH` reference solvers where they exist.
4. `warpSPHCore` gains `Gaussian` and `HOCT4` kernels via the onboarding
   pipeline, and the pipeline (specs + audit + CI test) runs green over
   every spec'd kernel.
5. The `warpSPH` compressible scheme offers a validated C&D 2010 switch
   and R&H 2012 conductivity as first-class options.
6. All replication code lives in this folder, imports kernels from
   `warpSPHCore`, and is re-runnable from a clean checkout.

## Where artifacts go

```
warpSPHCore/
├── scripts/kernels/
│   ├── kernel_specs.yaml      # per-kernel machine-readable specs (Phase 1)
│   ├── kernel_audit.py        # the audit battery (Phase 1)
│   └── README.md              # kernel-onboarding checklist (Phase 1)
├── tests/kernels/
│   └── test_kernel_audit.py   # CI wrapper, parametrized over specs (Phase 1)
├── src/warpSPHCore/kernels/kernelFunctions/
│   ├── gaussian.py            # Phase 3
│   └── hoct4.py               # Phase 3
└── higherOrderSPH/dehnen2012_convergence-without-pairing-instability/
    ├── data/da2012_reference.yaml   # paper Tables 1–2 + eq. 19 constants
    ├── scripts/                 # one deterministic script per figure/table
    ├── notebooks/               # optional exploratory notebooks
    ├── sim/                     # shared dynamic-test helpers (Phase 7)
    ├── figures/  results/       # outputs (git-ignored)
    └── REPORT.md                # Phase 8
warpSPH/                         # sibling repo
├── src/warpSPH/modules/shockCapturing/   # C&D 2010 validation + R&H 2012 (Phase 6)
└── src/warpSPH/cases/                      # Gresho-3D + Sod reuse (Phase 7)
```

Replication scripts (`higherOrderSPH/.../scripts/`) must import kernels
from the installed `warpSPHCore`; only the audit pipeline and the two new
kernel files touch `src/`.

## Phase 0 — scaffolding & literature sync

- [x] Create the layout above; add `figures/`, `results/` to `.gitignore`.
- [x] **Sync `read2010` from the dump** (in the `warpSPH` repo):
      `literature/dump/0906.0774v2.pdf` is Read, Hayfield & Agertz 2010,
      *Resolving mixing in Smoothed Particle Hydrodynamics* (MNRAS 405,
      1513) — the source of the HOCT4 kernel. Follow
      `warpSPH/literature/ADDING.md` (identify from the document, fields
      from the DOI record, rename to `read2010_<slug>.pdf`, update
      MANIFEST/references.bib/ABSTRACTS, run `scripts/check_literature.py`).
      Done 2026-09-17: `read2010_resolving-mixing-sph.pdf`, checker green.
- [x] Extract the HOCT4 definition (functional form, normalisation,
      σ ≈ 0.228343 H) from the synced PDF → reference data.
      Done: eqs. 46–51, verified numerically — N₃ = 6.5150499306 (paper
      6.52), D&A's σ = 0.228343 H is exactly the eq.-8-convention 3D σ of
      the verified kernel (0.228343408 H); the whole D&A Table 2 HOCT4
      row is reproduced. See `paper_notes.md` *Verified: HOCT4*.
- [x] Read & Hayfield (2012) artificial-conductivity paper: **synced
      2026-09-17** from the dump as `read2012_sphs-higher-order-dissipation-
      switch.pdf` (MNRAS 422(4), 3037); full SPHS scheme transcribed into
      `data/da2012_reference.yaml` (`rnh2012_sphs`) — supersedes the D&A
      §4.3 interim spec.
- [x] Transcribe Tables 1, 2, the eq. (19) (ε₁₀₀, α) constants, the
      Gresho–Chan IC (eq. 33) and the Sod Riemann data into
      `data/da2012_reference.yaml` (single source of truth). **Sod data
      correction:** the `warpSPH` reference problem (ρ,p,u) =
      (1, 1, 0) / (0.25, 0.1795, 0) is NOT consistent with D&A Fig. 11's
      quoted discontinuity positions (its exact solution: contact 0.1228,
      shock 0.3155 at t = 0.2). The **R&H 2012 problem** (ρ,p,u) =
      (1, 1, 0) / (0.125, 0.1, 0) matches (contact 0.1682, shock 0.3689
      vs the paper's ≈ 0.17 / ≈ 0.378) → working IC, confirm against the
      Fig. 11 exact-solution overlay on the first Phase-7 run. Both exact
      solutions recorded in the YAML.
- [x] `scripts/common.py`: kernel access helpers pinning the notation once
      — `h_paper = h_code / kernelScale(dim)`; host-side kernel evaluation
      tied to the shipped warp functions (evaluate `eval_k`/`eval_C_d` on
      a CPU warp device, not a re-transcription). Smoke test (Table 1,
      dim 3) green at machine precision for all six shipped kernels.

Acceptance: reference data + `common.py` importable; literature checker
green in `warpSPH`; HOCT4 definition transcribed into the reference data.
**MET 2026-09-17.**

## Phase 1 — kernel audit pipeline (repo-level, permanent)

The streamlined "add a kernel, prove it works" process. Lives at
`warpSPHCore` repo level, not in the replication folder — it outlives this
paper.

- [x] `scripts/kernels/kernel_specs.yaml` — per kernel: source citation
      (paper + table/figure), functional form (single expr in `q` with
      `pos(x) = (x₊)` covering the piecewise-polynomial representation;
      `shape_by_dim` for the Wendland family's `(1−r)^ℓ · P(r)` forms so
      any ψ_ℓk can be added later), C (ν=1,2,3), σ²/H² (ν=1,2,3, eq.-8
      convention), packingRatio + reference N_H + code correction factor,
      `ft_expected`, and documented `known_issues` (B7's non-standard
      scale).
- [x] `scripts/kernels/kernel_audit.py` — the battery, per spec'd kernel
      and ν = 1,2,3:
      1. functional form: code kernel vs spec polynomials on a fine grid
         (< 1e-13 in float64);
      2. normalisation: independent numerical
         `V_ν ∫₀¹ r^{ν−1} w(r) dr` → compare with code `C_d`;
      3. moments: σ²/H² from `V_ν/ν ∫₀¹ r^{ν+1} w(r) dr`;
         H/h = 1/(2√(σ²/H²)); compare both with the spec and with the code
         `kernelScale`;
      4. derivatives: code `_dkdq/_d2kdq2/_d3kdq3` vs automatic
         differentiation of `_k` (mirrors
         `scripts/gradcheck/kernel_sanity_native.py` — factor the shared
         logic out rather than duplicating);
      5. support boundary: zero outside support, smooth join at the edge;
      6. Fourier sign analysis (optional flag): `w̄(k)` via eq. (14),
         non-negativity check + first zero κ₀ for kernels expected to have
         one;
      7. packing bookkeeping: `sphKernelN_H` at the spec's packingRatio
         reproduces the reference N_H (allowing for the 3-decimal
         rounding), `sphKernel_xi = H·d_nn`.
      Exit non-zero on any failure; print a per-kernel verdict table.
- [x] `tests/kernels/test_kernel_audit.py` — pytest wrapper (session
      audit in a float64 subprocess so the session precision is untouched)
      + per-kernel verdict tests + broken-spec canary, picked up by CI
      (`.github/workflows/tests.yml` already runs `pytest tests/`).
- [x] `scripts/kernels/README.md` — the onboarding checklist:
      (1) add `kernelFunctions/<name>.py` in the existing pattern,
      (2) register in the `KernelFunctions` enum,
      (3) wire the dispatch in `kernels/eval_kernel.py` (7 dispatch
      functions), (4) add the spec entry, (5) run the audit, (6) run the
      CI tests and commit kernel + dispatch + spec together.

Acceptance: audit + CI test run green over the kernels spec'd in Phase 2
(known discrepancies reported as explicit findings, not masked); a
deliberately-broken test kernel fails the audit (sanity check of the
sanity check). **MET 2026-09-18** — all 8 spec'd kernels audited (b7/b8 split the
former B7 entry; 12/12 CI tests green, ~6 s; ~20 s with `--ft`); the
cubic-C canary fails the audit as required; the former B7 scale
mismatch is the single KNOWN deviation that was ever documented —
resolved 2026-09-18 by adopting the shape-derived scales, so the audit
now has zero KNOWN rows.

## Phase 2 — audit the shipped kernels (D&A Figs 1–2, Table 1)

- [x] Spec entries from `data/da2012_reference.yaml` for Wendland2/4/6,
      Cubic/Quartic/Quintic Spline (and B7/B8 — neither is in Table 1:
      b8 spec'd against the former code B7 shape, b7 derived from the
      family formula) — done in Phase 1 (its acceptance requires the
      Phase-2 kernels spec'd).
- [x] Run the audit → per-kernel verdict — done in Phase 1: all Table 1
      kernels pass every check; the **B7 identity is resolved (2026-09-18)**:
      the code B7 shape is exactly the D&A eq.-11 family member b₈
      (order 8, degree 7 — "B7" is named by degree; classical b₇ is a
      different, 4-term degree-6 spline, max |diff| 5.3e-2), the code's
      C values DO normalise the shape (13.0031746 / 25.2517566 /
      49.6684683), the shape's own moments give
      H/h = 2.449490/2.481044/2.513123 (σ²/H² = 1/24, 0.040614, 19/480
      — 1D exactly on the family pattern 1/(3n)) vs the code scale
      (quintic row) 2.121321/2.158131/2.195775, and the 3D FT is
      non-negative up to the noise floor (κ̂ ~ 25) → pairing-stable.
- [x] Resolve the **B7 scale decision** — RESOLVED 2026-09-18: the
      shape-derived kernelScale was adopted. The former B7 (family b₈)
      was renamed **B8** (enum 33, the old value, so stored configs keep
      resolving to the same shape) with the shape-derived scales
      2.449490/2.481044/2.513123; the genuine classical b₇ (family
      formula, 4 terms, knots 1/7, 3/7, 5/7) was derived and added as
      the new **B7** (enum 34) with shape-derived C
      (823543/92160, 5764801/113149π, 5764801/61440π) and scales
      2.291288/2.325170/2.359700 (σ²/H² = 1/21, 7691281/166329030,
      11/245). The audit `known_issues` entry was removed; the audit
      now has zero KNOWN rows. Both keep the inherited 1.1425 packing
      factor (no paper reference; documented, overridable).
- [x] `fig01_kernel_shapes.py` — Fig. 1 (ν = 3, common h = 2σ scaling,
      linear + log panels, support arrows) for all Table 1 kernels +
      Gaussian + HOCT4. **DONE 2026-09-18**: 8 kernels at h = 1
      (H = scale₃), checks non-negativity / support / normalisation
      (exact piecewise + Simpson); HOCT4 orange per the paper's colour
      coding; figures/fig01_kernel_shapes.{png,pdf} (gitignored).
- [x] `fig02_fourier_transforms.py` — Fig. 2: numerical 3D FT (eq. 14);
      closed-form FT as cross-check — **the closed form is derived from
      the piecewise-polynomial definitions** (eq. 15's PDF transcription
      is garbled beyond repair, see findings log); assertions `w̄(0) = 1`,
      Taylor `w̄(k) = 1 − ½σ²k²` (eq. 17) to O(k⁴), non-negativity for
      Wendland + HOCT4 + Gaussian, sign changes for all B-splines (record
      first zeros — they set the pairing criterion κ₀ > κ_Nyquist).
      **DONE 2026-09-18**: primary = Simpson FT (n = 20001) on
      κ̂ = |k|h ∈ [0, 12]; closed form (exact, piecewise) cross-checks to
      ≤ 1e-11; w̄(0) = 1 to ≤ 5e-12 (HOCT4: paper's 10-dp rounded C₃);
      eq.-17 slope a₂ = −1/8 ± 1e-4 for all 8 (all overlap at small k);
      min w̄ ≥ −1e-10 on [0, 12] for C2/C4/C6/HOCT4/Gaussian; B-spline
      first zeros κ̂ = 6.8829 (κ = 12.5663 ≈ 4π), 11.1282 (κ = 22.4670),
      8.5762 (κ = 18.8314, flat crossing |w̄| ~ 1e-13, ±0.01); zero
      counting validated against the high-res numerical FT (tangential
      near-zeros of b5 at κ̂ ~ 7.78/15.56/23.34 correctly rejected).

Acceptance: all Table 1 rows verified (or discrepancy logged); Figs 1–2
match the paper (Fourier curves: max |Δw̄| < 1e-3 on a common k-grid,
control points re-digitised from the PDF if needed).
**MET 2026-09-18 (with one documented limitation)**: all Table 1 rows
verified via Figs 1–2 + the Phase-1 audit; the closed-form/numerical
cross-check is ≤ 1e-11 (far below the 1e-3 bar); qualitative paper
claims reproduced (all curves overlap at small κ̂ per eq. 17; B-splines
oscillate about zero, Wendland/HOCT4/Gaussian stay non-negative; first
zeros recorded). Limitation: the model has no image input, so a
pixel-level comparison against the paper's rendered Figs 1–2 (incl.
PDF re-digitisation) could not be performed — the PNGs are generated for
user visual check.

## Phase 3 — add Gaussian + HOCT4 to `warpSPHCore`

Onboarded through the Phase-1 pipeline; separate commit(s), clearly marked
as new kernels.

- [ ] `gaussian.py`: the "true Gaussian", **truncated at 16σ** (the paper's
      own convention for the stability work; note in the spec that any
      truncation invalidates FT non-negativity — paper footnote 10).
      Paper's Gaussian = N(0, σ²) with h = 2σ, so σ = H/16,
      shape `f(q) = exp(−0.5(16q)²)`, `kernelScale = H/h = 8.0`;
      `C_d = (128/π)^{ν/2}` (corrected 2026-09-18 — see findings log);
      `packingRatio = 1.337` (Table 2, N_h = 10 row — consistent:
      (N_H/V_ν)^{1/3}/kernelScale with N_H = 5120).
- [ ] `hoct4.py`: from the synced `read2010` definition (verified, see
      `paper_notes.md` *Verified: HOCT4*): `C_d = 6.5150499306`,
      `kernelScale = 2.189684410` (3D, from the eq.-8-convention
      σ = 0.228343 H); `packingRatio = 2.158` (Table 2, N_H = 442).
- [ ] Spec entries, audit run, CI green; Figs 1–2 regenerated including
      the two new kernels (they are the paper's comparison curves).

Acceptance: both kernels pass the full audit; the audit catches a
deliberate perturbation of each (spot-check); the paper's Figs 1–2 are
reproducible entirely from the shipped kernel set.

## Phase 4 — density estimation & N_H bookkeeping
(Fig. 3, eqs. 7, 18, 19, Table 2)

- [ ] `particle_configs.py`: densest-sphere packing (FCC, integer cell
      counts), glass proxy (seeded Poisson random), paired configuration
      (each FCC point → two points, mean density unchanged).
- [ ] `fig03_density_estimation.py`:
      - `ρ̂/ρ` vs N_H (sweep H at fixed particle set) for every shipped
        kernel (incl. new Gaussian + HOCT4), in FCC and glass;
        expected: cubic under-estimates, Wendland over-estimates at low
        N_H, HOCT4 worst;
      - paired-configuration curve (the paper's "crosses"): verify the
        `ρ̂(N_H/f) < ρ̂(N_H)` pairing criterion of §5.1.1;
      - fit ε(N_H) = ε₁₀₀(N_H/100)^{−α} (eq. 19) to the Wendland C²/C⁴/C⁶
        3D over-estimation; compare with the paper's
        (0.0294, 0.977), (0.01342, 1.579), (0.0116, 2.236);
      - corrected estimate (eq. 18) with the paper's ε: within a few % of
        1 over the N_H range.
- [ ] N_H bookkeeping cross-check (Table 2) with both the paper's
      packingRatios and the code's (the deliberate × 1.0175 / × 1.1425
      deviations noted in the findings log).

Acceptance: Fig. 3 reproduced (kernel ordering of biases matches; ε fits
within a factor ≲ 1.5 of the paper's constants — the paper's glass
configurations are not fully specified; the proxy is documented in the
figure caption).

## Phase 5 — linear stability analysis (Figs 4–6, Appendix A)

Pure numerics on a static FCC equilibrium — no time integration.

- [ ] `stability_p_matrix.py`: P-matrix from eq. (23) with `t(k)` (24a),
      `T(k)` (24b) on an FCC lattice (converged shell sums); fixed-h and
      adaptive-h branches (Appendix A; for the constant-density FCC
      equilibrium ℋ̄ ≈ Π̄ ≈ 1, Ξ̄ ≈ 0, but compute them). EOS P = Kρ^{5/3}.
- [ ] Analytic cross-checks before plotting: k → 0 (eqs. 25–26, A19),
      continuum limit (eq. 28), resolved waves (eq. 29), untruncated
      Gaussian stable.
- [ ] `fig04_fig05_stability_contours.py`: ω²_∥/c²k² (top),
      ω²_⊥2/c²k² (bottom) over (|k|·d_nn, h/d_nn), N_H (N_h for the
      Gaussian) on the right axis; wave directions k ∝ (1,1,1),
      k ∝ (1,1,0); Fig. 4 = Gaussian + cubic, Fig. 5 = quartic + quintic
      + HOCT4 (left), Wendland C²–C⁶ (right).
- [ ] `fig06_sound_speed.py`: c_SPH/c vs |k| for the ten Table-2
      kernel-N_H combinations, three wave directions, λ = 8h marker.

Acceptance: stability boundaries — cubic ≲ 55, quartic ≈ 67, quintic
≈ 190 (+ small-N_H island near 100), Wendland C² island near 40, clean
otherwise, HOCT4 island near 150; Fig. 6: cubic few-% error, quartic
(N_H = 60) < 1%, Wendland improving with N_H, λ = 8h ≲ 1%.

## Phase 6 — frontend build-out in `warpSPH` (C&D 2010 + R&H 2012)

Prerequisite for Phase 7, valuable on its own. The frontend's
`src/warpSPH/modules/shockCapturing/CullenDehnen2010.py` already exists
(384 lines, wired into the compressible schemes) but carries an open
"the signs here should have been wrong, double check!" note and
commented-out alternate formulations from prior experimentation. The
compressible scheme itself is Monaghan-style with rudimentary switch
support — now is the time to build this properly.

- [ ] Validate `CullenDehnen2010.py` equation-by-equation against Cullen
      & Dehnen 2010 (local: `warpSPH/literature/cullen2010_inviscid-sph.pdf`):
      resolve the sign question (the R/Ξ limiter of their eqs. 17–18),
      the shear target-α and the l = 0.05 decay integration; remove or
      demote the dead alternate formulations.
- [ ] Add a Read & Hayfield (2012) artificial-conductivity module
      (paper ingested in Phase 0 if obtainable; D&A's §4.3 description as
      interim spec) — its job in the paper: suppress the thermal-energy
      overshoot at the contact discontinuity (accept the over-smoothing
      of e and ρ, the paper says so explicitly).
- [ ] Wire both into the compressible scheme as first-class, selectable
      options (switch on/off, per the paper's usage: C&D switch always on,
      conductivity on for the shock test); keep the Monaghan-style path
      as baseline.
- [ ] Control test (cheap, 2D): the existing `greshoVortex` case
      (CRKSPH, steady exact solution) with and without the switch —
      the switch must remove the shear-driven artificial dissipation
      (D&A intro / Springel 2010 argument; the paper's Fig. 10 cubic
      spline matches the no-viscosity case precisely because of the
      switch).

Acceptance: C&D module matches the paper's equations (sign question
resolved and documented); R&H module in place; control test passes; a
short smoke Sod run shows the contact overshoot suppressed.

## Phase 7 — D&A dynamic tests (Figs 7–13)

Vehicle: the `warpSPH` frontend (Phase 6 puts the needed closures there).
All runs record N, N_H, kernel, h/d_nn, box, dt rule, seed, runtime;
reproducible.

**Hardware note:** the box GPU has **96 GB VRAM, available only when the
local LLM (LM Studio) is not running** — schedule large runs outside an
active LLM session (or on another machine). Memory is not the constraint
at any of the paper's sizes (6.7×10⁷ particles ≈ tens of GB); runtime is.

- [ ] 7a — pairing relaxation (§4.1, Figs 7–8): 32 000 particles, 3D
      periodic, FCC + 1D-Gaussian offsets (σ = d_nn), v = 0, viscous
      heating suppressed, evolve to glass equilibrium. No reference in
      `warpSPH` — build the IC + driver in `sim/`.
      - `fig08_pairing_relaxation.py`: min_i q_min,i (eqs. 30–32) vs N_H,
        all kernels (Wendland with the eq. 18/19 correction), up to
        N_H = 700; Fig. 7 (positions) is a by-product.
- [ ] 7b — Gresho–Chan vortex (§4.2, Figs 9–10, 13): reuse
      `caseUtils.sampleGreshoVortex` — the existing case
      (`cases/greshoVortex.py`) is 2D CRKSPH with the time-independent
      exact solution; D&A's test is 3D, N₁D = 51, 102, 203, 406, t = 1.
      Build the 3D variant (cylindrical IC, axial uniformity); if the
      N₁D = 406 cube (≈ 6.7×10⁷ particles) is impractical on runtime,
      use a documented axial-slab reduction.
      - `fig09_vortex_profiles.py`: v_φ(R) at t = 1, N₁D = 51, four
        kernel-N_H combos + the P₀ = 0 control run.
      - `fig10_vortex_convergence.py`: L₁ velocity error vs N₁D, all
        kernels at their best stable N_H.
- [ ] 7c — Sod shock tube (§4.3, Fig. 11): 3D, glass-like ICs, single N,
      the six kernel-N_H combos of Fig. 10, t = 0.2. Cross-validate
      against the `warpSPH` reference (`cases/sodND.py` +
      `caseUtils/compressible/sod/sodSolution.py` exact solution): run
      the same IC through both the reference solver and the
      replication's configuration, compare profiles — divergence is a
      bug in exactly one of them.
      - `fig11_sod_shock.py`: v, ρ, e_thermal vs exact; L₁ velocity error
        on −0.4 < x < 0.5.
- [ ] 7d — cost (Figs 12–13): `fig12_cost.py` — single-step wall-clock vs
      N_H at N = 2.2×10⁵ (paper) or a documented reduced N; relative
      scaling only (this machine ≠ ALICE). Fig. 13 = replot of Fig. 10
      against measured cost.

Acceptance:
- 7a: cubic pairs gradually beyond N_H ≈ 55, quartic > 67, quintic > 190,
  Wendland clean to the max N_H tested;
- 7b: kernel ranking of Fig. 10 reproduced (Wendland C⁶ N_H = 400 best,
  B-splines saturating, cubic's degraded slope);
- 7c: replication and reference solver agree to integration tolerance;
  post-shock noise decreases with N_H; L₁ error plateau at large N_H;
- 7d: sub-linear cost in N_H below ≈ 400.

## Phase 8 — reporting

- [ ] `REPORT.md`: per-figure comparison (paper vs replication), the
      discrepancy table from the findings log, and implications for
      `warpSPHCore` kernel defaults (Wendland2 + its N_H, the B7→B8
      rename + new classical B7 outcome (resolved 2026-09-18), the
      eq.-18 ε constants as a candidate `renorm` feature — the existing
      `calibrateNormalization` lattice correction is a *different*
      correction; do not conflate) plus a summary of the Phase 3 kernel
      additions and the Phase 6 frontend work.
- [ ] Decisions on any `src/` fixes as separate, explicit steps.

## Findings log

| Date | Finding | Status |
|---|---|---|
| 2026-09-17 | **Eq. 8 carries a 1/ν factor** (`σ² = ν⁻¹∫\|x\|²W d^νx`); the initial HOCT4 "σ discrepancy" (D&A 0.228343 H vs the \|x\|²-moment 0.3955024 H) was this factor being missed. With it, D&A's σ is exactly the verified kernel's 3D value (0.3955024/√3 = 0.2283434 H) and the whole Table 2 HOCT4 row reproduces. All σ in the reference data use the eq.-8 convention; `kernelScale(HOCT4, 3D) = 2.189684410`, not 1.264215. | resolved — informational for all phases |
| 2026-09-17 | **D&A's Sod IC is not the `warpSPH` `sodND` problem.** The paper's quoted discontinuity positions (contact ≈ 0.17, shock ≈ 0.378 at t = 0.2) match the R&H 2012 problem (1,1,0)→(0.125,0.1,0), γ = 5/3 (contact 0.168239, shock 0.368895), not (1,1,0)→(0.25,0.1795,0) (contact 0.122843, shock 0.315505). Working IC for Phase 7; confirm against the Fig. 11 exact-solution overlay on the first run. | open — confirm Phase 7 |
| 2026-09-17 | **CT/HOCT central cores are linear with a cusp** (f′(0) ≠ 0 — the "constant central core to the kernel gradient"). The CT core printed in the typeset read2010 PDF is garbled; the C²-continuous reading (linear core 11/9 − 2x, α = 1/3) normalises to the paper's own N formula with ratio 1.000000, proving the reading. | resolved — informational |
| 2026-09-18 | **`sphKernelN_H` could not compile under a float64 build** (the kernel audit's precision): bare `np.pi` / `4 * np.pi / 3` constant BinOps trace as int32×float32, and `packingRatio**dim` is `pow(float64, int32)` (no overload; only the float32 build ever compiled this function). Fixed in `src/warpSPHCore/kernels/properties.py` (module-level constant leaves + `wp.pow(x, scalar_t(dim))`); semantically identical in float32. Found by the audit — evidence the float64 path was untested. | resolved 2026-09-18 |
| 2026-09-17, resolved 2026-09-18 | **Code B7 identity: it is the D&A family's b₈ (order 8, degree 7), not the classical b₇.** The paper's eq. 11 (Schoenberg B-splines, closed form b_n(q) = Σᵢ(−1)ⁱC(n,i)((n−2i)/n−q)₊^{n−1}) was derived from Table 1 and verified to machine precision against the shipped b4/b5/b6; the code B7 shape (knots ¼/½/¾, 4 terms, degree 7) matches family n = 8 to 8.3e-17 in all dims — the code name "B7" is by degree, D&A index by order. The 1D pattern σ²/H² = 1/(3n) (stated in the paper: b_n → N(0, H²/3n)) holds exactly for all n incl. n = 8 (1/24) and n = 7 (1/21), so it is NOT a red herring — it is the code shape's own 1D moment. The classical b₇ (4 terms, degree 6, (1−q)⁶ − 7(5/7−q)₊⁶ + 21(3/7−q)₊⁶ − 35(1/7−q)₊⁶) differs from the code shape by 5.3e-2 max; its moments: C = 823543/92160, 5764801/113149π, 5764801/61440π (8.936013/16.217493/29.866425), σ²/H² = 1/21, 7691281/166329030, 11/245, H/h = 2.291288/2.325170/2.359700, 3D FT min −3.17e-6 (barely pairing-unstable, first zero κ̂ ≈ 21.96). Code B7 (= b₈): C = 4096/315, 589824/7435π, 16384/105π (all correct for the shape), σ²/H² = 1/24, 531453/13085600, 19/480, H/h(shape) = 2.449490/2.481044/2.513123, 3D FT non-negative (pairing-stable). `B7_kernelScale` copied the quintic (b6) row (2.121321/2.158131/2.195775) — off by ~13–15 %. **RESOLUTION (2026-09-18, user-directed):** shape-derived scales adopted; the former B7 was renamed **B8** (enum 33, the old value — stored configs keep resolving to the same shape) with the corrected scale, and the genuine classical b₇ was added as the new **B7** (enum 34). Audit now has zero KNOWN rows; the audit's `known_issues` entry was removed. | resolved 2026-09-18 — scale adopted, rename + new kernel shipped |
| 2026-09-18 | **Warp tracer evaluates Python-float BinOps inside traced `@wp.func` in float32** — `scalar_t(5.0/7.0)` becomes the float32 constant 0.7142857313156128 in a float64 build, silently corrupting any non-exactly-representable constant (~1.2e-7 shape error for b₇'s 5/7 knot). Found when the new classical b₇ shape failed its audit form check at 1.25e-7 (machine precision elsewhere). Safe forms (verified by probe): full-precision float literals, module-level Python-float constants referenced by name; inexact literals like `scalar_t(0.6)` are unaffected. Same class as the `properties.py` constant-BinOp finding. Fixed in `kernelFunctions/B7.py` (module-level knot constants); **`src/warpSPHCore/util/support.py:86` has the same pattern** (`scalar_t(np.pi * 3.0 /4.0)`, `scalar_t(1.0/3.0)`) — unfixed, out of scope, candidate `src/` fix (Phase 8 decisions). | resolved for B7; support.py instance open — Phase 8 |
| 2026-09-18 | **Gaussian convention in the reference YAML was wrong** (shape and C rows): it had f(q) = e^{−0.5(8q)²}, C_d = (32/π)^{ν/2}, i.e. σ = H/8 — contradicting the paper (Gaussian = N(0,σ²) with h = 2σ, truncated at 16σ → σ = H/16) and the YAML's own σ²/H² = 1/256 row. With the old convention the Gaussian's own h would be √2, not 1, and its eq.-17 slope would be −κ̂²/4 instead of the universal −κ̂²/8 (it would NOT overlap the other kernels at small k, contrary to the paper's Fig. 2 statement). Corrected: f(q) = e^{−0.5(16q)²}, C_d = (128/π)^{ν/2} (C₃ = 260.0699, W(0) = 0.50795 at h = 1; truncation f(1) = e^{−128} ≈ 3e-56). kernelScale = 8.0, packingRatio 1.337 unchanged. Found by the Phase-2 eq.-17 Taylor check (fit gave a₂ = −0.4994 instead of −1/8). | resolved 2026-09-18 — YAML, ft_kernels, fig01, Phase-3 spec updated |
| 2026-09-18 | **Eq. 15 (B-spline closed-form 3D FT) is garbled beyond repair** in the PDF text layer, and the printed form (as either transcription) is mathematically wrong: it has poles at sin(nκ) = 0 where the true FT is smooth and diverges at κ → 0 while w̄(0) = 1. Superseded by a derived closed form: every kernel except the Gaussian is a piecewise polynomial, so ∫_a^b r^m sin(κr) dr per piece by exact antiderivatives (Taylor series in κ for κ < 5 to avoid cancellation, antiderivative for κ ≥ 5); piecewise transcription validated against the shipped shapes at < 1e-12. Cross-check closed vs numerical: ≤ 1e-11 on the Fig.-2 grid (cubic's closed form also matches the hand-derived 384(−2κsin(κ/2) + κsinκ − 16cos(κ/2) + 4cosκ + 12)/κ⁶). | resolved 2026-09-18 — `ft_kernels.py` |
| 2026-09-18 | **B-spline 3D FT zero structure (common h = 2σ, κ̂ = |k|h)**: b₄ — 7 sign-change zeros in [0,30] at 6.8829/9.8446/13.7657/16.9252/20.6487/23.8897/27.5318 (4nπ/H factor zeros + tan(κ/4) = κ/4 factor zeros); b₅ — 3 sign-change zeros at 11.1282/19.1320/27.0047 PLUS 3 tangential near-zeros (w̄ touches ~0 from one side only, |w̄| ~ 1e-11, at κ̂ ~ 7.78/15.56/23.34 — the closed form's ~1e-16 rounding flips the sign across them; the high-res numerical FT confirms no crossing); b₆ — 4 sign-change zeros at 8.5762/12.2783/21.1094/29.7957. First zeros (pairing criterion κ₀): κ = 12.5663 (≈ 4π) / 22.4670 / 18.8314 — b₅ agrees with the Phase-1 audit (22.46). b₆'s first crossing is flat (|w̄| ~ 1e-13, slope ~ 2e-11/κ̂) and is only resolvable to ±0.01 in κ̂ (closed form and high-res Simpson agree within that). | resolved 2026-09-18 — recorded for Phase 4 (stability) |
| 2026-09-17 | Code `packingRatio` deviations from Table 2: CubicSpline × 1.0175 (Price 2012 alignment), QuinticSpline, B7 & B8 × 1.1425 (CRKSPH alignment). Deliberate per in-code comments; replicate both variants. | open — Phase 4 |
| 2026-09-17 | Code `h` (kernel functions) = paper's support radius H, not paper's h = 2σ. Notation map in `paper_notes.md`. | informational — all phases |
| 2026-09-17 | `warpSPH` `CullenDehnen2010.py` carries an unresolved sign note ("the signs here should have been wrong, double check!") plus dead alternate formulations. | open — Phase 6 |
| 2026-09-17 | Existing `greshoVortex` case is 2D CRKSPH; D&A's test is 3D conservative SPH — build a 3D variant, keep the 2D case as cross-check. | open — Phase 7 |

## Risks & open questions

- ~~**HOCT4 functional form**~~ — **resolved 2026-09-17**: read2010
  synced, definition verified (see findings log); Phase 3 unblocked.
- ~~**Read & Hayfield (2012)** not in the collection~~ — **resolved
  2026-09-17**: read2012 synced, SPHS fully transcribed; Phase 6 builds
  from the paper, not D&A's description.
- **D&A Sod IC** is a working assignment (R&H 2012 problem); if the first
  Phase-7 run's exact-solution overlay disagrees, re-derive the IC from
  Fig. 11's digitised discontinuity positions.
- **Fig. 3 glass configurations** are not fully specified; Poisson proxy
  documented as a deviation.
- **Timing figures (12–13)** are machine-dependent; relative scaling only.
- **GPU availability**: 96 GB VRAM only when the local LLM is stopped —
  large Phase-7 runs must be scheduled accordingly.
- PDF text layer garbles Table 1 (C column) and eq. (15); code
  transcription used, independently verified in Phase 1/2.
- Environment: conda env `warp`; cap BLAS threads for all NumPy/SciPy
  numerics (shared box); `wp.init()` once per process; Phases 1–5 run on
  CPU alone.

## Phase overview

| Phase | Paper content / scope | Cost | Output |
|---|---|---|---|
| 0 | scaffolding, read2010 sync, reference data | minutes | `data/`, literature sync |
| 1 | kernel audit pipeline (repo-level, CI-gated) | hours | onboarding process |
| 2 | Figs 1–2, Table 1, B7/B8 flag (resolved 2026-09-18) | minutes (pure math) | kernel-identity verdict |
| 3 | Gaussian + HOCT4 in `warpSPHCore` | hours | two new kernels |
| 4 | Fig. 3, eqs. 18–19, Table 2 | minutes–hours | density-bias + ε verdict |
| 5 | Figs 4–6, Appendix A | hours (eigenvalue sweeps) | stability/sound-speed verdict |
| 6 | C&D 2010 + R&H 2012 in `warpSPH` | days | validated dissipation controls |
| 7 | Figs 7–13 | hours–days (dynamics) | convergence + cost verdicts |
| 8 | `REPORT.md` | — | findings, decisions |

Phases 2–5 deliver the core "kernels exactly right and working right"
verdict; Phases 1, 3 and 6 are the durable infrastructure this replication
exists to motivate.
