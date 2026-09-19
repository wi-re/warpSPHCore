# Design note — D&A 2012 eq. 18/19 density self-term correction ("renorm ε")

> **BUILT — closeout item 7 (2026-09-19); item 4 scoped it.** Originally
> written to answer the six questions in `closeout_plan.md` item 4; the
> build then followed this note (warpSPHCore `bf14e56` + warpSPH
> `3de37e7`). Where the build deviated from the scope, the section says
> so. `REPORT.md` (item 6) carries a summary + recommendation; this file
> is the detail.
>
> Paper: Dehnen & Aly (2012) eqs. 18/19; constants in
> `data/da2012_reference.yaml` → `density_correction`.

## 1. What the correction is

- **Eq. 18:** `ρ̂_corr,i = ρ̂_i − ε·m_i·W(0, h_i)` — "simply the original
  estimate (1) with a fraction ε of the self-contribution subtracted".
  Motivation (paper §3.2): at a particle position the self-term
  `m_i W(0, h_i)` biases the estimate high; visible in Fig. 3 as the
  over-estimation of the Wendland/HOCT4/Gaussian curves at small N_H.
- **Eq. 19:** `ε = ε₁₀₀·(N_H/100)^(−α)`; 3D constants — W2
  (0.0294, 0.977), W4 (0.01342, 1.579), W6 (0.0116, 2.236).
- **Consistency property (paper):** replacing ρ̂ by ρ̂,corr in the
  Lagrangian leaves the equations of motion otherwise identical, because
  `h^ν·ρ̂,corr = h^ν·ρ̂ − ε·m·C_d·f(0)` — the correction contributes a
  constant (in h) to `h^νρ̂`, so `∂(h^νρ̂,corr)/∂ln h = ∂(h^νρ̂)/∂ln h`
  and the conservation properties are unaffected. This is the
  justification for the "correct after the fact, evolve the raw estimate"
  implementation (§3).
- **Our refit** (exact FCC lattice sums, `W0(H)` convention): ε₁₀₀ =
  0.02949 / 0.01361 / 0.01131, α = 0.999 / 1.633 / 2.218 (the fig03
  check-#4 values; the item-7 multi-dim refit reproduces them at x1.005
  — it is the identical computation) vs the paper's 0.0294/0.977,
  0.01342/1.579, 0.0116/2.236 — within ×1.04.

## 2. Naming

**Do not name it `renorm`.** `warpSPHCore/renorm.py` is *gradient
renormalization* (covariance matrix → L, the `RenormalizationState`
path); a "renorm ε" feature would collide. Recommended code name:
`densityCorrection` / `selfTermCorrection` (e.g.
`warpSPHCore/util/densityCorrection.py`,
`OperationProperties.densityCorrection`, config param
`schemeConfig.densityCorrection`).

## 3. Hook point — recommended: post-process OUTSIDE the density operator

The correction needs the *finished* raw density (ε depends on N_H, which
depends on ρ̂), so it is a per-particle remap
`(ρ̂, m, h, kernel) → ρ̂,corr`, not a term inside the neighbour sum.

**Density paths in warpSPH (after the 2026-09-19 port — option B):** ALL
schemes that compute a kernel-sum density now go through
`modules/density/density.py::computeDensities` — `monaghan.py` and
`compSPH.py` were ported from direct `warpOperation(Density)` calls onto
it (warpSPH, user decision 2026-09-19). `computeDensities` gained an
optional `supportMode` parameter, **default Gather** (the C&D switch E.1
rationale in its docstring); monaghan passes `config.supportMode`
(default **SuperSymmetric**) so its behavior is **unchanged**, and compSPH
uses the Gather default (it had hardcoded Gather). Verified **bit-exact**
against the old direct calls on a non-uniform-support state, both modes
(warpSPH `.tmp/probe_density_port_equiv.py`; mode spread 1.3e-1 rel
confirms the supports exercise the mode). The other schemes (deltaSPH,
divergenceFree, omniIncompressible, dfsphReference, band2018pb) already
used it; crkSPH has no kernel-sum density path. The ε hook is therefore a
**single call site** inside `computeDensities`.

**Options (A/B as originally scoped; B landed in the ported form):**
- **A. Inside the operator** (flag on `OperationProperties`, applied in
  `coreOperations/wp_density.py` after the sum). Single call site, but the
  correction must be replicated in the forward-mode JVP/HVP density kernels
  (`wp_densityJVP.py`, `wp_densityHVP.py`) and the CRK density path —
  three implementations of the same math.
- **B. Post-processing outside (recommended, LANDED).** A small pure-torch
  elementwise function in warpSPHCore, applied inside `computeDensities`
  immediately after the raw `warpOperation` result and **before the EOS**
  (`idealGasEOS` in monaghan.py reads `currentState.densities`), so the
  EOS, the force terms, `divergence = −drhodt/ρ` and any switch reading the
  scheme's density all see ρ̂,corr — matching the paper's "replace ρ̂ by
  ρ̂,corr in the Lagrangian". The continuity update (`drhodt`) stays the
  scheme's standard one: the correction is a per-step remap of the
  estimate, consistent with the §1 constant-in-h argument.

  - One implementation, trivially testable (pure torch, CPU/GPU, both
    precisions), no AD-path duplication (forward-mode density JVP is not
    used by the D&A compressible use case; if it ever is, the correction's
    own JVP is elementwise).
  - Cost: three elementwise tensor ops on N vectors per density step —
    negligible next to the density operator.
- **Config (as built):** a `DensityCorrection` dataclass
  (`enabled` default **off**, optional `eps100`/`alpha` overrides) as a
  SIBLING FIELD of `SimulationConfig.calibrateNormalization` (user
  decision 2026-09-19: "making it a sibling next to
  calibrateNormalization is probably the easiest option") — not the
  scheme-side module-configuration pattern. `buildConfig` accepts the
  bool shorthand or the dataclass; the runner exposes
  `--densityCorrection/--no-densityCorrection` via a plain CaseSpec
  bool; the nested-dataclass encode/decode branches are fields-driven
  and generic (any future nested config type round-trips unchanged).

Out of scope unless a module reads the scheme's density: internal
re-computations inside modules (e.g. the C&D switch's own E.1 density
estimate) — they do not automatically pick up the correction.

## 4. N_H at density time

Per particle, from the **raw** estimate (the only density available at the
hook; the true ρ the paper uses in `N_H = V_ν H^ν ρ/m` is unknown at
runtime):

```
N_H,i = V_ν · h_i^ν · ρ̂_i / m_i        (V_1, V_2, V_3 = 2, π, 4π/3)
```

- Using ρ̂,corr instead of ρ̂ would shift ε by O(α·ε) ≈ 3–6 % of an
  already 1–3 % correction — second order, negligible; **no iteration**.
- Adaptive h (monaghan's `evaluateOptimalSupport`) is handled naturally —
  the formula is per-particle and per-step.
- **Runtime-convention note (load-bearing, added by the build):** the ε
  that runs at runtime is evaluated at this N_H,est from the RAW
  estimate — on a defect-free lattice that is the exact N_H × ρ̂, NOT
  the exact N_H. Mid-window the difference is negligible (bias ~1e-3);
  at the window edges the bias reaches ~5 % (W6 3D at N_H ≈ 40), which
  shifts ε by O(α·bias) and the residual by α·bias² — that term
  dominates the measured edge band. The refit's bands are therefore
  measured with the runtime convention (dense sweep over the CLOSED
  window; the 34-point fit grid does not even contain N_H = 40.0).
- The paper fits **3D only**. The build extended to 2D (refit on the
  densest 2D lattice — shipped); 1D was refit too and NOT shipped: the
  raw 1D estimate is already within 1e-6 of exact over 40 ≤ N_H ≤ 400
  (the 1D continuum limit is immediate) and the implied ε is not a power
  law — it crosses zero, so no fit exists.

## 5. W(0, H) convention — load-bearing gotcha

`W0 = C_d · f(0) / h^ν` with **h = the code support = the paper's H**
(the notation pin in `data/da2012_reference.yaml`: the shipped kernels are
unit-support, `W(r) = C_d f(r/H)/H^ν`, f supported on [0,1]).

- Library building blocks (host-callable `@wp.func`, as used by
  `delta.py`): `eval_k(0.0, dim, kernel)` = f(0) and
  `eval_C_d(dim, kernel)` — both in `warpSPHCore.kernels.eval_kernel`.
- **Gotcha (findings log 2026-09-18, fig03):** evaluating at
  `h = H/kernelScale` is wrong by `kernelScale^ν` (×7.26 for Wendland C²)
  and breaks the fit.
- The replication already ships both pieces (name-string keyed) in
  `scripts/common.py`: `W0(name, dim, h_code)` and
  `eps_density_correction(name, N_H, ref)` (raises `KeyError` for kernels
  without constants). The library version keys on the `KernelFunctions`
  enum instead of names and owns its own constants table.

## 6. Constants — the shipped table (as built)

The build refit all eight kernels × 3 dims (`scripts/eps_constants_
multidim.py`; all 24 fits in `results/eps_constants_multidim.json`,
window 40–400, bands measured with the §4 runtime convention) and
shipped **six entries**, validity window 40 ≤ N_H ≤ 400 on every
entry:

| (dim, kernel) | ε₁₀₀ | α |
|---|---|---|
| 2D Wendland2 | 3.0233e-3 | 1.5016 |
| 2D Wendland4 | 3.5065e-4 | 2.4815 |
| 2D Wendland6 | 8.0060e-5 | 3.4709 |
| 3D Wendland2 | 2.9488e-2 | 0.9990 |
| 3D Wendland4 | 1.3612e-2 | 1.6333 |
| 3D Wendland6 | 1.1308e-2 | 2.2185 |

- The 3D row is the paper's fit on the same lattice: within ×1.04 (§1)
  and TIGHTER — corrected band over the closed window (dense sweep,
  runtime convention) 0.27–1.0 % (3D) / 3e-5–2.2e-4 (2D), vs ~5 % for
  the paper's constants in 3D; mid-window ~1e-4 (3D) / 1e-5 (2D).
- **NOT shipped, with the measured reason:**
  - 1D (all kernels): the raw estimate is already within 1e-6 of exact
    over 40–400 (the 1D continuum limit is immediate) and the implied
    ε is not a power law — it crosses zero, so no fit exists.
  - B-splines (2D + 3D): they under-estimate and their lattice bias
    OSCILLATES with N_H (the spline offset changes sign, cf.
    `latticeDensity`), so a single (ε₁₀₀, α) mis-corrects: the fitted
    power law's log residuals are 0.78–1.29 decades and the "corrected"
    band is WORSE than the raw one for 8 of the 10 entries (e.g. 3D b8:
    raw 5.4e-2 → "corrected" 1.2e-1); quartic-3D even fits a NEGATIVE
    α (−0.121).
  - HOCT4/Gaussian: no paper fit; the Gaussian's small-N_H
    over-estimation (×2.2 at N_H = 500, ×54 at N_H = 20 — self-term
    dominated while H/d_nn < 2) is outside a power-law ε's reach, which
    is why the paper leaves it out of Fig. 3.
- No constants → `KeyError` with the shipped list (mirrors the
  replication script); the `eps100`/`alpha` overrides exist for refits
  (e.g. on top of `calibrateNormalization` — §7).
- **Wendland2 remains the default-kernel candidate** (the replication's
  working kernel and warpSPH's common choice).

## 7. Coexistence with `calibrateNormalization` — different axes, refit rule

They are **different corrections; do not conflate**:

| | `calibrateNormalization` (LATTICE_DENSITY_PLAN.md) | ε self-term correction |
|---|---|---|
| what | constant **1/L** rescale of the whole lattice sum, `L = latticeDensity(kernel, n_h, dim)` | per-particle **ε(N_H)** fraction of the self-term |
| depends on | (kernel, n_h) only | ρ̂, m, h per particle |
| applied | kernel level (autograd `arg_extract`), on the Density operator | post-operator remap (§3) |
| fixes | ideal-lattice quadrature offset (integral vs lattice sum) | self-contribution bias at small N_H |

They partially **overlap at small n_h** (the lattice offset is dominated
by the self-term), so enabling both with the paper's ε constants
double-counts part of the self-term. **Rule:** both default off; do not
enable both in one run unless ε is refit on the 1/L-corrected lattice sum
(the fig03 machinery does the refit). The D&A replication uses the ε
correction alone (standard kernel normalisation).

## 8. Tests (as built)

1. **Unit (warpSPHCore, `tests/operations/test_densityCorrection.py`,
   46/46 pass, CPU):** the W0 code-support convention (incl. the
   `h/kernelScale` gotcha — the ratio must equal kernelScale^−dim, §5);
   the table transcription (rel 1e-12); paper agreement x1.04; the
   KeyError paths (Gaussian/HOCT4/splines/B7/B8 in 2D+3D; all Wendland
   in 1D); the corrected densest lattice within the refit band at
   N_H = 40/100/200/400 (2D+3D × W2/W4/W6, the refit's own lattice and
   the test's own per-point reference sum) with the RAW estimate ≥ 2×
   the band at the window edge; the manual formula term-by-term
   (incl. the eps100 override) and the zero-ε identity; bad dim raises.
2. **Frontend (warpSPH, `tests/test_densityCorrection.py`, 45/45 pass,
   CPU float32):** every OFF spelling (absent / `False` /
   `DensityCorrection()`) bit-exact with the raw operator; ON bit-exact
   with `applyDensityCorrection` on the raw estimate (and not a no-op);
   the eps100 override reaches the hook; the bool form equals the bare
   dataclass; a non-shipped kernel raises; the corrected densest lattice
   through the REAL operator + Verlet list on a lattice-box periodic
   domain within the refit band at N_H = 40/100/400, raw clearly
   outside at 40; the config serialization round-trip (incl. a
   pre-field dict).
3. **Cross-repo consistency:** the refit script self-tests 3D against
   fig03 check #4 at x1.005 (identical computation — a transcription
   guard, not a new measurement) and the paper at x1.5.
4. **Deferred (GPU):** a scheme-level Monaghan smoke (corrected initial
   ρ̂,corr ≈ 1 within the band, mass conservation) — the runner
   auto-selects cuda:0 and the GPU is held by the local LLM; the CPU
   frontend test exercises the same operator path.

## 9. Build record (built 2026-09-19, closeout item 7)

- `warpSPHCore/util/densityCorrection.py` (~180 lines incl. the module
  docstring: the feature, the shipped table with the shipping rationale,
  the W0 convention, the distinction from calibrateNormalization):
  `densityCorrectionConstants` / `selfTermW0` (one-time host-call
  `C_d f(0)` cache) / `applyDensityCorrection` (pure torch — three
  elementwise ops; the warp kernel is untouched). Exported via
  `util/__init__.py`.
- warpSPH frontend (~90 lines over 4 files): the `DensityCorrection`
  dataclass + SimulationConfig sibling field + buildConfig bool
  shorthand + the generic nested-dataclass encode/decode branches
  (simulationConfig.py); the CaseSpec bool + help line; the runner
  kwarg; the `computeDensities` hook (~15 lines, §3).
- Refit: `scripts/eps_constants_multidim.py` (+ results JSON/TXT,
  gitignored) — 8 kernels × 3 dims × 3 fit windows, dense band sweeps
  with the §4 runtime convention.
- Tests: 46 (warpSPHCore) + 45 (warpSPH), all pass on CPU; §8.
- Commits: warpSPHCore `bf14e56` (refit + util + unit tests); warpSPH
  `3de37e7` (frontend wiring + frontend tests).
