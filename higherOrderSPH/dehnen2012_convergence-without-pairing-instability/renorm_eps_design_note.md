# Design note — D&A 2012 eq. 18/19 density self-term correction ("renorm ε")

> **Closeout item 4 — SCOPE only, NOT built.** This note answers the six
> questions in `closeout_plan.md` item 4 so the feature can be built later
> without re-deriving anything. `REPORT.md` (item 6) carries a summary +
> recommendation; this file is the detail.
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
- **Our Phase-4 refit** (exact FCC lattice sums, `W0(H)` convention):
  ε₁₀₀ = 0.02834 / 0.01335 / 0.01220 vs the paper's 0.0294 / 0.01342 /
  0.0116 — agreement within ×1.04 (findings log 2026-09-18).

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
- **Config:** a `DensityCorrection` module-configuration (pattern:
  `shiftProperties` / `viscositySwitchParams`): `enabled` (default **off**),
  per-kernel constants table lookup, optional `(eps100, alpha)` override.

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
- The paper fits **3D only**; 1D/2D would need their own refits
  (out of scope).

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

## 6. Constants — table, paper values, W2 as the default candidate

- Ship **only the three Wendland 3D pairs** (the paper's fits). Rationale:
  the correction *subtracts* a fraction of the self-term, so it only helps
  kernels that **over-estimate**. Our Phase-4 exact lattice sums:
  Wendland/HOCT4/Gaussian stay ≥ 1 (over-estimate); the B-splines dip
  **below** 1 (under-estimate) — a positive ε would make them *worse*.
  Do not add B-spline constants.
- HOCT4/Gaussian: no paper constants. The Gaussian's small-N_H
  over-estimation (×2.2 at N_H = 500, ×54 at N_H = 20 — self-term
  dominated while H/d_nn < 2) is probably outside a power-law ε's reach,
  which is why the paper leaves it out of Fig. 3. Leave both out; a future
  refit via the fig03 machinery is the route if ever wanted.
- Config override `(eps100, alpha)` per run (for refits, e.g. on top of
  `calibrateNormalization` — §7). No constants → `KeyError` (mirrors the
  replication script).
- **Wendland2 is the default-kernel candidate** for the feature (the
  replication's working kernel and warpSPH's common choice).

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

## 8. Tests (for when it is built)

1. **Unit (warpSPHCore, pure torch):** on FCC lattices (reuse the fig03
   exact-sum machinery), the corrected estimate is within **0.10–2.40 % of
   1 over 40 ≤ N_H ≤ 400** for W2/W4/W6 (the fig03 corrected-curve result);
   the uncorrected output reproduces the fig03 solid curves (W2 minimum
   etc.); the W0-convention check (support-radius value vs the
   `h/kernelScale` gotcha, §5).
2. **Cross-repo consistency:** the library function reproduces
   `scripts/fig03_density_estimation.py`'s corrected curves (the script
   could then delegate to it).
3. **Integration (warpSPH, Monaghan scheme):** lattice/glass IC with the
   correction on — the initial ρ̂,corr reads ≈ 1 within the band (vs the
   raw over-estimate); conservation smoke (total mass exact to round-off;
   EOS pressures shift as the corrected density implies).

## 9. Build estimate (when it is actually built)

- `warpSPHCore/util/densityCorrection.py`: the elementwise function +
  `selfTermW0(kernel, dim, supports)` + the 3-entry constants table
  (~80 lines, pure torch).
- warpSPH: `DensityCorrection` module-configuration (~15 lines) + the
  single call-site wiring inside `computeDensities` (post-port, §3;
  ~5 lines) + config plumbing.
- Tests per §8, reusing the fig03 FCC machinery.
