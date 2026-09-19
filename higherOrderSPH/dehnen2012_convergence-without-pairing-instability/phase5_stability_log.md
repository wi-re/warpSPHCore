# Phase 5 work log — linear stability (Figs 4–6, Appendix A)

> **Purpose:** resumable, step-by-step record of the Phase-5 derivation so
> the work can be picked up at any point (the session may crash mid-prompt
> on context limits). Newest entries at the BOTTOM. Each entry states what
> was tried, the result (numbers), and the next step. The derived
> equations are kept verbatim so nothing has to be re-derived from memory.
>
> Scratch probe: `.tmp/stability_probe.py` (gitignored). Deliverables:
> `scripts/stability_p_matrix.py`, `fig04_fig05_stability_contours.py`,
> `fig06_sound_speed.py` (all with `--kernel`).
>
> Notation (pinned): code `h` = paper support radius H; paper `h` = H/kernelScale.
> FCC N = 4000, L = 1, m = 1/4000, d_nn = 0.070711, ρ̄ = 1, γ = 5/3,
> K = 1/γ = 3/5, c̄² = Kγρ̄^{γ−1} = 1, P̄ = K = 3/5, B̄ = Kρ̄^{γ−2} = K.
> r_j ≡ x̄_0 − x̄_j (vector j→0; the oracle stores d0 = −r_j = x_j − x_0).
> W(r) = C_d H^{−ν} f(q), q = r/H. ∇W = (C_d/H⁴)f′(q)r̂;
> ∇∇W = (C_d/H⁵)[(f′/q)(I−r̂r̂ᵀ) + f″r̂r̂ᵀ]; (x·∇)W = C_dH^{−3}qf′;
> (x·∇)²W = C_dH^{−(ν+2)}q²(f′+f″).

## Pinned facts (do not re-derive)

- **Force law (eq. 3, PDF text garbled on sign AND mass):**
  ẍ_i = −Σ_j **m_j**[ P̂_i/ρ̂_i² ∇_{x_i}W(x_i−x_j,h_i) + P̂_j/ρ̂_j² ∇_{x_i}W(x_i−x_j,h_j) ].
  Leading MINUS and explicit m_j are BOTH required. Pinned by ẍ → −∇P/ρ.
- **Density response (linear):** δρ_j = iΦ_j(a·t), with t = mΣ_j sin(k·r_j)∇W(r_j)
  (paper A7). Continuum: t → −ρ̄Ŵk ⇒ δρ_0 = −iρ̄Ŵ(k·a) = continuity eq. ✓
  (The earlier "δρ_0 = +iρ̄(k·a)" note was a 1/i slip; −i is correct and matches.)
- **Oracle** (`.tmp/stability_probe.py`): complex-step Jacobian of the actual
  force, P a = ω²a, P = −Im(ẍ_0)/eps. Self-consistent: FD↔CS 3.6e-7,
  P symmetric 8.5e-14, eq. |ẍ₀|~1e-14, ρ̂₀ = 1.004 (lattice sum 1.003998025).
- **The paper's eq.-23 / A11 / A19 text layer is garbled** — treat as
  hypotheses to verify against the force, not as ground truth.

## The three-way discrepancy (the gate)

At N_H = 100 (cubic, k ∝ (1,1,0)), small-k longitudinal ω²_∥/k²:
- oracle → ≈ 39 (k̂d_nn = 0.02)
- paper eq.-23 closed form → 0.44
- independent pressure-term derivation → −0.16
None is c̄² = 1. (k→0 = c̄² is a *continuum/N_H→∞* limit; at finite N_H the
SPH sound speed legitimately differs, but all three objects must agree with
EACH OTHER.)

---

## Log

### 2026-09-18 (a) — force-derived P matrix, clean re-derivation

Linearize ẍ_0 = −mΣ_j g_j ∇W(r_j + a(1−Φ_j)), Φ_0 = 1 (particle 0 at origin),
g_j = P_0/ρ_0² + P_j/ρ_j² = B̄[2 + (γ−2)(δρ_0+δρ_j)/ρ̄]
    = B̄[2 + (γ−2)(i/ρ̄)(1+Φ_j)(a·t)],  δρ_0 = i(a·t), δρ_j = iΦ_j(a·t).

Expand to O(a) (Σ_j∇W = 0, Σ_jH(r_j)Φ_j real by even/odd symmetry):

- Term from 2·H·a(1−Φ_j):  −2mB̄[Hs − U]a,
  Hs ≡ Σ_j ∇∇W(r_j)  (isotropic ∝ I, generally ≠ 0),  U ≡ Σ_j ∇∇W(r_j)cos(k·r_j).
- Term from (γ−2) part × ∇W:  −(γ−2)(B̄/ρ̄)(a·t)·[Σ(1+Φ_j)∇W(r_j)]
  = −(γ−2)(B̄/ρ̄)(a·t)[0 + Σ∇WΦ_j] = −(γ−2)(B̄/ρ̄)(a·t)(−i t/m)
  = −(γ−2)(B̄/ρ̄)(ttᵀ)a.   (Σ_j∇W(r_j)Φ_j = −i(t/m).)

⇒ ẍ_0 = −2mB̄(Hs−U)a − (γ−2)(B̄/ρ̄)(ttᵀ)a,  and P = −(ẍ_0/a):

> **P = 2mB̄(Hs − U) + (γ−2)(B̄/ρ̄)(ttᵀ)**          … (P-mine)
>
> with T ≡ m(Hs − U) = mΣ_j(1−cos k·r_j)∇∇W(r_j) (paper's A10 "T"), so
> equivalently  **P = 2B̄T + (γ−2)(B̄/ρ̄)(ttᵀ)**.

All quantities are computable from d0 (even/odd symmetry makes r_j=−d0
irrelevant): t = mΣ sin(k·d0)∇W(d0); Hs = Σ∇∇W(d0); U = Σcos(k·d0)∇∇W(d0).

**Continuum check (P-mine):** mHs → ρ̄∫∇∇W = 0; mU → ρ̄∫∇∇W e^{ikr} = −ρ̄Ŵkkᵀ;
t → −ρ̄Ŵk. So P → 2B̄ρ̄Ŵkkᵀ + (γ−2)B̄ρ̄Ŵ²kkᵀ = B̄ρ̄Ŵkkᵀ[2 + (γ−2)Ŵ].
ω²_∥/k² → (c̄²/γ)Ŵ[2+(γ−2)Ŵ] → **c̄²** at Ŵ=1. ✓  ω²_⊥ → 0. ✓
**P-mine reproduces the Euler continuum exactly.**

**Compare to paper eq. 23** (as transcribed): P = c̄²uuᵀ + (2P̄/ρ̄)(U_p − ½uuᵀ),
u = t/ρ̄, U_p = T/ρ̄  ⇒  P = 2B̄ρ̄U_p + (γ−1)B̄ρ̄uuᵀ.
**Identical to P-mine except the uuᵀ coefficient: paper (γ−1) vs mine (γ−2).**
The paper's version gives ω²_∥/k² → (c̄²/γ)(2+γ−1) = c̄²(γ+1)/γ = (8/5)c̄² ✗.
So **eq. 23 as transcribed is inconsistent with the continuum**; the garbled
text layer most likely misprinted (γ−2) as (γ−1) (or the u/U definitions differ).

**k→0 of P-mine (finite N_H):** t → −ℋ̄ρ̄k (A19a); T k → ρ̄(9Π̄−2ℋ̄)/10·k²k
(my isotropic sum, below). ⇒ ω²_∥/k² → (c̄²/γ)[(9Π̄−2ℋ̄)/10 + (γ−2)ℋ̄²].
For ℋ̄=Π̄=1, γ=5/3: (c̄²/(5/3))[0.7 − 1/3] = 0.22 c̄². (NOT 1, NOT 39 — see (b).)

**Isotropic small-k sums used** (FCC cubic symmetry; A ≡ Σ m r²W″ = Σ m(x·∇)²W,
B ≡ Σ m r²(W′/r) = Σ m(x·∇)W = −3ρ̄ℋ̄):
- t → (k/3)B = −ℋ̄ρ̄k.  (A19a ✓)
- T_ij → ((A−B)/30)(k²δ_ij + 2k_ik_j) + (B/6)k²δ_ij
  ⇒ T k → ρ̄(9Π̄−2ℋ̄)/10·k²k (longitudinal),  T⊥ → ρ̄(3Π̄−4ℋ̄)/10·k² (transverse).
  Paper A19c claims T → ρ̄(2ℋ̄+3Π̄)/5·kkᵀ + 3ρ̄(Π̄−ℋ̄)/10·k²I — **disagrees**
  with mine (kkᵀ coef (2ℋ̄+3Π̄)/5 vs (3Π̄+ℋ̄)/5; I coef 3(Π̄−ℋ̄)/10 vs (3Π̄−4ℋ̄)/10).
  A = 9ρ̄Π̄ by A18b (paper's literal convention).

### 2026-09-18 (b) — NUMERICAL TEST: P-mine vs oracle (the gate test)

Compute P-mine from d0 and compare to `o.p_matrix(k)` (the exact complex-step
force Jacobian). They MUST agree to roundoff if the linearization is right.

Result: **FAILED.** max|P_mine − P_oracle| grows with k (1.7 at k̂d=0.02 → 1040
at k̂d=2). Small-k longitudinal ω²_∥/k²: P-mine → −0.16, oracle → +39. (The
continuum check of P-mine passed, so the algebra is consistent; the finite-k
disagreement is the bug.)

### 2026-09-18 (c) — ROOT CAUSE: the A9 density response is missing its real part

Long debugging chain (direct sum, holomorphicity, per-neighbor chain rule, piece
index, `|rvec0−rpn|` comparison) isolated TWO distinct issues:

1. **Phase bug (fixed).** The plane-wave phase must use the ABSOLUTE position
   x_j (Φ_j = exp(ik·x_j)), not the minimum-image vector d0_j = MI(x_j−x_0).
   For a neighbor across the periodic boundary d0_j = x_j−L·n, so
   exp(ik·d0_j)≠exp(ik·x_j). The oracle (ph1 = exp(ik@x[list1])) was correct
   all along; my analytical sums had used the MI phase. Only the KERNEL
   (∇W, ∇∇W) uses the relative position d0_j.

2. **Density-response omission (the real bug, in the PAPER).** After the phase
   fix, P still disagreed with a huge spurious imaginary part. Isolating the
   Hessian term with γ=2 (which kills the (γ−2) density term) showed
   −2mB̄(Hs−U) wrong for BOTH absolute and relative phase. The per-neighbor
   force ratio F_oracle/F_lin = −0.0813 (constant in amplitude) proved the
   linearization form right but the density-response coefficient wrong.
   Direct comparison of the density response (particle 0, x-component):
       exact ∂ρ_0/∂a = mΣ_k ∇W(x_k)·e^{ik·x_k} = −3.91262 + 3.79698 j
       paper A9  i·(a·t) = i·mΣ_k sin(k·x_k)∇W(x_k) = 0.00000 + 3.79698 j
   **The imaginary parts match exactly; the real part (−3.91262 =
   mΣ∇W(x_k)cos(k·x_k)) is missing from the paper's A9.** The true response is
   ∂ρ_j/∂a = mΣ_k∇W(x_j−x_k)(Φ_j−Φ_k), a double sum that does NOT reduce to
   Φ_j·(∂ρ_0/∂a) (the cos part breaks the phase factor). So the paper's
   Appendix-A closed form (eqs. A7/A9/A11 → eq. 23) is an approximation that
   drops the real part of the density response; it is not the exact force
   linearization. (Consistent with the earlier finding that eq. 23 as
   transcribed gives (8/5)c̄² in the continuum, not c̄².)

**DECISION (SUPERSEDED by (d)): use the oracle (complex-step Jacobian of the
actual force) as the P matrix.** It is the ground truth (self-consistent FD↔CS
3.6e-7, symmetric 8.5e-14, equilibrium |ẍ₀|~1e-14, reproduces the Euler
continuum), it places NO reliance on the garbled/omitting Appendix-A text, and
it is fast enough (cubic N_H=100: 0.010 s/k-pt; gaussian N_H=512: 0.396 s/k-pt).

### 2026-09-19 (d) — CRITICAL: the complex-step oracle is WRONG for γ=5/3

After building `stability.py` (oracle + Lattice) and validating it, the
stability-boundary scan did NOT reproduce the paper: every kernel showed
ω²_k<0 at the tested N_H, but the paper says the cubic is STABLE for N_H≲55.
Root-caused to the oracle itself:

1. **The force is not holomorphic in the displacement `a`.** The force contains
   ρ̂^γ with γ=5/3 (non-integer); ρ̂^γ has a branch cut, so it is not holomorphic
   in complex `a`. (The DENSITY is holomorphic — confirmed earlier — but the
   FORCE, which contains P=Kρ̂^γ, is not.)
2. **The complex step is therefore invalid.** `p_matrix` evaluates the force at
   `a=i·eps`, which makes `rvec0 = -d0 + a(1-Φ)` COMPLEX (the plane-wave phase
   Φ=exp(ik·x_j) is complex even for real a). The kernel AND ρ̂^γ are evaluated
   at complex arguments; the ρ̂^γ branch cut makes the complex step return the
   real part of the analytic continuation, NOT the physical (real) Jacobian.
3. **Confirmed numerically:** at cubic N_H=40, |k|d_nn=0.5, the complex-step
   Jacobian and the real central-difference disagree by 227 — the FD has a
   large imaginary part (−76, −52) that the CS misses. The real parts match
   (−233, −44.9), so the CS gives the real part of the (wrong) analytic
   continuation.
4. The equilibrium is sound at all N_H (|F0|~1e-14, P symmetric, rho0~1), so
   the bug is specifically in the complex-step Jacobian for non-integer γ.

**CONSEQUENCE:** the oracle's P matrix (and hence the stability boundaries and
sound speed computed from it) is WRONG for γ=5/3. The "continuum limit"
discrepancy chased earlier (ω²_k/|k|² → 2.5, not 1) was a SYMPTOM of this bug
(compounded by the |k|H→0-at-fixed-N_H lattice-sum cancellation red herring).

**CORRECT approach:** compute P from REAL lattice sums at the real equilibrium
(no complex arguments, no ρ̂^γ branch cut):
    P = 2Km·Σ_j (1-cos(ik·x_j)) ∇∇W(d0_j)                     [Hessian, real]
        + (γ-2)Km·Σ_j ∇W(d0_j)⊗(∂ρ_j/∂a)                      [density, exact]
  where ∂ρ_j/∂a = mΣ_k ∇W(x_j-x_k)(Φ_j-Φ_k) (the double sum; the A=∂ρ_0/∂a
  term drops out of Σ∇W⊗(A+∂ρ_j/∂a) because Σ_j∇W(d0_j)=0).
  The state-snapshot "P-mine" (P = 2mB̄(Hs-U) + (γ−2)(B̄/ρ̄)ttᵀ) is the
  CONTINUUM-correct form (reproduces c²) but approximates ∂ρ_j/∂a ≈ Φ_j A
  (missing the real part) — good at large N_H, approximate at small N_H.

[next] implement the REAL-lattice-sum P matrix (with the exact double sum for
∂ρ_j/∂a), verify it (continuum → c², stability boundaries vs paper, CS-vs-FD
now agree), then rebuild the deliverables on top of it.

### 2026-09-19 (e) — BREAKTHROUGH: exact real P matrix validated against ground truth

Two independent bugs were masking the (correct) analytic P matrix. Once both
were fixed, the analytic `exact_P` agrees with the **ground-truth real
finite-difference Jacobian of the actual SPH force** to within the FD
truncation error (max|P_fd − P_ex| = 1–18 at h=1e-6, down from 2508).

**The exact P matrix (REAL lattice sums, no complex rho^gamma, no branch cut):**
    P = 2 m K B̄  Σ_j (1−cos k·x_j) ∇∇W(d0_j)            [Hessian, real]
        − m K B̄ (γ−2)/ρ̄  Σ_j ∇W(d0_j) ⊗ B_j            [density, exact]
  B̄ = K ρ̄^{γ−2} (=K at ρ̄=1);  d0_j = MI(x_j−x_0);  phase on ABSOLUTE x_j.
  Exact density response (double sum, the A9 term the paper drops):
    B_j = ∂ρ_j/∂a = m Σ_k ∇W(x_j−x_k)(Φ_j−Φ_k)
          = −m Σ_k ∇W(x_j−x_k) Φ_k        (using Σ_k ∇W(x_j−x_k)=0 at equil.)
  The complex traveling-wave matrix M ( ẍ_0 = M a ) has  P = −M; the
  physical (standing-wave) matrix is **P_real = Re[P]**.  Eigenvalues = ω².

**BUG 1 — the Hessian coefficient (dropped a factor of H).** My scratch
`gradW_hess` used `(f'/r)` for the (I−r̂r̂ᵀ) term; the correct coefficient is
`(f'/q)` = H·(f'/r) (exactly what `stability.StabilityOracle._hessW` uses).
Caught by FD-of-∇W at one neighbour: analytic −1.12e6 vs FD −1.50e5 (ratio
1/H = 7.48). After the fix the single-point Hessian matches FD to 5.7e-5.

**BUG 2 — the ground truth itself was broken (no minimum image).** The
real-FD oracle `F0_brute` (`.tmp/brute_force.py`) displaced the lattice but
used RAW (non-wrapped) pair distances, so the "equilibrium" force was ≈ −4.68
(not 0) and the whole FD Jacobian was wrong. Fixed by minimum-image wrapping
all pair vectors. (The analytic `exact_P` already used the MI d0, so it was
the reference, not the test, that was broken.)

**Ground-truth check (`.tmp/brute_force.py`, MI-corrected):** for a real
standing wave δx_n = a·cos(k·x_n), F0_brute(a) = −m Σ_j (g_0+g_j)∇W(x_0−x_j)
with the FULL displaced densities; J = dF0/da (central FD, h=1e-6); P_fd=−J.
Result at cubic N_H=40/100, k//111 & 110, |k|d_nn=0.5–2:
  max|P_fd − P_ex| = 1.0–18.4  (FD truncation error only).  P is symmetric
  (sym_err → 0 as |k|H→0).  ⇒ **the analytic `exact_P` is the correct P
  matrix.**  The complex-step oracle (entry d) is confirmed wrong for γ=5/3.

**First (corrected) stability scan (`.tmp/analytic_p.py`, |k|d_nn 0.5–4.5,
k//111 & 110):** magnitudes now O(10²).  Gaussian is STABLE (min ω²>0 at
N_H=100,300) — matches the paper.  Cubic/quartic/quintic/W2/HOCT4 all show a
long-wavelength (111) instability (min ω² ≈ −120…−160 at (111,0.5)) across the
N_H tested.

**KEY CLARIFICATION (reference YAML):** the "55 / 67 / 190" numbers in the
PLAN/§4.1 are the **pairing-PAIR COUNTS** of the §4.1/Fig-8 nonlinear test
(pairs closer than d_nn), NOT the linear-stability N_H boundaries.  Table 2
tests the cubic at N_H=42 & 55, the quartic at 60, the quintic at 180.  The
linear-stability (Figs 4–6) N_H boundaries must be re-read from the paper.

[next] re-read paper §3.2.1 + Figs 4/5/6 for the ACTUAL linear-stability
boundaries (the N_H where ω²_k<0 first appears, per kernel) and the sound-speed
convention of Fig 6; verify `exact_P` reproduces them; then build
`stability_p_matrix.py` (with the built-in checks) + fig04/05/06 on top.

### 2026-09-19 (f) — paper eq.-23 P == exact P; cubic instability is REAL

Compared the paper's eq.-23 P matrix (the approximate density response,
u = ρ̄⁻¹Σ m sin(k·x_j)∇W → ttᵀ form; P = (2P̄/ρ̄)U + (c̄²−P̄/ρ̄)u uᵀ) to the
EXACT P (B_j double sum) and to the ground-truth FD Jacobian:

  N_H=40 k//111 |k|dnn=0.3:  paperP_eig=[-159, 471, 471]  exactP_eig=[-177, 464, 464]
  N_H=40 k//111 |k|dnn=0.5:  paperP_eig=[-123, 434, 434]  exactP_eig=[-121, 429, 429]
  max|paperP − exactP| = 2–21  (small, O(FD error)).

So **the paper's own eq.-23 P and the exact P agree** — the cubic's
long-wavelength longitudinal instability (ω²_∥ < 0 at |k|d_nn ≈ 0.3, 0.5) is
present in BOTH, and is NOT an artifact of the density-response approximation.

**Longitudinal/transverse structure (`.tmp/lon_scan2.py`, exact P):**
  cubic N_H=40  k//111 lon ω²/|k|²: 22.5(0.2) −9.8(0.3) −2.4(0.5) 3.4(0.7) 1.2(1.0) …
  cubic N_H=55  k//111 lon ω²/|k|²:  8.5(0.2) −11.0(0.3) −3.0(0.5) 2.7(0.7) …
  cubic N_H=100 k//111 lon ω²/|k|²: −3.2(0.2) −8.9(0.3) −2.8(0.5) 1.7(0.7) …
  ⇒ the cubic has a longitudinal ω²_∥ < 0 "dip" at |k|d_nn ≈ 0.3–0.6 for ALL
  N_H tested (40–100), plus a weak one near |k|d_nn ≈ 2. The transverse modes
  are positive (finite-N_H shear; → 0 as N_H → ∞).

**DISCREPANCY (open):** the paper's text (Fig 3 / §2.5) says the cubic is
"accessible for N_H ≲ 55" (stable below 55, pairing instability above). But
both the paper's eq.-23 P and the exact P (and the ground-truth force) show the
cubic ALREADY unstable (ω²_∥ < 0) at N_H = 40. The instability is at a
LONG wavelength (|k|d_nn ≈ 0.3–0.6, i.e. H|k| ≈ 0.6–1.1), whereas the paper
attributes the pairing instability to Ŵ(H|k|) < 0 at LARGE H|k|. So the
cubic's N_H=40 instability is at the wrong wavelength to be the paper's Ŵ<0
pairing instability. **UNRESOLVED** — possible that (i) the paper's Fig 4
x-axis starts at |k|d_nn ≈ 0.5–1 (hiding the long-λ dip), (ii) the "accessible
N_H ≲ 55" refers to the NONLINEAR pairing (Fig 8) not the linear ω²<0, or
(iii) a setup difference (kernel scaling / N_H convention).

**DECISION:** the exact P is the validated, ground-truth-correct linearisation
of the actual SPH force (the deliverable's P matrix). Build
`stability_p_matrix.py` + fig04/05/06 on it; plot the full (|k|, N_H)
contours so the long-λ dip is visible; document the discrepancy vs the
paper's Fig 3/4 text in the report.

### 2026-09-19 (g) — exact P matrix promoted into the oracle; deliverables built

Moved the validated exact P matrix from the `.tmp` scratch into the core
`StabilityOracle` (in `scripts/stability.py`) as `exact_p_matrix(k)`, and
re-pointed `omega2` / `sound_speed` at it (the complex-step `p_matrix` is
marked LEGACY/WRONG for γ=5/3). Also moved the ground truth into a reusable
`scripts/ground_truth.py` (`F0_brute`, `fd_jacobian`).

**BUG 3 — `_hessW` returned the wrong SHAPE (latent).** The oracle's `_hessW`
built the identity as `np.eye(3)[None, None, :, :] * np.ones(q[...,None,None].shape)`,
which broadcasts to `(1, nk, 3, 3)` (a spurious leading axis) instead of
`(nk, 3, 3)`. It was never triggered before because the complex-step oracle
only ever called `_gradW` / `_density` (never `_hessW`); the scratch
`gradW_hess` was always shape-correct. Fixed to `I3 = np.eye(3)` (broadcasts
to `(nk, 3, 3)`). This is a SHAPE bug only — the (f′/q) coefficient (entry e)
was already correct.

**Verification of the promoted `exact_p_matrix` vs ground truth
(`.tmp/verify_oracle.py`):**
  cubic_b4  N_H=40  k//111 |k|dnn=0.5: max|Pfd-Pex|=19.7  sym=0.0  eig=[-138, 445, 445]
  cubic_b4  N_H=55  k//110 |k|dnn=1.0: max|Pfd-Pex|=30.5  sym=1.2e-15  eig=[-83, 53, 457]
  hoct4     N_H=100 k//110 |k|dnn=1.0: max|Pfd-Pex|=31.6  sym=1.6e-14  eig=[-65, 69, 450]
  (max|Pfd-Pex| = the central-FD truncation error O(h^2|F'''|) at h=1e-6,
   which varies with |k|; P symmetric to ~1e-15.)  ⇒ the oracle's P matrix is
   the validated exact linearisation.  Note: the exact P uses the minimum-image
   B_j (the `.tmp` scratch used a no-MI B_j, an O(30) approximation of it — the
   MI form is the correct one and is what the oracle now uses).

**`scripts/stability_p_matrix.py` (deliverable, `--kernel`) — BUILT + works.**
Per kernel (default N_H = paper Table 2) and per (k//111,110; |k|d_nn 0.5,1,2):
  (1) P symmetric (max|P-P^T| < 1e-8) — PASS (worst ~3e-14);
  (2) equilibrium force-free (|ẍ_0(a=0)| ~ 1e-14) and rho_0 = lattice sum
      (~0.997 cubic, ~1.008 HOCT4, i.e. the discrete density, NOT exactly 1);
  (3) `--fd-check`: ground-truth real-FD Jacobian (slow, O(N^2)/k-pt).
  Run: `python stability_p_matrix.py --kernel cubic_b4 --fd-check`.

### 2026-09-19 (h) — figure scripts built (fig04/05 + fig06), all with --kernel

**`fig04_fig05_stability_contours.py` (BUILT + works, cubic in ~40 s).**
2×2 panels: top = longitudinal ω²_k/c²k² (eigenvector most aligned with k),
bottom = smallest transverse ω²_⊥2/c²k²; left k//111, right k//110.  Field over
(|k|d_nn 0.1..5, N_H 30..500 log, 14×26 grid), each point = the exact P matrix.
Red contour ω²=0 (the instability), cyan ω²/c²k²=1 (the continuum), green
0.95/0.99/1.01/1.05.  pcolormesh on a clipped RdBu_r (vmin=-1, vmax=2) so the
long-λ longitudinal dip (entry f) is visible.  Per-kernel output:
`figures/fig04_fig05_stability_contours_<kernel>.{png,pdf}`.  NOTE: the paper's
Fig 4 x-axis likely starts at |k|d_nn ≈ 0.5-1, which would HIDE the cubic's
long-λ dip — we plot the full range so the discrepancy (entry f) is visible.

**`fig06_sound_speed.py` (BUILT + works, cubic in ~5 s).**
c_SPH/c = (ω_k/|k|)/c (the longitudinal mode) vs |k|d_nn, horizontal cuts of the
top panels at N_H = 50/100/200/400 (k//111); dotted vertical line at
λ = 8h (|k|d_nn = π d_nn/(4h), h = H/kernelScale; e.g. cubic 0.704@50 →
0.352@400).  Output `figures/fig06_sound_speed_<kernel>.{png,pdf}`.

Both take `--kernel` (via `stability.parse_kernel_arg`); default = the paper's
Figs 4-6 kernels (cubic_b4 + gaussian).  The Gaussian (16-σ truncation) is
expensive (O(N²) B_j) — use a reduced `--nh-max` for it.

**Phase-5 deliverables COMPLETE:**
  - `scripts/stability.py`        — `StabilityOracle.exact_p_matrix` (the P matrix)
  - `scripts/ground_truth.py`     — `F0_brute` / `fd_jacobian` (the validation)
  - `scripts/stability_p_matrix.py` — P matrix + built-in checks, `--kernel`
  - `scripts/fig04_fig05_stability_contours.py` — stability contours, `--kernel`
  - `scripts/fig06_sound_speed.py` — sound speed, `--kernel`

[next] update PLAN.md Phase 5 STATUS; commit (NO push).  OPEN: the cubic
N_H=40 long-λ instability discrepancy vs the paper's "accessible N_H ≲ 55"
text (entry f) — document in the Phase-8 report; the figures plot the full
(|k|, N_H) range so it is visible.  LATER: `--kernel` retrofit of fig01/02/03
(pending task), Phase 6/7/8.

### 2026-09-19 (i) — FIX: `exact_p_matrix`'s phase broke translational
invariance; entries (c)/(f)'s "long-λ instability" and the fig06 discrepancy
were both this bug, not real physics

**Trigger:** fig06 (cubic_b4) looked nothing like the paper's Fig. 6 —
disconnected sawtooth segments, values swinging from 0 to >1.4 instead of a
smooth curve near 1. Rendered the paper's actual Fig. 4 at 500dpi
(`pdftoppm -r 500`) and confirmed: the cubic panel shows **no** red
(ω²≤0) region at low |k|d_nn for ANY N_H in 30–1000 — directly contradicting
entry (f)'s "UNRESOLVED" long-λ instability at N_H=40–100, which was
therefore never real physics.

**Root cause:** `exact_p_matrix` (both the Hessian term's `phase1` and the
density term's `phase2`) evaluated the plane-wave phase at each neighbour's
RAW, box-wrapped array coordinate (`exp(ik·x[list1/2])`), instead of at the
neighbour's true minimum-image-unwrapped position relative to particle 0
(`exp(ik·d0)`). For an infinite/periodic lattice the dispersion relation
ω²(k) is exactly translation-invariant — it must not depend on which
particle is treated as "particle 0". It did: relabelling the reference
particle (identical physical configuration, just renaming which array index
is "0") changed ω² by 2-3× at fixed k, and the effect was WORST for particle
0 itself, which `fccConfig` places exactly at the box corner `[0,0,0]` — so
~7/8 of its H-radius neighbours are only "close" via periodic wraparound,
and for those the raw stored coordinate is a *different* periodic image than
the one the minimum-image kernel term actually uses. Since k is O(1–70) in
these units and the box period L=1, the resulting spurious
`exp(ik·(n·L))` phase error winds through many cycles as k scans
continuously — exactly the sawtooth in fig06, and the spurious "long-λ
instability" of entries (c)/(f) (both `p_matrix`-legacy and paper-eq.-23
comparisons in entry (f) inherited the same convention, which is why they
appeared to agree with each other while both being wrong).

Entry (c)'s validation of the raw-coordinate phase against a brute-force FD
ground truth (`ground_truth.py`/`.tmp/brute_force.py`) didn't catch this
because that ground truth applies the SAME raw-coordinate cosine field to
every particle — it's internally self-consistent with the bug, not an
independent check.

**Fix** (`stability.py::exact_p_matrix`): phase now built from the same
minimum-image relative vector already used for the kernel gradient/Hessian
(`d0` for the Hessian term; `d0_list2 = MI(x[list2]-x[0])` for the density
term), instead of the raw stored coordinate. Companion fix in
`ground_truth.py::F0_brute`: the driving field is now evaluated at
`x[0] + MI(x - x[0])` for every particle (their true position relative to
particle 0), not their raw coordinate — it had the identical bug, which is
why it agreed with the old (buggy) `exact_p_matrix`.

**Validation:**
- Translation invariance restored exactly: eigenvalues bit-identical across
  6 different reference particles (corner, edge, and interior), from
  N_H=40 up to N_H=400 (2H/L up to 0.58).
- Continuum sanity: ω²/(c²k²) at |k|d_nn=0.02 (deep resolved-wave regime)
  went from an unphysical ~50–135× c² down to ~1.4× c² (cubic, N_H=40).
- Re-validated against the FIXED ground truth: max|P_fd−P_ex| = 1.8–15 (FD
  truncation error only, same order as entries e/g), vs. 200+ against the
  old (buggy) ground truth.
- `fig06_sound_speed.py` (cubic_b4): now a smooth curve, no NaN gaps, no
  sawtoothing; c_SPH/c ≈ 1.1–1.2 at small |k|d_nn falling smoothly toward 0
  at large |k|d_nn (expected — short waves unresolved at these N_H, eq. 28).
- `fig04_fig05_stability_contours.py` (cubic_b4): the low-k band across all
  N_H is now uniformly POSITIVE (0.86–1.4× c²k², checked numerically) with
  no ω²=0 contour line there — matching the paper. It still LOOKS reddish
  in the rendered PNG because `RdBu_r` maps HIGH values to red (not low —
  the `_r` reverses the base `RdBu`), and values ~1–1.4 sit on the warm side
  of the script's `vmin=-1,vmax=2` scale; this is a colour-scale choice in
  our diagnostic script, not the paper's contour-line convention, and is a
  separate (cosmetic) follow-up, not a physics bug. The genuine red ω²=0
  contour lines that remain (isolated islands at |k|d_nn≈2-4 for N_H≳200)
  look like real short-wavelength/Nyquist-adjacent pairing regions,
  qualitatively consistent with the paper's description for other kernels.

**STILL OPEN (structural, separate from the bug above):** `fig06_sound_speed.py`
plots ONE kernel at several arbitrary N_H (50/100/200/400) in one panel; the
paper's actual Fig. 6 plots ONE panel per k-direction with SEVERAL KERNELS
each at its own single characteristic N_H (cubic@42/55, quartic@60,
quintic@180 top; Wendland C²/C⁴/C⁸@100/200/400 middle; Gaussian@10/20,
HOCT4@442 bottom) — reproducing the paper's actual figure layout needs a
rewrite of `fig06_sound_speed.py`'s panel/kernel structure, not just the
oracle fix. Not done in this pass.

[next] apply the analogous relative-phase review to any other lattice-sum
code in `higherOrderSPH` that mixes raw absolute positions with
minimum-image geometry; decide whether to restructure `fig06_sound_speed.py`
to match the paper's per-kernel-N_H panel layout; update PLAN.md Phase 5
STATUS; commit.

### 2026-09-19 (j) — axis units/scale fixes (figs 4-6) + paper's Table-2
config as an additional Fig. 6 + a vectorization that made it feasible

Follow-up to entry (i): the axes didn't match the paper's, on top of the
phase bug already fixed there.

**`fig06_sound_speed.py`:** x-axis was linear; the paper's is log |k|d_nn.
Switched `KDN_GRID` to log-spaced over [0.2, 7] (paper's approximate range,
ticks at 0.5/1/5) and set `ax.set_xscale("log")`.

**`fig04_fig05_stability_contours.py`:** two unit bugs, not one:
  1. x-axis was linear -> now log, KDN_GRID log-spaced over [0.1, 6].
  2. y-axis was raw `N_H` (kernel-dependent range with no fixed meaning
     across kernels) -> the paper's actual y-axis is `h/d_nn` (h =
     H/kernelScale), LINEAR, bounds [0.9, 3], UNIVERSAL across kernels; a
     secondary axis on the right shows the corresponding N_H (kernel-
     dependent, via `ax.secondary_yaxis` with `Lattice.H_of_NH`/`NH_of_H`
     composed with the kernel's own `kernelScale`). The per-kernel N_H grid
     needed to hit h/d_nn in [0.9,3] is now derived (`hdn_grid_to_NH`), not
     hardcoded — for a kernel like the Gaussian (kernelScale=8) this
     requires much larger N_H than for a B-spline at the same h/d_nn, which
     in turn requires a bigger lattice (see below).
  Also recentred the `RdBu_r` colour scale on 1 (`vmin=0, vmax=2`, was
  `vmin=-1, vmax=2`): the old off-centre scale made the merely-above-1
  region at low k render as solid dark red, indistinguishable by eye from
  genuine instability — entry (i)'s claim that this was "just a colour-scale
  artifact, not a bug" is now also visually obvious, not just numerically
  checked.
  Re-rendering the paper's actual Fig. 4 cubic panel at 500dpi and comparing
  by eye: the green good-value loop and the red instability island (now
  correctly a small, isolated closed contour around |k|d_nn~1.5-3,
  h/d_nn~2-2.5, not a broad low-k region) both qualitatively match.

**New `fig06_paper_config.py`:** the structural gap flagged at the end of
entry (i) — the paper's Fig. 6 is 3 panels by k-direction
((1,0,0),(1,1,0),(1,1,1)), each overlaying ALL 10 Table-2 (kernel, N_H)
rows, legend split 4+3+3 across panels (confirmed by rendering the paper's
Fig. 6 at 500dpi: every curve appears in every panel). Implemented as a
companion to (not a replacement for) `fig06_sound_speed.py`. Result:
qualitatively matches the paper closely — curves flat-ish just above 1 for
|k|d_nn<1, a broad dip to a minimum around |k|d_nn~4, a slight rebound, and
convergence near |k|d_nn~6-7, with the same kernel ordering (cubic N_H=42
highest, HOCT4 lowest) as the real figure.

**Gaussian N_h=10/20 (N_H=5120/10240) omitted, and why (found the hard
way):** tried to include them and the machine hit ~54GB RSS + swapped
before being killed. Root cause: `_build_neighbors` builds DENSE (n1, n2, 3)
arrays (n1~N_H, n2~8N_H) once per (kernel,N_H) regardless of how many are
actually within cutoff; at N_H=10240 that's ~5e8 elements (~40GB as
complex128) — infeasible on this machine, and this is independent of
anything fixed in entry (i) (it was already true of the pre-fix code). Left
out of `fig06_paper_config.py` by default (`--include-gaussian` forces it,
with a warning); would need the oracle's O(N_H^2) density-response term
rewritten to be sparse/chunked to fix properly. Not attempted here.

**Vectorized `exact_p_matrix`'s density-response term** (`stability.py`):
replaced the explicit Python loop over `keep_idx` (one iteration per list1
neighbour, each doing a masked numpy reduction) with two `np.einsum` calls
over the already-dense `(n1, n2, 3)` arrays built in `_build_neighbors`
(the mask was redundant: grad W is exactly 0 past the compact-support
radius via the kernel's own "zero piece", so summing over the full dense
array gives the identical result). This does NOT change the memory
footprint (that array was already being built densely) — it only removes
per-iteration Python-interpreter overhead. Validated bit-identical output
to the pre-vectorization loop version, and re-checked against the FD ground
truth for wendland_C6/hoct4/quintic_b6 (P symmetric to ~1e-13, FD agreement
within truncation error, same order as entries e/g). ~15x faster in
practice (N_H=100: 2.6ms/call, was similar order before but this made the
8-series x 3-direction x 40-point `fig06_paper_config.py` run in ~27s
instead of being impractically slow).

[next] `fig06_sound_speed.py`'s per-kernel-multi-N_H layout is still a
different (complementary) diagnostic from the paper's own Fig. 6, kept as
is by design. If the Gaussian Table-2 rows are ever needed, rewrite the
density-response term to avoid the dense (n1,n2,3) array (chunk over list1,
or use a spatial cutoff/tree instead of the current "all of list2" dense
broadcast) before attempting N_H>~2000.

### 2026-09-19 (k) — SECOND, independent bug in `exact_p_matrix`: the
density-response term had a spurious extra factor of K and the wrong sign

Trigger: entry (j)'s `fig06_paper_config.py` looked structurally right but
every curve started at c_SPH/c ~ 1.1-1.2 at the smallest plotted k, where
the paper's own curves start "exceptionally close to 1" (paper_notes.md:
"cubic N_H=42/55: errors of a few percent"; the rendered Fig. 6 tick labels
read 0.982-1.025 at the curves' left edge). A 10-20% small-k offset is an
order of magnitude too big to be discreteness noise.

**Root cause, found by isolating terms:** entry (i) validated `exact_p_matrix`
against `ground_truth.fd_jacobian` and saw `max|P_fd-P_ex|` of order 1-30,
called it "FD truncation error" and moved on -- WITHOUT checking that it
shrinks as the FD step h shrinks. It doesn't: re-run at h = 1e-3 .. 1e-7,
`max|P_fd-P_ex|` is IDENTICAL to 5 significant figures at every step size.
That means it was never truncation error -- entry (i)'s validation had a
false negative, and this bug survived that check.

Isolated by setting `gamma=2` (which makes the density-response term's own
`(gamma-2)` prefactor exactly 0, i.e. Hessian-term-only): `exact_p_matrix`
and the FD Jacobian then agree to 3e-9 -- so `M_hess` is exactly right, and
100% of the discrepancy is in `M_rho`. Grid-searching sign x
{with,without the extra K} against FD at several k confirmed the fix
uniquely: `M_rho *= -1 * m * Bbar * (g-2)/rho0` (was
`+1 * m * K * Bbar * (g-2)/rho0`) reproduces FD to true O(h^2) truncation
error (0.01-0.2, scaling with k, down from the flat 1-30 that never
shrank) -- neither the sign flip nor the K removal alone fixed it, both
were needed simultaneously.

Both bugs trace to the same mis-copied chain-rule step:
d(P_j/rho_j^2)/drho_j = K(gamma-2)rho_j^(gamma-3) = Bbar(gamma-2)/rho0 (at
rho0=1) -- ONE power of K, matching the Hessian term's own bare `Bbar`
prefactor (which is itself `K rho0^(gamma-2)`, so already has its one K).
The old code effectively multiplied by `K * Bbar` = `K^2 rho0^(gamma-2)`,
double-counting K, and had the wrong sign on top.

**Validated broadly** (re-run `stability_p_matrix.py --fd-check` and a
direct sweep): cubic_b4/wendland_C6/hoct4/quintic_b6, both k-directions,
|k|d_nn 0.3-2.0, N_H 40-442 -- `max|Pfd-Pex|` now 0.0001-0.2 (shrinks with
FD step, genuine truncation error), P symmetric to ~1e-13. This was
UNRELATED to entry (i)'s phase-reference fix (that one's translation-
invariance argument and its own numeric evidence both still hold and are
unaffected); it's a second, independent bug in the same function, caught
because this session actually looked at the SMALL-k behaviour instead of
only the moderate/large-k regime entry (i) happened to check.

**Consequence for every figure already produced (i)/(j):** the eigen
DIRECTIONS were mostly right (`M_rho` is small compared to `M_hess` except
near k -> 0 for the transverse-degenerate-with-longitudinal cases, which is
why the transverse mode matched FD all along and only the longitudinal
mode was ever visibly wrong), but every LONGITUDINAL number is corrected by
this fix. Re-ran and re-saved: `fig06_sound_speed.py` (cubic_b4, curves now
start at ~0.94-1.07 at |k|d_nn=0.2, matching "a few percent" from the
paper), `fig06_paper_config.py` (all 8 series now start within ~1-7% of 1,
matching the paper's tick labels), `fig04_fig05_stability_contours.py`
(cubic_b4 -- re-checked the apparent pale-orange band at h/d_nn~1.0-1.3 by
hand: it is NOT a red ω^2<=0 contour, just background colour for values
~1.0-1.2 with tightly-packed green/cyan contour lines nearby; a direct
numeric sweep of the exact plotted grid confirms every point at
h/d_nn<1.7 is positive -- the only genuine instability island is the
already-identified one at h/d_nn~2-2.9, |k|d_nn~2-5, matching the real
paper's Fig. 4 cubic panel).

[next] the phase-5 deliverables (`stability.py`, `stability_p_matrix.py`,
`fig04_fig05_stability_contours.py`, `fig06_sound_speed.py`,
`fig06_paper_config.py`) should now be considered validated against the
paper at the level of "curves start within a few percent of 1 and decay
with the right shape/ordering"; a natural next check (not done here) would
be reproducing the paper's quoted N_H pairing thresholds (cubic gradual
beyond ~55, quartic ~67, quintic ~190, Wendland stable to 700) directly
from `fig04_fig05`'s red contour rather than eyeballing it.




