# Dehnen & Aly 2012 — extracted paper content

- **Title:** Improving convergence in smoothed particle hydrodynamics
  simulations without pairing instability
- **Authors:** Walter Dehnen & Hossam Aly (University of Leicester)
- **Venue:** MNRAS 425(2), 1068–1082 (2012); arXiv:1204.2471v2
- **PDF:** `/home/lu26029/dev/warpSPH/literature/dehnen2012_convergence-without-pairing-instability.pdf`
- **Extracted:** 2026-09-17, via `pdftotext` (two passes: `-layout` and raw).
  Where the text layer garbled a table (Table 1's `C` column), the value is
  taken from the `warpSPHCore` transcription in
  `src/warpSPHCore/kernels/kernelFunctions/*.py` (comments there cite this
  paper) and is marked *(code)* — Phase 1 of `PLAN.md` verifies every value
  independently by numerical integration.

## One-paragraph summary

SPH force errors from unavoidable particle disorder ("E0 errors", worst in
shear) degrade convergence below O(h²). Raising the neighbour number N_H
reduces E0 errors, but conventional B-spline kernels become unstable to the
pairing (clumping) instability at large N_H. The paper shows (linear
stability analysis + 3D test simulations) that a *necessary* condition for
pairing stability at large N_H is a non-negative kernel Fourier transform,
that the Wendland functions satisfy this (and are stable for all N_H), and
that Wendland kernels with large N_H give the best convergence in a strong
shear test and shock test. It also gives a simple density-estimate
correction (subtract a fraction ε(N_H) of the self-contribution) that fixes
the low-N_H over-estimation of the Wendland kernels without touching
stability, and argues that the conventional B-spline scaling h̃ = 2H/n is
inappropriate — the resolution scale should be h = 2σ.

## Definitions used throughout the paper

| Symbol | Meaning | Paper ref |
|---|---|---|
| `W(x, h) = H^{-ν} w(|x|/H)` | isotropic kernel; `w(r) = 0` for `r ≥ 1`, support radius `H` | eq. (6) |
| `σ² = ν^{-1} ∫ x² W(x) d^νx` | kernel standard deviation | eq. (8) |
| `h = 2σ` | **the** smoothing/resolution scale used in this paper | eq. (10) |
| `N_H = V_ν H^ν (ρ̂_i / m_i)` | average number of neighbours within the support sphere | eq. (7) |
| `N_h ≡ (h/H)^ν N_H` | neighbours within distance `h` | §2.1 |
| `h̃ = 2H/n` | conventional Monaghan–Lattanzio B-spline scale (paper argues it is not a resolution scale) | §2.2.1 |
| `d_nn` | nearest-neighbour distance of the unperturbed (densest-sphere) packing | §3.2 |
| `w̄(k) = F_ν[w]` | Fourier transform; for spherical kernels depends only on `H|k|`; eq. (14) is the 3D form `4πκ^{-1} ∫₀^∞ sin(κr) w(r) r dr` | §2.4 |

### ⚠ Notation clash with `warpSPHCore`

`warpSPHCore`'s `sphKernel_(x, h, kernel)` (in
`src/warpSPHCore/kernels/kernel.py`) computes `q = |x|/h` and cuts off at
`q > 1`: **the `h` of the code's kernel functions is the paper's support
radius `H`, not the paper's `h = 2σ`.** Conversion used in all replications:

```
h_paper = h_code / kernelScale(dim)        # kernelScale = H/h_paper, Table 1
N_H     = V_ν (h_code · d_nn)^ν            # == sphKernelN_H
```

The code's `packingRatio()` is the paper's Table-2 quantity
`h_paper (ρ̂/m)^{1/3}` at the kernel's chosen N_H, and
`sphKernel_xi = packingRatio · kernelScale = H·d_nn` (neighbour-search
cutoff in units of d_nn).

## Table 1 — kernel functions and scale quantities

`w(r) = C·b_n(r)` (B-splines, eq. 11) or `w(r) = C·ψ_ℓk(r)` (Wendland,
eq. 12) with `ψ_ℓk(r) = I_k (1−r)_+^ℓ`, `I[f](r) = ∫_r^∞ s f(s) ds`,
`ℓ = k + 1 + ⌊ν/2⌋`; `(·)_+ = max{0, ·}`.

| Kernel | Function w(r) (before C) | C (ν=1, 2, 3) | σ²/H² (ν=1, 2, 3) | H/h (ν=1, 2, 3) |
|---|---|---|---|---|
| cubic spline b₄ | `(1−r)³₊ − 4(½−r)³₊` | 8/3, 80/(7π), 16/π *(code)* | 1/12, 31/392, 3/40 | 1.732051, 1.778002, 1.825742 |
| quartic spline b₅ | `(1−r)⁴₊ − 5(⅗−r)⁴₊ + 10(⅕−r)⁴₊` | 3125/768, 46875/(2398π), 15625/(512π) *(code)* | 1/15, 9759/152600, 23/375 | 1.936492, 1.977173, 2.018932 |
| quintic spline b₆ | `(1−r)⁵₊ − 6(⅔−r)⁵₊ + 15(⅓−r)⁵₊` | 243/40, 15309/(478π), 2187/(40π) *(code)* | 1/18, 2771/51624, 7/135 | 2.121321, 2.158131, 2.195775 |
| Wendland C², ν=1 | ψ₂,₁ = `(1−r)³₊(1+3r)` | 5/4 *(code)* | 2/21 | 1.620185 |
| Wendland C⁴, ν=1 | ψ₃,₂ = `(1−r)⁵₊(1+5r+8r²)` | 3/2 *(code)* | (1/15 — shares H/h with quartic 1D) | 1.936492 |
| Wendland C⁶, ν=1 | ψ₄,₃ = `(1−r)⁷₊(1+7r+19r²+21r³)` | 55/32 *(code)* | (verify) | 2.207940 |
| Wendland C², ν=2,3 | ψ₃,₁ = `(1−r)⁴₊(1+4r)` | —, 7/π, 21/(2π) *(code)* | —, 5/72, 1/15 | —, 1.897367, 1.936492 |
| Wendland C⁴, ν=2,3 | ψ₄,₂ = `(1−r)⁶₊(1+6r+35/3 r²)` | —, 9/π, 495/(32π) *(code)* | —, 7/132, 2/39 | —, 2.171239, 2.207940 |
| Wendland C⁶, ν=2,3 | ψ₅,₃ = `(1−r)⁸₊(1+8r+25r²+32r³)` | —, 78/(7π), 1365/(64π) *(code)* | —, 3/70, 1/24 | —, 2.415230, 2.449490 |
| HOCT4 (Read et al. 2010) | not tabulated; central spike, no inflection point; σ ≈ 0.228343 H | —, —, 6.5150499306 *(verified)* | —, —, 0.0521407118 *(verified)* | —, —, 2.189684 *(verified)* |
| Gaussian | `N(0, σ²)` (truncated at 16σ: H = 16σ) | — | 1/256 | 8.0 |

Consistency identity (used to cross-check the transcription):
`H/h = 1/(2√(σ²/H²))`. Holds for every row above.

Notes:
- The 1D Wendland C² kernel ψ₂,₁ is the kernel of Lucy (1977); the code's
  `wendland2_k` uses ψ₂,₁ for `dim==1` and ψ₃,₁ otherwise — matching the
  paper (which lists the 1D variants separately and notes ψ₂,₁ "for 3D
  simulations, when it is not a Wendland function").
- B-spline pattern in 1D: σ²/H² = 1/(3n) for n = 4, 5, 6 — and in fact
  for ALL n in 1D (the paper states b_n → N(0, H²/3n); the pattern is
  exact, verified n = 4…8). The eq.-11 family in support-1 units has the
  closed form b_n(q) = Σᵢ₌₀^{⌊(n−1)/2⌋} (−1)ⁱC(n,i)((n−2i)/n − q)₊^{n−1},
  which reproduces the shipped cubic/quartic/quintic to machine precision.
- The former code `B7` is **not in the paper's Table 1 but IS a family
  member: exactly b₈** (order 8, degree 7 — the old code name is by
  degree, the paper indexes by order): b₈(q) = (1−q)⁷₊ − 8(¾−q)₊⁷
  + 28(½−q)₊⁷ − 56(¼−q)₊⁷, matched the old code shape to 8.3e-17
  (2026-09-18). Its σ²/H² = 1/24, 0.0406136, 19/480 (1D exactly
  1/(3·8)); C (4096/315, 589824/(7435π), 16384/(105π) *(code)*)
  normalise it correctly. It had reused the quintic row's kernelScale
  (2.121321, 2.158131, 2.195775) instead of the shape-derived
  2.449490/2.481044/2.513123. **Resolved 2026-09-18 (user-directed):**
  the shape-derived scale was adopted and the kernel renamed **B8**
  (enum 33 — the old value, so stored configs keep resolving to the
  same shape). The **genuine classical b₇** (order 7, degree 6:
  (1−q)⁶₊ − 7(5/7−q)₊⁶ + 21(3/7−q)₊⁶ − 35(1/7−q)₊⁶, C = 823543/92160,
  5764801/(113149π), 5764801/(61440π), σ²/H² = 1/21,
  7691281/166329030, 11/245, H/h = 2.291288/2.325170/2.359700, 3D FT
  min −3.2e-6, first zero κ̂ ≈ 21.96 → barely pairing-unstable) was
  derived from the family formula and added as the new **B7** (enum 34)
  — a different spline from the b₈ shape: max |diff| 5.3e-2.

## Table 2 — kernel-N_H combinations used in Fig. 6 and the §4 tests

| Kernel | N_H | N_h | h/d_nn | h(ρ̂/m)^{1/3} |
|---|---|---|---|---|
| cubic spline | 42 | 6.90 | 1.052 | 1.181 |
| cubic spline | 55 | 9.04 | 1.151 | 1.292 |
| quartic spline | 60 | 7.29 | 1.072 | 1.203 |
| quintic spline | 180 | 17.00 | 1.421 | 1.595 |
| Wendland C² | 100 | 13.77 | 1.325 | 1.487 |
| Wendland C⁴ | 200 | 18.58 | 1.464 | 1.643 |
| Wendland C⁶ | 400 | 27.22 | 1.662 | 1.866 |
| HOCT4 | 442 | 42.10 | 1.923 | 2.158 |
| Gaussian | ∞ | 10.00 | 1.191 | 1.337 |
| Gaussian | ∞ | 20.00 | 1.500 | 1.684 |

`d_nn` for densest-sphere packing with number density
`n = 2^{1/2} d_nn^{-3}` (FCC). Code cross-check:
`packingRatio()` values are exactly the last column for Wendland2 (1.487),
Wendland4 (1.643), Wendland6 (1.866), QuarticSpline (1.203), QuinticSpline
(1.595 × 1.1425 "CRKSPH" factor), CubicSpline (1.292 × 1.0175 Price-2012
factor; the paper value is 1.292). N_h and h/d_nn are derived quantities:
`N_h = (h/H)^ν N_H`, `h/d_nn = packingRatio`.

## Verified: HOCT4 kernel (Read, Hayfield & Agertz 2010, eqs. 46–51)

`W(r) = N/H³ f(x)`, `x = r/H`, `nk = 4`, `β = 0.5`, `γ = 0.75`, with exact
rationals `A = 16/5 = 3.2`, `B = −94/5 = −18.8`:

```
f(x) = P x + Q                              0 ≤ x ≤ α
f(x) = (1−x)⁴ + A(γ−x)⁴ + B(β−x)⁴          α < x ≤ β
f(x) = (1−x)⁴ + A(γ−x)⁴                     β < x ≤ γ
f(x) = (1−x)⁴                               γ < x ≤ 1
f(x) = 0                                    x > 1
```

- `α = 0.214108111463` — smaller root of `(1−a)² + A(γ−a)² + B(β−a)² = 0`
  (eq. 51; the other root 0.6078096968 falls outside the piece domain).
- `P = −2.154228492776` (eq. 49), `Q = 0.981018558675` (eq. 50).
- Normalisation: `N₁ = 2.0684349363`, `N₂ = 3.7158334191`,
  `N₃ = 6.5150499306` (paper Table 1: 6.52 ✓).
- σ²/H² (eq.-8 convention): 1D 0.0505294124, 2D 0.0520086521,
  3D 0.0521407118 → `H/h₃ = 2.189684410`. **D&A's quoted σ ≈ 0.228343 H is
  exactly √0.0521407118 = 0.228343408** — the earlier "discrepancy" was
  the missing 1/ν factor of eq. 8 (full second moment is 3× that:
  0.1564221355 = 0.3955024²).
- D&A's Table 2 HOCT4 row is fully reproduced by these values:
  `N_h = 442/2.189684410³ = 42.0996` (paper 42.10), `h/d_nn = 1.92261`
  (paper 1.923), `h(ρ̂/m)^{1/3} = 2.15806` (paper 2.158).
- Structure (verified numerically): C²-continuous at α, β, γ; monotonically
  decreasing; central cusp `f(0) = Q = 0.9810`, `f'(0) = P ≠ 0` — the
  "constant central core to the kernel gradient" (triangular central spike)
  is by design. `f(α) = 0.5197807644`, `f(β) = 0.075`,
  `f(γ) = 0.00390625`, `N₃ f(0) = 6.3913848926`.
- Method cross-check: the CT kernel of the same paper (C², linear central
  core) was derived from its C² constraints — the printed core is garbled
  in the typeset PDF, but the C²-continuous reading (linear core
  `11/9 − 2x`, α = 1/3 forced by C²) normalises to the paper's own
  `N = 8/[π(6.4α⁵ − 16α⁶ + 1)]` with ratio 1.000000.
- **Core-library parameters (Phase 3):** `kernelScale = 2.189684410` (3D,
  NOT the 1.264215 from the |x|² misreading), `C_d = N₃ = 6.5150499306`,
  `packingRatio` from the densest-sphere convention.

## R&H 2012 SPHS — transcribed (read2012, MNRAS 422(4), 3037)

Full equation transcription (switch eq. 21, relaxation eqs. 22–25, 10×10
moment-matrix gradient estimator §4.2, viscosity eqs. 29–31, Balsara
eq. 32, entropy dissipation eq. 33–35, mass dissipation eqs. 36/39–40,
parameters ns = 0.05, α_max = 1, α_min = 0.2, Balsara 10⁻⁴) is in
`data/da2012_reference.yaml` (`rnh2012_sphs`). Key confirmations from the
PDF layout: eq. 21's denominator carries the `h_i²|∇(∇·v_i)|` term; eq. 33
carries the `(ρ_j/ρ_i)^{γ−1}` factor (bbox-verified). R&H 2012's own Sod
setup (used for the Phase-6 conductivity validation): [−0.5, 0.5],
(ρ,P,v) L = (1.0, 1.0, 0), R = (0.125, 0.1, 0), γ = 5/3, 3D,
32×32×400 + 16×16×200 lattice, N₁D = 600, t = 0.2, initial α = 1 on
−0.05 < x < 0.05.

## Key equations

| Eq. | Content |
|---|---|
| (1) | density estimator `ρ̂_i = Σ_j m_j W(x_i − x_j, h_i)` |
| (3) | conservative SPH acceleration (Nelson & Papaloizou 1994 form, with ℋ factors) |
| (7) | `N_H = V_ν H^ν (ρ̂_i/m_i)` |
| (10) | `h = 2σ` |
| (14) | 3D Fourier transform `w̄(κ) = 4πκ^{-1} ∫₀^∞ sin(κr) w(r) r dr` |
| (15) | B-spline FT (closed form, includes normalisation): `F₃[b_n](κ) = (3/nκ)^{n+2} [sin(nκ) − nκ cot(nκ)]^{n/2}` (transcription garbled — use numerical FT as primary, closed form as cross-check) |
| (16) | Wendland FT via `F₃[ψ_ℓk](κ) = −κ^{-(k+1)} d/dκ F₁[(1−r)_+^ℓ](κ)` |
| (17) | small-k Taylor: `w̄(k) = 1 − ½σ²k² + O(|k|⁴)` (all kernels, after common h scaling) |
| (18) | corrected density: `ρ̂_i,corr = ρ̂_i − ε m_i W(0, h_i)` |
| (19) | `ε = ε₁₀₀ (N_H/100)^{−α}` with (ε₁₀₀, α) = (0.0294, 0.977) Wendland C²; (0.01342, 1.579) C⁴; (0.0116, 2.236) C⁶ — all ν=3 |
| (21) | dispersion relation `a·P(k) = ω²a` |
| (23) | P-matrix for conservative SPH, general EOS, any ν (see Appendix A) |
| (24a,b) | `t(k) = ρ̂^{-1} Σ m sin(k·x⃗_j) ∇W(x⃗_j, h̄)`, `T(k) = ρ̂^{-1} Σ m (1−cos(k·x⃗_j)) ∇⁽²⁾W(x⃗_j, h̄)` |
| (25, 26) | resolved-wave limit (`|k|h ≪ 1`, isotropic): `ω²_∥ = c̄²k² + (2P̄/ρ̄)·{3ν/(ν+2)}·Ξ̄...` — longitudinal/transverse eigenvalues; error quality set by the density estimate |
| (28) | continuum limit (`h ≫ d_nn`): `P → c²k⁽²⁾w̄ + (2P/ρc²)(1 − w̄ − ν^{-1}k·∇_k w̄)` — instability iff `w̄ < 0` |
| (29) | combined limit: `ω²_∥ = c²k²[1 + σ²k²(γ−1){2/ν} + 1 − γ]`; 3D adaptive h → bracket term `5/3 − γ` (vanishes for γ = 5/3) |
| (30) | `r_min,i = min_{j≠i} |x_i − x_j|/H_i` |
| (31) | `r_min ≤ (2^{5/2}π/(3 N_H))^{1/3}` (upper bound at densest packing; text garbled, re-derived) |
| (32) | `q_min,i = r_min,i / (2^{5/2}π/(3N_H))^{1/3} ≈ min |x_i − x_j|/d_nn,grid` — ≈1 densest packing, ∼0.7 glass, ≈0 pairing |
| (33) | Gresho–Chan vortex IC, see below |

### Gresho–Chan vortex initial conditions (eq. 33), P₀ = 5

With R the cylindrical radius:

```
P(R)    = P₀ + 12.5 R²                         0 ≤ R < 0.2
P(R)    = P₀ + 12.5 R² + 4 − 20R + 4 ln(5R)    0.2 ≤ R < 0.4
P(R)    = P₀ + 2(2 ln 2 − 1)                   0.4 ≤ R

v_φ(R)  = 5R                                    0 ≤ R < 0.2
v_φ(R)  = 2 − 5R                                0.2 ≤ R < 0.4
v_φ(R)  = 0                                     0.4 ≤ R
```

Centrifugal-balance check (re-derived 2026-09-17): `dP/dR = v_φ²/R` in both
non-zero branches (branch 2: `25R² + 4 − 20 + 4/R`... explicitly
`dP/dR = 25R − 20 + 4/R = (2−5R)²/R` ✓), and both P branches are continuous
at R = 0.2 (P₀ + 0.5) and R = 0.4 (P₀ + 4 ln 2 − 2 = P₀ + 2(2 ln 2 − 1) ✓).
Use this as a transcription sanity check in the replication.

## Figure inventory — what each figure evaluates

| Fig. | Content | Evaluates | Replication approach |
|---|---|---|---|
| 1 | Kernel shapes w(r) for Table 1 + Gaussian + HOCT4, scaled to common h = 2σ, ν = 3 (top linear w/ support arrows, bottom log) | visual kernel comparison: support, central value, small-scale emphasis | **done 2026-09-18** — `scripts/fig01_kernel_shapes.py` (checks: non-negativity, support, normalisation) |
| 2 | Fourier transforms `w̄(k)` of the same set, negative parts as broken curves; **x-axis is `|k|σ` (σ = h/2), log `|w̄|` axis 1e-6–1, range [0, 3π]** | the non-negativity condition for pairing stability | **done 2026-09-18** — `scripts/fig02_fourier_transforms.py`: numerical 3D FT (eq. 14, primary) + derived piecewise closed form as cross-check (eq. 15 garbled — see PLAN.md findings); eq. 17 Taylor a₂ = −1/8 verified for all 8; first zeros recorded: cubic 3.4414 (just over π, as in the paper), quartic 5.5641, quintic 4.2881 in `|k|σ` units (= 12.5663 ≈ 4π, 22.4670, 18.8314 in H units) |
| 3 | Density estimate vs N_H for densest-sphere packing (solid), glass (squares), pairing (crosses); dashed = corrected estimate (eq. 18) for Wendland | density-estimator bias per kernel vs N_H; basis for the ε(N_H) correction | static particle configurations (FCC lattice; Poisson-random as glass proxy; paired = each point split into a close pair); fit ε(N_H) and compare to eq. 19 constants |
| 4 | Stability contours, Gaussian (truncated at 16σ) + cubic spline: `ω²_∥/c²k²` (top), `ω²_⊥2/c²k²` (bottom) over (`|k|·d_nn`, `h/d_nn`), two wave directions k ∝ (1,1,1) and (1,1,0) | pairing-stability regions + sound-speed accuracy (P ∝ ρ^{5/3}, FCC equilibrium) | implement P-matrix (eq. 23, 24) on FCC lattice, eigenvalues; pure numerics |
| 5 | As Fig. 4 for quartic/quintic spline + HOCT4 (left) and Wendland C²–C⁶ (right) | same; isolates the small-N_H instability islands (quintic N_H ≈ 100, Wendland C² N_H ≈ 40) | same machinery |
| 6 | `c_SPH/c = ω_∥/|k|` vs `|k|` for the 10 Table-2 kernel-N_H combinations, three wave directions, vertical line at λ = 8h | numerical resolution of sound waves per kernel-N_H | horizontal cuts through Fig. 4/5 machinery |
| 7 | final x,y positions (|z| < d_nn/2) of §4.1 tests; symbol size ∝ z; overlap = pairing | qualitative glass vs pairing outcome | output of the §4.1 relaxation run |
| 8 | final `min_i q_min,i` (eq. 32) vs N_H, all kernels (Wendland with density correction) | **pairing in practice**: cubic pairs gradually beyond N_H > 55; quartic > 67; quintic > 190; Wendland no pairing up to N_H = 700 | full 3D SPH dynamics: 32 000 particles, noisy FCC IC (1D Gaussian offset σ = d_nn), periodic, P = Kρ^{5/3}, conservative SPH + ℋ, Cullen & Dehnen (2010) viscosity switch, Read & Hayfield (2012) conductivity, viscous heating suppressed, evolve to glass equilibrium |
| 9 | Gresho–Chan v_φ at t = 1, lowest resolution N₁D = 51, four kernel-N_H combos (incl. a P₀ = 0 run) | E0-error noise in strong shear at fixed resolution | full dynamics; IC from eq. 33; densest-sphere packing; N₁D = 51, 102, 203, 406 |
| 10 | L₁ velocity error vs N₁D for all kernels (each at its best stable N_H) | convergence in strong shear; cubic slope degraded by too-low v_φ at R < 0.2; B-splines saturate for N_H ≳ 200; Wendland C⁶ N_H = 400 best | same runs; L₁ over all particles |
| 11 | Sod shock tube, 3D glass-like ICs, single N, six kernel-N_H combos: v, ρ, u-thermal at t = 0.2 vs exact solution | E0-error noise in post-shock region; shock/contact smoothing trade-off vs N_H; L₁ velocity error on −0.4 < x < 0.5 | full dynamics; Riemann data not stated in body text — verify from Fig. 11 caption/PDF page; standard Sod (ρ_L = 1, u_L = 0, P_L = 1; ρ_R = 0.125, u_R = 0, P_R = 0.1) assumed until confirmed |
| 12 | wall-clock time per SPH step, N = 220 (×10³) particles, vs N_H, 4 processors | sub-linear cost growth in N_H (data-dominated regime) | timing study on our hardware; report *relative* scaling only (machine differs from ALICE) |
| 13 | Fig. 10 with computational cost on x-axis | accuracy-per-cost; highest N_H makes optimal use of resources | replot of 10 + 12 |

## Key results to reproduce (acceptance anchors)

1. **Stability (Figs 4–6, 8):** B-splines unstable at large N_H; quartic
   pairs beyond N_H ≈ 67, quintic beyond ≈ 190, cubic degrades beyond
   ≈ 55 (gradual). Wendland kernels: no pairing up to N_H = 700. Small-N_H
   instability islands exist for quintic (≈100) and Wendland C² (≈40) in
   the *linear* analysis but did not trigger in the §4.1 runs.
2. **Sound speed (Fig. 6):** cubic spline N_H = 42/55: errors of a few
   percent; quartic N_H = 60: < 1% and resolves shorter waves than cubic
   N_H = 55; λ = 8h resolved to ≲ 1%.
3. **Density (Fig. 3):** cubic under-estimates; Wendland (and Gaussian)
   over-estimate at low N_H; HOCT4 worst (generic for spike kernels);
   corrected estimate (eq. 18–19) lands on ≈ 1.
4. **Shear test (Figs 9–10):** Wendland C⁶ N_H = 400 best convergence;
   quintic N_H = 180 ≈ Wendland C⁴ N_H = 200; B-splines saturate for
   N_H ≳ 200 (loss of resolution balances E0 reduction); cost of jumping
   quintic-180 → Wendland C⁶-400 ≈ factor 1.5.
5. **Shock (Fig. 11):** cubic N_H = 55 shows strong post-shock velocity
   noise (particle re-ordering after anisotropic compression); noise
   decreases with N_H but L₁ velocity error does not approach zero (shock
   smoothing); energy overshoot at the contact prevented by the
   (deliberately over-smoothing) conductivity.
6. **Cost (Fig. 12):** sub-linear in N_H below ≈ 400.

## Appendix A — linear stability derivation (structure)

- A1 (fixed h): second-order density correction from the Lagrangian;
  `t(k)` (eq. A7), `T(k)` (eq. A10); result eq. (A11).
- A2 (adaptive h, h^νρ̂ = const): η₁, η₂ corrections (eqs. A14–A15),
  `ℋ̄ ≈ 1`, `Π̄ ≈ 1`, `Ξ̄ ≡ Π̄/ℋ̄ − 1 ≈ 0` (eq. A17); result eq. (A16) →
  the P-matrix of eq. (23).
- A3 (k → 0 limit): `t → −ℋ̄ρ̄k`, `T → (2ℋ̄+νΠ̄)ρ̄/(ν+2) k⁽²⁾ + ν(Π̄−ℋ̄)ρ̄/(2(ν+2)) |k|² I`
  (eq. A19) → eqs. (25)–(26).

## Open items found during extraction

- **HOCT4 source paper** (Read, Hayfield & Agertz 2010, *Resolving mixing
  in Smoothed Particle Hydrodynamics*, MNRAS 405, 1513; arXiv:0906.0774):
  **RESOLVED 2026-09-17** — synced into the literature collection as
  `read2010_resolving-mixing-sph.pdf` (checker green). Functional form
  extracted from eqs. 46–51 and verified numerically (see
  *Verified: HOCT4* below). The synced typeset reprint is identical to
  arXiv v2; arXiv v1 is an earlier draft (different equation numbering,
  "SPHS" naming) with the same kernel structure — no conflicting version.
- **Sod Riemann data**: not explicit in the body text. **RESOLVED
  2026-09-17 (working assumption, confirm on first Phase-7 run):** the
  `warpSPH` reference solver problem (ρ, p, u) = (1, 1, 0) /
  (0.25, 0.1795, 0) is NOT consistent with Fig. 11's quoted discontinuity
  positions — its exact solution puts the contact at x = 0.1228 and the
  shock at x = 0.3155 at t = 0.2, far from the paper's "≈ 0.17" and
  "≈ 0.378". The **R&H 2012 Sod problem** (ρ, p, u) = (1, 1, 0) /
  (0.125, 0.1, 0), γ = 5/3, matches: contact 0.168239 (paper ≈ 0.17),
  shock 0.368895 (paper ≈ 0.378, within SPH smearing of the velocity
  jump). D&A's "artificial conductivity similar to Read & Hayfield
  (2012)" makes adopting their exact test plausible. Both exact solutions
  are recorded in `data/da2012_reference.yaml`
  (`sod.working_ic` / `sod.alternative_ic_warpSPH_sodND`).
- **Read & Hayfield (2012)** conductivity paper: **RESOLVED 2026-09-17** —
  synced into the collection as `read2012_sphs-higher-order-dissipation-
  switch.pdf` (MNRAS 422(4), 3037). The full SPHS scheme is transcribed in
  `data/da2012_reference.yaml` (`rnh2012_sphs`) — supersedes D&A's §4.3
  description as the spec for the conductivity module.
- Table 1 `C` column garbled by the PDF text layer → code transcription
  used; **VERIFIED 2026-09-17** by `scripts/common.py` smoke test: C,
  σ²/H² (eq.-8 convention) and H/h all match to ≤ 1.4e-17 for cubic,
  quartic, quintic and Wendland C²/C⁴/C⁶ (dim 3).
- Eq. (15) B-spline closed-form FT garbled → use numerical FT as primary.
- Fig. 3 "glass" configurations: source not stated (presumably relaxed
  random configurations); Poisson random is an acceptable static proxy, to
  be noted in the replication.
- N_H grid used in Fig. 8 (spacing, max = 700) and the §4.1 box size /
  runtime: not stated; choose practical values and record them.
- Eq. (31) bound garbled; re-derived as `r_min ≤ (2^{5/2}π/(3N_H))^{1/3}`
  from eq. (7) + FCC density `n = 2^{1/2} d_nn^{-3}` — consistent with
  `q_min ≈ min|x_i − x_j|/d_nn`.
