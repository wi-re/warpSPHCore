# TGV error analysis (Pass-2 deep dive, 2026-09-23)

Why the `tgv` benchmark case converges at only **0.69 order** (vs Sod 1.31,
Gresho 1.54), where the wall is, and what the delta+-SPH comparison leg shows.
All probes were fixed-`dt` runs of the real frontend cases (scratch scripts
under `.tmp/`, method summarized below so this note stands alone).

## 1. The error model (from the fixed-dt probe)

At fixed `nx=64` with `adaptiveDt` off, the velocity error vs the analytic
field is **U-shaped in dt with a minimum at dt ≈ 5e-3** — not the monotonic
decrease a plain time-integration error gives:

| dt | 1e-3 (default) | 5e-4 | 2e-4 | 5e-3 | 1e-2 | 2e-2 |
|---|---|---|---|---|---|---|
| err_l2 | 1.456e-2 | 1.640e-2 | 1.976e-2 | **1.184e-2** | 1.854e-2 | 3.730e-2 |

The decomposition consistent with all six points:

```
err(h, dt) ≈ 0.19·h^1.5  +  ~3e-3  +  ~2.8e-4·n_steps^0.4  +  C·dt
             spatial      constant  per-step             O(dt) time
             (~1.5 order) bias     accumulation (n = t/dt)  error
```

- Only the **spatial** term improves with refinement. The **~3e-3 constant
  bias** is independent of h and dt — a systematic O(1) scheme error no
  refinement removes. The **per-step** term grows with step count (why
  smaller dt is worse); the **O(dt)** term grows with larger dt. The 0.69
  observed order is just the spatial term shrinking toward the non-converging
  floor.
- Consequence: the default `dt=1e-3` is suboptimal — `dt≈5e-3` cuts the error
  18 % and is 5× faster (400 vs 2000 steps).

## 2. The bias derivation: it is residual compressibility

Round 1 ruled out the candidates one by one (nx=64, dt=5e-3):

- **Amplitude**: `v_sim = α·v_an + v_⊥` with α = 1.00725 — the amplitude is
  essentially exact (0.7 % under-dissipation); 83 % of the error energy is
  structural, not amplitude.
- **Density**: RMS(ρ−1) = 6e-4, max 6e-3; corr(|v_err|, |ρ−1|) = 0.023,
  hot-spot ratio 0.96. The density field is clean and carries no bias.

Round 2 (divergence + Fourier content of the velocity field) localizes it:

- **Divergence**: FD div of `v_sim` on a 32² periodic grid (cell-averaged,
  central differences; analytic-field calibration reads 7.8e-16, i.e. the
  measurement is clean) = **RMS 1.59e-2, max 6.2e-2**. The DFSPH flow is
  compressible at the ~1.6 % level.
- **Fourier modes** (particle samples, `c_k = mean_i v_i e^{-ik·x_i}`):
  the primary (1,±1) modes match almost exactly (sim 2.42e-1 vs an 2.40e-1,
  phases identical → **no shift**; the best-fit shift from the mode-phase
  ratios is δ ≈ (0, −8e-5) and shifts the error by 0.0 %). But the sim
  carries spurious modes the analytic field does not have:
  **(2,0)-u = 3.12e-3, (0,2)-w = 3.14e-3** (an ≈ 8e-6), (3,3) ≈ 4e-4.
- **The key argument**: a divergence-free field satisfies
  `kx·u_k + ky·w_k = 0` for *every* Fourier mode. For the (2,0) mode that is
  `2·u_{20} = 0` → u_{20} must vanish; for (0,2), `2·w_{02} = 0` → w_{02}
  must vanish. The spurious (2,0)-u and (0,2)-w modes therefore **cannot
  exist in a divergence-free flow — they are the compressibility itself** —
  and their ~3.1e-3 amplitude is the ~3e-3 constant bias of the error model.

**Conclusion:** the non-converging floor of the DFSPH TGV run is the scheme's
residual failure to keep ∇·v = 0, concentrated in the (2,0)/(0,2) compressible
modes. This is the standing **vd+ps problem** of the incompressible solver
(the coupling of incompressibility, particle shifting and the
divergence-free pass) — a frontend scheme issue outside the scope of the
benchmark effort, not a harness or metric artifact. Per the user's note, the
DFSPH pressure derivation is built on semi-implicit Euler, so a higher-order
integrator addresses the dt-terms of §1 but not this term-2 bias.

## 3. Delta+-SPH comparison (tgv-wc leg)

The frontend `tgvWeaklyCompressible` case (`tgv-wc`) integrates the *same*
vortex (L = 2π, k = 2 → kTgv = 1, phase π/2, uMag = 1, nu = 0.01, t* = 2.0,
same analytic) with delta-SPH + particle shifting (the delta+ variant;
Wendland4 kernel, RK2). Same ladder 32/48/64/96, same `dt = 1e-3` (the case's
`targetDt`).

The legacy back-solve (neither `soundSpeed` nor `machTarget` set) fixes
`dt = targetDt` and inverts `c0 = 0.3·h/(kernelScale·dt)` with `h ∝ dx`, so
**c0 ∝ dx ∝ 1/nx and Ma ∝ nx** across the ladder (the frontend docstring's
"~1/dx" has it inverted): c0 ≈ 108 → 36, Ma ≈ 0.009 → 0.028 from nx=32 to 96.
All well below 0.1, so the weakly-compressible density fluctuations
(O(Ma²), ~8e-5 → 8e-4 up the ladder) do not mask the scheme error —
the leg measures the delta+ spatial/temporal discretization, not the Mach
number.

Note the tgv-wc start is *already* a delta+-relaxed configuration:
`buildSystem` runs `shuffleParticles(shuffleIters=128, jitter=1.0)` — a 1·h
Gaussian jitter plus 128 rounds of delta-SPH shift relaxation (the same
glass construction as the unused `SamplingScheme.optimal` sampler). A
well-relaxed start has near-uniform summation density, so the spurious
initial pressure c0²(ρ−ρ0) is small and the initial pressure build-up stays
weak.

### Results (float64, dt = 1e-3, 2001 steps)

| nx | 32 | 48 | 64 | 96 | 128 | 160 |
|---|---|---|---|---|---|---|
| delta+ err_l2 | 7.57e-2 | 3.86e-2 | 2.23e-2 | 1.23e-2 | 8.24e-3 | 8.67e-3 |
| DFSPH err_l2 (benchmark) | 2.55e-2 | 1.79e-2 | 1.46e-2 | 1.19e-2 | 1.08e-2 | 1.17e-2 |
| delta+ pairwise order | 1.66 | 1.90 | 1.47 | 1.40 | ≈0 | |

(The first delta+ pass ran in float32 because of the §4 float64 blocker —
now fixed; the float32 leg agrees at every point within the unseeded-start
run scatter: 7.68/3.83/2.42/1.26/8.37/8.60 e-2...e-3.)

- **Delta+ converges cleanly at ~1.5–1.9 order** from nx=32 to 128 — in
  contrast to DFSPH's 0.69, which is its non-converging compressibility bias
  of §2.
- **Overtake confirmed (nx=128/160 runs):** at nx=96 DFSPH is still 5%
  better (1.19e-2 vs 1.23e-2); by nx=128 delta+ is 24% better
  (8.24e-3 vs 1.08e-2) and by nx=160 26% better (8.67e-3 vs 1.17e-2).
  Crossover between nx=96 and 128, as the extrapolation predicted.
- **Both schemes are now on their respective floors.** DFSPH sits on its
  ~1.1e-2 plateau (the §2 compressibility bias + per-step accumulation).
  Delta+ flattens at ~8.5e-3 and even rises slightly 128→160 — a constant
  floor cannot rise, so its floor is at least partly the *growing* Ma²
  weakly-compressible physics error (the legacy back-solve makes Ma ∝ nx:
  0.036 → 0.045 across 128→160, Ma²: 1.3e-3 → 2.0e-3) plus the fixed-dt
  (1e-3) time error. (tgv-wc's start jitter is unseeded, so runs carry a
  few-percent run-to-run scatter; the 9% DFSPH 128→160 rise is at that
  level too.) Delta+'s true spatial order past its floor would need dt
  refined with nx.
- nx=64 diagnostics: density RMS(ρ−1) = 6.9e-4 (consistent with Ma² ≈ 3.4e-4
  at this resolution); primary (1,±1) modes match (2.39e-1 vs an 2.40e-1).
- **The delta+ flow carries no excess (2,0)/(0,2) content**: its (2,0)-u
  (7.0e-4) and (0,2) modes match the analytic field *sampled at the same
  displaced particle positions* (6.9e-4 / 8.4e-4) — the apparent spurious
  modes are aliasing of the glass-like sampling, and the true excess over
  that aliasing is ≤ 3e-4, an order of magnitude below DFSPH's 3.1e-3
  (§2).

### Grid-divergence control (why the raw divergence number is misleading here)

The ladder probe's raw grid divergence for delta+ (1.47e-2) looked as bad as
DFSPH's (1.59e-2), contradicting the Fourier result. A control run (nx=64)
separates the pipeline from the flow:

| quantity | div RMS |
|---|---|
| div(CIC v_sim) | 1.41e-2 |
| div(CIC v_an @ sim positions) — glass-sampling control | 7.65e-3 |
| div(CIC v_an @ ideal lattice) — lattice control | 5.7e-16 |

So at glass-like positions the CIC+FD pipeline *itself* produces 7.65e-3 for
a truly divergence-free field: the raw delta+ number is ~half measurement
artifact, the rest real **high-k (particle-scale)** divergence — consistent
with the Fourier finding that there is no *low-k* (2,0)/(0,2) excess. (For
DFSPH the positions are lattice-like, so its 1.59e-2 is essentially all real
low-k compressibility, as §2's mode analysis shows.) Lesson: the grid
divergence metric needs a sampling-matched control, or it overstates a
glass-sampled flow.

### Start A/B: does the optimal (relaxed) start lower the initial error?

The user's hypothesis: the optimal (delta+-relaxed) sampling keeps the
initial error low by avoiding the initial pressure build-up. Two short runs
(t = 0.25, nx=64):

| start | err_l2 | err_linf | RMS(ρ−1) |
|---|---|---|---|
| A: default (1·h jitter + 128 delta+ shifts) | 4.74e-3 | 1.95e-2 | 1.0e-4 |
| B: perfect lattice (shuffleIters=0) | 2.87e-3 | 4.2e-3 | 1.6e-4 |

Findings: (i) the initial error is small for *both* starts — at Ma ≈ 0.02
the acoustic transient is weak either way, so the initial pressure build-up
is not the dominant error source (the t*=2.0 error mostly accumulates over
the run: 2.9e-3 at t=0.25 → 1.26e-2 at t=2.0 for nx=96). (ii) The lattice
start is even slightly better at t=0.25, but it is the unstable SPH
equilibrium the case deliberately shuffles away from (pairing-instability
lattice noise grows later — `buildSystem`'s docstring). (iii) The
`SamplingScheme.optimal` sampler (the validated 0.1·dx-jitter glass recipe)
is **not wired into tgv-wc**: `samplingScheme` is dead config for the
WCSPH path (`setupBasicWeaklyCompressibleInitialState` always calls
`sampleRegularParticles(jitter=0.0)`), and the case's own relaxation uses a
much coarser 1·h jitter. So "start from the optimal sampling" is not a
turn-key knob today; exercising it needs either a frontend hook or a
probe-level stand-in.

## 4. float64 blocker: found, root-caused, FIXED (one line)

Running `tgv-wc` under `warpSPHCore_PRECISION=float64` initially failed at
`warpSPH/modules/deltaSPH/densityDiffusion.py:75`
(`delta * currentState.supports / xi * c0`, `xi = sphKernel_xi(...)`):

- `sphKernel_xi` is a `@wp.func` (warpSPHCore); called eagerly it returns a
  **Python float in float32 mode but a Warp float64 scalar in float64 mode**.
- Warp 1.17.0 has no `div(Tensor, float64)` builtin, so the
  `torch_tensor / warp_scalar` fallback raised
  `RuntimeError: Couldn't find a function 'div' compatible with the
  arguments 'Tensor, float64'`.

**Root cause is a missing eager-scalar coercion, not a deep precision gap.**
The DFSPH path is float64-audited (the whole Pass-2 benchmark runs it); the
delta-SPH path simply had one unwrapped eager `@wp.func` call where every
sibling call site (shifting, WCSPH timestep, cases) already wrapped the same
helper in `float(...)`.

**Fix (one line, in the frontend `warpSPH` repo):**
`densityDiffusion.py` — `xi = float(sphKernel_xi(config.kernel.value,
config.dim))`, matching the established `float(...)` convention. After it,
`tgv-wc` runs cleanly in float64 end-to-end; the full 6-point float64 ladder
(§3) matches the earlier float32 leg at every point within unseeded-start
run scatter, so the fix is numerically inert. With the blocker gone, `tgv-wc`
satisfies the harness's float64 contract and can be registered as a
first-class `CASES` entry.
