# Native core integration with the 0.7.1 simulation stack — 2026-11-03

## Integrated source and execution boundary

Core `main` at `c06ea96` is merged into native development in `cf99ff0`.
The owned-row/chunked multi-GPU baseline was already present in `82dac97`;
the remaining upstream release changes were the version and changelog.
The paired frontend is pinned at `3d755aa` and Integrators at `33df9b9`, both
0.7.1. No primary frontend/integrator checkout, installed editable package,
deployment image or concurrent worktree was changed.

An explicit `WARPSPHCORE_OPERATORS=native` selection in a Warp process now
routes public core operations and CRK volume/moment/consistency-density
accumulations to C++. It borrows the existing Torch states and CSR arrays,
including the frontend's Morton permutations. No particle-sized conversion,
validation scan or host readback is added. The 0.7.1 projection initializer's
calibrated density is supported by applying the shared cached lattice factor
to the native result. The pure native API remains unchanged.

Frontend-specific force/work, diffusion, shifting, pressure iteration and CRK
coefficient kernels still use their previous implementations. Explicit frontend
Verlet construction is unchanged. This is a **mixed full-simulation path**,
not a completely native fluid frontend. Warmed-step dispatch profiles distinguish
native work from operations used only during initialization.

Native operators reject chunked adjacency rather than gathering it into a global
CSR or silently falling back. The upstream multi-GPU/multi-node path remains
available with Warp operators. Porting the C++ traversal to owned rows and
qualifying distributed derivatives are separate work. Warp-specific explicit
JVP/HVP and CRK-JVP adapters also reject native operator selection; native Torch
first-order AD and explicit field linearizations retain their prior scope.
This campaign uses eager execution; full native frontend graphs are unqualified.

No compiled numerical source changed. The previously qualified stable-ABI
CUDA library was reused after the loader verified its compiled-source digest.
Python/package selection checks and focused operator parity qualify the new
dispatch; recompiling every dimension/kernel combination adds no coverage here.

## Complete trajectory comparisons

Each numerical case runs in a fresh process per operator backend, with the same
seed, configuration and UID roster. States are compared in UID order at
initialization, every accepted timestep and the final output. Positions,
supports, masses, kinds, velocity, density, pressure, internal energy or wave
fields are checked as applicable; categorical values match exactly and floating
outputs must be finite. The FP32 bound is
`max(abs(native - warp)) / max(1, max(abs(warp))) <= 2e-4` per field.
Clocks are compared separately at a precision-scaled tolerance.

The FP32 matrix is Delta-SPH TGV in input and Morton order (256 particles),
Morton CompSPH/CRKSPH acoustic waves (32 particles), and wave-equation evolution
in 1D/2D/3D (32/256/512 particles). Each trajectory has ten accepted steps.
IISPH TGV is a separate projection control, not included among passing fluid
qualifications below.

| Complete trajectory selection | Passing pairs | Largest scaled field difference |
|---|---:|---:|
| FP32 CPU, Delta/Comp/CRK/wave matrix | 7/7 | 2.38e-7 |
| FP32 CUDA, same matrix | 7/7 | 1.19e-7 |
| FP32 CUDA, larger Morton TGV/Gresho and 3D wave (16K/16K/32K) | 3/3 | 1.19e-5 |
| FP64 IISPH control, CPU and CUDA | 2/2 | 5.23e-9 |

Each of these pairs compares all ten accepted states. Separately, all seven
qualified CPU scaling pairs and three `n_h=2.57` controls pass their saved
initial/final-state comparison (largest scaled difference `1.19e-5`).
The clock bound is `max(1e-12 seconds, 64 * dtype_epsilon * abs(t))`, allowing
negligible differences from the state-dependent physical step limiter.

### Projection discrepancy retained as a failed gate

The coarse, unrelaxed 144-particle FP32 IISPH fixture does **not** satisfy
cross-backend trajectory parity. Identical initial saved states develop a
first-step CPU density difference of `3.58e-7`, a pressure difference of about
60 and a velocity difference of `1.89e-3` at requested `dt=1e-4`. Differences
then grow. Increasing the requested timestep to `.01` still fails. The upstream
physical step limiter may subsequently shorten the steps despite
`adaptiveDt=False`, so the report preserves actual clocks too.

FP64 CPU and CUDA controls at requested `dt=.001` pass the tighter `1e-8` field bound. For CPU,
the maximum scaled difference is `4.12e-9`, with clock difference `6.25e-17`; CUDA gives `5.23e-9` and `1.39e-16` respectively.
This is evidence of strong precision/timestep sensitivity in the coupled
projection, consistent with small density changes being amplified by the
pressure source's inverse timestep squared. It does not qualify FP32 IISPH or
establish that this coarse trajectory is a physically stable reference.
The failed FP32 records are retained; no tolerance was relaxed to pass them.

## Warmed full-step CPU measurements

CPU: Intel Core Ultra 9 285K, 24 available cores, Torch scheduler set to
24 threads for both backends. Precision FP32, Wendland4, `n_h=4`.
CRK and wave runs warm five steps and record fifteen. The fixed-sound-speed TGV
runs and `n_h=2.57` controls warm three and record seven; setup and compilation
are excluded. The primary runner timer includes integrator stages, neighborhood
work, and ordering/layout work. The accepted-boundary interval additionally
contains runner diagnostics/output overhead. No snapshots or profiler are
enabled in performance runs. These are local paired observations, not isolated
hardware ceilings or a long physical validation campaign.

The reported neighborhood degree is the mean CSR **candidate** degree, including
the frontend's Verlet padding, not the count inside the physical kernel support.
The fixed wave list has no Verlet padding: 257 neighbors at `n_h=4` and 81 at
`n_h=2.57` on its cubic lattice.

### Stiff-EOS TGV control, excluded from the qualified timings

The first larger TGV runs left the upstream sound-speed selection at its default:
it back-solves sound speed from the requested `dt=1e-4`. Those runs have identical
initial states, but at 16K particles the final density differs by one FP32 ULP
(`1.19e-7`), velocity by `3.44e-5`, and pressure by `.00877`. The pressure field
fails the fixed comparison bound. The 64K control likewise fails on pressure.
These results remain in the evidence archive and are excluded from the primary
qualified scaling figure. The comparison tolerance is unchanged.

The repeated TGV runs explicitly select `soundSpeed=10` (initial speed near one,
Mach about `.1`) independently of the fixed timestep. Their complete saved
initial/final states pass at all three sizes, with maximum scaled difference
`1.19e-5`. The report compares the same physical configuration across resolutions
instead of changing compressibility with resolution. The largest-case result is
therefore a full-step comparison with an explicitly specified EOS.

| Simulation | Particles | n_h | CSR degree | Warp core [ms] | Native core [ms] | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| Delta-SPH 2D | 1,024 | 4 | 99.0 | 43.02 | 43.59 | 0.99x |
| Delta-SPH 2D | 16,384 | 4 | 98.8 | 616.89 | 501.48 | 1.23x |
| Delta-SPH 2D | 65,536 | 4 | 98.8 | 3,368.86 | 2,785.01 | 1.21x |
| Delta-SPH 2D | 16,384 | 2.57 | 45.0 | 297.96 | 242.47 | 1.23x |
| CRKSPH 2D | 1,024 | 4 | 101.0 | 154.75 | 182.27 | 0.85x |
| CRKSPH 2D | 16,384 | 4 | 101.0 | 2,384.75 | 1,799.63 | 1.33x |
| CRKSPH 2D | 16,384 | 2.57 | 45.0 | 1,085.20 | 856.10 | 1.27x |
| Wave 3D | 4,096 | 4 | 257.0 | 262.40 | 54.10 | 4.85x |
| Wave 3D | 32,768 | 4 | 257.0 | 3,061.24 | 228.37 | 13.40x |
| Wave 3D | 32,768 | 2.57 | 81.0 | 797.99 | 114.40 | 6.98x |

All fluid rows retain Warp frontend kernels. The 1K CRK case is slower in
native execution; this is not a blanket speedup. At larger sizes the fluid
gain is about 20–33%, while the operator-dominated 3D wave benefits much more.

For the largest three primary cases, the corresponding accepted-boundary
intervals **including runner diagnostics** are:

| Case | Warp [ms] | Native core [ms] | Interval speedup |
|---|---:|---:|---:|
| Delta-SPH 2D, 65,536 | 5,175.02 | 4,470.20 | 1.16x |
| CRKSPH 2D, 16,384 | 2,385.95 | 1,800.83 | 1.32x |
| Wave 3D, 32,768 | 3,713.40 | 293.97 | 12.63x |

Warmed dispatch profiles over two accepted steps record:

| Case | Native dispatches |
|---|---|
| Delta-SPH | 4 continuity divergences |
| CRKSPH | 4 apparent-volume, 4 moments, 4 consistency-density, 4 corrected-gradient calls |
| Wave | 8 Laplacians in RK4; 2 gradients in diagnostics |

This explains the limited fluid gains: custom force/work and diffusion loops
remain Warp. Sorting is now integrated, but converting those frontend operators
is the next substantial performance opportunity. Native owned-row traversal
is required before the upstream multi-GPU simulation can use this selection.

## Warmed full-step GPU measurements

GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, sm_120, Torch
2.13.0+cu130 and Warp 1.17.0. The primary sweep starts after the compute GPU
was freed. Every backend runs in a fresh process, with ten warmup steps and
forty measured steps. Configuration matches the qualified CPU models: FP32,
Wendland4, fixed sound speed 10 for TGV, Morton-ordered fluids and fixed wave
geometry. Full eager steps use the runner CUDA-event timer; these are not
isolated kernel timings. All nine primary final-state comparisons pass the same
field bound, with largest scaled difference `3.26e-5`.

| Simulation | Particles | n_h | CSR degree | Warp core [ms] | Native core [ms] | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| Delta-SPH 2D | 16,384 | 4 | 98.8 | 3.904 | 3.760 | 1.04x |
| Delta-SPH 2D | 65,536 | 4 | 98.8 | 4.001 | 3.687 | 1.09x |
| Delta-SPH 2D | 262,144 | 4 | 98.7 | 8.412 | 7.270 | 1.16x |
| Delta-SPH 2D | 65,536 | 2.57 | 45.0 | 4.532 | 3.641 | 1.24x |
| CRKSPH 2D | 16,384 | 4 | 101.0 | 8.223 | 6.793 | 1.21x |
| CRKSPH 2D | 65,536 | 4 | 100.1 | 10.140 | 8.721 | 1.16x |
| CRKSPH 2D | 262,144 | 4 | 100.0 | 28.141 | 22.147 | 1.27x |
| CRKSPH 2D | 65,536 | 2.57 | 44.1 | 8.343 | 7.003 | 1.19x |
| Wave 3D | 32,768 | 4 | 251.0 | 3.941 | 0.995 | 3.96x |
| Wave 3D | 262,144 | 4 | 251.0 | 10.995 | 3.952 | 2.78x |
| Wave 3D | 1,000,000 | 4 | 254.4 | 44.885 | 16.532 | 2.72x |
| Wave 3D | 262,144 | 2.57 | 81.0 | 3.761 | 1.331 | 2.83x |

The smaller-neighborhood GPU controls are repeated with CPU analysis stopped;
all three final-state comparisons pass. Clean reverse-order confirmations
also stop this task's CPU analysis during the measured runs:

| Case, n_h=4 | Primary speedup | Clean reverse-order speedup |
|---|---:|---:|
| CRKSPH-2d-256-morton | 1.16x | 1.17x |
| deltaSPH-2d-256-morton | 1.09x | 1.05x |
| deltaSPH-2d-512-morton | 1.16x | 1.10x |
| waveEquation-3d-64-input | 2.78x | 2.94x |

The launch-bound Delta case has a small benefit and noticeable run-to-run
variation; an earlier exploratory repeat reached approximate parity. The larger
Delta case shows about 10–16%, CRK is steadier, and the wave gain remains large.
Neither the fluid data nor the 1K CPU regression supports a blanket speedup claim.

Mean step cost retains rare search/layout and other slow steps, unlike a median:

| Largest primary case | Warp mean [ms] | Native mean [ms] | Mean-cost speedup |
|---|---:|---:|---:|
| Delta-SPH 2D, 262,144 | 8.316 | 7.271 | 1.14x |
| CRKSPH 2D, 262,144 | 28.243 | 22.597 | 1.25x |
| Wave 3D, 1,000,000 | 45.069 | 16.383 | 2.75x |

The mean is over the recorded finite window, not a guaranteed long-run rebuild
frequency. The state and solver diagnostics can add further overhead; their
accepted-boundary intervals are retained separately in every result JSON.

GPU telemetry is sampled throughout every run. No unrelated compute PID is
observed. One preceding benchmark context remains visible during the next
process's startup, 4.3 seconds before that next measured window; it does not
overlap measurement. Ordinary desktop display activity remains. Earlier
`gpu-small`/`gpu-repeat` runs are exploratory and superseded by the controls
and clean confirmations for final claims. The audit and raw observations are
preserved rather than discarded.


## Qualification and reproduction

New regression coverage is limited to two parity scenarios per device (all core
routes/calibration/unsupported-adapter guards, and CRK preparation routes), plus
one lightweight backend-selection scenario. Existing backend/device suite
separation is retained; no additional large Cartesian test matrix is introduced.

| Focused check | Result |
|---|---|
| Python/package/import isolation | 24 passed |
| New core/calibration/CRK dispatch parity, CPU + CUDA | 4 passed |
| Native derivative chain/affine/fixed-origin/import checks, CPU + CUDA | 7 passed |
| Upstream Warp chunked construction and storage-lifetime controls | 5 passed |
| CMake standalone package configuration | version 0.7.1 generated |

The compiled numerical source is unchanged; focused import, dispatch, AD and
upstream ownership checks suffice without rerunning the entire kernel matrix.

Use the paired source roots on `PYTHONPATH`; no global reinstall is required:

```bash
python scripts/benchmarks/bench_simulations.py --operators native \
  --case tgv-wc --scheme deltaSPH --device cpu --nx 128 \
  --n-h 4 --order morton --sound-speed 10 --threads 24 --warmup 3 --steps 7 \
  --out /tmp/paired/tgv-native
# Repeat with --operators warp and --out /tmp/paired/tgv-warp.
python scripts/benchmarks/compare_simulations.py /tmp/paired \
  --pattern tgv --out /tmp/paired/comparison.json

python scripts/benchmarks/bench_simulations.py --operators native \
  --case waveEquation --dim 3 --nx 100 --device cuda --order input \
  --n-h 4 --warmup 10 --steps 40 --out /tmp/paired/wave-native
# Repeat with --operators warp and --out /tmp/paired/wave-warp.
python scripts/benchmarks/plot_simulations.py /tmp/paired \
  --device cuda --out /tmp/paired/figures
```

For numerical trajectories add `--snapshots`. For dispatch evidence run a separate
short `--profile` case; it records two warmed steps, excluding initialization.
Do not interpret profiled or correctness-run timings as performance results.

Initial CUDA correctness runs occurred under another GPU job and their incidental
timers are excluded from performance conclusions. After an idle window became
available, a separate GPU performance campaign and reverse-order confirmations
were run. No AMD hardware or physical multi-node topology is qualified here.

Evidence, final figures, source manifests, commands, complete state comparisons
and the deliberately failed controls are retained at
`/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-frontend-071-20261103`.
The graphs are `figures/cpu-scaling.png` and `figures/gpu-scaling.png` (SVG also
available), with matching summary JSON. `SHA256SUMS` covers the archive. Only the
separate `gpu-performance`, controls and confirmation runs supply GPU benchmark
results; earlier CUDA correctness files are numerical qualification records.
