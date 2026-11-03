# Frontend stage costs with mixed native core operators — 2026-11-03

This follows the [0.7.1 integration qualification](native_frontend_071_2026_11_03.md).
The comparison changes only `WARPSPHCORE_OPERATORS=warp|native`; the frontend,
integrator, spatial ordering, neighborhood and physical parameters are paired.
Frontend `3d755aa` and Integrators `33df9b9` remain unchanged. No production
numerical kernels or frontend profiling gates are modified.

## Attribution and timing scope

The benchmark reuses the numbered frontend `record_function` gates to identify
substeps. A benchmark-only wrapper partitions each runner timer into exclusive
intervals. Entering a nested stage or a native/Warp dispatch closes the previous
interval; summing inclusive nested profiler ranges would double count that work.
The wave RHS lacks a numbered gate, so a benchmark-only scope encloses its
Laplacian call. Timed steps include integrator, neighborhood preparation, layout
preparation and the CRK continuation-layout adoption timer. Initialization,
diagnostics, output hooks and state comparison are outside these timers.

Colors group related substeps. Texture identifies execution:

- Solid: a native core C++/CUDA operator call, including its dispatch/allocation.
- Diagonal hatch: a Warp kernel launch and its execution interval.
- Dots: remaining scaffolding, Torch work, tensor preparation and dispatch.
  This is not a claim that Torch's own operators lack native implementations.

CRK factors are deliberately mixed: apparent volume, moments and consistency
density dispatch native, while the fused coefficient solve still dispatches
Warp. A whole frontend stage is never called native merely because it contains
a native call. Raw gate names and launch/operator counts remain in the evidence;
the exported CSV preserves individual gate/engine timing before aggregation.

CPU uses wall-clock boundaries. CUDA records reusable timing events on the
current Torch stream and checks Warp launches use that same stream. Events are
read after the runner's existing end-of-timer synchronization; no per-stage
synchronization is inserted. CUDA intervals include host submission gaps and
therefore describe elapsed step cost, **not kernel-active time or utilization**.
Events, Python scopes and attribution bookkeeping perturb execution. Separate
matched ordinary runs, plotted as black diamonds, quantify this effect. The
colored stack is the measured instrumented total; it is not rescaled to the
ordinary timing. Independent panel scales are linear and explicitly labelled.

Each bar uses arithmetic mean exclusive costs over accepted measured steps,
which add to a mean total. Independently taking medians of the stages would not
in general sum to the median step. The uninstrumented controls run the exact
same number of steps and warmup, and save the same final state.

## Configurations and evidence

FP32, Wendland4, `n_h=4`, eager execution, no CUDA graphs. Fluid fixtures use
Morton order; Delta-SPH pins sound speed to 10 independently of the requested
`dt=1e-4`. Wave geometry and adjacency remain fixed. CPU uses 24 Torch threads
on the Core Ultra 9 285K; Warp CPU kernels remain serial, so the thread count
refers to Torch scheduling. GPU is RTX PRO 6000 Blackwell. CPU and GPU campaigns
run sequentially; analysis and CPU simulations are stopped during GPU timing.

CPU: 16,384 particles for each fluid, 32,768 for the 3D wave; five warmup and
seven measured steps. GPU: 262,144 particles for each fluid, 1,000,000 for the
3D wave; ten warmup and thirty measured steps. Larger GPU fixtures reduce the
relative influence of event instrumentation. These different sizes are shown
on each panel; the figure is not a CPU-versus-GPU scaling comparison.

## Measured results

The native-core bars retain the following exclusive elapsed-time shares. These
are **instrumented** means; they identify where the remaining work runs, rather
than predicting speedup from a hypothetical complete native port.

| Device / case | Instrumented step [ms] | Native core | Remaining Warp | Scaffolding / Torch / dispatch |
|---|---:|---:|---:|---:|
| CPU / deltaSPH | 502.15 | 3.3% | 95.5% | 1.3% |
| CPU / CRKSPH | 1785.91 | 3.5% | 88.3% | 8.2% |
| CPU / waveEquation | 251.49 | 84.3% | 0.0% | 15.7% |
| CUDA / deltaSPH | 8.04 | 5.7% | 71.6% | 22.7% |
| CUDA / CRKSPH | 23.41 | 7.8% | 78.1% | 14.1% |
| CUDA / waveEquation | 15.89 | 96.1% | 0.0% | 3.9% |

The CPU Delta-SPH native-core run still spends 180.1 ms in Warp density
diffusion, 171.0 ms in Warp velocity diffusion and 126.8 ms in Warp pressure
force. Continuity changes from 131.2 ms in Warp to 16.5 ms in native code,
but the unported stages dominate the complete step.

For CPU CRKSPH, Warp pressure/viscosity acceleration costs 861.2 ms and pair
work costs 464.8 ms, together about 74% of the mixed step. Warp optimal-support
kernels add 233.9 ms. Native factor accumulation is 45.2 ms and corrected
velocity gradient 17.8 ms. The fused CRK coefficient solve remains Warp and is
represented by hatching within the factor stage.

On GPU CRKSPH, the same unported acceleration and pair-work stages cost 11.18 ms
and 3.96 ms, together about 65% of the mixed step. Native CRK accumulations
cost 1.28 ms. This makes frontend force/work the clearest next conversion target
for this fixture. It does not establish the same priority for every scheme or
claim that porting those stages reproduces the core-operator speedups.

Wave RHS timing has no Warp launches after the native switch. Its current
spatial computation is the native Laplacian; residual cost is integration and
scaffolding. The saved dispatch counts explicitly exclude diagnostics outside
the integrator timer, so diagnostic gradients are not counted as RHS work.

Instrumented versus ordinary mean differences are -0.3% to +8.1% on CPU,
with the +8.1% observation in native wave. GPU fluids show +4.6% to +7.4%; GPU
wave is -1.5% / +0.1% for Warp / native. These paired differences include
run-to-run variation, so negative values are not negative profiling overhead.
The chart shows each ordinary mean directly rather than subtracting or hiding
these differences. The remaining force-stage costs need not be identical
between separate processes: these are elapsed dispatch regions, including
submission gaps, not a kernel-only hardware-counter measurement.

## Validation

All twelve instrumented runs have bit-identical initial and final saved states
to their matched ordinary controls. All six Warp/native final-state pairs pass
the unchanged `2e-4` scaled field bound; worst observed error is `3.20e-5`.
The saved states are canonicalized by UID before comparison. This campaign
checks final states; the earlier qualification contains per-step trajectories.

The exclusive partition sums match the runner timer sum to within 0.16% on
CUDA and 0.005% on CPU; the small difference is timer-wrapper boundary overhead.
The independently collected frame timer means match the runner trajectory
means. No nested ranges are added twice. One deterministic instrumentation
unit test checks nested stage/dispatch attribution, leaving integration
coverage in the actual paired campaigns rather than a new test matrix.

GPU telemetry contains 86 samples with no unrelated compute PIDs, including
measurement windows. The evidence is retained independently at:

`/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-frontend-stages-20261103`

The primary visual is `figures/simulation-stages.png` (also SVG); individual
frontend gate values are in `figures/frontend-gates.csv` and raw per-step
partitions and dispatch counts are under `results/*-stages/stage-timing.json`.


## Reproduction

Use the paired source roots and the previously qualified prebuilt native CUDA
library described in the integration report. Each run uses a fresh process.
For example, from the core checkout with that `PYTHONPATH` and prebuilt loader:

```bash
python scripts/benchmarks/bench_simulations.py \
  --operators native --case gresho --scheme CRKSPH --device cuda \
  --dim 2 --nx 512 --n-h 4 --order morton --sound-speed 10 \
  --warmup 10 --steps 30 --stage-profile \
  --out /tmp/stages/timing-cuda-CRKSPH-native-stages
```

Repeat without `--stage-profile` into the matching `-control` directory;
repeat for Warp core. Use schemes `deltaSPH` (`--case tgv-wc`) and `waveEquation`
(`--case waveEquation --dim 3 --nx 100 --order input`) for the other GPU panels.
CPU uses `--device cpu`, `--nx 128` for the fluids and `--nx 32` for the wave,
with five warmup/seven measured steps. Once all pairs are present:

```bash
python scripts/benchmarks/plot_simulation_stages.py /tmp/stages --out /tmp/stages/figures
```

`stage-timing.json` contains exclusive individual gates, actual dispatch counts,
per-timer partitions and accepted step indices. `result.json` retains ordinary
runner timers, source digests and arguments; `states.pt` contains canonical
UID-ordered initial/final states. `stage-summary.json`, `frontend-gates.csv`, PNG
and SVG exports are generated from these raw records. The evidence archive also
contains the exact sequential campaign, GPU telemetry, qualification script,
source pins and checksums.
