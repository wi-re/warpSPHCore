# Native operator and radius-search optimization, 2026-10-31

The saved starting point is `d97d24e` on `codex/native-backend`. Optimization stays
in the isolated native worktree. Warp kernels remain unchanged as the readable
behavioral reference; manual backward and adjoint source files are unchanged.
Native/Torch is the primary execution path for this work. Existing standalone
code remains compatible but was not extended into a second optimization project.

## Implemented changes

- Prepare Wendland2 support coefficients once per Gather row. Evaluate primal
  values and gradients directly instead of constructing unused derivative jets.
  Other kernels and support schemes retain the general numerical path.
- Cache periodic query geometry per row. Common wrapped-coordinate and minimum-image
  cases avoid repeated floor/division operations. General periodic images retain
  the floor fallback and the existing shifted-scalar half-box convention.
- Explicitly use float floor/square-root functions for FP32 in ordinary C++.
  Global C math calls otherwise promoted float to double on the host compiler.
- Use shared eight-lane cooperative GPU rows with fixed field-component counts
  for all seven primals. Shared CPU/GPU numerical helpers remain under the core
  native tree; redundant interpolation-specific CUDA helpers were removed.
- Use CUB merge sorting for CUDA compact-hash index keys. The lexicographic
  bucket/cell/original-ID ordering is unchanged. HIP retains the portable merge implementation.
- Collect GPU neighbors without serial row sorting, then sort with cooperative
  register bitonic networks for degrees up to 32/64. Longer rows retain heap sort.
  Native CSR rows still contain ascending original reference IDs.
- No numerical-content checks, CSR verification, debug flags, or new public
  runtime parameters were introduced. Qualification stays outside timing.

## Measured result

All entries below are native/Torch completed wall times in milliseconds for one
million particles in 2D, approximately 33 neighbors per particle. Each entry is
`before → optimized (speedup)`. These are desktop measurements, not portable guarantees.

| Operation | GPU FP32 | GPU FP64 | CPU FP32, 16 workers | CPU FP64, 16 workers |
|---|---:|---:|---:|---:|
| Search | 10.01 → 6.10 (1.64×) | 32.64 → 11.53 (2.83×) | 619.03 → 556.99 (1.11×) | 708.93 → 684.28 (1.04×) |
| Density | 1.28 → 1.01 (1.26×) | 6.19 → 2.09 (2.96×) | 146.50 → 27.50 (5.33×) | 173.50 → 39.71 (4.37×) |
| Interpolate | 1.56 → 1.54 (1.02×) | 6.27 → 2.66 (2.36×) | 150.72 → 94.36 (1.60×) | 307.42 → 124.29 (2.47×) |
| Gradient | 2.06 → 1.69 (1.21×) | 9.28 → 3.17 (2.93×) | 163.48 → 64.35 (2.54×) | 310.64 → 137.35 (2.26×) |
| Divergence | 1.84 → 1.72 (1.07×) | 8.68 → 3.01 (2.88×) | 158.24 → 64.44 (2.46×) | 304.12 → 118.73 (2.56×) |
| Curl | 2.04 → 1.65 (1.24×) | 8.62 → 2.99 (2.88×) | 164.60 → 84.25 (1.95×) | 307.78 → 130.41 (2.36×) |
| Laplacian | 2.30 → 1.89 (1.22×) | 9.22 → 3.47 (2.66×) | 200.72 → 111.94 (1.79×) | 307.03 → 161.67 (1.90×) |
| Covariance | 1.57 → 1.23 (1.28×) | 8.50 → 2.93 (2.90×) | 115.27 → 39.05 (2.95×) | 217.81 → 56.47 (3.86×) |

FP64 GPU primals in this 2D case improve by 2.36–2.96× over the checkpoint and
are 2.98–7.34× faster than the contemporary Warp measurements. FP32 GPU gains
are smaller: interpolation is effectively unchanged; most primals remain slightly
slower than Warp, while covariance is faster. This is still an optimization opportunity.
CPU primals improve by 1.60–5.33× in FP32 and 1.90–4.37× in FP64.
CPU fresh search changes little (1.04× FP64, 1.11× FP32); this pass does not claim
to have solved its scaling. Its serial compact-index sorting/build loops remain
candidates for a separate stage-level CPU profile and parallel implementation.

## Experiment and measurement scope

- RTX PRO 6000 Blackwell, sm_120; Core Ultra 9 285K, 24 physical cores.
  CPU cases use 16 workers pinned to CPUs 0–15, matching the useful worker setting
  established by the earlier thread-count sweep. Warp CPU kernels launch serially.
- Torch 2.13.0+cu130, Warp 1.17.0, Python 3.13.14.
  JIT uses `-O2`, host `-march=native`, CUDA architecture 12.0. No fast-math flag was added.
- Matched before/after matrices contain 224 timing records each: GPU FP64 in
  1D/2D/3D and FP32 in 2D/3D, CPU in 2D for both precisions, at N=32,768 and 1,000,000.
  Search rebuilds the complete compact index and CSR; operators reuse identical sorted CSR.
- GPU timings use five measured batches of ten operator calls (one search call),
  following two warmup batches. CPU uses three one-call batches after one warmup.
  Compilation, fixtures, transfers, canonicalization and qualification are excluded.
- Every raw sample, median, min/max, qualification, source hash and activity
  snapshot is retained. Min/max ranges are not confidence intervals.
- The first optimized FP64 GPU sweep contained a non-reproducing 8.96 ms
  Laplacian result. A separate seven-batch repeat measured 3.46 ms, and a complete
  repeat measured 3.47 ms. Both full sweeps and the focused repeat are archived;
  the comparison consistently uses the second full FP64 GPU sweep, without
  choosing individual per-operation minima. The degenerate 1D zero-curl case
  varied from about 23 to 32 microseconds and is not a useful throughput result.
- A 32-lane-per-row candidate was rejected: FP32 interpolation was slightly
  slower and FP64 interpolation regressed from 3.23 to 4.72 ms relative to the
  corresponding eight-lane candidate. Those candidate results are retained.
- Final Nsight Compute FP64 profiles for interpolation/Laplacian/divergence show
  94.6–95.3% L2 sector hits, 6.0–7.9% external-memory throughput, and 85.2–85.8%
  reported SM throughput. Occupancy is 36.9–49.2%, with 66/94/78 registers per
  thread. These counters suggest arithmetic/instruction and latency work rather
  than external-memory saturation; SM throughput is not a measured FLOP rate.
  Profiling runs are separate from ordinary timing evidence.

## Correctness and numerical qualification

- Final native Torch CPU/CUDA suite: **2,681 passed, 207 skipped**. This includes
  manual backwards/adjoints, all kernel families, layouts and support policies.
- New independent edge-sum/AD oracles cover ragged degrees, scalar/vector/matrix
  fields, both precisions, dimensions, both CSR widths, support/direction policies,
  multiple periodic images, partial GPU blocks, and nondefault streams.
  Additional regressions cover half-box rounding and sorted search rows at
  degrees 31/32/33 and 63/64/65, empty rows and longer-row heap fallback.
- Python scaffolding suite: **19 passed**. Standalone CMake compatibility checks:
  **4/4 CPU**, **6/6 CUDA**. HIP library builds and shared cooperative operator/
  sorting instantiations compile for **gfx942 and gfx1100**; AMD runtime is untested.
- FP64 search membership exactly agrees with Warp for every measured case.
  At one million FP32 particles, native/Warp membership differs on 4 directed
  edges in 2D CUDA, 5 in 2D CPU, and 10 in 3D CUDA, out of roughly 33 million.
  Independent FP64 distances on rounded inputs put every difference within
  1.69e-9 of the cutoff. The benchmark records these as cutoff-rounding
  equivalence, rather than exact CSR equality. This discrepancy also occurs at
  the saved baseline. No cutoff predicate was changed to squared-distance testing.
- Benchmark FP64 operator qualification retains `rtol=atol=3e-9`. FP32 uses
  relative L2 ≤3e-4 plus an amplitude-scaled absolute tolerance for zero-crossing
  fields; measured maximum relative L2 is 2.70e-5. Unit-test tolerances are unchanged.
- High-resolution 1D FP32 derivatives are excluded from this measurement matrix
  because Warp/native periodic arithmetic becomes visibly differently conditioned.
  An independent FP64 edge oracle measured native gradient relative L2 2.73e-5
  versus Warp 4.38e-5 on the diagnostic fixture; the evidence is archived.
  This pass does not assert bitwise floating-point equivalence.

## Reproduction and evidence

Use `scripts/benchmarks/bench_backends.py --precision float32|float64`; the primary
Warp/native Torch paths are now the default and need no standalone bridge.
`analyze_optimization.py` exports every matched before/after case and generates
the CPU/GPU comparison. `analyze_backends.py` generates per-dimension scaling
graphs and labels the selected precision correctly. Detailed commands are in
`scripts/benchmarks/native/README.md`.

Local archive: `warpSPHDevelopment/core/experiments/native-performance-20261031/`.
It contains `before-after.png`, `before-after.csv`, per-precision scaling graphs,
raw baseline/optimized/candidate JSONs, source snapshots, profiler CSVs, build
configuration, binaries, test logs and the command driver. Generated bulk evidence
is outside Git under the development-artifact policy.
