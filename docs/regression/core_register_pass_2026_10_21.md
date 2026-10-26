# Shared SPH register pass — 21 October 2026

The useful common optimization is to expose tensor dimensions to Warp's
compiler. Forcing lower register counts is not a reliable optimization.
This pass temporarily takes priority over the MFM/MFV register work; AC remains
deferred. The MFM/MFV architecture changes already made remain in place.

## Changes retained

- `math/wp_outerTensorProduct.py`: derive the loop extent and flattened input
  stride from the concrete input vector type.
- `math/wp_divdot.py`: derive the loop extent and contraction stride from the
  concrete output vector type.
- `crk/crk_moments.py`: use the moment vector's static spatial dimension for
  the rank-three derivative loop and flattened index.

The particle order, summation order, tensor layouts, correction policies and
public operator interface are preserved. Runtime shape arguments remain in the
internal ABI. The CRK moment operator here is the shared core/AD operator;
ordinary frontend CRKSPH also has fused assembly kernels, so these numbers do
not predict its whole timestep improvement.

## Measurements

RTX PRO 6000 Blackwell, SM120; Warp 1.17.0, Torch 2.13.0+cu130. Periodic equal-h
lattices, 262,144 particles, target 55 neighbors, Wendland2. Preparation is
excluded. One GPU process at a time. CUDA event intervals around a direct,
preallocated operator launch include launch gaps; these are not runner or
whole-timestep timings. CPU derivative work finished before repeated timings.

Three process-isolated A/B trials alternate source order, with 21 samples per
trial and four warmups. The table gives the median of trial medians for CSR:

| CRK moments | Lanes | Before, ms | After, ms | Time reduction |
|---|---:|---:|---:|---:|
| 2D FP32 | 1 | 0.555 | 0.370 | 33.2% |
| 2D FP32 | 32 | 0.687 | 0.572 | 16.7% |
| 3D FP64 | 1 | 7.465 | 6.274 | 16.0% |
| 3D FP64 | 32 | 11.174 | 9.832 | 12.0% |

The wider screen also covers 2D FP64, 3D FP32, serial/tiled kernels and both
traversals. Grid CRK moments improve in the measured 2D cases and 3D FP64;
3D FP32 tiled grid is approximately unchanged. Initial 2D FP32 gradient and
divergence measurements improve roughly 5–10%; their FP64 gains are small.

NCU independently measures 3D FP64 serial CSR moments at 7.384 → 6.225 ms.
Registers rise from 232 to 255, while achieved occupancy stays at 16.6%.
Local-memory load sectors fall from 281,935,872 to 35,192,832 (87.5%); store
sectors fall from 170,262,528 to 47,120,384 (72.3%). Allocated local bytes fall
from 352 to 216 per thread. This supports targeting dynamic tensor indexing
and local traffic rather than minimizing the register number in isolation.
Local bytes, local traffic and compiler spill counts are distinct measures.

Corrected offline ptxas results expose the tradeoff: the original 3D FP64
moment forward entry has no compiler spills, while the retained serial entry
has 280 bytes of spill stores / 192 bytes of spill loads (328 / 224 tiled).
Despite these spills, its measured total local traffic and duration decrease.
It remains at the 255-register ceiling; reducing accumulator state further is
an open target, rather than an excuse to impose an unqualified register cap.

## Experiments kept out of production

`probe_register_variants.py` makes source-isolated, forward-only kernel clones
without changing production dispatch. It tests traversal constants, policy
constants, a lean launch and register caps, retaining the original pair math.

- **Register caps:** NCU covariance FP32, 3D serial CSR: 99 → 64 registers,
  achieved occupancy 31.7% → 58.3%, duration 656 → 738 microseconds. FP64 CRK
  moments capped at 128 registers nearly double occupancy but run slightly
  slower and increase local traffic. A grid case improves with a cap, so the
  result is workload dependent; no blanket cap was adopted.
- **Policy specialization:** 3D FP32 covariance drops to 72 registers and NCU
  measures 624 microseconds, but other cases regress. Specializing every
  combination would also multiply compilation variants. No automatic policy
  specialization was added.
- **Traversal specialization:** some CSR cases improve, but 3D FP32 tiled grid
  covariance regresses from about 10.1 to 30.9 ms. Keep the shared pair math;
  separately qualify grid and CSR outer loops before adopting a common split.
- **Static distance loop:** serial density improves markedly, but tiled grid
  covariance regresses. This candidate remains isolated from production.
- **Lean launch:** mixed throughput results and sometimes higher register
  allocations; no global launch-policy change.

The existing ptxas parser also needed repair: helper-function property lines
could overwrite a kernel's frame and spill counts with zeros. It now tracks
entry identity, retains cumulative stack size separately, and has a regression
fixture. Re-parsing all 88 archived inventory entries confirms those original
forward entries had no spills, but corrects their stack-frame fields. The new
FP64 moment spills above were hidden by that bug. Original reports are retained
with correction notices, alongside corrected records and raw log hashes.

The first AST-transformer prototype was invalid: Warp's generic overload
construction discarded its transformers. It is explicitly marked invalid in
the evidence. The retained probe materializes transformed source and records
concrete source hashes, kernel options and actual driver resource queries.

## Qualification and remaining scope

Saved before/after outputs are bitwise identical across 32 source-paired cases:
2D FP32/FP64 and 3D FP64, both lane counts, with CSR for density/gradient/
covariance/moments (plus FP32 divergence) and grid for moments. Every experimental
clone also checks its outputs against the original kernel for the same case.

The final CPU derivative gate passes 77 checks, including a new raw-moment
finite-difference canary in 1D/2D/3D for CSR and grid. Additional final gates pass:

| Gate | Result |
|---|---|
| FP32 operators, corrections, traversal and streams | 1,558 passed; 49 skipped |
| Focused FP64 gradient/divergence/CRK/correction checks | 50 passed; 18 skipped; 222 deselected |
| CPU derivative gate, including native finite differences | 77 passed; 41 deselected |
| Benchmark/resource-parser contracts | 15 passed |

These are focused gates, not a rerun of the complete repository suite.

Next shared targets are reducing live correction state where measured and
removing repeated grid lookup/stencil work. `checkOffset` recomputes query-cell
state and scans earlier offsets to prevent periodic duplicates; any fast path
must retain the small-periodic-grid alias guard and enlarged-stencil behavior.
Those algorithmic traversal changes were not mixed into this register pass.
MFM/MFV end-to-end qualification and architecture cleanup remain active.

## Evidence and reproduction

Raw timings, saved outputs, NCU reports/CSV, compiler artifacts, source snapshots,
commands and qualification logs live outside Git at:

`../warpSPHDevelopment/core/experiments/core-register-pass-2026-10-21/`

The archive manifest hashes the evidence. Probe reports include source hashes,
precision, GPU/runtime versions and source-root identity. Use an isolated source
root containing `src/warpSPHCore` to compare revisions:

```bash
python scripts/benchmarks/probe_register_variants.py \
  --source-root /path/to/checkout --dim 3 --n 262144 \
  --operations density gradient covariance crk_moments \
  --traversals adjacency grid --lanes 1 32 \
  --variants runtime traversal policy lean cap64 --output /path/to/results.json

warpSPHCore_PRECISION=float64 python scripts/gradcheck/gradcheck_crk_moments_dimensions.py
```

Use `--profile-range` with NCU `--profile-from-start off`; counters and profiler
durations are kept separate from the unprofiled repeated timing samples.
