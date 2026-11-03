# Native execution scaling through one million particles — 2026-10-30

The corrected comparison covers **54 workload shapes and 1,674 timing
records**, on CPU and CUDA in 1D/2D/3D. It measures fresh neighborhood search and
all seven core SPH primals through Warp/Torch, native/Torch, standalone allocation,
and standalone output reuse. Every measured case passed neighbor-set and float64
primal parity checks before timing.

The result is presented as scaling graphs. Generated figures, the graph report
`graphs.md`, CSV and complete raw samples live in the development archive at
`core/experiments/native-backend-20261030-million/`, relative to
`WARPSPH_DEVELOPMENT_ROOT`. The local shared root is
`/home/lu26029/dev/warpSPHDevelopment`. The key figures are `search.png` and
`interpolate.png`; `operators-cpu.png` and `operators-cuda.png` include all seven
operators. Curves are medians, shaded bands are sample min/max, and both axes are
logarithmic. No fitted or extrapolated timings are shown.

## Execution design correction

The earlier comparison included repeated particle-content and CSR verification in
the native execution path. That was inappropriate runtime work and made the
comparison unfair. The machinery has been removed, including the validation enum,
context field, debug switches, verification hooks, host-copy utilities, stale-index
rescans, stored reference keys and recount-status buffers. There is no debug
parameter carried by an operator. Entry points retain dispatch metadata checks.
Optional diagnostics live independently under `tests/verify` and are never imported
by the execution library.

The caller owns data validity and index invalidation. Reference motion across
cells, domain changes or support growth beyond the indexed bound requires an
explicit rebuild. Search construction still reduces support bounds and returns
the edge total needed for allocation; these are required algorithm steps.

## Interpolation and compiler concern

At 32,768 particles in 3D, CUDA interpolation now measures
0.392 ms for Warp/Torch,
0.531 ms for native/Torch,
0.547 ms for allocating standalone,
and 0.523 ms with standalone
output reuse. The original roughly tenfold checked-call gap disappears.

The standalone CUDA build already uses `-O3` and architecture `sm_120`; the Torch
JIT uses `-O2` and the same GPU target. No fast-math change was needed. A focused
same-input comparison at this size found Warp interpolation at
0.501 ms with one lane per particle, versus
0.373 ms with its default 32 cooperative lanes.
The native one-thread-per-particle implementation took
0.497 ms. This points to the existing cooperative
neighbor-loop optimization as the main remaining difference at that size, rather
than a missing optimization-level flag. These are completed-call timings, not
isolated GPU instruction profiles.

At one million particles in 3D, interpolation measures
2.989 s for Warp on CPU versus
1.065 s for native/Torch and
1.259 s for standalone
reuse. On CUDA the corresponding times are
10.941,
9.227, and
9.179 ms.
The complete dimension and operator comparisons are in the graphs.

## Workload and timing contract

The particle counts are 1, 16, 256, 4,096, 32,768, 131,072, 262,144, 524,288, 1,000,000. Uniform random points
occupy a periodic unit box. Support radii target 32 other particles plus self,
so the mean degree is about 33 at the larger sizes: approximately 33 million
directed CSR entries at one million particles. Supports vary slightly between
particles. Tiny cases have a capped radius and fewer neighbors. This is constant
neighbor-count scaling across dimensions, rather than a fixed support-to-spacing
ratio or a full production resolution sweep.

All paths use float64 particle/field data, int64 offsets/indices and int32 counts.
Operators consume an identical sorted CSR with Wendland2, Gather, Difference,
Brookshaw and TrueAllToToAll. Fields are vectors; density and covariance keep their
usual scalar/matrix outputs. 1D curl is a degenerate zero operation.

Search includes fresh index construction, count, scan, allocation and collection.
Warp uses fused collection, input query order, N buckets, no COO rows and no spare
capacity. Operators exclude search. Standalone iteration loops run in C++ without
Torch or Python calls. Fixtures and descriptors are prepared outside timing.
Standalone allocation uses malloc/cudaMalloc, while Torch uses its own allocator;
output reuse is a separately labeled integration policy.

Small cases use five measured batches of ten operator calls after two warmup
batches. Extended cases use three measured single-call batches after one warmup.
Search batches contain one call. Compilation, input construction, parity checks
and CSR canonicalization are excluded. Completed wall time includes final stream
completion. Raw per-record samples, iteration counts, warmup counts and per-run
source/build provenance are retained. No numerical or CSR verification is timed.

The machine is an Intel Core Ultra 9 285K and RTX PRO 6000 Blackwell Workstation,
with Python 3.13.14, Torch 2.13.0+cu130, Warp 1.17.0 and CUDA 13. Torch CPU threads
and standalone OpenMP rows use four threads. CPU models differ: the Torch JIT
uses `-O2 -march=native`, standalone uses portable Release `-O3`, and Warp uses its
ordinary CPU execution. CPU affinity and clocks are not fixed. Warp CUDA keeps
its per-operator defaults: 32 lanes except curl/Laplacian at one lane; native uses
one thread per particle.

Desktop CPU/graphics activity continued. 35 records have a worst sample over
1.5 times their median; raw ranges remain visible. Close differences are not
isolated throughput certifications. These measurements do not establish AMD,
H100 or GH200 performance. HIP remains compile/link qualified for gfx942/gfx1100.

## Qualification and reproduction

After removing runtime verification, the native Torch CPU/CUDA matrix passed
2,559 tests, standalone CPU/CUDA independent oracles passed 1,512, all six C++
CPU/CUDA tests passed, and the pure CPU build passed all four C++ tests. The final
host-helper cleanup passed 233 focused adapter/trace tests. A trace regression
checks that normal operator calls launch no verification reductions or scalar
reads. Separate verifier tests passed independently. AMD execution is unmeasured.

Use `scripts/benchmarks/bench_backends.py` and the optional CMake timing bridge;
commands are in `scripts/benchmarks/native/README.md`. The graph generator accepts
multiple non-overlapping compatible runs, retaining per-run provenance:

```bash
python scripts/benchmarks/analyze_backends.py small-results.json large-results.json \
  --output-dir /path/to/scaling-results
```

The previous verification-heavy report is retained only as superseded evidence.
