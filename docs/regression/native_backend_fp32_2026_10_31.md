# Native FP32 GPU exploration, 2026-10-31

Starting checkpoint: `0560de5`, branch `codex/native-backend`, isolated worktree
`/home/lu26029/.codex/worktrees/native-backend/warpSPHCore`.
The goals were to explain the narrow FP32/FP64 interpolation ratio, improve
FP32 operators and compact-hash radius search, and qualify the resulting paths
without changing Warp's readable numerical reference or the public API.
Native/Torch is the primary CPU/GPU architecture. This pass does not add a
second optimized framework-free runtime.

## Result

The following are completed public-call wall-time medians in milliseconds:
one million particles, 2D, approximately 33 neighbors per particle. Native times
include output and temporary allocation and reference preparation. Search builds
a fresh index and CSR, including count, scan, collection and row sorting.

| Operation (FP32) | Checkpoint | Final | Warp | Speedup over checkpoint |
|---|---:|---:|---:|---:|
| Search | 6.149 | 2.784 | 5.103 | 2.21× |
| Density | 1.043 | 0.332 | 0.995 | 3.14× |
| Interpolate | 1.487 | 0.332 | 1.493 | 4.48× |
| Gradient | 1.691 | 0.350 | 1.518 | 4.83× |
| Divergence | 1.642 | 0.374 | 1.492 | 4.39× |
| Curl | 1.686 | 0.358 | 1.647 | 4.71× |
| Laplacian | 1.974 | 0.377 | 1.771 | 5.24× |
| Covariance | 1.231 | 0.334 | 2.107 | 3.68× |

| Operation (FP64) | Checkpoint | Final | Warp | Speedup over checkpoint |
|---|---:|---:|---:|---:|
| Search | 11.866 | 9.450 | 11.494 | 1.26× |
| Density | 2.116 | 2.127 | 7.092 | 0.99× |
| Interpolate | 2.685 | 2.661 | 7.950 | 1.01× |
| Gradient | 3.181 | 3.203 | 11.772 | 0.99× |
| Divergence | 3.006 | 3.011 | 10.795 | 1.00× |
| Curl | 3.009 | 3.000 | 9.076 | 1.00× |
| Laplacian | 3.478 | 3.473 | 15.886 | 1.00× |
| Covariance | 2.937 | 2.910 | 21.570 | 1.01× |

Large-case FP32 operators now exceed the contemporary Warp reference in this
fixture. The FP64/FP32 interpolation ratio increases from 1.81× to 8.01×.
FP64 SPH operators keep the checkpoint implementation; the radius-search
change applies to both precisions. These are measurements on one GPU and
particle distribution, not universal speedup guarantees.

Graphs and all sample ranges are in the external development archive:
`core/experiments/native-fp32-exploration-20261031/paired-million.png` and
`fp32-scaling.png`. `paired-summary.csv` contains all 80 matched native cases.

## Why FP32 had little advantage

Nsight Compute on the unchanged checkpoint's 2D vector interpolation found
FP32 L2 throughput at 95.27% of peak, DRAM throughput at only 13.40%, and a 95.30%
L2 hit rate. Repeated scalar gathers, address instructions and reference-side
work consumed the advantage of faster arithmetic. A large cache hit rate alone
was not evidence that memory access was cheap.

On the same million-particle fixture, the packed FP32 traversal reduces L2 read
sectors from 205,613,716 to 44,493,643 (4.62× fewer) and memory instructions from
735,948,130 to 340,988,885 (54% fewer). Its separate preparation pass reads another
784,284 L2 sectors. Traversal duration falls from about 1.43 ms to 0.299 ms;
preparation takes 0.018 ms. Those are profiler kernel times, distinct from the
public-call wall times above. The lower traversal L2 hit rate (78.75%) reflects
eliminated redundant traffic, while L2 throughput remains high (92.98%).

The FP64 checkpoint traversal takes about 2.64 ms with 49.61% L2 throughput and
85.41% SM throughput. These counters explain why the nominal 64:1 arithmetic
throughput ratio does not predict this workload's timing ratio. SM throughput
is not a direct measurement of achieved algorithmic FLOPs. Unsupported or
misnormalized profiler percentages above 100% were not used in this diagnosis.

## Promoted implementation

### Packed reference records for FP32 primals

An O(Nreference) preparation kernel gathers positions and coefficients into
aligned float4 records. Density stores mass; interpolation stores field times
mass/density; derivative operators store volume and the original reference
field. Query fields and geometry are cached per cooperative row. Eight lanes
share each query. Neighbor-index width is resolved at launch, preserving 32/64-bit
CSR indices and offsets in every combination.

The derivative operators reuse the existing pair equations through a packed
argument adapter. The manual backward and adjoint sources are unchanged. No
approximate math or fast-math compiler option is introduced. Moving reference
arithmetic outside the edge loop may change floating-point rounding, so parity
is tested numerically rather than promised bit for bit.

Selection uses tensor metadata only: FP32, Wendland2, Gather support,
TrueAllToToAll filtering, and at least 32,768 query particles. Derivative operators
also require Difference mode, and Laplacian requires Brookshaw. 1D zero curl
keeps its original path. Cases with empty, excessively large, disproportionately
large, or too sparsely used reference populations retain the general traversal.
The reference address range is bounded before selecting narrower address math.
Other kernels, schemes, precisions and CPU execution retain their existing path.

The prepared storage is per call and uses the current Torch device and stream.
There is no persistent cache or caller-managed lifetime. It adds one allocation
and preparation launch. At one million reference particles, the temporary cost
is 16 MB for 2D density/interpolation/covariance and 32 MB for 2D vector derivative
operators. In 3D, density/covariance need 16 MB, vector interpolation/derivatives
32 MB; matrix interpolation needs 48 MB and matrix derivatives 64 MB.
The local GPU binary grows from 34,990,936 to 38,298,072 bytes (+9.45%).

The initial 1,024-query cutoff regressed 1K–4K calls by about 2–4 microseconds.
The cutoff was raised to 32K after a dedicated 256/1K/4K/16K sweep; those rejected
results are retained. Below the cutoff the single-pass path avoids the additional
launch and scratch cost. The cutoff is an empirical choice for this hardware,
not a device-independent optimum. Reusable workspaces or capture could change
that tradeoff, but are outside this pass. The final small-case sweep shows
operator ratios of 0.94–1.12× over the checkpoint, without the systematic
2–4 microsecond packing penalty.

### Four-lane, cell-ordered compact-hash search

Four lanes split each candidate cell's points. Ballot/popcount prefixes assign
accepted output slots without atomics or a second edge array. Self searches visit
queries in the index's existing cell order and map output rows back to original
IDs. Distinct query/reference populations retain input query order. The existing
minimum-image geometry, strict distance/support predicate and ascending original
neighbor-ID contract remain in force. Count and collect share the same traversal.

There is no new index sort, permutation buffer or public scheduling parameter.
The existing final CSR row sort is retained. These changes apply only to the
native/Torch GPU adapter; the standalone GPU executor remains compatible with
shared headers but uses its previous traversal. CPU search is unchanged.

## Experiment design and rejected candidates

The private interpolation harness isolates metadata specialization, narrower
indices, coefficient preparation, loop unrolling, vector loads, subgroup width
and block size. It uses a single independently qualified FP64 fixture/CSR for
both precisions, with no particle-content checks in timed execution. It rotates
candidate order and retains all samples. Private scratch/output allocation is
excluded, so promotion is judged using the ordinary public API instead.

Clean FP32 screening found approximately 1.42–1.47 ms for the checkpoint-like
row, 1.22 ms with fixed metadata, 0.94 ms with vector loads, 0.68 ms with weighted
field preparation, 0.47 ms for vector loads plus weighted preparation, and
0.285 ms for packed position/weighted-field records. The latter includes its
preparation launch. Width eight and block size 128 were retained. FP64 packing
showed no compelling benefit and was not promoted. Unrolling and wider groups
were screened rather than enabled indiscriminately.

The archive retains contended FP64 private screening and intermediate artifacts;
those runs do not support FP64 speedup claims. Early public-wrapper runs imported
the package before selecting CSR64, silently choosing the then-unoptimized CSR32
fallback. They are explicitly quarantined in `superseded-csr32-wrapper/` and
excluded from every result above. The wrapper now sets and asserts the intended
dtypes before execution. Both CSR widths are supported by the final implementation.
Pre-cutoff and pre-index-width results and their compatible binaries are retained.

## Measurement scope

- RTX PRO 6000 Blackwell workstation GPU, sm_120; Core Ultra 9 285K.
  Python 3.13.14, Torch 2.13.0+cu130, Warp 1.17.0. Native builds use `-O2`,
  CPU `-march=native`, GPU architecture 12.0, without fast math.
- Paired checkpoint/final artifacts are loaded in fresh processes and hashed.
  FP32 covers 2D/3D; FP64 covers 1D/2D/3D; both cover 32,768 and 1,000,000
  particles. Each stage contains 160 records across Warp and native/Torch.
- Seven measured batches follow two warmup batches. Operators use ten calls per
  batch; search uses one. Fixtures, compilation, transfers and independent
  qualification are outside timing. Operators use the same sorted CSR.
- The user supplied an idle GPU window. Every final activity snapshot shows no
  other compute process. Snapshots cannot guarantee exclusive ownership between
  samples; desktop activity, clocks and scheduler noise remain. Min/max whiskers
  show observed sample ranges, not confidence intervals.
- FP32 search agreement uses the existing cutoff-rounding qualification,
  not bitwise CSR identity: differing pairs must lie within the documented
  FP32 boundary band. This fixture has 32,996,277 native FP32 edges versus
  32,996,295 native FP64 edges. This qualification is outside timed execution.
- High-resolution 1D FP32 remains excluded for the previously documented cutoff
  rounding/conditioning issue; 1D FP64 and small correctness fixtures are covered.
- A reversed-order 32K FP64 recheck used eleven batches of fifty calls after some
  original ranges overlapped. Density/gradient/curl/Laplacian ratios were
  0.995/1.007/1.022/0.997. The original full sweeps are preserved unchanged;
  this did not establish a persistent FP64 regression.
- CPU behavior is regression-tested rather than rebenchmarked. AMD support is
  compile-qualified only. Other CUDA architectures need independent tuning and
  measurement; the fixed threshold and subgroup choices may not be optimal there.

## Correctness and build qualification

- Final CPU/CUDA native Torch suite: **3,023 passed, 589 skipped**; 18 existing
  Torch JIT deprecation warnings. Separate Python-scaffolding suite: **19 passed**.
- Independent FP64 Torch equations qualify all seven packed primals and manual
  VJPs in dimensions 1/2/3, scalar/vector/matrix fields where defined, every
  index/offset width combination, periodic images and padded/empty rows. Tests
  force the final 32K dispatch regime while bounding oracle edge work.
- Ragged interpolation additionally covers support/filter fallbacks and gradients;
  nondefault-stream tests exercise scratch lifetime. Existing radius-search
  tests cover exact CSR parity, periodic minimum-image geometry, distinct
  populations, support schemes and sorted-row thresholds.
- CPU CMake compatibility: **4/4 CTest tests passed**. CUDA CMake compatibility:
  **6/6 passed**, including standalone GPU and compact-index checks.
- Offline HIP compilation instantiates packed operators, cooperative ballot
  search, dimensions 1/2/3, index widths and both reordered/Gather variants for
  **gfx942 and gfx1100**. Compilation passes with loop-unrolling warnings.
  This does not qualify a linked ROCm wheel or AMD runtime behavior.
- Warp kernels and backward/adjoint source files have no changes. No CSR scans,
  particle-content validation, CPU transfers for verification or public debug
  arguments were introduced into operation execution.

A new deliberately ill-conditioned FP32 Laplacian VJP stress geometry exposed
four pointwise tolerance failures near close pairs. The untouched checkpoint
reproduced the same four failures and worst errors (largest absolute error
0.00849; one small-valued component had relative error 0.00136). The original
fixture, checkpoint log and optimized log are retained in
`conditioning-diagnostic/`. This is a pre-existing numerical limitation, not a
resolved issue. The regular packing oracle uses separated clouds at unchanged
tolerances to test packing without that conditioning confounder. Existing
regression tolerances were not loosened.

## Reproduction and evidence

Run instructions and the private diagnostic sources are in
`experiments/native_gpu/README.md`. Full commands, activity, raw samples,
correctness results, artifact hashes, profiler CSVs/traces, build flags, compile
harnesses and final source snapshots are retained under:

`/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-fp32-exploration-20261031/`

`paired-graphs.md` indexes the figures; `SHA256SUMS.json` inventories the archive.
The final report uses the complete last paired sweeps rather than selecting
individual minima from successive runs. Historical run paths in JSON identify
the original locations; preserved binary copies in the corresponding historical
subdirectories should be matched by SHA-256.
