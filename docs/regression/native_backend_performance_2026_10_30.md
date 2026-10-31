# Initial native backend performance pass — 2026-10-30

**Superseded:** these measurements included inappropriate per-call content/CSR
verification. They do not represent the corrected normal execution API. The
original raw data is retained as historical evidence; use the corrected production
comparison for performance conclusions.

The standalone core removes substantial calling overhead for small problems.
Native execution currently gives mixed throughput results: CPU search and 3D
SPH improve, while larger CUDA search and SPH still need optimization. A native
Torch call does not automatically run faster than the existing Warp path.

The completed pass contains **1,140 qualified timing records**: CPU and CUDA,
1D/2D/3D, N = 1/16/256/4096/32768, fresh compact-hash search and all seven core
primals (density, interpolation, gradient, divergence, curl, Laplacian,
covariance). Every neighborhood comparison passed exactly after canonicalizing
rows. Native Torch and standalone primal outputs agreed with Warp at float64
`rtol=atol=3e-9`. Both checked and metadata-only standalone calls were qualified
before timing. This pass makes no numerical implementation changes.

## Scope and reproducibility

The machine was an Intel Core Ultra 9 285K with an NVIDIA RTX PRO 6000 Blackwell
Workstation GPU. Python 3.13.14, Torch 2.13.0+cu130, Warp 1.17.0, CUDA nvcc 13,
and CUDA architecture 120 were used. Torch CPU threads and standalone OpenMP
threads were both set to four; CPU affinity was not fixed. These implementations
use different CPU parallelism models. The native Torch JIT uses `-O2` and
`-march=native`; the standalone Release build uses `-O3` and portable CPU code.
This compares the present integration paths, including those build choices.

Inputs are deterministic uniform random points in a periodic unit box. Supports
produce approximately 32 neighbors per particle once N is large enough, with a
small per-particle variation. At N=32768 the three dimensions contain
1.081–1.084 million directed CSR entries, or about 33 neighbors per particle.
All paths use float64 data, int64 CSR offsets and indices, and int32 counts.
Operators consume an identical sorted CSR, vector fields, Wendland2, Gather,
Difference, Brookshaw and TrueAllToToAll. Density/covariance retain their usual
scalar/matrix outputs; 1D curl is a degenerate zero operator.

Warp CUDA uses its existing per-operator lane defaults: 32 for density,
interpolation, gradient, divergence and covariance; one for curl and Laplacian.
The native GPU implementation uses one thread per particle. This is a comparison
against the current Warp path, including its existing cooperative kernels, rather
than an ablation with the same thread mapping. The CPU Warp path uses its ordinary
non-tiled launch.

The search includes fresh index construction, count, scan, output allocation and
collection. Warp uses fused collection, input query order, no COO rows, no spare
capacity and N hash buckets. Native search uses N buckets as well. Search is
excluded from operator timings.

Five measured batches follow two warmup batches. A batch contains ten operator
calls or one search. Completed wall time includes final stream completion;
compilation, fixture construction, correctness checks, descriptor setup and CSR
canonicalization are excluded. Standalone iterations run inside C++ and call no
Torch/Python APIs. Torch owns the equivalent resident fixture memory outside the
measured C++ interval. The timing bridge and core shared library link neither
framework.

Three standalone policies are separate: full content checking with fresh output
allocation, full checking with reused output, and metadata checking with reused
output. The last policy assumes validated retained inputs. Fresh standalone GPU
allocation uses cudaMalloc/cudaFree; Torch uses its caching allocator. The
metadata/reused result changes more than the language boundary. The checked/reused
series helps distinguish allocation from repeated validation.

Raw samples, source hashes, binary hashes, activity snapshots, build configuration,
CSV, tables and four scaling figures are in the development archive:
`core/experiments/native-backend-20261030/`, relative to
`WARPSPH_DEVELOPMENT_ROOT`. The measured harness is preserved there as well. On the
local machine the shared root is `/home/lu26029/dev/warpSPHDevelopment`.
A preliminary run with mixed CSR widths was stopped, retained as incomplete, and
excluded from all conclusions and figures.

## Small-problem latency

N=1 is a self-neighbor probe: it bounds calling overhead and includes minimal
computation. Across all seven operators and dimensions, CPU medians were
114–157 µs for Warp/Torch, 55–67 µs for native/Torch, 0.40–0.49 µs for checked
standalone allocation, and 0.28–0.32 µs for prepared standalone reuse. These are
steady C++ loop timings and do not include a process launch or Python/C boundary.

A less degenerate example is **3D density, N=16**, with 88 directed neighbor
entries. Times below are median **microseconds per completed call**:

| Device | Operation | Warp / Torch | Native / Torch | C++ checked + allocate | C++ checked + reuse | C++ metadata + reuse |
|---|---|---:|---:|---:|---:|---:|
| cpu | density | 150.67 | 95.17 | 6.21 | 6.20 | 5.98 |
| cuda | density | 154.92 | 375.82 | 160.50 | 112.33 | 51.19 |

Here checked standalone CPU is about 24× faster than Warp; GPU metadata/reuse is
about 3× faster. Checked standalone GPU with fresh allocation is approximately
level with Warp. Validation and allocator policy therefore materially affect
whether a small native GPU call improves latency.

## Larger-problem throughput

For **3D, N=32768**, times below are median **milliseconds per completed call**:

| Device | Operation | Warp / Torch | Native / Torch | C++ checked + allocate | C++ checked + reuse | C++ metadata + reuse |
|---|---|---:|---:|---:|---:|---:|
| cpu | search | 458.32 | 64.11 | 94.43 | — | — |
| cpu | density | 26.16 | 13.86 | 13.45 | 13.40 | 11.95 |
| cpu | gradient | 52.65 | 15.29 | 15.06 | 15.39 | 13.72 |
| cuda | search | 5.63 | 11.27 | 11.05 | — | — |
| cuda | density | 0.33 | 0.88 | 2.75 | 2.67 | 0.46 |
| cuda | gradient | 0.62 | 1.20 | 3.25 | 3.14 | 0.69 |

The native Torch CPU search is about 7.1× faster than Warp in this case, while
standalone search is about 4.9× faster. Standalone is not always the fastest native
interface: the adapter and standalone paths perform different validation and
allocation work. CPU gradient improves by roughly 3.4–3.8× across native modes.
These comparisons also include the different CPU execution and compiler settings.

CUDA search depends strongly on dimension. At N=32768, standalone fresh search is
0.99 ms versus Warp's 1.69 ms in 1D; 2.16 versus 1.91 ms in 2D; and 11.05 versus
5.63 ms in 3D. The current native index/traversal is not a uniform GPU improvement.

For density/gradient at this size, even prepared standalone GPU remains slower
than Warp: 0.46 versus 0.33 ms for 3D density and 0.69 versus 0.62 ms for 3D
gradient. Fully checked standalone density takes 2.75 ms. Source inspection shows
that its content checks copy required floating inputs and the live CSR to host,
then scan those values; metadata mode skips that work. Native Torch validation
has its own device checks and synchronization costs. The measurements establish
that validation is expensive, but are not a per-stage GPU profile.

## Interpretation and next work

The small-problem goal is supported by this pass, especially for direct CPU calls
and GPU calls retaining validated buffers. Simply switching a Python call from
Warp to the current native Torch wrapper is insufficient to obtain that full
benefit, and currently regresses the illustrated CUDA SPH calls.

The next useful performance work is a retained validation/workspace contract
accessible from native Torch, followed by GPU validation summaries and allocation
reuse. Profiling the native 3D hash build/traversal and cooperative neighbor
processing should follow. The present GPU index uses a portable merge sort and
native operators use a serial neighbor walk within each particle's thread;
these are concrete candidates for investigation, not measured stage attribution.

No other GPU compute process appeared in the per-case snapshots besides the
benchmark itself. Desktop graphics and CPU activity continued. Twenty-six of
1,140 records had a worst batch above 1.5× their median; the largest such ratio
was 4.73×. Raw samples and min/max plot bands retain these outliers. This is an
initial scaling/latency pass, not an isolated throughput certification or a claim
about close percentage differences.

The timing bridge builds in CPU-only, CUDA and HIP configurations, and the
CPU-only bridge passed an independent smoke run. HIP compiled and linked for
both gfx942 and gfx1100; AMD execution and performance remain unmeasured. A final
HIP-only cleanup annotation suppresses an ignored-result warning; the measured
CPU/CUDA code is unchanged and the measured source is archived.

Reproduction commands and detailed timing contracts are in
[`scripts/benchmarks/native/README.md`](../../scripts/benchmarks/native/README.md).
The driver is [`bench_backends.py`](../../scripts/benchmarks/bench_backends.py);
[`analyze_backends.py`](../../scripts/benchmarks/analyze_backends.py) produces
CSV, plots and complete tables from the recorded JSON without running workloads.
