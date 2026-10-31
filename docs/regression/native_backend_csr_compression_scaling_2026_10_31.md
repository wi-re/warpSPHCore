# Compressed particle lists: prior work and scaling to 16M

The current byte codecs are useful conservative GPU prototypes, but their storage efficiency is below the published small-gap codecs. Extending the actual interpolation measurements to 16 million particles confirms the previous dense 3D throughput ranking: CSR32 remains faster; compression principally saves memory. No native kernel, Warp kernel, derivative implementation or production dispatch was changed in this extension.

## Relevant prior work

**Band, Gissler and Teschner (2019 online / 2020 journal)** compress exact sorted particle lists. They keep the first ID uncompressed and encode gap-minus-one: original gaps 1/2 need only two control bits, gaps 3..256 use one payload byte, and larger gaps use four. Their authors' version reports 0.851 B/neighbor at 29.88 neighbors and 0.579 B/neighbor at 255.52 neighbors; the reported roughly 1% overhead concerns a complete CPU DFSPH step, not one GPU interpolation. [Authors' manuscript, Sections 3.1/5.2 and Tables 3/5](https://cg.informatik.uni-freiburg.de/publications/2019_CGF_CompressedNeighbors.pdf), [journal record](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.13890).

**Stream VByte** separates control/data streams and stores two length bits plus 1..4 data bytes per integer. Its CPU SIMD decode-rate results are not GPU interaction throughput. [Lemire, Kurz and Rupp](https://arxiv.org/html/1709.08990).

**Thaler and Keller (2026 preprint)** combine shared cluster lists with nibble coding: one bit for a unit gap and special codes for small gaps. They report 3.6 B/particle on GH200 and 2.4 B/particle on MI300A at roughly 200 neighbors, with typical 1-5% compression overhead versus uncompressed clustered lists. Their GH200 results also show full uncompressed lists outperforming clusters in some regimes. These are shared conservative cluster lists, not independently stored exact particle lists; bytes/particle cannot be compared directly with bytes/neighbor. [Paper, Sections II-B/III-C/III-D/V](https://arxiv.org/html/2602.19873v1).

Our recent cell-relative prototype still stores every exact neighbor as a byte plus range descriptors; it does not share one list across many query particles. Approaching clustered-list storage requires the cluster approach explored earlier, in addition to a better integer codec. Different neighbor counts, cluster/super-cluster sizes, periodic metadata and symmetry policies also affect any cross-study comparison.

## Same-input storage comparison

To remove scene differences, `model_sizes.py` generates the same Morton-sorted FP32 3D fixtures as the timing study, samples 32,768 query rows uniformly without replacement (seed 51031), and searches those queries against the full reference state. Every exact ID in each sampled row contributes to the offline formulas. No scan is inserted into interpolation.

All following estimates include the same 12 B/row metadata, a uint32 initial ID where applicable, control/payload byte rounding and nibble segment masks. Our implemented chunk32 model additionally includes code padding, independent restart bases and payload directories. The row-stream models carry the previous ID between segments within a row; they do not retain independent random access to every 32-entry chunk. The nibble formula is applied to full particle lists, not shared cluster lists.

| Format / storage model, 3D n_h=4 | B/neighbor, 1M particles | B/neighbor, 16M particles |
|---|---:|---:|
| Implemented independent chunk32 layout | 1.429 | 1.427 |
| Row-stream byte escapes | 1.119 | 1.117 |
| Stream VByte model | 1.324 | 1.323 |
| Band-style model | 0.524 | 0.522 |
| Nibble model on full lists | 0.401 | 0.401 |

Band and nibble numbers are **storage estimates, not implemented or timed GPU consumers**. The implemented chunk32 model predicts the full measured allocation within 0.1% in both dense fixtures, providing a check on the sampling/layout accounting. It is not a throughput prediction for any other codec.

About 76.8% of consecutive gaps are one and 83.3% are one or two. Our codec spends a byte on each, while special coding can use fewer bits. Row restarts, directories and padding also contribute to the difference. The current dense byte layout uses about 1.43 B/neighbor; the same-data models suggest roughly 0.52 B with Band-style coding or 0.40 B with full-list nibble coding. At 16M, this projects to about 2.25 GB / 1.73 GB, compared with the implemented byte format's measured 6.14 GB. Decoder dependency and register costs remain to be measured before claiming a speedup.

At n_h=2.57, corresponding Band/nibble estimates are about 0.81/0.70 B/neighbor. All model results and assumptions are retained in `codec-models.json` and the source. No published cluster-storage figure is substituted for an exact-list estimate.

![Codec storage models](/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-csr-compression-scale-20261031/codec-size-models.png)

## Measurements beyond 1M

N=1M/2M/4M/8M/16M, dimensions 1/2/3, n_h=4/2.57, FP32/FP64: **60 canonical fixtures**. Kernels, native/planner shared objects and schedules remain pinned from the preceding study; schedules were screened at 262K and are not retuned here. Each fixture runs in a fresh process to avoid retaining earlier large topologies. Nine rotated batches use a native64 pilot to choose 1..100 calls, targeting about 30 ms per batch. Actual iteration counts/calibration samples are recorded. Planning is measured once per representation here and is not treated as a precise new construction-cost benchmark.

At **16M particles, dense 3D n_h=4 FP32**, there are **4,305,491,568 directed neighbors**, averaging 269.093 per particle. This exercises edge offsets beyond uint32 and byte-code buffers larger than 4 GiB while particle IDs still fit int32.

| Format | Allocating call ms | Matched resident call ms | Retained adjacency GB |
|---|---:|---:|---:|
| CSR64 | 25.914 | 25.571 | 34.636 |
| CSR32 | 18.841 | 19.376 | 17.414 |
| Byte deltas | 21.674 | 20.527 | 6.140 |
| Cell-relative bytes | 20.740 | 21.041 | 7.342 |

Allocating raw controls are production native/Torch calls. Allocating compressed calls use the prototype wrapper. Private resident controls share the prototype's packing, arithmetic and C++ launch path, with independently screened schedules. Includes field packing and output work; excludes sorting, search, CSR32 conversion and compression. GB is decimal and includes actual retained row metadata/padding; state/field/output/scratch are excluded consistently.

Dense 3D FP32 allocating-call scaling:

| Particles | Native/Torch CSR32 ms | Byte deltas ms | Cell-relative bytes ms |
|---|---:|---:|---:|
| 1M | 1.186 | 1.367 | 1.360 |
| 2M | 2.431 | 2.675 | 2.663 |
| 4M | 4.775 | 5.338 | 5.233 |
| 8M | 9.147 | 10.890 | 10.743 |
| 16M | 18.841 | 21.674 | 20.740 |

Throughput is approximately flat across these sizes. At 16M the compressed medians take about 15%/10% longer than production CSR32, while using about 2.84x/2.37x less adjacency memory. FP64 dense 3D allocating calls are about 251 ms for CSR32, 253 ms for deltas and 259 ms for cell bytes with their frozen speed-first schedules. Their FP64 retained storage is 9.15/8.80 GB because those schedules use shorter chunks than FP32; the ID format itself is precision-independent.

Larger size does not make the current compressed variants overtake CSR32 in dense 3D. Some 1D/2D byte-code configurations remain beneficial, but the graphs retain all cases instead of claiming a uniform win. Small public-call gains also include wrapper/scheduling differences; matched resident controls are provided separately.

![FP32 matched throughput](/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-csr-compression-scale-20261031/throughput-float32-matched.png)

![FP32 allocating throughput](/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-csr-compression-scale-20261031/throughput-float32-public.png)

![FP64 matched throughput](/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-csr-compression-scale-20261031/throughput-float64-matched.png)

![FP32 memory at 16M](/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-csr-compression-scale-20261031/storage-float32.png)

Throughput uses actual directed counts, linear zero-based y-axes and observed batch ranges. FP32 large 1D coordinates in a unit box are quantized: at 16M mean counts are 9.302/5.963 for n_h=4/2.57, versus FP64's 9.000/6.139. Precision modes therefore do not have exactly identical 1D neighbor graphs.

## Cache and memory traffic at 16M

This GPU reports 128 MiB L2. The packed FP32 3D reference array alone grows from 32 MB at 1M to 512 MB at 16M, crossing that capacity. A successful hardware-counter-only Nsight run measured the following traversal traffic at 16M:

| Traversal | DRAM GB | L2 GB |
|---|---:|---:|
| csr64 | 35.871 | 71.110 |
| csr32 | 18.586 | 53.094 |
| delta | 7.265 | 22.371 |
| cell | 8.473 | 24.655 |
| native32_public | 19.175 | 56.765 |

Traffic falls with compression, but that reduction does not yield the fastest traversal. The 16M traffic is broadly proportional to the earlier 1M counters, consistent with Morton ordering keeping reference reuse local even when the full arrays exceed L2. This is an interpretation of the observed scaling, not a claim that all data remains cached. Register/instruction costs from the previous study are unchanged by this extension.

The initial full-counter Nsight run hit CUDA launch timeout 702 during SASS instrumentation/replay and was discarded. Its partial reports/diagnostics remain in `profiling-full-failed`. The hardware-only run completed in one pass per kernel. Reported ordinary-call timings come from the separate unprofiled runs, not from profiler replay.

## Correctness and reproducibility

All eight timed paths in all 60 canonical fixtures pass output checks against native CSR64 before timing. The reused immutable kernel artifact is the same one that passed the previous 3,860-case codec/numerical suite and CUDA memcheck/racecheck; no CUDA code was changed or rebuilt. The offline models have hand-counted consecutive-row and 1/2/3/254/255/256/257 gap-boundary checks.

The initial larger 1D trial inherited the toy key encoder's 21-bit limit. Canonical 1D fixtures now use exact stable coordinate order, which is the 1D Morton order, and support-sized cells without that limit. Only those 20 fixtures were rerun. Superseded JSON/logs remain in `initial-1d-gridcap`; they are excluded from aggregates and plots. Dimensions 2/3 retain the original refined Morton planner.

Every canonical fixture has an independent PID, library/source hashes, all event/wall samples, calibration, actual counts, retained storage by tensor and GPU activity snapshots. No competing compute process was observed. Desktop/system variability is not excluded; plots preserve the observed ranges. The peak Torch-managed allocation in the dense 16M cases was about 70 GB FP32 / 78 GB FP64 while several representations coexisted; native search scratch/desktop allocations are not included in that counter. The prototype still constructs CSR before compression, so retained representation size is not construction peak memory.

Native/prototype libraries are unchanged from the earlier archive. The controls and prototype are copied into this archive for reproducibility. Script/source snapshots, author manuscript, raw Nsight exports, failed-profile diagnostics, CSV, model inputs and seven figures are retained under:

`/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-csr-compression-scale-20261031`

The next focused codec experiment is a unity/small-gap representation on exact lists. It should measure GPU decoding and staging/register costs rather than assuming that the modeled storage saving becomes the same throughput gain.
