# Byte-compressed particle CSR experiments

Both proposed encodings now run in an isolated CUDA interpolation prototype: independent byte-delta chunks and cell-relative byte offsets with run bases. Production dispatch, Warp kernels, backward/adjoint implementations and the default CSR representation are unchanged.

The measurements support a memory/performance tradeoff, not a universal replacement for CSR32. Byte deltas improve some large FP32 1D/2D cases. In dense 3D both formats substantially reduce adjacency storage and beat CSR64, but remain slower than production CSR32. FP64 gains little speed from compressing IDs because its arithmetic pipeline is already busy.

## Dense 3D result

One million random particles, refined Morton order, periodic unit box, n_h=4, Wendland C2 Gather interpolation of a D-component field, FP32. Actual mean neighbors: 269.079094.

| Format | Retained adjacency MB | Allocating call ms | Matched private call ms |
|---|---:|---:|---:|
| CSR64 | 2164.6 | 1.599 | 1.663 |
| CSR32 | 1088.3 | 1.118 | 1.213 |
| Byte deltas | 384.4 | 1.294 | 1.275 |
| Cell-relative bytes | 460.4 | 1.257 | 1.269 |

MB is decimal and includes actual code padding, payload directories, bases/masks, int64 row directories and int32 counts. State/field/packing/output buffers are excluded for every representation. The earlier ~0.35 GB delta estimate excluded some padding/row metadata; the implemented representation actually retains **384.4 MB**. Cell-relative encoding retains **460.4 MB**. Against CSR32, these are approximately **2.83x** and **2.36x** smaller; against CSR64, **5.63x** and **4.70x** smaller.

Allocating raw controls are the production native/Torch operator. Allocating compressed prototypes use their experimental Python/ctypes wrapper and allocate packed-reference/output storage per call. Private controls use the same C++ launch loop, arithmetic, field packing and resident scratch/output as the prototypes; schedules are screened independently for each format. Production scheduling can differ from the private controls. Public/private differences must not be attributed solely to allocator cost.

The allocating compressed medians provide about **24%/27% higher throughput than CSR64**, while taking about **16%/12% longer than CSR32** for deltas/cell bytes. Observed batch ranges are retained; small differences with overlapping ranges are not presented as robust wins.

## Scaling and dimensions

The final scaling suite uses held-out N=32,768/131,072/524,288/1,000,000, dimensions 1/2/3, n_h=4/2.57 and both precisions. A separate N=262,144 screen chooses each schedule before these measurements. It explores chunk sizes 4/8/16/32, scalar or packed-four-code decoding, 128/256 threads per CUDA block and several batching factors. No schedule is retuned at the held-out sizes.

One-million-particle FP32 allocating calls:

| Dimension | n_h | Native/Torch CSR32 ms | Byte deltas ms | Cell bytes ms |
|---|---:|---:|---:|---:|
| 1 | 4 | 0.048448 | 0.039834 | 0.046400 |
| 1 | 2.57 | 0.044614 | 0.034682 | 0.038656 |
| 2 | 4 | 0.198554 | 0.189562 | 0.202541 |
| 2 | 2.57 | 0.114477 | 0.098464 | 0.105850 |
| 3 | 4 | 1.117773 | 1.294483 | 1.256736 |
| 3 | 2.57 | 0.387475 | 0.426214 | 0.439622 |

The 1D public-call gains include differences in wrapper/dispatch cost. Matched private controls still show byte-delta gains of about 14% higher throughput in 1D n_h=4 and 21% in 2D n_h=2.57 at one million particles. These results do not imply gains at every smaller size. Cell-relative encoding is less consistently beneficial.

![Matched FP32 throughput](/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-csr-compression-20261031/throughput-float32-matched.png)

![Allocating FP32 throughput](/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-csr-compression-20261031/throughput-float32-public.png)

![Matched FP64 throughput](/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-csr-compression-20261031/throughput-float64-matched.png)

All throughput axes start at zero. Whiskers show the observed batch range, not confidence intervals. Plots use actual directed neighbor counts. Main-suite native/compressed outputs pass numerical qualification against the native CSR64 output before timing.

Actual FP32 retained storage, using the fastest screened configuration for each format:

| Dimension | n_h | CSR32 MB | Deltas MB | Cell bytes MB |
|---|---:|---:|---:|---:|
| 1 | 4 | 48.12 | 43.59 | 48.88 |
| 1 | 2.57 | 36.66 | 34.98 | 46.97 |
| 2 | 4 | 217.01 | 124.00 | 136.01 |
| 2 | 2.57 | 99.02 | 63.89 | 91.83 |
| 3 | 4 | 1088.32 | 384.40 | 460.36 |
| 3 | 2.57 | 300.39 | 139.70 | 240.64 |

![FP32 retained storage](/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-csr-compression-20261031/storage-float32.png)

Short neighborhoods amortize headers poorly: cell-relative storage can exceed CSR32 in 1D. Speed-first screening also sometimes chooses short chunks in FP64, sacrificing compression for scheduling. Precision itself does not require a different ID representation.

## Fixed 32-entry memory-oriented configurations

To separate chunk sizing from precision, additional 3D runs hold compressed chunks at 32 entries. Batch/thread/vector choices still come from the independent screen, constrained to that chunk size. The FP32 dense result above already uses 32-entry chunks and is not remeasured.

| Precision | n_h | Encoding | Retained MB | Allocating compressed ms | Native/Torch CSR32 ms, same run |
|---|---:|---|---:|---:|---:|
| float32 | 2.57 | delta | 133.25 | 0.586 | 0.383 |
| float32 | 2.57 | cell | 179.65 | 0.572 | 0.383 |
| float64 | 4 | delta | 384.40 | 16.466 | 15.751 |
| float64 | 4 | cell | 460.36 | 16.545 | 15.751 |
| float64 | 2.57 | delta | 133.25 | 5.718 | 4.716 |
| float64 | 2.57 | cell | 179.65 | 5.705 | 4.716 |

Dense FP64 can retain the ~384/460 MB encodings with a roughly 5% call-time penalty relative to CSR32 in that run. Larger chunks hurt sparse 3D throughput more substantially. A fixed chunk size is therefore a memory/performance policy, not an obvious universal default.

## What Nsight shows

Focused profiles use one million 3D particles, n_h=4, the final screened schedules, no forced clocks and no cache flushing. These counters are for traversal alone; the timed calls above also pack reference values.

| FP32 traversal | DRAM GB | DRAM throughput % | Registers/thread | Achieved occupancy |
|---|---:|---:|---:|---:|
| csr64 | 2.255 | 95.0% | 61 | 63.3% |
| csr32 | 1.175 | 66.6% | 61 | 63.8% |
| delta | 0.453 | 22.9% | 80 | 46.6% |
| cell | 0.531 | 26.9% | 80 | 46.7% |
| native32_public | 1.168 | 71.0% | 64 | 64.6% |

Compared with production CSR32, deltas/cell bytes reduce measured DRAM traffic by approximately **2.58x/2.20x**. L2 traffic also falls. The smaller traffic is real, rather than just a smaller allocation whose decoder rereads the same data excessively.

The packed compressed kernels use 80 registers/thread versus 64 for production CSR32 and achieve about 47% occupancy versus 65%. Executed warp instruction counts are about 1.082/1.007 billion versus 0.941 billion. No local-memory spilling is reported. Long-scoreboard stalls fall, but delta decoding adds scan-related dependencies; shorter-register-lifetime record staging and cheaper metadata access are reasonable follow-up experiments. These counters support an interpretation that decoding/dependency/register costs replace much of the saved bandwidth bottleneck; they do not isolate a single cause.

FP64 profiles show approximately 84-85% FP64 pipe activity across raw and compressed paths. ID traffic decreases, but compression does not remove the dominant arithmetic work.

## Representation and planning cost

Deltas store an absolute uint32 first ID per chunk, byte gaps, and an escape code 255 with uint32 payloads. Every chunk has an independent payload directory entry. Cell-relative chunks store byte offsets, a run-start bitmask and uint32 source-range bases. Source cells with more than 256 particles are split into contiguous <=256-particle ranges; no neighbor is truncated. Each chunk independently identifies its range bases.

The packed variant loads four consecutive codes as one aligned uint32 per thread. Deltas scan four-entry totals across fewer lanes. Cell decoding reuses a range base within each four-entry group until its boundary mask changes. Both retain a scalar alternative and matched raw controls.

For dense FP32 3D, explicit conversion from existing CSR takes median wall times of about **3.41 ms for deltas** and **3.53 ms for cell bytes**, each based on three construction samples. This includes GPU count/fill passes, Torch prefix sums, allocation, sizing synchronization and the transient particle-base map for cell encoding; search and Morton sorting are excluded. Relative to the measured allocating CSR64 control, conversion amortizes after roughly **12/11 interpolation calls** respectively. Neither amortizes through a runtime saving against CSR32 in this dense case.

The prototype first builds ordinary CSR. The conversion peak therefore includes that CSR plus the new representation and temporary buffers. Interpolation keeps no CSR IDs, CSR offsets or particle-to-cell map, but integration into neighbor search would be needed to avoid the large CSR intermediate altogether. The prototype does not claim end-to-end simulation speedups.

All content verification and host readbacks live in planning/standalone verification scripts. No runtime debug argument, numerical scan or CPU copy was added to an operator. The encodings are lossless, so the reconstructed neighbor IDs can be used by individual derivative operators; this experiment only implements the primal interpolation consumer.

## Correctness and provenance

- Full standalone suite: **3,860 codec/numerical configurations**, including exact scalar and packed-vector GPU roundtrips and independent scalar CPU decoding.
- Dense FP64 numerical references; bit-identical output against raw CSR under the same schedule; both precisions, dimensions 1/2/3, mixed periodicity, variable/zero supports and nonuniform masses/densities.
- Empty, singleton and partial chunks; byte escape boundaries 254/255; IDs up to INT32_MAX; multiple escapes per quartet; dense source cells exceeding 256 particles; nondefault stream.
- CUDA memcheck: **zero errors**; racecheck: **zero hazards/errors/warnings**, on the 52-configuration quick suite.
- The 48 main held-out fixtures and three extra fixed-chunk fixtures qualify every timed output against native CSR64 outside timing.
- GPU process snapshots around every fixture report no competing compute process. These snapshots cannot rule out all transient system/desktop activity; batch ranges preserve observed timing variability.

GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition. Torch 2.13.0+cu130, CUDA 13, nvcc -O2 sm_120, FP32/FP64. Fixed fixtures use the same seeds and refined Morton sorting as the preceding profiler study. Nine rotated timing batches of five calls are used in the main and fixed-chunk suites; screening uses seven batches of five calls. Resident private calls and allocating public calls are kept distinct.

The archive retains immutable native/planner/prototype libraries, screening and final JSON, individual event/wall samples, conversion samples, storage by tensor, Nsight exports, diagnostics, CSV, figures and source snapshots. `source-v1` and `screen-v1-*` preserve the preliminary scalar-only screen; final results use `compression-v2.so`, which contains both scalar and packed-four variants. Original neighbor-gap distributions remain in `experiments/profiler_tuning/index_gaps_2026_10_31.json`.

Source and reproduction commands: [experiment README](../../experiments/csr_compression/README.md). Raw archive: `/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-csr-compression-20261031`.
