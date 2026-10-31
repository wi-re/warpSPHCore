# Profiler-guided native interpolation tuning

Based on the hardware-FLOP study committed as `2cd4bf0`, in the isolated `codex/native-cluster-pairs` worktree. This study changes native interpolation forward execution and permits fixed-adjacency CUDA Graph capture. Warp kernels, numerical kernel formulas and backward/adjoint implementations are unchanged.

## Retained changes

- FP32 interpolation exposes two independent neighbor-index/record gathers before consuming either, with independent sums. Dense 3D rows use 16 threads per query; 1D/2D use four. The sparse 3D and matrix-field paths retain the original single-record loop. Selection uses tensor metadata, not data scans or readbacks.
- FP64 scalar/vector interpolation now prepares aligned position/weighted-field records once per reference. This removes repeated mass/density division and weighting from incident edges. Fields with more than D components and unsupported support/direction/kernel combinations retain the old fallback.
- CUDA Graph capture is allowed for interpolation with supplied fixed adjacency. The former per-tensor blanket capture rejection is removed; variable-sized radius search retains one explicit rejection. No automatic graph cache, new operator parameter, persistent tensor conversion or runtime numerical validator is added.
- A missing host/device annotation on the pure `field_outputs` constexpr helper was exposed by a clean CMake build and fixed. The JIT had previously hidden it through relaxed constexpr compilation.

## Main measured result

One million random particles, refined Morton order, unit periodic box, Wendland C2 Gather interpolation, D-component field. `n_h` is support radius divided by mean spacing. Both neighborhood sizes, both precisions, dimensions 1/2/3, and held-out 32K/1M sizes are measured. The schedules were screened independently at 262K. The following are full native/Torch calls including scratch/output allocation, packing and device-visible dispatch gaps, excluding neighbor search and index conversion.

| 3D case | CSR IDs | Before ms | After ms | Throughput ratio |
|---|---:|---:|---:|---:|
| float32, n_h=4 | 64 bit | 2.496 | 1.604 | 1.556× |
| float32, n_h=4 | 32 bit | 1.971 | 1.175 | 1.678× |
| float32, n_h=2.57 | 64 bit | 0.535 | 0.533 | 1.004× |
| float32, n_h=2.57 | 32 bit | 0.389 | 0.389 | 1.000× |
| float64, n_h=4 | 64 bit | 19.297 | 15.857 | 1.217× |
| float64, n_h=4 | 32 bit | 19.349 | 15.820 | 1.223× |
| float64, n_h=2.57 | 64 bit | 5.838 | 4.803 | 1.215× |
| float64, n_h=2.57 | 32 bit | 5.823 | 4.781 | 1.218× |

Dense FP32 3D improves by **1.56× with CSR64** and **1.68× with CSR32**. FP64 improves by approximately **1.22×** in both 3D neighborhoods. Sparse FP32 3D is effectively unchanged. Improvements are not uniform: the FP32 2D CSR64 `n_h=4` median is 0.94×, with overlapping observed sample ranges, while CSR32 improves by 1.15×. The full data and range whiskers are retained rather than presenting every configuration as a win.

![Native speedup](/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-profiler-tuning-20261031/native-speedup.png)

## What the profiler resolved

For the original FP32 3D CSR64 traversal, Nsight reports about **88% long-scoreboard stalls**, 48 registers/thread, 70% achieved occupancy and 60% DRAM throughput. It has enough active threads to do useful work, but often waits for dependent index → record → arithmetic loads. The warning about uncoalesced traffic is consistent with those gathers. Generic profiler estimated-speedup suggestions are not treated as measured gains.

After the change, the traversal uses 64 registers/thread and about 64% occupancy: occupancy goes down, while performance improves. Long-scoreboard stalls fall to **72%** and DRAM throughput rises to **95.3%**. This is the intended trade: more independent memory requests per warp hide latency better. No spills are reported.

| Native CSR64, 3D n_h=4 | Original counted GFLOPs/call | Revised counted GFLOPs/call | Revised profiled traversal ms |
|---|---:|---:|---:|
| float32 | 11.492 | 10.784 | 1.381 |
| float64 | 17.201 | 12.376 | 14.380 |

Counts include executed add/mul/FMA arithmetic with FMA=2, following the previous study. The revised FP64 path executes approximately **28% fewer FP64 FLOPs**. It can run faster while reporting fewer FLOP/s: less division/weight preparation work is being done. Its FP64 pipe remains busy (~85%), so reducing repeated work is more productive than treating the gap to ideal FMA peak as idle capacity.

The nominal device references are [125 TFLOP/s FP32 and 1.792 TB/s DRAM](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf), with [FP64 at 1/64 of FP32](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf). For 269,079,094 directed pairs, CSR64 IDs alone occupy 2.153 GB. Merely streaming those IDs at nominal peak takes **1.20 ms**, before reading particle records or computing a kernel. Using the revised FLOP count, that ID-only memory roof is about **9 TFLOP/s**, or **7% of nominal FP32 FMA peak**. It is therefore inappropriate to expect this CSR representation to approach 125 TFLOP/s. CSR32 halves ID traffic, but still has record accesses and load dependencies. No index conversion cost is hidden inside a call; the lists are prepared beforehand.

## Cluster experiments

The baseline super-cluster kernel already has ~99% occupancy, 40 registers/thread and no spills. Its FP32 global-memory stalls are only ~5%; synchronization and shared-memory/integer instruction work are more prominent.

Six private variants are retained:

| Variant | Change |
|---:|---|
| 0 | Independent metadata/source reads, removing one block barrier |
| 1 | Variant 0 with hardware `cp.async` copies |
| 2 | Two-stage `cp.async` double buffering |
| 3 | Variant 0 with four-way loop unrolling |
| 4 | Variant 3 with periodic images decoded once into two-bit fields |
| 5 | Variant 0 with the same image decoding |

The focused 3D FP32 profile measured 1.336 ms for the baseline traversal, 1.366 ms for variant 0, 1.500 ms for single-stage async, 1.376 ms for double buffering, and 1.309 ms for unrolling alone. Async staging preserves ~99% occupancy and 40 registers/thread; double buffering doubles shared storage from 6.4 to 12.9 KB. Barrier stalls only fall from ~33% to ~29–30%. The exposed global-load latency was already small, so hiding it did not compensate for staging overhead. These are profiler observations, not replacements for ordinary call timings.

Periodic images were originally decoded from base three inside every pair loop. Decoding them once per source cluster and using shifts/masks removes repeated integer work without changing `(query - source) - image` floating-point order. This variant wins the independent FP32 3D screen. Held-out results:

| FP32 3D, 1M | Original resident ms | Image-bit variant resident ms | Original allocating call ms | Image-bit allocating call ms |
|---|---:|---:|---:|---:|
| n_h=4 | 1.435 | 1.273 | 1.564 | 1.243 |
| n_h=2.57 | 0.733 | 0.611 | 0.765 | 0.647 |

The dense resident improvement is approximately 13%; the allocating-call medians differ by about 26%, with overlapping ranges. The smaller neighborhood improves by about 20% resident / 18% allocating-call median. Treat the exact percentages cautiously: public/private sample variation is visible. These optimized cluster variants remain in the isolated experiment; no production cluster backend is introduced. FP64 did not show a compelling comparable image-decoding win.

## Modern CUDA features

- **CUDA Graphs: tested and enabled for fixed-adjacency interpolation.** At 32K particles, completed FP32 calls fall from 19.7 to 6.6 microseconds in 1D and 32.2 to 11.0 microseconds in 2D. The 3D 32K case and 1M cases are mostly kernel-bound, with little improvement. Replay recomputes packing and observes changes to both field values and reference masses. Search, sorting and capture are excluded from replay timing. This is caller-owned graph capture; topology, tensor storage and execution shape must remain valid.
- **Async global-to-shared copies: implemented and measured, no dense 3D win.** [NVIDIA documents](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html) small 4/8/16-byte async transfers and the need for completion synchronization. This matches the irregular source clusters better than pretending they are a large dense tensor.
- **TMA: plausible with a different padded source layout, not benchmarked here.** TMA is a data-copy feature and does not require tensor-core arithmetic. Bulk copies impose alignment/transfer-size constraints; arbitrary cell-bounded cluster starts do not automatically satisfy them. A local compile-only probe for `cp.async.bulk.tensor...tile::gather4` succeeds for both `sm_120` and `sm_100` with CUDA 13.0, consistent with the installed [NVIDIA CCCL instruction wrapper](https://nvidia.github.io/cccl/unstable/libcudacxx/ptx/instructions/cp_async_bulk_tensor.html). This is compiler availability, not runtime qualification. A credible next test would first pack aligned fixed-size source-cluster rows and include its padding/packing cost. Current global-load stalls are small, so there is no evidence that a copy-engine replacement alone would yield a large speedup.
- **Thread-block clusters / distributed shared memory: reviewed, not implemented.** Sharing source data across CTAs can be useful on Hopper/Blackwell, but requires coordinated CTA groups and extra synchronization. Here the per-CTA data already fits comfortably in shared memory; adding a hierarchy is not justified by this profile. See [NVIDIA tuning guidance](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html). Particle clusters in this experiment are unrelated to CUDA thread-block clusters.
- **Tensor cores: not a drop-in strict-FP32 optimization.** [TF32 uses reduced-precision inputs and FP32 accumulators](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#alternate-floating-point). Three-dimensional distances, support tests, square roots and the kernel polynomial remain scalar work, and multiplying the resulting weight tile by only D field components is a small, poorly utilized matrix product. Multi-pass full-FP32 emulation exists in principle, but has not been justified here. No reduced-precision or fast-math path is introduced.

## Qualification and reproducibility

- Clean CUDA CMake build at `-O2`, `sm_120`, with line information; stable Torch ABI. No fast math or forced clock changes.
- Final native CUDA primal, VJP, fallback and stream/graph suites: **512 passed, 20 skipped**. Separate Python scaffolding suite: **19 passed**.
- Independent dense-reference private verification: **4,752 configurations**, including FP32/FP64, dimensions 1/2/3, both CSR widths, ragged rows, partial/overfull/empty cells, variable supports/masses/densities, grouping/culling choices, near-periodic-cutoff regressions and a nondefault stream.
- Final private racecheck and memcheck: **zero hazards/errors**, 70 quick numerical configurations each. Earlier four-variant results are also retained.
- Final packed templates compile offline for **gfx942 and gfx1100**. AMD runtime performance remains unmeasured.
- Before/after native artifacts are loaded in separate processes with identical current Python wrappers, sorting and lists. FP32 is measured baseline then candidate; FP64 reverses the order. Eleven rotated batches of ten calls are retained per case. Both CSR widths are tested as persistent input representations; search/index conversion is outside timing.
- Exact raw Nsight reports, CSVs, source snapshots, immutable binary artifacts, compiler/driver metadata, activity snapshots and checksums are retained. Preliminary `final-*` and `promoted-*` sweeps precede the final paired matrix and are excluded from the main before/after figures. The early test-oracle field-shape error is retained as a superseded harness failure; it was corrected before final tests.

Archive:

`/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-profiler-tuning-20261031/`

Repository drivers live in `experiments/profiler_tuning`. The main CSV is `summary.csv`; figures are `native-speedup.png` and `graph-overhead.png`. Commands and artifact hashes are recorded in the archive.
