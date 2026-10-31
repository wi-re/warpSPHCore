# Measured interpolation FLOPs: FP32 and FP64

Hardware-counter study on the existing `codex/native-cluster-pairs` experiment, based on `30807f2`. No production kernel, dispatch, gradient or CRKSPH behavior changes. GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (188 enabled SMs, compute capability 12.0).

## Device reference

NVIDIA specifies [125 TFLOP/s FP32](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf). Its [RTX PRO Blackwell architecture whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf) specifies FP64 at 1/64 of FP32 (two FP64 cores per SM), giving approximately **1.953 TFLOP/s FP64**, and **1.792 TB/s DRAM bandwidth**. These are scalar CUDA-core FMA peaks. No tensor-core rate is used.

A separate eight-chain, register-only FMA reference measured **133.59 TFLOP/s FP32** and **1.635 TFLOP/s FP64** using ordinary CUDA events. The counter-profiled instances measured 127.56 and 1.807 TFLOP/s respectively. Clocks were left under normal driver control: nominal specifications and measured clocks are different references, and this microbenchmark is not a guarantee for other instruction mixes. The primary counter run observed roughly 2.74–2.86 GHz on the 3D traversal kernels. The archive also reports a peak derived from each kernel's measured clock, 188 SMs and 128/2 FMA lanes per SM.

## What is measured

- One million random particles, periodic unit box, refined Morton order, directed Gather Wendland C2 interpolation of a D-component field. Dimensions 1/2/3; support radius / mean spacing `n_h=4` and `2.57`; FP32 and FP64. Both CSR implementations use the same native-generated neighbor list. The fixed schedules come from the previous independent 262K-particle screen; no new tuning is performed.
- Nsight Compute dynamically counts predicated-on thread instructions: `FADD + FMUL + 2*FFMA`, separately `DADD + DMUL + 2*DFMA`. This follows the [NVIDIA FLOP-count metric mapping](https://docs.nvidia.com/nsight-compute/2022.4/pdf/NsightComputeCli.pdf). No percentage of SM activity is relabeled as FLOPs.
- These counts include compiler-generated arithmetic, reductions, periodic geometry and rejected-candidate arithmetic. Integer instructions, comparisons, shuffles, memory and special-function instructions are excluded. Division/square-root implementation contributes any executed add/mul/FMA instructions; no arbitrary equivalent-FLOP cost is assigned to a special-function instruction. Auxiliary FP32 arithmetic in an FP64 operation is retained separately in the raw summary.
- Main achieved rates divide the counted FLOPs by a separate, uninstrumented full-call CUDA event median: nine rotated samples of five calls, including preparation, allocations and device-visible dispatch gaps. Neighbor search, sorting and plan construction are outside the call. Kernel replay duration and kernel-only rates are also retained separately; they are not substituted for ordinary event times.
- Profiler capture starts only around a warmed complete interpolation call. All of its kernels are included, including output zeroing and packed-field preparation. Caches are not deliberately flushed and clocks are not locked. Additional 3D pipeline metrics are collected separately. Profiled kernel-time sums are generally shorter than the ordinary full-call events; both are reported.
- Numerical parity is checked before timing, not inside operation execution. The existing random FP32 1D Warp periodic-rounding discrepancy remains unqualified and its plot bars are omitted. All other 58 comparisons pass. GPU activity snapshots contain no other compute process.

## Primary 3D load: n_h=4

Actual mean neighbor count: 269.079094 (self included). TFLOP/s below uses complete-call event time; percentage is relative to the nominal FMA peak for that precision.

| Implementation | FP32 ms | FP32 TFLOP/s | FP32 peak | FP64 ms | FP64 TFLOP/s | FP64 peak | FP64/FP32 time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native/Torch CSR | 2.423 | 4.742 | 3.8% | 19.155 | 0.898 | 46.0% | 7.90× |
| Warp CSR | 11.140 | 4.713 | 3.8% | 53.707 | 1.314 | 67.3% | 4.82× |
| Cell shared | 1.519 | 17.437 | 13.9% | 41.762 | 0.638 | 32.7% | 27.50× |
| Cluster raw super-shared | 1.421 | 10.461 | 8.4% | 31.459 | 0.510 | 26.1% | 22.14× |
| Cluster compressed super-shared | 1.675 | 8.990 | 7.2% | 31.337 | 0.512 | 26.2% | 18.71× |

| Implementation | FP32 GFLOPs/call | FP64 GFLOPs/call | FP32 FLOPs/true pair | FP64 FLOPs/true pair |
|---|---:|---:|---:|---:|
| Native/Torch CSR | 11.492 | 17.201 | 42.71 | 63.93 |
| Warp CSR | 52.502 | 70.555 | 195.12 | 262.21 |
| Cell shared | 26.484 | 26.660 | 98.42 | 99.08 |
| Cluster raw super-shared | 14.866 | 16.059 | 55.25 | 59.68 |
| Cluster compressed super-shared | 15.059 | 16.059 | 55.97 | 59.68 |

The FLOP count is implementation work, not a common useful-work numerator. Warp executes roughly 4.6× as many FP32 add/mul/FMA FLOPs as native CSR here. Consequently, their approximately equal achieved FP32 TFLOP/s does not imply equal interpolation performance. The cell kernel executes more arithmetic than the cluster kernel but reports higher TFLOP/s. The cluster kernel still finishes sooner. FP64 native CSR uses the existing cooperative uncompressed-field implementation while FP32 uses packed-field traversal; this is a comparison of the existing precision-specific dispatches, not an otherwise identical kernel with only its scalar type changed.

## Pipeline interpretation

Focused 3D `n_h=4` traversal-kernel counters:

| Implementation | FP32 DRAM throughput | FP32 L2 throughput | FP32 long-scoreboard stalls | FP32 barrier stalls | FP64 pipe active |
|---|---:|---:|---:|---:|---:|
| Native/Torch CSR | 60.0% | 25.6% | 88.1% | 0.0% | 86.2% |
| Warp CSR | 12.9% | 83.5% | 63.2% | 0.0% | 89.4% |
| Cell shared | 2.1% | 14.4% | 13.0% | 26.5% | 86.5% |
| Cluster raw super-shared | 3.1% | 10.5% | 5.1% | 32.5% | 85.8% |
| Cluster compressed super-shared | 2.2% | 9.3% | 5.9% | 36.5% | 85.8% |

Stall percentages describe scheduler warp states, not fractions of total wall time. Pipeline active measures occupied cycles, not useful FLOPs or active lanes. These metrics cannot be added together or converted directly into achievable speedups.

FP32 CSR is dominated by memory dependencies: native CSR has 88% long-scoreboard stalls, while Warp has much higher L2 traffic and 84% L2 throughput. The cell and cluster kernels greatly reduce DRAM traffic and memory stalls, but introduce shared-memory staging and synchronization. At this load their FP32 throughput is far below the ideal FMA peak; that difference is not all available arithmetic headroom.

FP64 pipe activity is already 86–89% for all five paths. Lower counted FLOP/s relative to an ideal FMA ceiling can coexist with busy pipes: add/mul performs one FLOP rather than two, floating-point comparisons and other instructions are excluded from the numerator, and masking/divergence and dependencies reduce useful lane work. The counters support investigating instruction mix and wasted work rather than assuming a free 2–4× gain from the nominal-peak gap.

Whole-call FP32 DRAM traffic at this load is approximately 2.28 GB for native CSR, 2.28 GB for Warp CSR, 0.080 GB for cell shared, 0.104 GB for cluster raw and 0.089 GB for cluster compressed. L2 traffic is approximately 5.33, 66.82, 1.96, 1.45 and 1.45 GB respectively. These are warm profiler replay observations; DRAM traffic varies with cache state.

## Smaller neighborhood: 3D n_h=2.57

| Implementation | FP32 TFLOP/s | FP64 TFLOP/s | FP32 ms | FP64 ms |
|---|---:|---:|---:|---:|
| Native/Torch CSR | 6.417 | 0.824 | 0.499 | 5.787 |
| Warp CSR | 6.742 | 1.081 | 2.148 | 17.890 |
| Cell shared | 10.144 | 0.480 | 0.744 | 16.159 |
| Cluster raw super-shared | 8.534 | 0.438 | 0.628 | 12.799 |
| Cluster compressed super-shared | 7.471 | 0.446 | 0.727 | 12.667 |

The complete 60-case table and per-kernel counters cover both neighborhood sizes in 1D/2D/3D. In 1D, launch/setup costs and short neighbor rows matter more. In 3D at `n_h=4`, tile reuse is much more valuable. Neither FLOP utilization nor this interpolation-only timing establishes whole-simulation performance.

## Reproduce and retained evidence

Run `experiments/cluster_pairs/profile_flops.py --help` for the driver. `analyze_flops.py ARCHIVE` joins counters and independent timing samples, verifies matching artifact hashes and edge counts, and produces the figures. `fma_ceiling.cu` is a standalone scalar reference, compiled with `nvcc -std=c++17 -O3 -arch=sm_120`.

Full commands, binary Nsight reports, raw metric CSVs, ordinary timing samples, activity snapshots, exact driver sources, scalar FMA source/binary and checksums are retained at:

`/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-flops-20261031/`

The native and cell/cluster binary hashes are checked against the retained previous-study artifacts. That previous archive is an input to reproduction; it is not overwritten. New scripts and this report are the only repository changes.
