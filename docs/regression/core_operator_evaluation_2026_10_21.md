# Core SPH operator performance evaluation — 21 October 2026

This is the preparation-phase record: suite setup, CPU qualification, GPU-free compiler resource surveys and separately labeled historical measurements. The GPU later became available; **the current [operator inventory and fresh scaling report](core_operator_scaling_2026_10_21.md) supersedes the pending-run statements below**. No production operators or defaults were changed.

Simulation relevance is established in the current inventory. In particular, CRK moments on grid traversal are unused in simulations, and the current frontend uses fused CRK local assembly for ordinary execution. The core Torch CRK stages below describe library API coverage and the differentiation fallback; they must not be interpreted as ordinary timestep costs.

## Deliverables and evidence

All generated evidence for this session is in [`/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/`](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/). This is an ignored, workspace-local staging directory; it can subsequently be copied into the shared development archive described in [the artifact policy](../docs/development-artifacts.md). The source, tests, and this report stay in the repository.

| Tool | Purpose |
| --- | --- |
| [`_operator_cases.py`](../../scripts/benchmarks/_operator_cases.py) | Shared workloads, option matrix, physical pair counts, output checks |
| [`bench_core_operators.py`](../../scripts/benchmarks/bench_core_operators.py) | Import-only planning, CPU/CUDA qualification, explicit CUDA timing and profiling ranges |
| [`survey_operator_resources.py`](../../scripts/benchmarks/survey_operator_resources.py) | Offline CUDA compilation, physical register allocation, stack, spill, shared-memory reports and occupancy ceilings |
| [`survey_operator_dispatch.py`](../../scripts/benchmarks/survey_operator_dispatch.py) | Untimed CPU census of Torch dispatcher operations and Warp launch requests |
| [`plot_core_operators.py`](../../scripts/benchmarks/plot_core_operators.py) | PNG/SVG timing and throughput graphs, IQR bands, raw-point/provenance manifest; accepts earlier benchmark schemas |
| [`run_operator_evaluation.py`](../../scripts/benchmarks/run_operator_evaluation.py) | Separate precision/dimension processes; defaults to plans; gates timing on current source-matched qualification |
| [`test_core_operator_suite.py`](../../tests/benchmarks/test_core_operator_suite.py) | Work-count, option, compiler-parser, occupancy-model and plot-denominator contracts |

The baseline plan has **4,334 cases**: 1,182 per precision in 2D and 985 per precision in 3D. Particle counts are 1,024 through 1,048,576 in 2D and 512 through 2,097,152 in 3D. Neighbor targets are 25, 55, 110, 220; traversal is prebuilt CSR or prebuilt compact hash grid; lane settings are 1 and 32. Local matrix tests are deduplicated across irrelevant topology and neighbor options. Plans are saved in `plans/` and can be generated with a Python installation that has neither Torch nor Warp.

Additional options are supported for targeted sweeps: lanes 2/4/8/16, implicit grid rebuilds, frozen renormalization/CRK corrections on derivative operators, gradient and Laplacian formulations, support formulations, and several kernel families. Precision is selected before import with `warpSPHCore_PRECISION`. Run those targeted comparisons separately to keep each graph readable. AD, JVPs, prepared execution paths, arbitrary tensor-valued fields, irregular clouds, heterogeneous supports, mixed particle kinds, and index-width sweeps remain extensions; this is the primal all-fluid lattice baseline. Linear solves are excluded.

## Workloads and timing contract

The seven core operators are density, interpolation, gradient, divergence, curl, Laplacian and covariance. Scalar fields are used except for vector divergence/curl and the Dot Laplacian (which requires dimension-compatible vector inputs); covariance produces a matrix. Renormalization measures covariance, fallback processing, inverse and diagnostics together. CRK is measured as volume, fused moments, consistency density, frozen local terms, and the complete factor call. Local matrix comparisons include `pinvSymmetric`, fused `applyPinv`, the dimension-dependent production `pinv_warp` policy, and `torch.linalg.pinv`.

Geometry, values and matrices are deterministic. Matrices are nontrivial SPD batches with every seventh row rank deficient. A qualification pass compares native inverse/application against a float64 Torch reference with matched cutoff, and compares each traversal/lane/form against the matching serial CSR call. Full output structures are checked for finite values. Frozen correction construction is excluded from consumer timings, and included in the full correction-builder timings.

Inputs use a periodic equal-support lattice. Support must be smaller than half the box extent to avoid ambiguous periodic pair accounting. The exact CSR edge count is collected outside timing; self entries are excluded from the reported directed interaction count. A pair `(i,j)` and its reverse are distinct. Fused CRK moments count as one traversal despite their multiple outputs. Full CRK factors count three physical pair traversals. These counts normalize physical SPH work; they do not count every instruction or candidate visit made by a grid kernel. Candidate visits remain unavailable unless actually observed. Standalone local algebra uses **matrices/s**, with no fabricated pair rate. A pair-based builder's throughput includes its local work in the denominator.

Timing is explicit CUDA-only and uses a separate Torch/Warp stream, warmup, repeated synchronized public calls, raw samples and their median. The total wall interval includes Python/dispatch/allocation and synchronization plus event recording; the CUDA event interval is saved separately and can contain launch gaps. Profiling runs use a separate mode and produce no timing samples. The earlier datasets also contain graph-device intervals, which can be plotted with `--timing graph`; these are never substituted for total wall time. Independent stage timings are not an additive partition of a full call.

Current defaults are four warmups and nine samples. Each result is saved atomically after a case, starts as `complete=false`, records errors, and becomes complete only after all cases and a source-hash check pass. The plotting tool refuses incomplete input, invalid samples and inconsistent medians. Memory diagnostics include Torch peak allocated/reserved memory and total/free device memory; Torch allocator counters exclude allocations made directly by other runtimes. Device clocks, power, utilization, other processes and thermal state should be recorded alongside the later timing run; the suite does not enforce exclusive GPU use.

## Offline registers, memory and occupancy

Resources come from current-source production overloads compiled by Warp's NVRTC toolchain (reported toolkit 12.9) for **SM120**, then assembled by **ptxas 13.0.48**. The installed package reports Warp 1.17.0 and Torch 2.13.0+cu130. The compilation target follows the RTX PRO 6000 Blackwell in the historical evidence; hardware identity cannot be verified from this session. Compiled PTX, assembly logs, cubins, and per-artifact hashes are retained under `compiled/`.

Tiny CPU fixtures establish actual dimensional and field-type overloads. The survey intercepts launch requests, explicitly selects the corresponding CUDA tiled kernel because normal CPU dispatch disables it, and compiles without executing GPU kernels. CSR/grid selection, support and correction flags live in runtime state, so these resources cover the shared compiled traversal path rather than independently optimized CSR/grid branches. Outputs and numerical parity of CUDA tiled execution still require later GPU qualification.

The compiler's register count is the allocated physical count per thread, **not PTX virtual register declarations**. Stack bytes, spill loads/stores and static shared bytes are recorded separately. Stack allocation can exist with zero reported spills; that does not establish zero local-memory traffic. Torch/vendor kernels, including the 3D production inverse policy and CRK local algebra, cannot be assigned Warp register counts by this offline procedure.

Float32 allocated registers per thread (serial / 32 lanes):

| Operator | 2D registers | 3D registers |
| --- | ---: | ---: |
| density | 64 / 64 | 64 / 68 |
| interpolate | 68 / 64 | 71 / 64 |
| gradient | 72 / 75 | 83 / 95 |
| divergence | 72 / 78 | 87 / 94 |
| curl | 72 / 80 | 96 / 96 |
| laplacian | 72 / 72 | 79 / 80 |
| covariance | 80 / 80 | 99 / 113 |
| crk_volume | 70 (serial only) | 72 (serial only) |
| crk_moments | 91 / 95 | 128 / 128 |
| crk_density | 71 / 72 | 72 / 76 |
| pinv_symmetric | 40 (local kernel) | 40 (local kernel) |
| apply_pinv | 40 (local kernel) | 40 (local kernel) |
| pinv_policy | 30 (local kernel) | Torch backend |

The focused float64 matrix survey shows:

| Operator | 2D registers: serial / 32 lanes | 3D registers: serial / 32 lanes | 3D optimistic occupancy ceiling: serial / 32 lanes |
| --- | ---: | ---: | ---: |
| covariance | 130 / 130 | 194 / 200 | 16.7% / 20.8% |
| crk_moments | 152 / 164 | 232 / 232 | 16.7% / 16.7% |
| pinv_symmetric | 64 (local kernel) | 64 (local kernel) | 66.7% |
| apply_pinv | 66 (local kernel) | 66 (local kernel) | 50.0% |
| pinv_policy | 48 (local kernel) | Torch backend | unavailable |

The four complete surveys contain 90 records (88 compiled-launch records, including repeated serial CRK-volume entries, and two Torch-only records). **No spill loads/stores were reported in these forward kernels.** Float32 3D moments have a 184-byte/thread stack frame; native inverses/applications have 72-byte float32 or 144-byte float64 frames. Static shared allocations are 1,024 bytes/block for the observed serial kernels and `4 × lanes` bytes/block for these tiled kernels. These are compiler observations, not measured traffic.

Occupancy values are deliberately optimistic **resource ceilings**, not achieved occupancy or exact driver occupancy predictions. The SM120 model uses 48 resident warps, 32 resident blocks and 65,536 32-bit registers per SM, following [NVIDIA's Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/). It omits allocation rounding, SM partitions, shared-memory restrictions and finite grid size. The exact theoretical limit should later be checked with the CUDA driver/Nsight occupancy calculator; achieved occupancy and stalls require execution.

The launch policy matters independently of register counts. Serial kernels use 256 threads/block, eight warps/block, with roughly `ceil(N/256)` blocks. Tiled kernels use `dim=[N, lanes]`, `block_dim=lanes`, one block per query particle. CRK volume has no tiled variant and stays serial under every lane override. Under the model, 4/8/16/32-thread blocks can occupy at most 32 physical warps, or 66.7% physical warp occupancy, before register limits. Their useful-thread ceilings are at most 8.3/16.7/33.3/66.7% respectively. At small N, serial grids may underfill the device; at large N, tiled grids launch N tiny blocks. These competing effects explain why an occupancy figure alone cannot choose the fastest kernel.

The inventory records these launch layouts without changing them. Their occupancy ceilings are resource diagnostics rather than evidence of a simulation bottleneck.

## Untimed dispatch census

The CPU census observes dispatcher calls and Warp launch requests, after input/correction preparation and warmup. A Torch dispatcher operation can emit multiple CUDA kernels; the following are not CUDA kernel counts or performance samples.

| Workload | 2D Warp launches / Torch dispatches | 3D Warp launches / Torch dispatches |
| --- | ---: | ---: |
| Renormalization | 2 / 13 | 1 / 19 |
| Complete CRK factors | 3 / 263 | 3 / 263 |
| CRK local terms | 0 / 248 | 0 / 248 |
| Native symmetric inverse | 1 / 3 | 1 / 3 |
| Production inverse policy | 1 / 6 | 0 / 12 |

CRK terms perform a determinant, a general `torch.linalg.pinv`, many contractions and limiter operations. A CPU scalar read is observed in CRK terms/factors. Source inspection identifies `if mask.any():` in `crk_terms.py`, an unconditional Python truth test of a device reduction on CUDA. This requires a host readback even when no limiter warning is printed, and prevents graph capture of the full terms call. Its elapsed cost still needs CUDA profiling. In 3D the production renormalization policy computes diagnostic eigenvalues with Torch and uses the selected symmetric Torch inverse; it does not dispatch the standalone Warp Jacobi inverse. The census shows why standalone native inverse timing does not describe these full core wrappers. Ordinary frontend CRK execution uses a different, fused local assembly path.

## Existing scaling evidence and graphs

These results are from **15 October 2026**, float32, periodic 2D lattice, RTX PRO 6000 Blackwell Workstation Edition. Baseline operators use 55 requested / 57 actual CSR neighbors. Source hashes and dependencies are preserved in each figure manifest. Source drift is expected; these measurements support hypotheses for the new sweep rather than current-checkout speed claims.

| Operator | Total wall ms at N=1,024 | Total wall ms at N=1,048,576 | G directed interactions/s at N=1,048,576 | Paired timing/throughput graph |
| --- | ---: | ---: | ---: | --- |
| Density | 0.173 | 1.267 | 46.34 | [PNG](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/historical-figures/density-2d-float32-wall.png) |
| Interpolation | 0.181 | 1.534 | 38.27 | [PNG](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/historical-figures/interpolate-2d-float32-wall.png) |
| Gradient | 0.201 | 1.822 | 32.23 | [PNG](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/historical-figures/gradient-2d-float32-wall.png) |
| Divergence | 0.205 | 1.841 | 31.90 | [PNG](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/historical-figures/divergence-2d-float32-wall.png) |
| Curl | 0.283 | 1.713 | 34.28 | [PNG](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/historical-figures/curl-2d-float32-wall.png) |
| Laplacian | 0.309 | 1.833 | 32.04 | [PNG](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/historical-figures/laplacian-2d-float32-wall.png) |
| Covariance | 0.182 | 2.570 | 22.85 | [PNG](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/historical-figures/covariance-2d-float32-wall.png) |
| Complete CRK factors | 1.320 | 42.466 | 4.15 (three traversals) | [PNG](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/historical-figures/crk_factors-2d-float32-wall.png) |
| CRK local terms | 0.542 | 36.652 | 28.61 million matrices/s instead | [PNG](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/historical-figures/crk_local_terms-2d-float32-wall.png) |

SVG versions and medians/IQR/raw-point manifests are saved beside the PNGs. Separate [55-neighbor](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/historical-lanes-55/) and [220-neighbor](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/historical-lanes-220/) figures show CSR/grid and lane effects for curl and Laplacian. Baseline wall graphs and these lane-option graphs come from different archived runs and are not merged into one measurement series.

At small N, the seven basic operators' wall times greatly exceed their graph-device intervals; e.g. gradient is 0.201 ms wall versus 0.0081 ms graph at N=1,024. At N=1,048,576 those intervals are 1.822 and 1.653 ms. This supports an investigation of dispatch/launch overhead at small sizes and kernel work at larger sizes. Only three baseline sizes were measured, so the knee is not precisely located.

The independently timed CRK local stage takes 36.652 ms versus 42.466 ms for complete factors at N=1,048,576. The ratio is about 86%; independent calls are not an additive time attribution. This core Torch CRK wrapper is not the ordinary fused frontend CRK execution path.

The archived lane sweep reaches N=4,194,304. At 55 requested neighbors, graph-device Laplacian time is 6.566 ms serial versus 8.393 ms with 32 lanes; curl is 5.762 versus 6.884 ms. At 220 requested neighbors the ordering changes modestly: Laplacian is 33.361 versus 31.621 ms and curl 30.239 versus 26.943 ms. Four/eight-lane configurations are substantially slower in these large cases. The archived grid sweep is more striking: at N=1,048,576 and target 55, Laplacian is 3.972 ms serial versus 15.037 ms with 32 lanes, and curl 2.952 versus 11.413 ms. These are **graph-device intervals**, supporting a traversal-specific investigation rather than a universal lane default.

Fresh per-operator scaling graphs for standalone inverses, renormalization and separated CRK stages remain pending the explicit GPU timing phase. The plotter produces them from the new suite's JSON; no curves have been invented for workloads without measured samples.

## Diagnostic inconsistency found during qualification

CRK moments' integer neighbor-count output differs by traversal. Its loop increments the count for every visited candidate after particle-kind filtering, even when the kernel contribution is zero outside support. On the 2D N=1,024 / target-55 fixture, CSR reports 58,368 entries while grid reports 273,529 visits. The six floating-point moment outputs agree within tolerance.

Qualification checks all six floating outputs and separately records both integer counts and their equality flag. It does not certify integer-count parity. Throughput continues to use physical CSR pairs. Grid CRK moments are unused in actual simulations, as clarified by the user. This remains an API diagnostic and is excluded from simulation bottleneck priorities. No production fix is included here.

## Follow-up GPU inventory

The [current report](core_operator_scaling_2026_10_21.md) contains fresh timing, live resource queries, launch traces and the hardware-counter permission result. The follow-up measures operator families and options; it does not perform implementation optimization experiments. Library support and actual frontend usage are listed separately.

## Reproduction

Use the existing environment and explicit output paths. The tools never schedule or start a later timing sweep automatically.

```bash
PYTHON=/home/lu26029/miniconda3/envs/warp/bin/python
PTXAS=/home/lu26029/miniconda3/envs/warp/bin/ptxas
OUT=.tmp/operator-evaluation

WARP_CACHE_PATH="$PWD/$OUT/pytest-warp-cache" "$PYTHON" -m pytest tests/benchmarks -q
"$PYTHON" scripts/benchmarks/run_operator_evaluation.py --stage plan --output-dir "$OUT/plans"
"$PYTHON" scripts/benchmarks/run_operator_evaluation.py --stage validate --output-dir "$OUT"
"$PYTHON" scripts/benchmarks/run_operator_evaluation.py --stage resources --ptxas "$PTXAS" --output-dir "$OUT"

# Explicit later phase, once the GPU is available. Source-matched qualification
# in OUT is required by the driver. No linear solves are invoked.
"$PYTHON" scripts/benchmarks/run_operator_evaluation.py --stage benchmark --output-dir "$OUT"
"$PYTHON" scripts/benchmarks/plot_core_operators.py "$OUT/benchmark-2d-float32.json" \
  --neighbors 55 --output-dir "$OUT/figures-2d-float32-55"
```

The baseline timing driver uses plain operators. For corrected consumers, formulations and additional lanes, first qualify the intended options with `bench_core_operators.py validate`, then use the same CLI options with `benchmark --device cuda:0`. Example targeted comparison:

```bash
"$PYTHON" scripts/benchmarks/bench_core_operators.py validate --ns 1024 \
  --neighbors 55 --lanes 1 --operations gradient laplacian \
  --forms plain renorm crk --output "$OUT/validate-corrections.json"
"$PYTHON" scripts/benchmarks/bench_core_operators.py benchmark --device cuda:0 \
  --ns 1024 65536 1048576 --neighbors 55 220 --lanes 1 8 32 \
  --operations gradient laplacian --forms plain renorm crk \
  --output "$OUT/benchmark-corrections.json"
```

A separate counter capture, executed only when GPU use is appropriate:

```bash
/home/lu26029/miniconda3/envs/warp/bin/ncu --profile-from-start off \
  --section LaunchStats --section Occupancy --section SpeedOfLight \
  --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats \
  --section SourceCounters --section InstructionStats --export "$OUT/laplacian-counters" \
  "$PYTHON" scripts/benchmarks/bench_core_operators.py benchmark --device cuda:0 \
  --profile-range --ns 65536 --neighbors 55 --operations laplacian \
  --traversals adjacency grid --lanes 1 32 --output "$OUT/laplacian-profile.json"
```

`--profile-range` surrounds one qualified/warmed public call with CUDA profiler start/stop and records no timing samples. Use a similar range for Torch/vendor stages; the static survey cannot describe their resources. Warp's [code-generation documentation](https://nvidia.github.io/warp/latest/user_guide/programming_model/code_generation.html) documents GPU-free AOT compilation; its [kernel-property API](https://nvidia.github.io/warp/v1.18/api_reference/_generated/warp.get_cuda_kernel_properties.html) describes driver-side resource queries for a later live-device cross-check.

## Validation completed

- **52 benchmark tests passed; 27 CUDA tests skipped**, with 14 existing Torch deprecation warnings. The first broad test attempt exposed the sandbox's read-only default Warp cache; rerunning with the documented workspace-local `WARP_CACHE_PATH` passed.
- **660 CPU cases passed**: 165 per dimension/precision combination, covering all 17 workloads, four neighbor targets, CSR/grid and plain/corrected derivative forms. Each completed qualification file matches the final suite source hashes.
- **288 additional 2D float32 option cases passed** across plain/renormalized/CRK gradient and Laplacian, Gather/Scatter/KernelMeanSymmetric, all gradient/Laplacian schemes and Wendland2/QuinticSpline. Dot Laplacian uses a vector field, as required by its API; field shape is recorded and kept distinct in plots.
- **Four offline resource surveys completed**, totaling 90 records; **two untimed dispatch censuses completed**. GPU kernels executed by the resource surveys: zero.
- **13 historical timing/throughput figures** were regenerated in PNG and SVG, with input hashes and plotted points in manifests. Representative figures were visually inspected.
- Import-free planning, Python compilation and whitespace checks passed. The production implementation is unchanged. CUDA lane parity, live-driver resource cross-checks, achieved occupancy and fresh timings remain for the GPU phase.

[`evaluation-summary.json`](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/evaluation-summary.json) indexes the evidence and hashes. Reproduction commands above prepare the remaining run; no automation was created.
