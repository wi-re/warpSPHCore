# GPU full-neighbor unity / small-gap codecs

Actual GPU encoders and fused interpolation consumers, isolated from production dispatch and AD.

Artifact archive: `/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-csr-small-gaps-20261031`. The throughput, storage and memory/throughput PNGs live there alongside raw JSONs.

## Scope and validation

* RTX PRO 6000 Blackwell Workstation, sm_120; PyTorch 2.13.0+cu130, CUDA 13 environment.
* Main sweep: 36 fresh-process fixtures, 1M/4M/16M particles, 1D/2D/3D, FP32/FP64, n_h=4/2.57.
* Schedules screened at 262,144 particles and frozen before held-out measurements.
* Nine randomized paired batches; CUDA events and wall times retained. Packing is included. Search, sorting, conversion, compression and correctness checks are excluded.
* Resident-buffer private controls and allocating wrappers reported separately. Native Torch wrappers are context for production call costs.
* Full correctness: 4,656 configurations, independent CPU readers, exact GPU index round trips, dense references, bit-exact matched raw controls, and a nondefault stream.
* Quick sanitizer matrix: 240 configurations; memcheck zero errors, racecheck zero errors/warnings.
* Main sweep plus four compact-layout fixtures and two rescreened bit-mask candidate fixtures: 42 held-out cases, all compute-process snapshots uncontended.
* All compared layouts coexist during measurement; peak Torch allocation is not the minimum memory needed by one layout.

## Actual large-case results

Below are allocating calls on 16M particles in 3D. Each table is one paired compact-layout run. GB are decimal, retained topology only, including counts/offsets/padding. The raw CSR graph has 4.305 billion edges at n_h=4 and 1.154 billion at n_h=2.57.

### float32, n_h=4, 269.093 neighbors per particle

| Layout | Time (ms) | Retained GB | Bytes/neighbor |
|---|---:|---:|---:|
| CSR32 | 19.147 | 17.414 | 4.045 |
| Byte delta | 21.205 | 6.140 | 1.426 |
| Cell relative | 21.303 | 7.342 | 1.705 |
| Two-bit gaps | 22.620 | 2.276 | 0.529 |
| Nibble, 32 entries | 26.056 | 1.752 | 0.407 |

### float32, n_h=2.57, 72.101 neighbors per particle

| Layout | Time (ms) | Retained GB | Bytes/neighbor |
|---|---:|---:|---:|
| CSR32 | 6.766 | 4.806 | 4.166 |
| Byte delta | 7.260 | 2.235 | 1.937 |
| Cell relative | 7.729 | 3.875 | 3.359 |
| Two-bit gaps | 7.419 | 0.967 | 0.838 |
| Nibble, 32 entries | 10.748 | 0.830 | 0.719 |

### float64, n_h=4, 269.093 neighbors per particle

| Layout | Time (ms) | Retained GB | Bytes/neighbor |
|---|---:|---:|---:|
| CSR32 | 252.673 | 17.414 | 4.045 |
| Byte delta | 254.450 | 9.146 | 2.124 |
| Cell relative | 259.157 | 8.797 | 2.043 |
| Two-bit gaps | 255.383 | 2.276 | 0.529 |
| Nibble, 32 entries | 281.369 | 1.752 | 0.407 |

### float64, n_h=2.57, 72.101 neighbors per particle

| Layout | Time (ms) | Retained GB | Bytes/neighbor |
|---|---:|---:|---:|
| CSR32 | 76.204 | 4.806 | 4.166 |
| Byte delta | 78.693 | 2.735 | 2.371 |
| Cell relative | 77.019 | 5.469 | 4.740 |
| Two-bit gaps | 80.050 | 0.967 | 0.838 |
| Nibble, 32 entries | 96.907 | 0.830 | 0.719 |

## Interpretation

The dense FP32 compact run reduces CSR32 storage by 86.9% with two-bit gaps and 89.9% with 32-entry nibble segments. Those calls cost 18.1% and 36.1% more time than paired native CSR32. Compression remains a capacity/throughput tradeoff here.

The faster nibble configuration selected 16-entry segments for dense FP32: 0.541 B/neighbor and 24.899 ms in the main run, compared with paired CSR32 at 18.595 ms. The fixed 32-entry compact configuration reaches 0.407 B/neighbor, close to the earlier 0.401 storage-only model. Dummy first controls and four-byte row alignment explain the small difference. Band reaches 0.529 versus the earlier 0.522 model.

For dense FP64, the speed-selected layouts are all approximately 253–256 ms versus CSR32 at 253.732 ms. The more compact 32-entry nibble variant costs more: 281.369 ms versus paired CSR32 at 252.673 ms. Precision alone therefore does not guarantee negligible decode overhead; scheduling and register lifetimes still matter.

At 16M in 1D with n_h=4, two-bit gaps take 0.923 ms versus paired native CSR32 at 1.257 ms, about 36% more throughput. In 2D, byte delta is the strongest of these compressed choices at that size. No single codec wins across dimensions and precision.

## Profile: 1M, 3D, n_h=4, FP32

These are traversal-only counters from three-pass hardware profiling. They exclude packing and wrapper costs and must not be substituted for ordinary whole-call measurements. Caches and clocks were uncontrolled consistently across paths; ordinary randomized event timings remain the performance comparison.

| Path | Traversal ms | DRAM GB | L2 GB | Registers | Occupancy % | Billion warp instructions |
|---|---:|---:|---:|---:|---:|---:|
| csr32 | 1.028 | 1.166 | 2.540 | 61 | 62.2 | 0.984 |
| delta | 1.162 | 0.454 | 1.397 | 80 | 46.6 | 1.082 |
| band | 1.221 | 0.200 | 1.811 | 72 | 55.3 | 1.501 |
| nibble | 1.341 | 0.205 | 1.206 | 64 | 62.4 | 1.959 |
| native32_public | 0.966 | 1.169 | 3.366 | 64 | 64.7 | 0.941 |

Two-bit and nibble consumers cut DRAM traffic by about 83% relative to matched private CSR32, but execute 53% and 99% more warp instructions. Band uses 72 registers and 55% occupancy versus raw at 61 and 62%; nibble reaches similar occupancy to raw yet remains slower. Thus occupancy is not the sole problem. Prefix sums, payload addressing, byte/nibble unpacking and the row-carried cursor/ID dependencies replace memory work with integer and shuffle work. The lower long-scoreboard stall fraction is consistent with reduced DRAM waiting, not an overall speed improvement.

## Rejected decoder candidate

A second implementation reads two-bit control words and uses population counts to locate payloads instead of a subgroup length scan. It passed the same full correctness matrix. Rescreening selected a new schedule, but held-out FP32 dense results were worse: 1.459 ms at 1M and 24.203 ms at 16M, versus 1.411 and 22.424 ms for the simpler decoder. Native paired baselines stayed similar (about 1.15 and 18.6–18.9 ms). The candidate source and binary are archived; it is not an extra production or experiment-dispatch option.

## Costs and next boundary

Compression is currently built from an exact CSR graph and needs its own count/fill passes and a planning prefix sum. Those costs are in each JSON under planning.wall_ms. The row-per-thread encoder favors clear race-free packing; it has not been optimized. Neither direct compressed construction during radius search nor backward/adjoint consumers are implemented. CSR construction remains the peak-memory constraint until that boundary changes.

The next performance experiment would be modest restart segments with explicit absolute IDs/payload offsets, permitting independent segment decode and gathers. That spends some of the memory saving to break row-carried dependencies. The current byte layout already shows that more bytes can buy faster interpolation.

## Files

`summary.json` / `summary.csv`: 432 main-sweep path measurements. `compact/`: four separate fixed-32 nibble comparisons. `bitmask-*`: candidate screens and held-out comparisons. `profile-summary.json`: selected counters; the Nsight report/CSV retain raw results. `source/`: measured sources and rejected candidate. The audit and checksums identify all retained artifacts.

Literature: [Band et al., Compressed Neighbour Lists for SPH](https://cg.informatik.uni-freiburg.de/publications/2019_CGF_CompressedNeighbors.pdf); [Thaler/Keller, 2026 preprint](https://arxiv.org/html/2602.19873v1). This experiment implements their codec ideas on exact particle lists; it does not reproduce the shared-cluster algorithm or claim their full-simulation performance.
