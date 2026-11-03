# Cell-bounded cluster-pair interpolation, 2026-10-31

The experiment is on `codex/native-cluster-pairs`, based on `62295d8`, in
`/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore`.
Production kernels, Warp references, native dispatch and backward/adjoint code
are unchanged. Native/Torch remains the intended primary CPU architecture.

Grouped cluster tiles are competitive with whole-cell traversal for dense FP32
3D interpolation, but are not a general replacement. At one million random
particles and `n_h=4`, they take **1.485 ms**, versus **1.557 ms** for shared
cells and **2.446/2.588 ms** for coarse/refined native CSR. Their cell/cluster
sample ranges overlap, so the small random-fixture difference is not treated as
a reliable improvement. On a regular 3D lattice the grouped tile takes
**1.167 ms**, versus **1.331 ms** for cells: an observed **14.1%
throughput gain**.

For dense random 3D, the selected grouped raw execution representation uses
**33.05 MB**, and compressing the same plan uses **18.68 MB**.
The 64-bit particle neighbor IDs alone require **2.153 GB**.
This representation is compact relative to particle CSR; it does not establish
a memory advantage over the cell metadata from the earlier experiment.

Whole-cell traversal remains faster for dense FP32 2D. Particle CSR wins at
lower neighbor counts and across the large FP64 tests. Cluster construction is
also much more expensive than constructing the simple cell plan. These findings
support retaining an experimental clustered option without adding production
dispatch complexity on the basis of this study alone.

## Method and relation to the literature

This tests small tile and list ideas from [Pall & Hess (2013)](https://arxiv.org/abs/1306.1737)
and [Thaler & Keller (2026)](https://arxiv.org/html/2602.19873). It is a **cell-bounded
Morton adaptation**, not a reproduction of the latter's Hilbert/octree search.
The implementation retains directed Gather accumulation, arbitrary positive
per-particle supports bounded by cell width, and the production Wendland2 helper.

Particles are sorted by coarse Morton cell plus up to three additional bits per
axis inside each cell. Clusters never cross cell boundaries. True partial lengths
are retained, so no dummy particle contributes. Query/source sizes 4/8 give four
tile shapes. Each query cluster has its own list, or up to eight consecutive
clusters share a list with an eight-bit query mask per source entry. Culling uses
either conservative boxes or boxes followed by an exact any-pair radius test.

The first three schedules use register broadcasts, warp-local shared memory, or
one query per thread with a serial source loop. The fourth assigns a CTA to a
query super-cluster, staging 32 source clusters at a time for reuse across all
query warps. This block-wide reuse is important: in dense FP32 3D at 1M, selected
register tiles take 2.417 ms and warp-local shared tiles
2.795 ms, versus 1.485 ms for the grouped block-wide schedule.
These are independently selected configurations.

Raw source IDs and periodic images occupy one 32-bit entry; query masks use one
byte. The optional codec uses ascending cluster-ID deltas in blocks of 32, with
a unity bitmask, info nibbles and variable-length payload nibbles. Periodic images
remain a separate byte per entry. Decoding is cooperative, with warp scans and
shuffles. Grouping, compression and source reuse are independently screened.

The selected 4x8 grouped dense-3D plan reduces visited pair work to
**2.64x** the true neighbor count, versus roughly **6.45x** for whole
cells before grid rounding. It still performs unnecessary distance calculations.
This is particularly costly in FP64 on the local RTX PRO 6000 Blackwell;
clustering reduces the whole-cell penalty but does not reach particle-CSR
throughput. No hardware-counter attribution of the remaining bottleneck is claimed.

## Completed calls at one million particles

All times are milliseconds. Calls include allocations, preparation and Python
dispatch; geometry/list plans are reused. Cells and cluster families use fixed
schedules selected at 262K, not per-row minima from the final sweep. Each family
can choose different tile sizes or grouping, so arbitrary differences between
raw/compressed columns cannot be attributed to the codec alone. All individual
schedules and samples remain in the archive.

**FLOAT32**

| D | n_h | Mean neighbors | Native coarse CSR | Native refined CSR | Selected cells | Raw clusters | Compressed clusters | Warp refined CSR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 9.03 | 0.074 | 0.094 | 0.130 | 0.090 | 0.112 | 0.558* |
| 1 | 2.57 | 6.17 | 0.066 | 0.068 | 0.185 | 0.110 | 0.161 | 0.560* |
| 2 | 4 | 51.25 | 0.329 | 0.329 | 0.221 | 0.273 | 0.341 | 1.156 |
| 2 | 2.57 | 21.75 | 0.180 | 0.182 | 0.187 | 0.198 | 0.227 | 0.805 |
| 3 | 4 | 269.08 | 2.446 | 2.588 | 1.557 | 1.485 | 1.696 | 11.275 |
| 3 | 2.57 | 72.10 | 0.589 | 0.589 | 0.770 | 0.643 | 0.828 | 2.116 |

**FLOAT64**

| D | n_h | Mean neighbors | Native coarse CSR | Native refined CSR | Selected cells | Raw clusters | Compressed clusters | Warp refined CSR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 9.00 | 0.798 | 0.730 | 0.813 | 1.063 | 1.038 | 3.939 |
| 1 | 2.57 | 6.14 | 0.630 | 0.581 | 0.730 | 1.149 | 1.201 | 3.982 |
| 2 | 4 | 51.25 | 3.577 | 3.490 | 4.811 | 5.097 | 4.823 | 10.291 |
| 2 | 2.57 | 21.75 | 1.826 | 1.890 | 2.602 | 3.495 | 3.069 | 5.775 |
| 3 | 4 | 269.08 | 19.618 | 19.346 | 41.958 | 31.169 | 31.020 | 53.700 |
| 3 | 2.57 | 72.10 | 5.924 | 5.831 | 14.358 | 12.464 | 12.519 | 17.987 |

\* Random high-resolution FP32 1D Warp fails the existing numerical parity gate
because of its periodic distance rounding. These curves are excluded from the
plots. Both `2^20`-particle binary-lattice controls pass Warp parity; particle
CSR still beats clustered interpolation there. No tolerance is relaxed.

Refining the within-cell ordering does not uniformly improve particle CSR. It
helps some lattice/low-dimensional cases and hurts others. The coarse/refined
columns prevent this separate ordering change being attributed to cluster tiles.
This study does not compare Morton against Hilbert or other global orderings.

## Fresh execution and storage

Fresh values include actual sorting, state permutation, GPU list construction,
optional compression and one interpolation, rather than adding stage medians.
Uncompressed fresh calls do not construct an unused compressed representation.

| Precision | D | n_h | Native refined CSR | Selected cells | Raw clusters | Compressed clusters |
|---|---|---:|---:|---:|---:|---:|
| float32 | 1 | 4 | 3.099 | 1.417 | 2.451 | 2.508 |
| float32 | 1 | 2.57 | 3.503 | 2.028 | 3.290 | 3.225 |
| float32 | 2 | 4 | 4.592 | 0.857 | 3.322 | 3.609 |
| float32 | 2 | 2.57 | 3.687 | 1.289 | 4.203 | 4.198 |
| float32 | 3 | 4 | 83.736 | 1.816 | 12.874 | 12.934 |
| float32 | 3 | 2.57 | 19.233 | 1.577 | 10.897 | 11.528 |
| float64 | 1 | 4 | 4.604 | 2.179 | 4.537 | 3.739 |
| float64 | 1 | 2.57 | 4.560 | 2.427 | 5.410 | 5.440 |
| float64 | 2 | 4 | 18.330 | 6.491 | 26.233 | 20.291 |
| float64 | 2 | 2.57 | 10.900 | 4.474 | 20.420 | 16.234 |
| float64 | 3 | 4 | 248.218 | 42.929 | 179.423 | 181.347 |
| float64 | 3 | 2.57 | 64.568 | 15.232 | 134.237 | 139.547 |

Dense FP32 3D illustrates the tradeoff: grouped clusters take
12.874 ms fresh, versus 1.816 ms for shared cells
and 83.736 ms for the retained native CSR builder.
The near-identical steady interpolation times do not compensate for the extra
cluster construction work in this fixture. The retained native CSR build includes
sorted-row work and is not claimed to be optimal.

The dense FP32 3D super-cluster controls select the same geometry and launch
schedule for both encodings. Here compression changes interpolation from
1.485 to 1.696 ms, an observed 14.3% runtime cost for the
smaller representation. Other raw/compressed columns may select different layouts.

The compressed representation can **increase** total storage for short 1D lists:
extra row offsets, block headers and periodic-image bytes exceed the index savings.
For `n_h=4` the selected 1D plan uses about 13.43 MB raw versus 14.49 MB compressed.
Compression should therefore remain optional rather than unconditional.

The 33.05/18.68 MB dense-3D numbers include query/source cluster ranges, query-to-list
mapping, row offsets, query masks, and either packed entries or compressed IDs
plus images. They exclude particle state, sorting/cell construction scratch,
bounding boxes/counts, output and packed field scratch. For this FP32 3D case,
packed scratch is another 24 MB and output 12 MB; retaining a sorted state/field
copy costs another 40 MB. The exploratory process holds multiple alternatives,
both encodings and both comparator CSRs simultaneously, so its peak allocated
memory is larger than the execution-representation numbers.

## Qualification, measurement limits and evidence

- **6,144 numerical/list/codec configurations** pass independent FP64 dense
  equations across FP32/FP64, 1D/2D/3D, both order refinements, grids 3/7, variable
  support/mass/density, partial/overfull/empty cells, both culling methods, both
  grouping sizes, four schedules, two block sizes and both encodings.
- Five synthetic codec rows cover empty rows, unity runs, partial 32-entry blocks,
  all direct tags, payload digit boundaries and high 27-bit source IDs. Both an
  independent CPU decoder and the GPU decoder match the original IDs exactly.
- Four additional interpolation cases exercise key-capacity handling. At 1M in
  1D with `n_h=2.57`, optional refinement reduces from eight to four to fit the
  existing 21-bit-per-axis encoder. Coarse cells and search radii are preserved.
  Nondefault-stream planning, preparation and interpolation also pass.
- Sixteen targeted periodic near-cutoff cases cover both precisions, culling
  modes, representations and register/block-wide execution. The old box test
  demonstrably loses a nonzero FP32 contribution; the corrected endpoint
  subtraction order matches the pair kernel and passes without relaxed tolerances.
- Compute Sanitizer checks the cluster namespace on the 256-configuration quick
  matrix plus codec/key/stream cases: racecheck reports zero hazards and memcheck
  reports zero errors. Other library kernels are outside the sanitizer filter.
- Final random sweeps contain 18 cases per precision: dimensions 1/2/3, `n_h=4`
  and 2.57, and N=32768/262144/1000000. There are seven rotated repetitions of
  five calls. Plan/fresh costs have three completed samples. Separate resident
  CUDA-event batches reuse output/scratch, include preparation, and exclude
  per-call Python setup; they are not substituted for public-call timings.
- Independent schedule screens use N=262144; those points are training/control
  sizes. The 32K/1M points are held out. Cell per-mode schedules come from the
  earlier independent screen; the aggregate cell family uses the separate 262K
  control to choose among its direct/shared/flat variants.
- Controls include four FP32 2D/3D lattice cases at 1M, two FP64 3D lattice cases,
  and two FP32 binary-lattice 1D cases. Lattice neighbor counts are recorded;
  for example, dense 3D FP32 has 253.52 CSR entries per query versus 269.08 random.
- GPU process snapshots show no competing compute job in either final random
  sweep or the lattice controls. Snapshots cannot establish exclusivity between
  samples; desktop activity, clocks and scheduling still affect measurements.
  Bands are observed min/max ranges, not confidence intervals. Small differences
  such as the random dense-3D cell/cluster tie are not interpreted as improvements.
- RTX PRO 6000 Blackwell (sm_120), Torch 2.13.0+cu130, Warp 1.17.0, Python 3.13.14;
  `nvcc -std=c++17 -O2 -arch=sm_120`, no fast math. CPU affinity 0-15 and Torch
  threads 16 are fixed. The native comparator is pinned to the retained artifact
  from `3a244a5`; Warp keeps its default neighbor-lane setting.
- CUDA-only, primal-only, unit periodic box, self population, Gather Wendland2
  and a D-component field. General domains, Hilbert/octree construction, global
  SFC clusters crossing cells, adaptive grids, distinct populations, symmetric
  updates, manual AD, moving-particle plan maintenance and HIP qualification are
  outside this experiment. No tests or input-content scans enter timed calls.

## Reproduction and interpretation

See [archived research sources](../docs/development-artifacts.md#research-sources) for commands.
All raw data, selected schedules, exact timed sources, artifacts, checksums,
sanitizer logs, CSV and plots are archived under:

`/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-cluster-pairs-20261031/`

`graphs.md` indexes the throughput and fresh-execution plots. `summary.json` uses
only independently selected schedules. `measured-source/` preserves the exact
timed source files, and the archive audit checks their hashes against each final
report. All final timing, correctness and sanitizer data use the same corrected
CUDA artifact. Earlier screens, the interrupted key-capacity sweep, and the
measurements preceding the periodic box correction are retained under
`before-periodic-box-correction/` as historical evidence, not merged into results.

The next useful production-facing evidence would be real solver particle
snapshots and their rebuild/operator-reuse cadence. This study establishes the
clustered representation and its costs; it does not justify replacing the existing
CSR/cell choices or adopting the complete Hilbert/octree architecture yet.
