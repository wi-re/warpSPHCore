# Cluster-pair interpolation experiment

An isolated CUDA primal experiment on `codex/native-cluster-pairs`, based on
`62295d8`. It adds no production dispatch, gradient implementation, or runtime
validation. The existing Warp implementations remain the readable references.

The design tests ideas from [Páll & Hess (2013)](https://arxiv.org/abs/1306.1737)
and [Thaler & Keller (2026)](https://arxiv.org/html/2602.19873): small interaction
tiles, query super-cluster masks, and compressed neighbor-cluster indices.
This is a cell-bounded Morton adaptation, not a reproduction of the latter's
Hilbert/octree search, symmetric accumulation, or multi-GPU machinery.

Measured outcomes, qualifications and selected schedules are in
[the study](../../docs/regression/native_backend_cluster_pairs_2026_10_31.md).

## Representation and execution

- Sort particles by a refined Morton key: the coarse cell key is unchanged,
  while additional low bits improve ordering inside each cell. Default
  refinement is up to eight subdivisions per axis; `--refinement 1` is the control.
  Very fine coarse grids use fewer refinement bits to stay within the existing
  21-bit-per-axis encoder. Each case records the actual refinement (four for
  the one-million-particle 1D `n_h=2.57` case).
- Split each occupied cell into query/source clusters of 4 or 8 particles.
  Partial clusters retain their true length; no padding particle participates.
  The four tile shapes are 4x4, 4x8, 8x4 and 8x8.
- Retain one list per query cluster or share a list among up to eight query
  clusters in the same cell. Each source entry has an eight-bit query mask.
- Compare conservative bounding-box culling with bounding boxes followed by an
  exact any-particle-pair test. Both preserve individual Gather support radii.
  The interpolation still tests every candidate particle distance.
- Raw entries pack a source-cluster ID and its periodic image into 32 bits.
  Cluster IDs are sorted by traversing occupied neighbor cells in sorted order.
- Compression uses blocks of 32 ascending source IDs: delta coding, a 32-bit
  unity mask, info nibbles for remaining deltas, and variable-length payload
  nibbles. Periodic images and query masks each remain a separate byte per entry.
  The GPU decoder uses warp scans and shuffles; the independent verifier also
  includes a CPU decoder and high-index synthetic codec cases.

Four execution schedules use the same numerical helper and plans:

| Mode | Source reuse |
|---:|---|
| 0 | A warp evaluates a small pair tile using register broadcasts |
| 1 | The same tile uses warp-local shared memory |
| 2 | One query per thread loops over source particles using broadcasts |
| 3 | A CTA handles eight query clusters and stages 32 source clusters at a time |

Mode 3 shares the source loads across query clusters, rather than only within
one warp. Query masks skip irrelevant cluster pairs. An 8x8 tile takes two
warps; other pair tiles take one. Normal pair/serial launches screen 4/8 warps
per CTA. Super-cluster launches use 8/16 warps as required by their tile shape.

The unit box is periodic in every dimension. Support must be positive and at
most the coarse cell width, with at least three cells per axis. There is one
self population, scalar mass/density, and a D-component field. The raw format
allows fewer than 2^27 source clusters. Directed Gather accumulation includes
self contributions and uses no atomics. Source records use the existing SoA
position/weighted-field preparation. No fast math is enabled.

## Build and verify

The library includes the previous cell experiment so its keys, cell planning
and interpolation comparators use the same compiled artifact. Adjust the
architecture for your GPU; this experiment has no HIP runtime qualification.

```bash
nvcc -std=c++17 -O2 -arch=sm_120 -Xcompiler=-fPIC -shared \
  -Isrc/warpSPHCore/native/csrc experiments/cluster_pairs/clusters.cu \
  -o /tmp/warpsph-clusters.so
python experiments/cluster_pairs/verify.py \
  --library /tmp/warpsph-clusters.so --output /tmp/cluster-verification.json
```

Verification covers FP32/FP64, 1D/2D/3D, coarse/refined order, periodic grids of
size 3/7, variable support/mass/density, empty/overfull cells, partial tiles,
both pruning strategies, both grouping sizes, all four execution modes,
compression, both block sizes, codec boundaries and a nondefault stream.
Sixteen additional periodic near-cutoff cases cover the box subtraction order;
the old planner fails this targeted regression and the corrected planner passes.
These checks run only in the verifier, outside operation execution and timings.

## Screen, measure and plot

Use the retained native/Torch artifact from `3a244a5`, and the previous cell
experiment's independent schedules. The scripts record source/binary hashes,
all individual samples and GPU process snapshots.

```bash
artifact=/path/to/optimized-native-torch.so
cell_archive=/path/to/native-cell-tiles-20261031
archive=/path/to/cluster-results
for precision in float32 float64; do
  python experiments/cluster_pairs/benchmark.py \
    --library /tmp/warpsph-clusters.so --native-artifact "$artifact" \
    --cell-selection-dir "$cell_archive" --precision "$precision" \
    --screen --ns 262144 --output "$archive/screen-$precision.json"
  python experiments/cluster_pairs/benchmark.py \
    --library /tmp/warpsph-clusters.so --native-artifact "$artifact" \
    --cell-selection-dir "$cell_archive" --precision "$precision" \
    --selection "$archive/screen-$precision.json" \
    --ns 32768 262144 1000000 --output "$archive/final-$precision.json"
done
python experiments/cluster_pairs/analyze.py "$archive"
```

Throughput plots use **higher is faster**; fresh execution time plots use
**lower is faster**. Both directions are stated in the figure titles and axes.

The primary load is `n_h=4`; `2.57` is included in every sweep. This means
compact support radius divided by mean particle spacing, not half-support.
Random fixtures reproduce the previous study's seeds and physical attributes.
The final run separately measures coarse-only native CSR and refined native
CSR, so improved ordering is not attributed to cluster execution.

Completed calls include output/scratch allocation, preparation and Python
dispatch. Resident-buffer measurements submit multiple operations in one C++
call and include preparation. Neither includes list construction. Fresh calls
include sorting, state permutation, count/scan/fill, optional compression and
one interpolation. Raw fresh calls do not build an unused compressed list.
Variable-size allocations synchronize during planning; interpolation performs
no host data reads. There is no plan maintenance or Verlet-skin lifecycle.

The screen retains all configurations and selects one schedule per execution
mode and representation. Final comparisons rotate order and retain sample
ranges. Aggregate plots choose cluster schedules from the independent 262K
screen; the cell-family aggregate uses its separate 262K control measurement.
Thus the 262K points are training/control sizes, while 32K and 1M are held out.

Memory statistics distinguish the representation needed during execution from
the larger exploratory process, which retains raw/compressed lists, construction
scratch and multiple alternatives simultaneously. Sorted particle state, packed
field scratch and output are additional to the reported list representation.
The known random high-resolution FP32 1D Warp periodic-rounding discrepancy is
reported explicitly; those unqualified Warp curves are omitted from plots.

The complete study archive, including raw samples, exact timed sources, binary
artifacts, plots, sanitizer logs and checksums, is at
`/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-cluster-pairs-20261031/`.
