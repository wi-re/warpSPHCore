# Morton cell-to-cell interpolation experiment, 2026-10-31

The toy lives on `codex/native-cell-tiles`, based on native checkpoint `3a244a5`,
in `/home/lu26029/.codex/worktrees/native-cell-tiles/warpSPHCore`.
It changes no production numerical kernel or dispatch. Native/Torch remains the
primary architecture. The primary backend benchmark now defaults to `n_h=4`,
with `--n-h 2.57` for comparison and `--neighbors 32` for historical reproduction.

## Literature survey

The closest match to the proposed method is **Goswami, Schlegel, Solenthaler and
Pajarola (2010), Interactive SPH Simulation and Rendering on the GPU**. Their
Section 3 sorts by Z-index, creates bounded particle blocks and stages neighboring
blocks into CUDA shared memory. Threads reuse the staged records across query
particles and retain a distance cutoff. This closely matches the suggested
organization; identifying it as the remembered paper is an inference.
[Author-hosted paper](https://www.ifi.uzh.ch/vmml/publications/interactive-sph/InteractiveSPH.pdf).

**Huang et al. (2019), A General Novel Parallel Framework for SPH-centric
Algorithms**, assigns each CTA's query particles to one cell and stages contiguous
neighbor data for shared-memory reuse. It also explores register reuse and task
assignment. It supports treating occupancy and thread assignment as part of the
algorithm, rather than assuming spatial sorting alone gives the benefit.
[Author/university publication record](https://researchconnect.suny.edu/en/publications/a-general-novel-parallel-framework-for-sph-centric-algorithms/),
[DOI](https://doi.org/10.1145/3321360).

**Algis, Bramas, Darles and Aveneau (2024), Efficient GPU Implementation of Particle
Interactions with Cutoff Radius and Few Particles per Cell**, compares direct
cell traversal, staged cells, multi-cell sub-boxes and pencils on CUDA/HIP GPUs.
It identifies occupancy, synchronization and small/uneven cell populations as
important limits; the proposed shared-memory schemes win in specific regimes.
This motivated a cached-load control and a variant that stages full tiles across
cell boundaries. [Full paper](https://arxiv.org/html/2406.16091v1).

**Gonnet (2014), Efficient and Scalable Algorithms for Smoothed Particle
Hydrodynamics on Hybrid Shared/Distributed-Memory Architectures**, expresses SPH
as cell self/pair tasks, with hierarchical subdivision and projected sorted
interactions. It is a complementary approach: better pair culling reduces the
spurious work, rather than relying entirely on reuse to make that work cheap.
Its mutual-update task scheduling is not implemented by this directed GPU toy.
[Full paper](https://arxiv.org/html/1404.2303v1).

## Neighborhood convention and geometry

Here `n_h = h / spacing`, where `h` is this repository's compact support radius,
not half that radius. For uniform random points in a unit periodic domain,
spacing is `N^(-1/D)` and the expected other-neighbor count is `V_D * n_h^D`:

| n_h | 1D | 2D | 3D |
|---|---:|---:|---:|
| 4 | 8 | 50.27 | 268.08 |
| 2.57 | 5.14 | 20.75 | 71.10 |

The self entry adds one for the random fixture. Lattices have discrete shell
counts, so the actual measured degree is always recorded. The toy has constant
support; the general backend benchmark retains ±2% support variation.

For cell width equal to support, the geometric candidate/ball-volume ratio is
`3^D / V_D`: **1.5×, 2.86×, 6.45×** in 1D/2D/3D. A periodic grid uses
`grid=floor(1/h)` and width `1/grid >= h`, so integer grid rounding can enlarge
that ratio. All recorded candidate counts use actual cell populations.
Morton order is a locality mechanism: consecutive Z indices are not assumed to
be the complete physical neighborhood.

## Toy design

1. Encode integer cell coordinates as 63-bit Morton keys on the GPU, sort keys
   with Torch, and permute all particle attributes and the vector field.
2. Build occupied-cell offsets and the exact periodic 3/9/27-cell stencil using
   coordinate offsets and binary lookup of occupied Morton keys. Empty cells
   have no particle range. Cell images are determined once per cell pair.
3. Split a query cell into tasks of at most `block_size / lanes_per_query`
   particles. Overfull cells create more tasks; no particles are dropped and no
   maximum occupancy is assumed. Reference cells are also processed in bounded
   tiles. Inactive query groups still participate in barriers.
4. Prepare SoA position/weighted-field records once per operation, using
   `field_j * mass_j / density_j`. The interpolation uses the production Wendland2
   numerical helper and returns results in Morton query order.
5. Compare cached global loads, one-cell-at-a-time shared tiles, and shared tiles
   spanning cell boundaries. The latter uses a small per-cell stencil prefix
   array to fill tiles; it does not materialize a particle CSR or a 27N reference
   list. All variants keep directed query accumulation without output atomics.

The final variants conservatively reject squared distances beyond an outward
rounded support bound before taking a square root. Surviving interactions use
the same square root and Wendland evaluation. No fast math or approximate
kernel is enabled. Earlier variants without this rejection remain in the source
and archive. Shared memory reduces repeated loads; it does not eliminate the
extra distance calculations or guarantee a speedup.

## Completed-call measurements

Completed wall times below include output/scratch allocation, descriptor setup,
coefficient preparation and synchronization. CSR or cell plans are already built
and reused. Values are milliseconds at **one million random particles**. Warp
columns use the same mathematical inputs; native and Warp CSR comparisons reuse
the same native-generated CSR within each particle ordering.

**FP32**

| D | n_h | Mean neighbors | Native input CSR | Native Morton CSR | Shared cells | Flat stencil | Warp Morton CSR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 9.03 | 0.101 | 0.093 | 0.191 | 0.133 | 0.559 |
| 1 | 2.57 | 6.17 | 0.082 | 0.071 | 0.203 | 0.192 | 0.559 |
| 2 | 4 | 51.25 | 0.454 | 0.346 | 0.284 | 0.219 | 1.238 |
| 2 | 2.57 | 21.75 | 0.220 | 0.186 | 0.294 | 0.226 | 0.755 |
| 3 | 4 | 269.08 | 2.471 | 2.508 | 1.706 | 2.194 | 12.169 |
| 3 | 2.57 | 72.10 | 0.758 | 0.605 | 0.932 | 0.792 | 2.261 |

**FP64**

| D | n_h | Mean neighbors | Native input CSR | Native Morton CSR | Shared cells | Flat stencil | Warp Morton CSR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 9.00 | 0.849 | 0.814 | 1.120 | 0.796 | 3.936 |
| 1 | 2.57 | 6.14 | 0.709 | 0.641 | 1.175 | 0.790 | 3.930 |
| 2 | 4 | 51.25 | 3.824 | 3.534 | 5.736 | 5.428 | 10.163 |
| 2 | 2.57 | 21.75 | 1.892 | 1.841 | 3.600 | 2.647 | 5.674 |
| 3 | 4 | 269.08 | 20.937 | 19.635 | 54.304 | 52.494 | 53.818 |
| 3 | 2.57 | 72.10 | 6.219 | 5.901 | 18.119 | 16.303 | 17.993 |

**Fresh build plus one interpolation, milliseconds**

| Precision | D | n_h | Sort + native CSR | Sort + shared cells | Sort + flat stencil | Warp fresh grid |
|---|---|---:|---:|---:|---:|---:|
| FP32 | 1 | 4 | 1.842 | 1.425 | 1.610 | 3.571 |
| FP32 | 1 | 2.57 | 2.106 | 1.879 | 1.891 | 2.952 |
| FP32 | 2 | 4 | 4.177 | 0.936 | 0.888 | 11.221 |
| FP32 | 2 | 2.57 | 2.733 | 1.260 | 1.175 | 7.034 |
| FP32 | 3 | 4 | 84.338 | 2.108 | 2.774 | 100.141 |
| FP32 | 3 | 2.57 | 18.281 | 1.535 | 1.351 | 34.890 |
| FP64 | 1 | 4 | 3.337 | 2.690 | 2.073 | 22.544 |
| FP64 | 1 | 2.57 | 3.037 | 2.999 | 2.447 | 17.482 |
| FP64 | 2 | 4 | 17.689 | 7.517 | 7.680 | 108.685 |
| FP64 | 2 | 2.57 | 10.009 | 5.301 | 4.823 | 58.071 |
| FP64 | 3 | 4 | 248.786 | 54.898 | 53.629 | 508.726 |
| FP64 | 3 | 2.57 | 65.459 | 18.929 | 17.277 | 196.562 |

The fresh columns are actual measured build-plus-interpolation calls, not sums
of independent stage medians. Morton CSR includes sorting/permutation before
native CSR construction. Tiled execution includes sorting, permutation, cell
metadata and task construction. Warp's fresh grid comparator avoids particle CSR
as well. Its retained-grid comparator is also measured, so the result is not
attributed solely to avoiding a CSR when comparing against Warp.

The useful operator-only wins are concentrated in **FP32 at n_h=4**:
flattened cells improve 2D from 0.346 to 0.219 ms (**1.58×** over already-sorted
native CSR), while shared cells improve 3D from 2.508 to 1.706 ms (**1.47×**).
Relative to input-order native CSR, those gains are 2.07× and 1.45×. Sorting alone
accounts for some of the 2D improvement, but little of the 3D improvement.
The lattice controls retain the pattern: 1.37× in 2D and 1.45× in 3D over sorted
CSR at n_h=4. Shared staging versus cached loads is an explicit measured control.

At n_h=2.57, sorted CSR is faster in the large FP32 2D/3D cases. In 1D, small
cell populations and large numbers of tiny CTAs make the tiled scheme slower;
the binary-lattice control confirms that this result does not depend on the
Warp periodic-rounding discrepancy. FP64's extra arithmetic is expensive on this
GPU: dense 3D flat tiles take 52.494 ms versus 19.635 ms for sorted native CSR.
The squared rejection control does not remove that tradeoff. These timings are
consistent with the extra arithmetic limiting reuse benefits; no hardware-counter
FLOP attribution was measured in this toy study.

The larger practical gain is **avoiding particle CSR construction**, especially
its count/collect and sorted-row work. Dense FP32 3D takes 2.108 ms for a fresh
shared-cell plan plus interpolation versus 84.338 ms for sorting, building native
CSR and interpolating (**40.0×**). This is a fresh-execution comparison, not a
40× improvement of interpolation alone. The saved CSR path was not modified to
omit row sorting, and this experiment does not establish that its current build
cost is optimal.

Plan reuse changes the choice. A simple model `fresh + (K-1)*interpolation` predicts
that sparse FP32 3D flat tiles lose their fresh advantage after about 91 repeated
interpolations on unchanged geometry; dense FP64 loses it around seven calls.
This model uses measured medians and assumes fixed geometry/order and no other
work. A simulation should choose using its actual operator mix, precision,
plan-update rate and memory budget. The experiment supports retaining a tiled
execution option, rather than replacing all CSR execution.

## Throughput and calling overhead

**FP32 useful billion interactions/second, one million particles**

| D | n_h | Shared completed call | Flat completed call | Shared resident + prep | Flat resident + prep |
|---|---:|---:|---:|---:|---:|
| 1 | 4 | 47.39 | 67.74 | 66.84 | 75.78 |
| 1 | 2.57 | 30.31 | 32.09 | 34.48 | 36.14 |
| 2 | 4 | 180.61 | 234.17 | 165.52 | 250.49 |
| 2 | 2.57 | 73.93 | 96.42 | 71.65 | 122.18 |
| 3 | 4 | 157.73 | 122.63 | 147.81 | 114.37 |
| 3 | 2.57 | 77.36 | 90.98 | 80.57 | 93.95 |

Useful pair throughput uses actual CSR entries; visited pair throughput also
counts rejected candidate pairs. These are different quantities. The separate
`resident.json` experiment reuses output/scratch and submits multiple iterations
in one C++ call. It reports both preparation-plus-interpolation and
interaction-only CUDA-event intervals. It excludes allocation and per-call
Python descriptor setup and must not be presented as a public API speedup.
Resident and completed-call data are separate runs; clocks, memory placement and
scheduling can make either median higher. Subtracting those medians does not
isolate exact Python overhead. `throughput.csv` retains every completed-call
pair rate and every sample range.

## Cost, scope and qualification

- Cell metadata contains Morton permutation/keys, counts/offsets, neighbor cell
  IDs, image shifts and stencil prefixes. It scales with occupied cells rather
  than accepted particle edges. Sorted state and task arrays are additional.
  For dense FP32 3D at 1M, metadata is 16.88 MB and shared-task arrays
  are 0.46 MB, versus approximately 2.15 GB for CSR
  neighbor IDs alone. A retained sorted state/field copy adds 40 MB in this case.
- Packed scratch uses `2*D` scalars per reference: 8/16/24 MB per million FP32
  particles in 1D/2D/3D; FP64 doubles that. Shared staging uses `2*D*block_size`
  scalars for the cell variant or `3*D*block_size` for the flattened variant,
  including image shifts. Output is another D-component array.
- Toy restrictions: CUDA, a unit periodic box, in-box finite positions, Gather,
  Wendland2, scalar mass/density and D-component fields. The benchmarks use
  self populations. Variable supports bounded by cell width are independently
  tested, but adaptive grids, distinct query/reference populations, AD and plan
  update/Verlet lifecycles are not implemented. A real timestep must rebuild or
  update the plan when particles change cells or the support bound changes.
- Sorting is an explicit architectural tradeoff. Callers may retain Morton order;
  restoring a single output to input order is separately measured. Integrating
  this into a solver would require all fields, IDs, boundary data and any future
  gradients to follow the same permutation contract.
- **864 numerical/planner configurations** pass independent FP64 dense Torch
  equations across dimensions, precisions, modes, block/lane choices, empty and
  overfull cells, periodic boundaries, grids of size 3/7 and partial query tasks.
  High Morton bits and nondefault-stream execution are covered. The Python
  scaffolding suite passes **19 tests**. CPU fixture checks cover both n_h values
  and the legacy fixed-degree override. No verification runs inside timed calls.
- Random high-resolution **1D FP32 Warp values fail the existing parity gate**
  because its `(dx+0.5)-0.5` periodic formula loses bits. Those timings are flagged
  and are not treated as numerically qualified speedups over Warp. The native
  and toy paths match. A separate 2^20-particle binary lattice passes Warp parity
  for both n_h values; all FP64 fixtures qualify. Standard-lattice 2D/3D controls
  are retained independently of the random matrix.
- RTX PRO 6000 Blackwell, sm_120; Torch 2.13.0+cu130, Warp 1.17.0, Python 3.13.14.
  CUDA builds use `-O2 -std=c++17 -arch=sm_120`, without fast math. CPU affinity
  0–15 and Torch thread count 16 are fixed. Warp uses its default 32 neighbor
  lanes; no per-fixture Warp lane sweep was performed. No CPU speedup or AMD
  runtime claim follows from this CUDA-only toy.
- Configurations are screened at N=262,144; final 32K/1M measurements are separate
  repetitions with fixed selections. Seven final batches use five calls each,
  with rotated comparator order. Fresh calls and planning use three samples.
  FP32 also includes 262K scaling, which is the training size. The final plots
  use complete last sweeps rather than selected individual minima.
- Activity snapshots show no competing compute process in all final random,
  lattice and resident runs. Snapshots cannot prove exclusivity between samples; desktop activity, clocks and scheduling noise remain. Error bands
  represent observed min–max ranges, not confidence intervals.
- An early `N^(-1/3)` fixture calculation rounded support just above an exact
  cell boundary, reducing the FP64 grid resolution. The final fixture computes
  spacing exactly for integer lattice sizes. The affected complete sweeps are
  retained under `before-exact-spacing/`; the squared-distance experiment is
  kept separate from that correction when interpreting gains.

## Reproduction and retained evidence

See [archived research sources](../docs/development-artifacts.md#research-sources). Code, source/binary hashes, selected
configurations, all raw timings, correctness results and graphs live in:

`/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-cell-tiles-20261031/`

`graphs.md` indexes figures. `SHA256SUMS.json` inventories the archive; historical
artifacts are matched by hash rather than by paths reused during experimentation.
The precomputed cell-image, generic periodic, flattened and squared-rejection
candidates are all retained. Production Warp, native kernels and manual
backward/adjoint files remain unchanged on this branch.
