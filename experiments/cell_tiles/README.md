# Morton cell-to-cell interpolation toy

This experiment compares particle CSR against directed cell-to-cell interpolation
using cached loads, shared-memory tiles, or shared tiles spanning cell boundaries.
It is separate from production dispatch. The literature, measured outcomes and
limits are in [the study](../../docs/regression/native_backend_cell_tiles_2026_10_31.md).

`n_h` means compact support radius / mean spacing. The primary load is 4;
2.57 is the comparison. Random and lattice fixtures record actual neighbor counts.
The main backend benchmark also defaults to `--n-h 4`; use `--neighbors 32`
when reproducing the historical fixed-degree results.

## Build and qualify

The CUDA library includes the existing numerical headers and links only CUDA.
The toy wrapper uses Torch for GPU allocation, sorting and permutation; `verify.py`
and `resident.py` do not require Warp. Warp is required for the comparison driver.
Use the same CUDA architecture as your GPU, without adding fast math:

```bash
nvcc -std=c++17 -O2 -arch=sm_120 -Xcompiler=-fPIC -shared \
  -Isrc/warpSPHCore/native/csrc experiments/cell_tiles/tiles.cu \
  -o /tmp/warpsph-cell-tiles.so
python experiments/cell_tiles/verify.py \
  --library /tmp/warpsph-cell-tiles.so --output /tmp/cell-tiles-verification.json
```

The independent verification covers FP32/FP64, 1D/2D/3D, periodic grids of size
3/7, empty and overfull cells, bounded tasks, high Morton bits and a nondefault
stream. It never runs as part of an interpolation call. The wrapper assumes
finite in-box data, positive support bounded by cell width, and a unit periodic
box. No input-content or CSR verification is added to timed paths.

## Compare and plot

Use a saved compatible native/Torch artifact from `3a244a5`. Loading an explicit
artifact avoids JIT compilation and pins the production comparator. Record the
artifact hash and keep it with the measurements. The driver sets precision and
CSR64 before importing the package.

```bash
export PYTHONPATH=src
artifact=/path/to/optimized-native-torch.so
archive=/path/to/cell-tiles-results
mkdir -p "$archive"

# Screen both precisions independently at 262K, then fix the selected schedule.
for precision in float32 float64; do
  python experiments/cell_tiles/benchmark.py \
    --library /tmp/warpsph-cell-tiles.so --native-artifact "$artifact" \
    --precision "$precision" --ns 262144 --screen --squared \
    --repeat 5 --iterations 3 --output "$archive/screen-squared-$precision.json"
  python experiments/cell_tiles/benchmark.py \
    --library /tmp/warpsph-cell-tiles.so --native-artifact "$artifact" \
    --precision "$precision" --ns 32768 1000000 --squared --grid-reference \
    --selection "$archive/screen-squared-$precision.json" \
    --output "$archive/final-$precision.json"
done
python experiments/cell_tiles/resident.py \
  --library /tmp/warpsph-cell-tiles.so --archive "$archive" --squared
python experiments/cell_tiles/analyze.py "$archive"
```

Final FP32 measurements additionally include N=262144 for scaling. The final
archive also contains `--layout lattice --dims 2 3 --ns 1000000` and the 1D binary
lattice control, `--layout lattice --dims 1 --ns 1048576`. Reuse the independently
screened FP32 selection for these controls. Random high-resolution 1D FP32 Warp
parity failures are explicitly flagged; binary-lattice and FP64 comparisons
qualify. No tolerance is silently changed for those cases.

The scheduling screen rotates candidate order; final repetitions rotate comparator
order. Fresh calls include sorting, permutation, metadata/CSR allocation and the
operator. Steady calls reuse their geometric plan but include output/scratch
allocation and coefficient preparation. Resident-buffer CUDA events are a separate
experiment, with one C++ batch per sample. They exclude per-call Python descriptor
setup and allocation and cannot be substituted for public-call times.

## Retained variants

| Mode | Traversal |
|---:|---|
| 0 / 1 | Generic minimum-image helper, cached loads / shared cells |
| 3 / 4 | Precomputed cell images, shared cells / cached loads |
| 5 | Precomputed images, full shared tiles across the stencil |
| 6 / 7 / 8 | Conservative squared rejection: shared cells / flat tiles / cached loads |

`--cell-images` screens modes 4/3/5; `--flat` retains the earlier 4/5 two-way
screen. `--squared` screens 8/6/7. Block sizes 32/128/256 and lane widths 1/4/8
are screened. Query tasks are bounded by block size / lane width; a physical
cell may have any occupancy and is split rather than truncated. The stencil
prefix stores only cell ranges, not accepted particle pairs.

The outward squared bound avoids dropping a contribution through a rounded
square-root comparison; surviving pairs use the original Wendland2 evaluation.
It is a coarse rejection optimization, not approximate kernel arithmetic.
The toy is CUDA-only and primal-only. General domains, adaptive grids,
query/reference populations, manual AD and moving-particle plan maintenance need
separate implementation and qualification before production integration.

All raw evidence, historical candidates, binaries and graphs are archived at:
`/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-cell-tiles-20261031/`.
