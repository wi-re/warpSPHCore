---
name: operation-matrix
description: Run scripts/operation_matrix.py, the forward-value diagnostic matrix for sphWarpCore's SPH operators (Density, Interpolate, Gradient, Divergence, Curl, Laplacian) across GradientScheme/LaplacianScheme variants, both traversal modes (adjacency/grid), and all three correction paths (none/CRK/renorm). Use a quick 2D/float32 smoke sweep for routine iteration on operator or kernel code, and the full precision x dim x jitter x device sweep sparingly -- before merging a change that touches kernel math broadly, not on every edit. Pair with the gradcheck skill, which checks backward-mode (gradients) instead of forward values.
---

# Running the operation matrix

`scripts/operation_matrix.py` builds a deterministic (or lightly jittered)
particle lattice with a linear test field, runs every operator/scheme/
traversal/correction combination against it, and prints an MAE-vs-analytic
pass/fail table. It has directly found 5+ real defects (Symmetric-scheme
double-volume bug, curl's CPU compile UB, Laplacian's gradient-scheme
incompatibility, renorm's mixed-backend launch, the `pinv2x2` desync -- see
`warpier_core.md`'s Reality Check section) that the pytest suite's
analytic-property checks didn't catch. It checks **forward values only** --
for gradients, use the `gradcheck` skill instead.

Two ways to run it, matching what CI actually gates on plus a documented
recipe for going wider:

## Quick smoke sweep (routine iteration -- run freely)

```bash
scripts/run_operation_matrix_sweep.sh --quick
# equivalent to:
python scripts/operation_matrix.py --device cpu --ci --verbose
```

2D, float32, non-jittered, `nx=32` (~1000 particles), ~10-15s. This is
exactly the first CI gate in `.github/workflows/tests.yml` -- if this is
red, CI will be red. `--ci` turns the diagnostic into a real exit-code gate
(non-zero on any `HIGH`/`ERR`/`NAN` cell or fatal build error); drop it if
you just want to eyeball the table without a hard pass/fail.

## What CI actually gates on (five targeted runs, not the full product)

Running the full precision x dim x jitter x device product on every push
would be too slow to be worth it as a per-push gate, so CI runs five
specific configurations chosen to hit the axes that have each hidden a real
bug (see `warpier_core.md`'s CI Wiring section for the reasoning behind
each):

```bash
python scripts/operation_matrix.py --device cpu --ci --verbose                                    # 2D float32 baseline
python scripts/operation_matrix.py --device cpu --precision float64 --nx 24 --ci --verbose         # float64: catches raw-literal type-promotion bugs
python scripts/operation_matrix.py --device cpu --dim 1 --nx 64 --ci --verbose                     # 1D: catches bugs 2D can't reach (e.g. the grid-path stride bug)
python scripts/operation_matrix.py --device cpu --jitter 0.01 --ci --verbose                       # confirmed-clean light jitter, actually exercises CRK/renorm a little
python scripts/operation_matrix.py --device cuda --dim 3 --nx 8 --ci --verbose                     # 3D, CUDA only -- see below
```

**3D must run on CUDA, not CPU.** Warp's CPU backend is single-core and
unoptimized; `dim=3` on CPU is prohibitively slow, not just "a bit slower"
(see the `feedback_3d_cpu_perf` note in this project's memory / the CI
workflow's runtime `torch.cuda.is_available()` guard, which no-ops the 3D
step entirely on CPU-only runners). If you don't have a CUDA device
available, skip the 3D case rather than substituting a CPU run with a
smaller `--nx` -- it's not a like-for-like check at any particle count.

## Full sweep (run sparingly, not per-edit)

```bash
scripts/run_operation_matrix_sweep.sh --full
```

Loops `--precision {float32,float64} x --dim {1,2} x --jitter {0.0,0.01} x
--device {cpu, cuda if available}` (all `--ci`-gated, all confirmed to stay
under threshold), then the `--dim 3` CUDA-only cases at the same
precision/jitter combinations, then one extra diagnostic pass at
`--jitter 0.3` **without** `--ci` for human inspection. Takes several
minutes (dozens of individual `operation_matrix.py` invocations, each
recompiling/relaunching every kernel variant) -- reserve it for validating
a change that touches kernel math broadly (a shared `@wp.func`, a traversal
path, a correction-path change), not for routine per-edit iteration; use
`--quick` for that instead.

**Jitter above 0.01 is diagnostic-only, not gated, on purpose.** The
`--jitter 0.3` pass at the end of `--full` is expected to print real `HIGH`
cells -- sound MAE thresholds for heavier jitter (0.15-0.3, the range that
actually stress-tests CRK/renorm the way the notebooks' jittered examples
do) are a still-open Phase-0 item (see `warpier_core.md`), not a regression
signal. Read that table with your eyes, don't key off its exit code. If you
need to sanity-check a specific heavier-jitter case yourself:

```bash
python scripts/operation_matrix.py --device cpu --jitter 0.15 --verbose   # no --ci: reports, doesn't gate
```

## Useful flags when running ad hoc

* `--nx` -- particles per axis (total is `nx**dim`); shrink for `--dim 3`
  (e.g. `--nx 8`), the default `32` is fine for `--dim 1/2`.
* `--n-h` -- particles per smoothing length per axis; the script converts
  this to an actual neighbor count via `n_h_to_nH`, which already accounts
  for `--dim` -- don't try to pick a flat neighbor count yourself across
  dimensions.
* `--threshold` -- MAE pass/fail cutoff (default `0.4`).
* `--verbose` -- also print notes for `HIGH`-error cells.
* `--seed` -- RNG seed for `--jitter` (default `0`).

Full flag reference: `python scripts/operation_matrix.py --help`.
