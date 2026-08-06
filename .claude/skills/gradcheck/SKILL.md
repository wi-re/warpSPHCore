---
name: gradcheck
description: Run sphWarpCore's torch.autograd.gradcheck regression scripts against the SPH operators (Density, Interpolate, Gradient, Divergence, Curl, Laplacian) -- all of them at once, or a single one while iterating on that operator's kernel or the AD bridge. Use whenever rewriting operator kernels, wp_autograd.py, or anything touching backward-mode differentiation, to catch silently-wrong gradients (adjoint bugs, ternary array-read zeroing, reentrancy/caching bugs) that forward-only checks (see the operation-matrix skill) cannot see.
---

# Running the gradcheck scripts

Six operators, six scripts, all under `scripts/`, all calling
`torch.autograd.gradcheck` directly against `warpOperation` with no manual
Jacobian and no workarounds -- if one of these fails, that's a real AD-bridge
or kernel-adjoint bug, not a flaky check. Every one of these bugs found so
far (reentrancy in `wp_autograd.py`, a silently-zeroed adjoint from a
ternary array read in Interpolate, a `float`-literal type-promotion compile
failure and an out-of-bounds `Dot`-scheme read in Laplacian) was invisible
to `scripts/operation_matrix.py`'s forward-only checks -- see
`warpier_core.md`'s "Backward-Mode (Reverse AD) Findings" and "Gradcheck
Script Rollout Plan" for the full history.

```
scripts/gradcheck_density.py             # legacy manual-Jacobian version, kept for extra coverage
scripts/gradcheck_density_native.py
scripts/gradcheck_interpolate_native.py
scripts/gradcheck_gradient_native.py
scripts/gradcheck_divergence_native.py
scripts/gradcheck_curl_native.py
scripts/gradcheck_laplacian_native.py
```

All of them hardcode `SPHWARPCORE_PRECISION=float64` (set via
`os.environ.setdefault` before importing `sphWarpCore` -- required, since
Warp bakes precision into every compiled kernel at first import and it
can't change mid-process) and run on `DEVICE = torch.device("cpu")`
(`scripts/_gradcheck_common.py`) -- there's no `--device`/`--precision` flag
to pass, unlike `operation_matrix.py`.

## Run all of them (all ops, ~30-45s)

```bash
pytest tests/operations/test_gradcheck_scripts.py -v
```

This is what CI runs on every push, as part of `pytest tests/`. Each script
runs as its own subprocess (required for the precision-baking reason above,
not a style choice -- see that file's docstring), asserted to exit `0`. Use
this as the default "did I break gradients anywhere" check.

## Run one operator directly (fast iteration loop)

While actively working on a single operator's kernel or its AD path, call
the script directly instead of going through pytest -- you get the full
stdout (per-case breakdown, e.g. self- vs. non-self-particle gradient terms
for Density; a `--plot` option on `gradcheck_density.py`) instead of a
pass/fail assertion:

```bash
python scripts/gradcheck_gradient_native.py
```

Swap in `curl`/`divergence`/`interpolate`/`laplacian`/`density` for the
operator you're touching. Each takes ~5-7s and prints one `PASSED`/`FAILED`
block per case (multiple `GradientScheme`/`LaplacianScheme` variants, field
ranks, and a "shared query==reference tensor" regression guard where
applicable -- see each script's own module docstring for exactly what it
covers).

## Adding a new operator or a new case

Follow the existing pattern rather than writing one from scratch:

1. Import shared fixtures from `scripts/_gradcheck_common.py`
   (`make_domain`, `single_particle_case`, `line_case`, `grid_case_2d`,
   `build_adjacency`) instead of duplicating setup.
2. Call `torch.autograd.gradcheck(f, (positions, supports, masses, ...))`
   directly against `warpOperation` -- no manual Jacobian, no per-call
   cloning (that workaround predates the reentrancy fix and is obsolete).
3. Register the new script's filename in `GRADCHECK_SCRIPTS` in
   `tests/operations/test_gradcheck_scripts.py` so it's picked up by CI
   automatically.

## A gotcha worth knowing before rewriting kernel code

A Python ternary assigned to a local var inside a `@wp.func`
(`fv = a[i] if cond else a[j]`) can compile fine and run the right branch at
runtime, yet silently produce a **zero adjoint** for the array read -- this
is exactly what Stage 2 (Interpolate) found. The explicit `if: ... else:`
block form doesn't have this problem and is what every other operator
already uses. If a gradcheck starts failing right after refactoring a
branch like this into a ternary, check that first.

## `repro_warp_grad_reentrancy.py` is not part of this

`scripts/repro_warp_grad_reentrancy.py` always exits `0` and prints an
illustrative 4-row PASS/FAIL matrix where three of the four rows are
*expected* to fail (only the "both fixes" row should be all-PASS) -- it's a
standing demonstration of the reentrancy bug and its fix, not a gate. Not
included in `test_gradcheck_scripts.py` for that reason. Re-run it by hand
if you touch `WarpFunctionWrapper.backward` / `StateAwareWarpFunction.backward`
in `wp_autograd.py`; it should always print `PASS`/`PASS` on the last row.
