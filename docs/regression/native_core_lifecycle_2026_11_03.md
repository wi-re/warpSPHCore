# Native core closure, stage 1 — 2026-11-03

The first closure stage adds Warp-free Verlet construction/reuse, device validity
metrics, filtering, capacity handling, and the remaining standard SPH options.
The frontend and installed editable packages are unchanged. The mixed frontend
continues selecting its existing explicit search calls until frontend integration.

## Implementation

Motion/support metrics and candidate filtering live in module-owned
`radiusSearch/native/csrc_verlet`. CPU uses Torch parallel work; CUDA/HIP use
block reductions and the same per-particle equations. No floating atomics are
needed. Validity is a device flag; eager rebuilding reads that scalar, while
filtering reads the scan total needed for variable-sized allocation. These are
algorithmic lifecycle decisions, not debug scans. No CSR or particle arrays are
copied to the CPU for validation.

Capacity policy preserves optional COO rows, CSR-only storage, spare -1 padding
and in-place array identity. Rebuild snapshots own their position/support storage.
Support-policy changes and changes to domain tensor identity/version force a
rebuild without comparing domain values on the host. Inference-mode domain
buffers lack mutation versions and therefore conservatively rebuild. Native
variable-size build/filter still rejects CUDA graph capture. Warp grid factories
and Warp-specific collection-order tunings remain backend-specific interfaces.

Normalization calibration uses the shared lattice-shell calculation with native
host kernel evaluation; no Warp import is needed. It scales floating operator
outputs and their derivatives while preserving integer count outputs. The
fixed-geometry linear field maps carry the same scalar in products/pullbacks.
Host support conversions were separated from their Warp decorators so these
utilities can be imported from a native-only process.

Positive-divergence Laplacians use Warp's per-pair gate. Primals and manual
JVP/VJP retain the active branch and the legacy matrix-channel behavior. The
option is nonlinear in fields and is rejected by the fixed *linear* field-map
API. This is a mathematical distinction, not a fallback to Warp. A shape check
rejects scalar positive-divergence fields outside 1D. Geometry derivatives at a
clipping transition are not claimed.

## Focused qualification

| Check | Result |
|---|---|
| Python scaffolding | 25 passed |
| CPU native lifecycle/options and existing primal/contract/hot-path checks | 181 passed, 4 skipped, 587 extended cases deselected |
| CUDA native lifecycle/options and existing primal/hot-path checks | 566 passed, 28 skipped |
| CPU Warp parity | Passing FP32 and separate FP64 processes |
| CUDA Warp parity | Passing FP32 |
| Offline HIP compilation | Verlet, corrected Laplacian and full differential GPU translation units compile for gfx942 |
| Strict prebuilt CUDA loader | Seven focused lifecycle/options tests pass with the rebuilt library and source receipt |
| Warp-blocked public example | Passing; includes Verlet reuse/filtering and calibrated density |

Independent oracles check periodic motion, multi-block reductions, separate
populations, support changes, all support policies, exact-cutoff filtering,
empty sets, CSR32/64 input storage, capacity/padding, snapshot mutation, domain/
policy invalidation, and non-default CUDA stream execution. Operator checks use
an independent lattice-point sum and dense differentiable pair equations;
clipped pullbacks cover positions, supports, masses, densities, fields and domain
bounds. Field-map transpose identities and forward AD are also checked.

New checks are seven focused native tests and one bounded parity test, with
rotated dimensions/precisions inside the relevant contracts rather than a new
exhaustive Cartesian suite. Existing unsupported AD/field combinations account
for the reported skips. No performance claim or full-simulation timing is made.
HIP is compile-qualified only; no AMD hardware or ROCm wheel was exercised.

Builds and evidence are external. The retained stage archive is
`/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-core-lifecycle-20261103`.
The next stage is renormalization, small matrices and solver dependencies; this
does not declare the complete core port or its merge qualification finished.
