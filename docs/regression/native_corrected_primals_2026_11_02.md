# Native corrected interpolation and gradient — 2026-11-02

Native interpolation and scalar/vector/matrix gradient now accept supplied CRK
coefficients through the existing `warpOperation`/`sphOperation` arguments.
CPU, CUDA and experimental HIP share the numerical policies and CSR traversal.
No Warp runtime is needed for these calls or their semantic correction states.

## Supported behavior

- Interpolation: query-side CRK A/B and explicit reference volumes.
- Gradient: CRK A/B/gradA/gradB, explicit volumes, grad-h reference-field
  scaling, and query gradient renormalization; all four gradient schemes.
- All 15 kernel families, five existing native support rules, dimensions 1–3,
  FP32/FP64, and scalar/vector/matrix fields.
- Independent CSR32/CSR64 neighbor and offset widths. Direction filtering and
  minimum-image periodic displacement retain their existing semantics.
- `referenceVolumes` defaults to `queryVolumes`, matching the public reference
  API. An explicit reference volume replaces mass/density weighting in
  interpolation and Naive/Difference/Summation gradients. Symmetric gradients
  retain their mass and query/reference density formula, even with volumes.
- `gradHState` divides the reference field by its reference omega. It does not
  divide the query field. Renormalization applies R @ correctedGradient.

`gradB[derivative, component]` is contracted as `gradB @ x_ij`. The CRK product
rule includes all four terms (A W B, A(1+B·x) gradW, (1+B·x) W gradA,
A W (gradB @ x)). Coefficients are prepared once per query/lane, then reused
across directed pair interactions. Outputs belong to the query; no pairwise
atomic updates are introduced.

Only query-side CRK data enters these operators. A `(query, reference)` CRK
state pair selects its first state. Native interpolation permits A/B alone;
unneeded derivative arrays can be `None`. Explicit volumes permit omitted
mass/density inputs for schemes that do not use them. Missing correction
inputs become real nullptrs, without dummy tensor allocations.

`CRKState`, `GradHState`, `RenormalizationState` and their semantic tangent
states now live in the shared Torch-only models. Existing `dataTypes` imports
remain aliases; even `dataTypes.corrections_t` can expose the models in a
native process without importing Warp.

## Architecture and execution boundary

CRK math resides in `crk/native/csrc_correction`; operation-specific policies
remain alongside their interpolation/gradient kernels. The shared corrected
adapters in `coreOperations/native/csrc_correction` reuse the existing row
executor, borrowed CPU views, raw GPU records, reduction and finalization.
They add two stable-ABI registrations with nullable tensor lists; existing
registrations and ordinary operation layouts are unchanged. A header-only
CRK source manifest ensures edits to the shared math participate in build
fingerprints and wheel/JIT compatibility checks.

Validation checks shapes, dtype, device and contiguous storage in C++. There
are no CSR content scans, device-to-host validation copies, debug flags or
persistent correction caches. The caller owns coefficients and their lifetime.

This implementation is **primal-only for corrected calls**. Used inputs that
request reverse-mode or forward-mode differentiation raise explicitly.
`torch.no_grad()` permits primal evaluation of existing differentiable state.
Ordinary uncorrected manual backwards continue to work. Corrected manual
JVP/VJP, native CRK factor construction, corrected divergence/curl/laplacian/
covariance, normalization calibration and PartialSymmetric support remain
separate work. Runtime capability reporting states these boundaries.

## Validation

- Final native CPU/CUDA suite: **11,291 passed, 651 skipped**. Existing
  uncorrected backwards are included.
- Corrected equation matrix: **1,908 cases per device**. An independent Torch
  oracle differentiates the corrected scalar kernel with virtual pair-local
  query displacements and spatially varying A/B; it does not reuse the native
  gradient formula. It covers kernel/support, rank, precision, filtering,
  disordered CSR rows with unused prefix storage, independent integer widths,
  unequal query/reference populations and unwrapped periodic coordinates.
- Additional contracts cover exact linear reproduction in 1D/2D/3D,
  nonsymmetric gradB, self interactions, empty populations, nullable mass/
  density, volume fallback, module-local calls and explicit AD failures.
  The linear reproduction coefficients come from an independent augmented
  moment solve differentiated with Torch.
- Warp reference parity: **288 CPU/CUDA comparisons per precision**, FP32 and
  FP64, including all ranks and gradient schemes with Gather/SuperSymmetric.
- Fresh CPU and CUDA JIT: **1,962 passed, 4 skipped each** on the corrected
  equation/contract matrix before the final three wrapper/fallback additions.
  Those six device cases also pass in the final full suite.
- CUDA Compute Sanitizer memcheck and synccheck: **1,962 passed, 4 skipped
  each; zero errors**, filtered to neighborhood executors.
- Python scaffolding: **20 passed**. Existing CPU standalone target:
  **4 CTests passed**.
- Modified GPU adapters compile with HIP for **gfx942**. No AMD runtime or
  ROCm wheel is qualified.
- CUDA wheel passes stable-ABI symbol/RPATH audit and strict CPU/GPU examples
  with Warp and compiler/JIT imports blocked. The corrected example reproduces
  a linear interpolant and its gradient using supplied coefficients. The wheel
  targets this local CPU and sm_120, pins Torch 2.13.0+cu130, and uses stable
  LibTorch ABI 2.10. It is not a GH200 build.

This pass qualifies capabilities and numerical correctness. Corrected GPU
execution currently uses the raw cooperative eight-lane row executor; packing,
shared-cell production dispatch and performance tuning are follow-up work.
No operator speedup claims are made here.

## Example and artifacts

Run the commented Warp-free example:

```sh
WARPSPHCORE_BACKEND=native python scripts/examples/native_corrected.py --device cpu
WARPSPHCORE_BACKEND=native python scripts/examples/native_corrected.py --device cuda
```

The example computes small initialization-only coefficients with Torch, then
passes contiguous coefficient arrays and explicit volumes to the native
operators. It omits unused mass/density arrays. Its coefficient solve is an
illustration, not a production replacement for `computeCRKFactors`.

Archive: `/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-corrected-primals-20261102`.
Includes committed source, native wheel, CMake configuration, build/JIT/HIP/
sanitizer/parity logs, source and code provenance, and an integrity manifest.
