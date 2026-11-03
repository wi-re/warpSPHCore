# Native CRK preparation — 2026-11-02

## Result

The native backend now builds the complete CRK primal preparation pipeline:
apparent volumes, six geometric moment tensors and counts, the existing Torch
factor solve, and CRK consistency density. The outputs feed corrected native
interpolation and gradient directly. This removes the need to construct or
supply CRK coefficients through Warp or an illustrative initialization solve.

Public exports are available from the selected native root and `crk.native`:
`computeCRKVolume`, `computeCRKMoments`, `computeCRKDensity`, `computeCRKTerms`,
`computeCRKFactors`, `prepareCRK` and `CRKPreparation`. The legacy three-value
`computeCRKFactors` result and `computeCRKTermsWarp` alias remain compatible.
The shared factor algebra no longer imports `type_config` or synchronizes to
print a numerical-limiter warning. Its existing numerical masks are retained.

## Numerical and integration contract

- Dimensions 1/2/3, FP32/FP64, all 15 kernels and five native support schemes
  in the primitive volume/moment passes. The high-level pipeline preserves
  Gather volume, Scatter moments and Scatter consistency-density semantics.
- Moments preserve `dm1[component, derivative]` and
  `dm2[derivative, component, component]`; factors preserve
  `gradB[derivative, component]`. Corrected gradients contract `gradB @ dx`.
- Volume is zero for an empty or masked row; otherwise it is the reciprocal
  kernel sum. Density accumulates mass times volume times corrected kernel
  and volume squared times corrected kernel before dividing once per query.
- Moment counts match Warp's directionally admitted edges, including
  zero-weight edges in an explicitly supplied broad CSR list. No extra CSR
  content validation runs in production.
- CSR particle IDs and offsets independently accept int32/int64; counts are
  int32. Nonzero storage prefixes, reversed rows and unused storage work.
- `prepareCRK` returns query volumes, reference volumes, diagnostic density
  and the shared `CRKState`, all in input population order. It owns no hidden
  sorting, permutation or topology cache. This is the boundary for upstream
  sorted frontend and solver integration.
- A distinct reference population estimates volumes against itself. Its
  optional `referenceAdjacency` is separate from the query/reference CSR.
  `referenceVolumeMode` retains the existing independent interaction rule,
  including the explicit override for aliased populations.
- A supplied query/reference CSR must cover both Gather and Scatter pairs.
  Union/SuperSymmetric search is suitable for unequal supports. With no list,
  the wrapper builds one union graph and reuses it for all three passes.
  Supplying both required lists triggers no hidden radius search.
- The frontend must recompute corrections when geometry, supports or category
  rules change, even while retaining topology. Permutation equivariance and
  independently supplied population topologies are tested.
- CPU uses Torch's parallel scheduler; CUDA/HIP uses the shared cooperative
  row executor on Torch's current stream. No primal scatter atomics are added.
  The Torch-only small matrix solve remains appropriate for the primary CPU
  architecture and shares the existing limiter with Warp.

Module-owned numerical policies live under `crk/native/csrc_volume`,
`csrc_moments` and `csrc_density`. `csrc_preparation` adapts nullable stable Torch
handles to shared loaders/executors. Generic optional-handle helpers live in
`native/csrc/torch/types.h`, rather than depending on a core correction operator.

A real JIT build exposed duplicate `nt_density_forward` object names across
modules. The shared JIT source helper now stages unique include wrappers only
for colliding translation units. Including the original file preserves local
header resolution and dependency tracking; source hashes also update Torch's
in-process versioner. Frontend modules can retain the agreed per-function
filenames without imposing a global basename registry.

## Qualification

| Check | Result |
|---|---|
| Full source native CPU/CUDA suite | 12,473 passed, 651 skipped |
| New independent preparation equations/contracts | 591 cases per device |
| Native/Warp preparation parity | 36 CPU/CUDA cases per precision; 72 total |
| Shared factor limiter/JVP/reverse gradchecks | 36 passed |
| Fresh CPU JIT preparation matrix | 591 passed |
| Fresh CUDA JIT preparation matrix | 591 passed |
| Compute Sanitizer memcheck | 141 passed, zero errors; neighborhood kernels filtered |
| Compute Sanitizer synccheck | 141 passed, zero errors; neighborhood kernels filtered |
| Python scaffolding | 23 passed |
| HIP offline compilation | Three new GPU adapters compiled for gfx942 |
| CUDA wheel | Stable-ABI/RPATH audit passed; strict CPU/CUDA examples and complete native suites passed |

The new equation oracle differentiates independent Torch moment expressions
rather than duplicating the native derivative formula. Complete-pipeline tests
reproduce linear fields and gradients in all dimensions and precisions, exercise
native automatic search as well as supplied CSR, and cover independent query and
reference ordering, aliased/reference-mode overrides, self interactions, nullable
unused masses/densities, empty populations, degenerate stencils and AD guards.

Gaussian tails can underflow in FP32; reciprocal sums near zero are ill-conditioned.
The general reciprocal equation fixtures use resolvable positive sums for Gaussian
and viscosity kernels. A separate precision-specific Gaussian test preserves the
actual underflow behavior rather than requiring a double-precision reciprocal
from an FP32 zero. No new clamping or numerical regularization is introduced.

The wheel targets the local x86_64 CPU and sm_120, pins Torch 2.13.0+cu130 and
uses stable LibTorch ABI 2.10. Warp and compiler/JIT imports are blocked while
running the complete preparation/interpolation/gradient example and native suites.
Packaged Python/C++ sources and compiled `.text`/`.nv_fatbin` hashes are checked
against the staged implementation. This does not qualify GH200 or AMD runtime.

## Use and remaining work

Run `WARPSPHCORE_BACKEND=native python scripts/examples/native_corrected.py
--device cpu` or `--device cuda`. The commented example now uses real native
preparation, including distinct reference volumes, rather than manually supplied
coefficients. The general frontend contract is documented in `docs/docs/native-backend.md`.

Preparation and corrected pair operators are primal-only. They reject reverse
and forward AD requests; `torch.no_grad()` permits primal execution of otherwise
differentiable states. The pure Torch factor algebra remains differentiable, but
manual volume/moment/density derivatives are needed for the complete chain.

Next local work is common-policy coverage and corrected behavior for the remaining
core operators, followed by production field-only JVP/VJP and transpose ownership.
Production clustered/compressed traversal, upstream sorted frontend integration,
multi-GPU ownership, AMD runtime qualification, GH200 qualification and corrected
performance tuning remain separate work. No performance claims are made here;
the previously recorded CPU FP64 vector-gradient regression remains unresolved.

Artifacts: `/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-crk-preparation-20261102`.
