# Remaining native corrected core primals — 2026-11-02

## Result

Native divergence, curl, Laplacian and covariance now accept the existing
correction state objects through `sphOperation`/`warpOperation`. Together with
interpolation and gradient, all six applicable corrected core primals have
shared neighborhood policies. Density retains its uncorrected equation and now
ignores a common correction bundle, matching Warp frontend calling conventions.

The upstream integration boundary is unchanged: shared particle/domain/CRK state
objects, existing keyword arguments, explicit reference volumes, and frontend-owned
CSR. Particle IDs and offsets independently support int32/int64. No hidden sorting,
particle permutations, CSR content scans or host validation readbacks are added.

## Preserved numerical distinctions

| Operation | Behavior |
|---|---|
| Interpolation | Query A/B and reference volumes; no spatial coefficient derivatives required |
| Gradient | Corrected query gradient, optional renormalization, reference-field grad-h scaling |
| Divergence | Same gradient correction, first/last tensor-axis contraction, both query/reference grad-h scaling |
| Curl | Same gradient correction and both grad-h roles; 1D zero, 2D scalar/vector and 3D vector/matrix conventions retained |
| Laplacian | All four schemes and gradient weightings; both grad-h roles; exact zero self contributions |
| Covariance | Geometric outer-product accumulation; ignores supplied fields and grad-h; optional positive corrected-kernel counts |
| Density | Ignores unused corrections and preserves its existing primal/backwards path |

The Naive Laplacian deliberately uses the raw kernel Laplacian, even if CRK or
renormalization is supplied. Brookshaw, Dot and Default instead consume the
corrected gradient. Naive/Summation/Difference Laplacian field weighting all use
reference-minus-query differences; Symmetric retains its distinct mass/density
weight. Exact self pairs remain zero even when CRK's gradient at the origin is
nonzero. No corrected Hessian or different Laplacian equation is introduced.

Warp parity exposed an important distinction from gradient: divergence, curl and
Laplacian scale `fi / omega_i` as well as `fj / omega_j`. Gradient scales only
`fj`. Query scaling is a compile-time policy choice in shared corrected-kernel
preparation, with its own nullable input pointer and consumed-input AD guard.

`consistentDivergence=True` selects the existing reference/query density weighting.
Its Symmetric scheme retains the original mass/density formula. With explicit
volumes, non-Symmetric consistency weighting can omit masses but requires both
population densities. Ordinary divergence continues to support both contractions.
Scalar 1D divergence is accepted and its existing reverse path is checked against
the scalar gradient pullback.

`covarianceReturnNumNeighbors=True` returns `(matrix, counts)` in one native pass.
Counts depend on `corrected_kernel > 0`, not CSR degree: zero weights, negative
CRK values and FP32 Gaussian underflow do not count. This differs from raw CRK
moment counts, which intentionally count every directionally admitted edge.
The optional count handle configures covariance only; no debug mode or general
execution flag is added to other mathematical policies.

## Implementation boundary

Shared `CorrectedKernel` owns cached query coefficient/renormalization loading,
kernel-gradient correction and reference grad-h scaling. Each module's
`csrc_<function>/nt_<function>_operator.h` owns its physical weighting,
contraction, accumulator and final store. The established CSR loader/executor
adapts these policies on CPU/CUDA/HIP without embedding neighbor decoding in math.
Covariance declares zero field components and loads no query/reference field.
Absent volumes, coefficients, omegas and matrices are real null pointers.

The four new stable Torch schemas are registered in existing module-owned forward
translation units. Public Python state types and the existing call signature do
not change. The private nullable correction layout gains query omega; source/header
fingerprints prevent loading an old incompatible packaged implementation.

Existing tuned ordinary CSR execution paths and uncorrected manual backwards are
retained. In particular, this change provides the common corrected policies; it
does not replace every tuned ordinary traversal or expose cluster/two-bit execution
as a public format. The existing experimental covariance policy is preserved.

## Qualification

| Check | Result |
|---|---|
| Final full native CPU/CUDA suite | **18,369 passed, 1,451 skipped** |
| New remaining-operator matrix/contracts | **2,948 passed, 400 skipped per device** |
| Warp parity, FP32 | 624 CPU/CUDA cases |
| Warp parity, FP64 | 624 CPU/CUDA cases |
| CPU JIT | 2,946 matrix/contract cases, plus two final integration/AD contracts passed |
| CUDA JIT | 2,948 passed, 400 skipped |
| Compute Sanitizer memcheck | 1,148 passed, zero errors; neighborhood kernels filtered |
| Compute Sanitizer synccheck | 1,148 passed, zero errors; neighborhood kernels filtered |
| Python scaffolding | 23 passed |
| Existing CPU standalone target | Four CTests passed |
| HIP compilation | Six affected forward adapters compiled for gfx942 |
| Strict packaged wheel CPU suite | 8,931 passed; Warp/compiler/JIT imports blocked |
| Strict packaged wheel CUDA suite | 9,437 passed; Warp/compiler/JIT imports blocked |

The independent oracle differentiates a corrected scalar kernel with virtual query
displacements and spatially varying coefficients. It then performs independent
field weighting and tensor contractions. The raw Laplacian oracle differentiates
radial shape polynomials and preserves the legacy regularization. Positive-weight
counts are evaluated in the requested precision, separately from the FP64 arithmetic
oracle; a dedicated Gaussian/negative-CRK test checks the resulting discrete behavior.

Coverage includes every kernel/support family, rank and precision; all four
Laplacian/gradient scheme combinations; both divergence contractions; density
consistency; direction filtering; broad/reversed CSR with unused prefix storage;
independent integer widths; unequal populations; unwrapped periodic coordinates;
nullable unused inputs; self/empty cases; constant-field annihilation; and explicit
reverse/forward AD failures. A complete native-preparation chain verifies linear
divergence/curl and identity covariance in all dimensions/precisions.

The commented `scripts/examples/native_corrected.py` now exercises preparation and
all six corrected core primals. The packaged example and full suites run while
Warp and compiler/JIT imports are blocked. ABI/RPATH audit passes; packaged source
bytes and compiled `.text`/`.nv_fatbin` hashes match the tested implementation.
The wheel is local x86_64/native CPU + sm_120, Torch 2.13.0+cu130, stable ABI 2.10.
HIP remains compile-qualified only; AMD runtime and GH200 are not qualified here.

An old density-volume rejection assertion was replaced with a genuinely unsupported
normalization-calibration case after intentionally matching Warp's ignored-bundle
behavior. The final native and installed-wheel suites include that updated contract.

## Remaining work

Corrected calls, consistency divergence and covariance-with-counts are primal-only;
`torch.no_grad()` permits primal execution of differentiable state. The unchanged
ordinary uncorrected operators retain their existing manual first-order backwards.

The next derivative milestone is production field-only JVP/VJP with cached CSR
transpose ownership for JFNK, followed by full geometry/support/correction and
preparation derivatives. Native renormalization/grad-h construction, broader
frontend operator ports and simulation integration remain separate work. Upstream
sorting/CSR integration can consume the current arrays and operators directly;
production cluster/compressed formats still need their traversal integration.

This is correctness and integration work. No new operator performance claims are
made. Corrected dispatch uses raw cooperative eight-lane GPU rows; packing,
specialization and profiling remain follow-ups. The previously documented CPU FP64
vector-gradient regression is still unresolved.

Artifacts: `/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-corrected-core-20261102`.
