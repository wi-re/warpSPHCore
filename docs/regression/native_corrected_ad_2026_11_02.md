# Corrected-core and complete CRK preparation derivatives

Implemented on `codex/native-cluster-pairs`, based on `a60581d` (field-only
JVP/VJP and cached CSR/CSC ownership). The three requested stages were implemented
and checked in sequence: correction inputs, corrected geometry/supports, then
complete CRK preparation.

## Delivered scope

| Stage | Delivered behavior |
| --- | --- |
| Correction inputs | Manual JVP/VJP for reference volumes, fallback mass/density weighting, CRK A/B/gradA/gradB, applicable query/reference grad-h, renormalization and fields |
| Corrected geometry | Position/support and periodic-domain-bound JVP/VJP for interpolation, gradient, divergence, curl, Laplacian and covariance, on fixed topology |
| CRK preparation | Native volume, moment and consistency-diagnostic derivatives connected through the existing Torch factor solve, including a consuming corrected operator |

Corrected `warpOperation`/`sphOperation` now support first-order Torch backwards
and `torch.autograd.forward_ad`. There is no new derivative flag in these calls.
Geometry differentiation retains the existing 11 smooth-kernel restriction;
coefficient and field derivatives support all 15 kernels. Precision, field layouts,
all five support policies, both divergence tensor contractions, all Laplacian
schemes and independent CSR widths retain their primal contracts.

Uncorrected calls keep their prior reverse bridge and forward-AD limitations;
ordinary density still ignores common correction bundles. `linearizeFields`
remains the explicit fixed-geometry map for repeated solver products. General
corrected requests that only differentiate fields reuse its existing optimized
native field-product/pullback policies.

## Numerical implementation

The common Torch bridge is `native/_differential.py`. Native pair algebra and
owned traversal live in `coreOperations/native/csrc_differential`. A general
`native/csrc/math/directional.h` scalar carries one first-order direction inside
one pair. Mixed-width CSR traversal uses the common `execution/csrView.h` adapter.
It is not a tape or a custom differentiation graph across operations.
Kernel tangents use existing analytic spatial/support derivatives; Naive Laplacian
uses the regularized radial derivatives through third order. VJPs are manual
contractions of the directed pair algebra.

The pullback reverses field weighting, grad-h divisions, query renormalization,
CRK value/gradient algebra and the raw kernel in that order. gradB keeps its
existing `[derivative, component]` axes. Query/reference roles are preserved under
transpose traversal. Query cotangents gather through CSR; reference cotangents
through CSC. Domain-bound contributions are query-owned partial sums reduced with
Torch. Native pair reductions use no floating scatter atomics. No per-edge
Jacobians, GPU-to-CPU readbacks or CSR content-validation scans are introduced.

Coefficient-only directions skip geometric contractions, including undefined
origin Hessians in force kernels. A fixed-origin Spiky coefficient regression
checks that zero geometric directions do not turn an otherwise finite derivative
into `NaN * 0`.

Native volume caches one kernel sum per query; consistency density caches its
numerator/denominator. Reciprocal and ratio finalizers have manual pullbacks and
explicit zero-denominator branches. Moments retain the six public tensor layouts
and discrete admitted-edge counts. Only row-sized state is retained; the Torch
factor solve saves its usual graph tensors. AD-enabled volume/diagnostic forwards
use the derivative executor's row sums for their finalization cache; no per-edge
forward cache is stored. Different row reduction layouts need not be bitwise
identical to inference, CPU or another precision.

Preparation shares a query `CSRTopology` across its query volume, moment and
diagnostic passes. Distinct reference self-volumes retain a separate topology.
CSC construction remains lazy and its storage is reused within the graph lifetime.
The bridge saves borrowed tensors with normal Torch version checks and rejects
changed CSR before backwards. Omitted AD directions remain real null pointers;
Torch's automatic zero-tangent materialization is disabled at this boundary.

## Branch and differentiation contracts

This is a first-order, piecewise differentiation contract:

- Topology, kinds, count outputs, periodic winding and limiter masks are discrete.
- Max-support ties retain the native query branch. No derivative of the branch
  selection is taken.
- Gradient-based Laplacians retain their exact-self zero branch and its zero
  branch derivative, including with nonzero CRK gradients. Naive Laplacian retains
  the raw regularized kernel Laplacian and ignores CRK/renormalization.
- Interpolation ignores query fields and CRK spatial coefficients; covariance
  ignores fields and grad-h. Explicit reference volumes override query volumes.
- The existing count/determinant/Schur limiter is unchanged. Its inactive identity
  correction has zero coefficient derivative. Crossing thresholds, changing rank,
  cusp/singular geometry and half-box discontinuities are not smooth derivative
  claims.

The factor solve keeps its existing Torch `pinv`, limiter and algebra. There is no
new inverse regularization or replacement solver. Numerical tests include inactive
limiter branches, aliasing and end-to-end derivative cancellation for affine
reproduction.

Backwards is once-differentiable. HVP, higher orders and `torch.func` transforms are
not supplied. When a JVP participates in a reverse graph, an explicit first-order
boundary rejects reverse-over-JVP. This prevents downstream Torch factor algebra
from returning a partial higher-order gradient that omits native Hessian terms.
Dual inputs to VJP are rejected as unsupported forward-over-backwards as well.
Ordinary first-order backwards through the primal is unaffected.

## Qualification

Local target: Torch `2.13.0+cu130`, CUDA 13, x86_64/native CPU and NVIDIA sm120.
Native Release build: `-O2 -DNDEBUG`, stable LibTorch ABI 2.10. AMD qualification
is offline gfx942 compilation only; no AMD runtime or ROCm wheel claim is made.

| Check | Result |
| --- | --- |
| New focused derivative suite | 126 scenarios per device, 252 CPU/CUDA passes |
| Full source/prebuilt Torch regression suite | 20,894 passed; 1,580 skipped |
| Python scaffolding | 23 passed |
| CPU and CUDA JIT | 126 focused scenarios passed on each |
| Installed CUDA wheel, CPU and GPU | 126 focused scenarios passed on each; complete example passed |
| Strict wheel environment | Warp and compiler/JIT imports blocked |
| Stable ABI audit | Passed; no direct ATen/c10/CPython symbols or absolute runtime paths |
| CUDA memcheck / synccheck | Focused derivative suite; zero reported errors |
| HIP compile | New derivative GPU translation unit compiled for gfx942 |

Tests rotate representative kernel/layout/support combinations rather than adding
another exhaustive Cartesian product. Existing primitive/primal matrices own the
broad kernel and layout qualification. New checks cover independent Torch
coefficient gradients, geometry finite differences by role, JVP/VJP identities,
all preparation subpasses, complete preparation and a consuming corrected-gradient
loss, FP32 geometry and a complete FP32 preparation/operator chain, empty populations, force-kernel fixed-origin coefficients,
aliased populations, inactive limiters, deterministic owned gathers, mutation
protection, affine derivative cancellation and unsupported higher-order requests.
The old AD-rejection tests were updated to check their newly supported behavior.
Test-matrix restructuring remains deferred, as requested. Packaging and JIT checks
use the focused suite instead of rerunning every inherited parameter combination.

`scripts/examples/native_crk_autograd.py` is a readable end-to-end example. On CPU
and CUDA its loss is approximately `0.0220177349253072`; the checked directional
derivative is approximately `0.0500405861100179`. The transpose-identity error is
below `7e-17`, and finite-difference errors below `1.5e-12`. Query and reference CSC
objects are reused. The prior field-only implicit example remains available.

Wheel source hashes and ELF `.text`/`.nv_fatbin` section hashes match the tested
CMake binary. The local wheel pins the qualified Torch version and targets this
CPU/GPU combination; this is not manylinux, GH200 or AMD runtime qualification.
No operation timings or throughput claims are made. The general geometry/coefficient
executor currently owns one row per GPU thread; tuning it is separate from this
capability/correctness milestone. Field-only requests retain their cooperative
executor.

## Next integration boundary

Upstream sorted CSR and existing semantic state objects can feed these calls
directly. The core now has corrected primal, coefficient, geometry and preparation
first-order derivatives ready for frontend integration. Remaining work includes
frontend-owned operators and complete simulation/JFNK integration, production
clustered traversal with its own ownership reversal, then two-bit traversal in the
previously agreed order. Nonsmooth geometry and higher-order contracts remain
separate work.

Evidence archive:
`/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-corrected-ad-20261102`.
Generated binaries are archived rather than committed.
