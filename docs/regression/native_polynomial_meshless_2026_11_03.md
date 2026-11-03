# Native core closure: polynomial reconstruction and meshless primals

The third ordered core stage ports polynomial neighbor sums and MFM/MFV kernel
passes to module-owned CPU/CUDA/HIP code. Existing Torch-level fit solves,
reconstruction weights, geometry construction and closure iteration logic stay
shared. Chunked/owned-row execution is the next stage, not a claim of this one.

## Polynomial assembly

`PolyFit`, `Reconstructor`, their read-outs and interface pairs import without
Warp in a native process. Compiled assembly supports arbitrary monomial tables,
packed symmetric moments, multicomponent right-hand sides, separate source
populations, explicit source volumes, LABFM subtraction, unit weights and the
existing 1D/2D sector stencils. Counts measure nonzero stencil weights rather
than nonzero monomials. The Torch scaffold retains equilibration, rank and
conditioning policies, TENO/WENO weighting and deficient-fit fallbacks.

Native pair derivatives cover positions, query supports, source mass/density,
explicit source volumes, fields and periodic domain lengths. Query reductions
use CSR; reference reductions use its cached CSC transpose, without floating
atomics or saved per-edge Jacobians. Integer topology and exponent tensors are
saved for AD mutation detection. Geometry AD follows the native smooth-radial
kernel qualification; unit-weight polynomial geometry has no radial derivative.
Sector/support switches are fixed branches. Raw assembly supports first-order
JVP/VJP; higher derivatives of native assembly are not claimed. The existing
implicit fit JVP remains a full-rank construction; this port does not extend its
rank-transition policy. Explicit JVP helpers work inside an existing dual level.

## Meshless primals

`warpSPHCore.mfm.MeshlessNative` uses the shared meshless controller. The legacy
`MeshlessWarp` name remains an alias in a native process for callers that keep
backend selection outside solver code. Moments, matrix gradients, slope factors,
guarded faces, closure residuals and weights, graph-Laplacian applications,
coarse stencils, Riemann fluxes and CFL estimates use native kernels. Segmented
multigrid reduction and retained closure buffers also run without Warp.

Canonical pair ordering preserves equal directed flux evaluations and conservative
assembly without atomically updating both particles. HLLC fallback estimates,
reconstruction, prediction, retry, MFV donor mass limit, centered weights, area
and face guards and bounded closure retain the Warp equations. The closure
potential has an arbitrary constant nullspace: compare potential differences or
remove its mean, not the raw potential from different preconditioners.

Compiled meshless execution remains forward-only, as the Warp implementation
was. `MeshlessGeometry` and the Torch meshless functions remain differentiable
and now work in a Warp-free process too. Retained CSR and optional closure weights
still consume edge memory; there is no claim of a particle-only memory bound.

## Qualification

| Check | Result |
|---|---|
| Six bounded native contracts | CPU: 6 passed; strict prebuilt CUDA: 6 passed |
| Warp import blocker around those six CPU contracts | 6 passed, including closure/multigrid and reference AD |
| Polynomial assembly and field pullback parity | CPU FP32, separate CPU FP64 and CUDA FP32 pass |
| Existing meshless reference contracts exercised externally with native execution | 17 CPU checks and 19 CUDA checks pass; legacy device/precision skips retained |
| Existing Warp meshless regressions after shared-controller extraction | 18 passed, 2 skipped on CPU |
| Offline HIP | Polynomial, meshless and closure/segmented GPU translation units compile for gfx942 |

The maintained additions are six native contracts and one bounded parity
contract. Dimensions, dtypes and kernels rotate inside the relevant contract;
the existing reference campaign was not duplicated into another maintained
matrix. Checks cover polynomial reproduction, finite differences and transpose
identities, source/query roles, periodic lengths, nonlinear reconstruction,
meshless geometry/rates/conservation/CFL, multigrid closure, bounded corrections,
stream ordering, graph replay and derivative rejection at the compiled meshless
boundary. No timings or AMD runtime results are claimed.

Source snapshots, binaries, logs (including failed and superseded probes), HIP
objects and adapted external contract probes are retained outside core under
`/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-core-polynomial-meshless-20261103`.
