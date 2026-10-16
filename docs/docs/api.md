# API reference

[Runnable examples](examples.md) shows these entry points composed into
checked solver, correction/AD, reconstruction and MFM/MFV clients.

For meshless geometry, reconstruction and conservative flux rates, see the
[MFM/MFV API and numerical contracts](mfm.md).

Every public function, by module. Signatures are abbreviated to the
arguments that matter; the detailed pages carry the equations, options,
and JVP scopes. Flat imports work for everything listed
(`from warpSPHCore import …` — see `__init__.py`'s `__all__`
aggregation).

CUDA operators, prepared launches, geometry/value JVPs, Density HVPs and the
autograd bridges follow Torch's current stream on the tensor's device. Warp
adjoint-buffer initialization, reverse launches and gradient clearing use that
same stream. Callers establish dependencies before moving data or reusing
scratch across streams; the bridge adds no device-wide synchronization.

## Type configuration — `type_config.py`

Configure before import/compilation. `warpSPHCore_PRECISION=float32|float64`
selects floating-point precision; `warpSPHCore_INDEX_DTYPE=int32|int64` selects
stored particle IDs, and `warpSPHCore_OFFSET_DTYPE=int32|int64` selects CSR offsets
and edge-address arithmetic. Both integer settings default to int32 independently
of precision. Per-row counts remain int32; search totals/prefix scans use int64,
with a range check before storing offsets or allocating neighbor capacity.

The optional `warpSPHCore_config` module uses `precision`, `dim`, `index_dtype`
and `offset_dtype`, overriding corresponding environment settings. Types cannot
be changed after import; start a new process for another configuration.
`get_type_config()` reports `scalar_t`, `dim_t`, `index_t`, `offset_t` and
`neighbor_count_t`; `get_torch_index_dtype()` and `get_torch_offset_dtype()` return
the matching Torch dtypes. Manually supplied adjacency/grid buffers must use the
configured types; kernels receive these buffers directly without dtype conversion.

Int64 offsets remove the repository's narrowing points but do not remove Warp's
current signed-int32 array-dimension limit. Flat neighbor capacity, including
padding/headroom, and particle populations are checked against that backend
limit before allocation/traversal. Grid cell keys retain int64 independently.

## Operators — `operations.py`

| function | what it does | details |
|---|---|---|
| `warpOperation(queryParticles, operationProperties, domain, queryValues=None, referenceValues=None, queryVolumes=None, referenceVolumes=None, adjacency=None, referenceParticles=None, preScatteredQuantities=None, crkState=None, gradHState=None, renormalizationState=None, consistentDivergence=False, covarianceReturnNumNeighbors=False)` | the unified entry point — dispatches on `operationProperties.operation` to one of the seven operators | one page per op: [Density](operations/density), [Gradient](operations/gradient), [Divergence](operations/divergence), [Curl](operations/curl), [Laplacian](operations/laplacian), [Interpolate](operations/interpolate), [Covariance](operations/covariance) |
| `warpOperationJVP(…, tangentQueryValues=None, tangentReferenceValues=None, queryTangentState=None, referenceTangentState=None, crkTangentState=None, renormalizationTangentState=None, tangentReferenceVolumes=None, …)` | forward-mode JVP of `warpOperation`: returns the **sum** of the requested pieces (value tangent, geometry tangent, correction tangents); scope rules per operator (which schemes/ops accept which tangents) | the JVP sections of the operator pages |
| `warpOperationHVP(…, v)` | the Density **position Hessian** $\nabla^2_{xx}\rho_i$ applied to a vector field — a dedicated kernel, not a composed JVP (only operator with an HVP) | [Density — HVP](operations/density#jvp--hvp) |
| `sphOperation_warp(…)` | the flat-tensor entry point (queryKinds/referenceKinds as keyword-only required tensors, `useCRK`/`useVolume`/`useGradHTerms`/`useGradientRenormalization` flags) — the interface the lower-level callers and the integrator repo use | [Autodiff machinery](autograd) (the bridge it resolves into) |

CUDA neighbor loops can use multiple cooperating lanes per query particle.
`setNeighborLanes(1)` selects serial traversal; `setNeighborLanes(32)` explicitly
selects tiled kernels where available. Other supported choices are 2, 4, 8 and
16. `setNeighborLanes(None)` restores the environment/per-operator defaults.
The environment equivalent is `WARPSPHCORE_NEIGHBOR_LANES`, read at import time.
CPU execution remains serial.

Density, interpolation, gradient, divergence and covariance retain their
existing 32-lane defaults. Curl and Laplacian have trial tiled variants and
retain serial defaults: explicit lane choices opt into them. Small-problem
speedups do not imply a large-problem speedup, so benchmark the intended size,
degree and traversal before selecting lanes. Ordinary and prepared launches
resolve the same policy. The existing correction and AD scope is preserved;
geometry JVPs continue to use their existing kernels. Changing a captured
operator's lane configuration requires rebuilding its graph or clearing its
owning solver workspace's capture cache.

## Neighbor search — `radiusSearch/`

| function | what it does | details |
|---|---|---|
| `radiusSearchCompactHashMap(queryParticles, domain, mode, hashMapLengthMode=NextPrime, fixedHashMapLength=4096, returnCompactHashMap=False, referenceParticles=None)` | build the compact hash map **and** collect an `AdjacencyList` | [Neighbor search](neighbor-search#the-compact-hash-pipeline-main-path) |
| `buildCompactHashMap(…)` | the hash-map half only (a `CompactHashMap`) | same |
| `radiusSearchOnCompactHashMap(…)` | the collection half only (hash map → `AdjacencyList`) | same |
| `buildVerletList(…, verletScale=1.0, priorNeighborhood=None)` | Verlet-list build with skin, reusing a prior list when still valid | [Neighbor search — Verlet](neighbor-search#verlet-lists) |
| `updateNeighborsVerlet(…)` / `filterVerletList(…)` | incremental pair additions between steps / dropping pairs that left the true support | same |
| `radiusNaive(x, y, hx, hy, …)` / `radiusNaiveFixed(…)` | the $O(N^2)$ all-pairs reference (validation, not production) | [Neighbor search — other backends](neighbor-search#other-backends) |
| `warp_radius_search_small(…)` | small-system variant | same |
| `checkOffset` / `getIndexRange` | the stencil-offset and neighbor-block accessors every dual-path operator kernel uses | [Neighbor search](neighbor-search) |

## CRK — `crk/`

| function | what it does | details |
|---|---|---|
| `computeCRKFactors(queryParticles, domain, kernel, operationMode=AllToAll, adjacency=None, referenceState=None, *, referenceVolumeMode=None)` | full CRK pipeline; optional independent reference-to-reference volume rule; returns `(apparentArea, crk_density, CRKState)` | [CRK overview](crk/crk-overview) |
| `computeCRKFactorsJVP(…)` | the pipeline's JVP (chained volume → moments → `torch.autograd.functional.jvp` through the terms solve) | [CRK overview — JVP](crk/crk-overview) |
| `computeKernelCRK(…)` / `computeKernelGradientCRK(…)` | the corrected kernel value / gradient $\widehat W_{ij}$, $\nabla\widehat W_{ij}$ | [CRK overview — corrected kernel](crk/crk-overview#the-corrected-kernel-and-gradient) |
| `computeKernelGradientCRKJVP(…)` / `correctGradientCRKJVP(…)` | the forward-mode building blocks the operator JVPs call for the CRK pieces | [CRK overview](crk/crk-overview) |

## Renormalization — `renorm.py`

| function | what it does | details |
|---|---|---|
| `computeRenormalizationMatrices(queryParticles, operationProperties, domain, …, returnEigVals=True)` | covariance (on a properties *copy*) → low-neighbor identity fallback → pseudo-inverse; returns `(C, eigVals, RenormalizationState)` | [Renormalization](renorm#computerenormalizationmatrices) |
| `computeRenormalizationMatricesJVP(…, queryTangentState, …)` | masked covariance tangent → selected spectral inverse tangent, including retained/discarded rotation | [Renormalization — JVP](renorm#jvp) |
| `pinv1x1` / `pinv2x2_warpBackend` (and the `pinv_warp` dispatcher, `pinv/`) | the dimension-dispatched pseudo-inverse (2D closed-form symmetric eigendecomposition, Warp kernel) | [Renormalization — step 3](renorm#computerenormalizationmatrices) |

## Polynomial fits and reconstruction — `polyfit/`

| entry point | what it does | details |
|---|---|---|
| `PolyFit.build(particles, domain, kernel, order, ...)` | local MLS/RKPM or LABFM fit; values, diffuse derivatives and interface states, with rank/condition diagnostics | [Reconstruction](reconstruction) |
| `Reconstructor.teno(...)` / `Reconstructor.weno(...)` | component-wise nonlinear candidate weighting; first-order fallback when no candidate is usable | [Reconstruction](reconstruction) |
| `interfacePairs(...)` | live directed/half pair lists and minimum-image vectors | [Reconstruction](reconstruction) |

## Linear algebra — `linalg/`

| function | what it does | details |
|---|---|---|
| `cg(...)` / `bicg(...)` / `bicgstab(...)` | Matrix-free solves with device stopping predicates, bounded restarts and verified original residuals | [Linear algebra](linalg) |
| `gmres(..., restart=30, preconditionerSide='right', accumulationDtype=None)` | Restarted Arnoldi/Givens solve with left/right preconditioning, bounded residual repair and verified convergence | [Restarted GMRES](linalg#restarted-gmres) |
| `minres(..., symmetric=True, precond=None, preconditionerSPD=False)` | Short-recurrence indefinite/singular symmetric solve, explicit inverse-SPD contract and verified original residual | [MINRES](linalg#short-recurrence-minres) |
| `richardson(..., omega=None, backtracking=True, maxBacktracks=8)` | Fixed/adaptive scalar updates, bounded true-residual search, finite best-iterate fallback and explicit trial diagnostics | [Richardson](linalg#bounded-richardson-iteration) |
| `LinearOperator(shape, dtype, device, matvec=None, matvecOut=None, adjoint=None)` | Callable operator metadata, supplied-output action and optional explicit transpose | [Operator metadata](linalg#explicit-operator-metadata) |
| `RestrictedSystem(operator, freeIndices)` | Owned coordinate restriction, Dirichlet RHS lifting/reconstruction, explicit reduced adjoint and inverse-action restriction | [Restricted systems](linalg#restricted-systems-and-prescribed-values) |
| `NullspaceSystem(operator, rightBasis, leftBasis, ...)` | Declared Euclidean nullspaces, explicit source compatibility/projection, projected inverse actions, gauges and original-source residual reports | [Nullspaces and gauges](linalg#nullspaces-source-compatibility-and-gauges) |
| `DiagonalPreconditioner(diagonal, ...)` / `BlockJacobiPreconditioner(blocks, ...)` | Owned inverse actions, explicit singular/SPD policies, adjoints, fixed-address updates and optional native small-block application | [Preconditioner helpers](linalg#diagonal-and-block-jacobi-helpers) |
| `CGWorkspace(b, maxiter, history=True, capture=False)` | Reusable native inference-only CG updates, optional history, owned results, supplied-output matvecs and cached CUDA chunks | [Native CG workspace](linalg#native-cg-workspace) |
| `BiCGStabWorkspace(b, maxiter, history=True, capture=False)` | Native right-preconditioned BiCGStab, reusable scratch, half/full guards, supplied-output operator/preconditioner actions and cached chunks | [Native BiCGStab workspace](linalg#native-bicgstab-workspace) |
| `pinvSymmetric(M, rtol=None)` / `applyPinv(V, M, rtol=None)` | Small symmetric pseudoinverse and fused application, with reverse/forward AD fallback | [Small symmetric matrices](linalg#small-symmetric-matrices) |
| `runStationary(...)` / `CheckSchedule(...)` / `residualStatistic(...)` | Shared stationary loop, checkpoint schedule and pressure residual statistics | [Stationary iteration](linalg#stationary-iteration) |

## Kernels — `kernels/`

| function | what it does | details |
|---|---|---|
| `sphKernel(x_i, x_j, h_i, h_j, kernelProperties, domain)` / `sphKernel_ij(…)` | $W_{ij}$ (lattice factor included) | [Kernel library](kernels#the-building-blocks) |
| `sphKernelGradient(…)` / `sphKernelGradient_ij(…)` | $\nabla_i W_{ij}$ | same |
| `sphKernelHessian(…)` | $\nabla^2 W_{ij}$ ($\varepsilon$-regularized) | same |
| `sphKernelDkDh(…)` / `sphGradientDkDh(…)` | $\partial W/\partial h$, $\partial\nabla W/\partial h$ (the grad-h building blocks) | same |
| `sphKernelLaplacian(…)` | the analytic kernel Laplacian (the `Naive` Laplacian scheme's estimator) | [Kernel library — naive Laplacian](kernels#the-naive-kernel-laplacian) |
| `sphKernelJVP(…)` / `sphKernelJVP_ij(…)` | $(W_{ij}, dW_{ij})$ — forward-mode building block | [Kernel library — JVPs](kernels#kernel-jvps) |
| `sphKernelScale` / `sphKernelC_d` / `sphKernelN_H` / `sphKernel_xi` | Dehnen & Aly packing/support *properties* (not evaluations — the lattice factor is deliberately not applied to them) | [Kernel library — packing](kernels#packing-and-support-properties) |
| `resolveNormalization(kernelProperties)` | the $1/L$ lattice factor (1.0 when the calibration is off) | [Lattice calibration](lattice-calibration) |

## Lattice density — `util/latticeDensity.py`

| function | what it does | details |
|---|---|---|
| `latticeDensity(kernel, n_h, dim, method='shells')` | $L(n_h)$ — what a perfect lattice at $h/s = n_h$ measures, in units of $\rho_0$ (cached) | [Lattice calibration — the math](lattice-calibration#the-math-three-ways--utillatticedensitypy) |
| `latticeDensityFactor(kernel, n_h, dim, method='shells')` | $1/L$ — the mass/kernel scaling | same |
| `latticeDensityShells` / `latticeDensityFourier` / `latticeDensityClosed` | the three evaluation methods (exact shell sum / Poisson identity / Wendland-only Epstein-zeta tail) | same |
| `latticeDensityIsStrictlyAbove1(kernel)` | Wendland ⇒ $L > 1$ for all $n_h$ (no root in $h$) | same |
| `shellCounts` / `jacobiR2` | $r_d(j)$ shell multiplicities (direct count / Jacobi's two-square closed form) | same |
| `kernelFourierTransform` / `epsteinZeta` / `WENDLAND_TAIL_COEFFICIENTS` | the $\widehat W$ and $Z_d$ constants behind the `fourier`/`closed` methods | same |

## Utilities — `util/`

| function | what it does | details |
|---|---|---|
| `generateNeighborTestData(nx, targetNumNeighbors, dim, periodic, device)` | the standard $n_x^d$ lattice test case (notebooks + most of `tests/operations/`) | [Neighbor search — test-data helpers](neighbor-search#test-data-helpers) |
| `volumeToSupport(volume, targetNeighbors, dim)` (and `_tensor`/`_warp` variants) | $h$ for a target neighbor count from a particle volume | same |
| `n_h_to_nH(n_h, dim)` / `nH_to_n_h(…)` | resolution-knob ↔ target-neighbor-count conversion | same |
| `computePairwiseSupport(h_i, h_j, scheme)` | the per-pair support resolution (the subgradient the JVPs mirror) | [Kernel library — support schemes](kernels#pairwise-evaluation-and-the-support-schemes) |
| `castTorchToWarp` / `castWarpToTorch` / `castTorchToWarpAsBuiltins` | torch↔Warp conversions (the "as builtins" variant preserves vector/matrix layouts) | [Autodiff machinery](autograd) |

## The autograd bridge — `autograd/`

| name | what it is | details |
|---|---|---|
| `StateAwareWarpFunction` (`warpWrapperStateaware`) | the `torch.autograd.Function` bridging Warp tapes and the torch graph (forward/backward/jvp) | [Autodiff machinery — the bridge](autograd#the-bridge-stateawarewarpfunction) |
| `warpWrapper2(launcher, kernel, outputSizes, outputDtypes, defaultStateArguments, additionalArguments=(), numThreads=None, jvp_fn=None)` | the legacy positional entry into `_launch` | [Autodiff machinery — call chain](autograd#call-chain) |
| `warpWrapper` / `WarpFunctionWrapper` | the older (flat-tensor-only) bridge, kept for compatibility | same |
| `extractStateInfo(…)` | the 36-slot flattening + `build_fn` builder (also where the lattice factor is resolved) | [Autodiff machinery — the 36-slot layout](autograd#the-36-slot-flat-layout) |
| `launch_kernel(kernel, output_shape, output_dtype, *args, numThreads=None)` | torch-allocated output + `wp.from_torch` view + `wp.launch` | same |
| `asScalarArg(value, *, device)` | scalar → device tensor for kernel arguments | — |
| `OperatorSpec` / `OutputSpec` / `ExtraSpec` / `ExtraKind` / `ShapeOf` / `ThreadSpec` / `JVPSpec` / `SPHContext` / `Corrections` / `EMPTY_CORRECTIONS` / `launchOperator` | the declared operator ABI | [Autodiff machinery — the declared ABI](autograd#the-declared-abi-operatorspec--sphcontext--launchoperator) |

## Data types and enums

All in [`dataTypes`](data-types) / [`enumTypes`](data-types#enums):
`ParticleState`, `ParticleTangentState`, `DomainDescription`,
`OperationProperties`, `AdjacencyList`, `CompactHashMap`, `CRKState`,
`CRKTangentState`, `GradHState`, `RenormalizationState`,
`RenormalizationTangentState`, `kernelState`, `Field`,
`SupportScheme`, `KernelFunctions`, `GradientScheme`,
`LaplacianScheme`, `WarpOperation`, `OperationDirection`,
`ParticleType`, `HashMapLengthMode`, `ExecutionMode`, `ViscosityTerms`.

## Version

`warpSPHCore.__version__` — 0.5.0.
