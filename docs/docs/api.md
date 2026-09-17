# API reference

Every public function, by module. Signatures are abbreviated to the
arguments that matter; the detailed pages carry the equations, options,
and JVP scopes. Flat imports work for everything listed
(`from warpSPHCore import …` — see `__init__.py`'s `__all__`
aggregation).

## Operators — `operations.py`

| function | what it does | details |
|---|---|---|
| `warpOperation(queryParticles, operationProperties, domain, queryValues=None, referenceValues=None, queryVolumes=None, referenceVolumes=None, adjacency=None, referenceParticles=None, preScatteredQuantities=None, crkState=None, gradHState=None, renormalizationState=None, consistentDivergence=False, covarianceReturnNumNeighbors=False)` | the unified entry point — dispatches on `operationProperties.operation` to one of the seven operators | one page per op: [Density](operations/density), [Gradient](operations/gradient), [Divergence](operations/divergence), [Curl](operations/curl), [Laplacian](operations/laplacian), [Interpolate](operations/interpolate), [Covariance](operations/covariance) |
| `warpOperationJVP(…, tangentQueryValues=None, tangentReferenceValues=None, queryTangentState=None, referenceTangentState=None, crkTangentState=None, renormalizationTangentState=None, tangentReferenceVolumes=None, …)` | forward-mode JVP of `warpOperation`: returns the **sum** of the requested pieces (value tangent, geometry tangent, correction tangents); scope rules per operator (which schemes/ops accept which tangents) | the JVP sections of the operator pages |
| `warpOperationHVP(…, v)` | the Density **position Hessian** $\nabla^2_{xx}\rho_i$ applied to a vector field — a dedicated kernel, not a composed JVP (only operator with an HVP) | [Density — HVP](operations/density#jvp--hvp) |
| `sphOperation_warp(…)` | the flat-tensor entry point (queryKinds/referenceKinds as keyword-only required tensors, `useCRK`/`useVolume`/`useGradHTerms`/`useGradientRenormalization` flags) — the interface the lower-level callers and the integrator repo use | [Autodiff machinery](autograd) (the bridge it resolves into) |

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
| `computeCRKFactors(queryParticles, domain, kernel, operationMode=AllToAll, adjacency=None, referenceState=None)` | the full CRK pipeline: apparent volume → moments → solve → limiter; returns `(apparentArea, crk_density, CRKState)` | [CRK overview](crk/crk-overview) |
| `computeCRKFactorsJVP(…)` | the pipeline's JVP (chained volume → moments → `torch.func.jvp` through the terms solve) | [CRK overview — JVP](crk/crk-overview) |
| `computeKernelCRK(…)` / `computeKernelGradientCRK(…)` | the corrected kernel value / gradient $\widehat W_{ij}$, $\nabla\widehat W_{ij}$ | [CRK overview — corrected kernel](crk/crk-overview#the-corrected-kernel-and-gradient) |
| `computeKernelGradientCRKJVP(…)` / `correctGradientCRKJVP(…)` | the forward-mode building blocks the operator JVPs call for the CRK pieces | [CRK overview](crk/crk-overview) |

## Renormalization — `renorm.py`

| function | what it does | details |
|---|---|---|
| `computeRenormalizationMatrices(queryParticles, operationProperties, domain, …, returnEigVals=True)` | covariance (on a properties *copy*) → low-neighbor identity fallback → pseudo-inverse; returns `(C, eigVals, RenormalizationState)` | [Renormalization](renorm#computerenormalizationmatrices) |
| `computeRenormalizationMatricesJVP(…, queryTangentState, …)` | the JVP: `dC` (masked) → $dL = -L\,dC\,L$ | [Renormalization — JVP](renorm#jvp) |
| `pinv1x1` / `pinv2x2_warpBackend` (and the `pinv_warp` dispatcher, `pinv/`) | the dimension-dispatched pseudo-inverse (2D closed-form symmetric eigendecomposition, Warp kernel) | [Renormalization — step 3](renorm#computerenormalizationmatrices) |

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
