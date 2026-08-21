import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any, Union
import torch

from .util import *

from .dataTypes import *
from .math import *
from .kernels import *


from .coreOperations import *
from .enumTypes import *
from .radiusSearch import buildCompactHashMap
from typing import Optional, Tuple
from torch.profiler import profile, record_function, ProfilerActivity


# States are the primary path here: `warpOperation` takes semantic state objects and
# dispatches straight to each operator's `_computeSPHX_stateBackend` -- no flattening to
# individual tensors and no reassembly back into states in between. `sphOperation_warp`
# is the flat-tensor "manual" entry point for callers that don't already have state
# objects on hand; it builds the same ParticleState/OperationProperties/CRKState/
# GradHState/RenormalizationState objects from its flat arguments and calls
# `warpOperation`, which does the actual dispatching. See warpier_core.md's call-graph
# notes for why this replaced the previous warpOperation -> sphOperation_warp ->
# compute<Op>_warpBackend -> _compute<Op>_stateBackend chain (every hop across that
# chain used to disassemble state into flat tensors only for the next hop to reassemble
# it).


def warpOperation(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    queryValues : Optional[torch.Tensor] = None, referenceValues : Optional[torch.Tensor] = None,
    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyListWarp, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    referenceParticles: Optional[ParticleState] = None,
    preScatteredQuantities: Optional[torch.Tensor] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
    consistentDivergence: bool = False,
    covarianceReturnNumNeighbors: bool = False,
):
    if operationProperties.operation == WarpOperation.Custom:
        raise ValueError("Custom operations are not supported in warpOperation. Please write your own caller.")
    referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
    referenceValues = referenceValues if referenceValues is not None else queryValues
    referenceVolumes = referenceVolumes if referenceVolumes is not None else queryVolumes

    if gradHState is not None and not isinstance(gradHState, GradHState):
        if isinstance(gradHState, tuple) and len(gradHState) == 2:
            queryOmegas, referenceOmegas = gradHState
        elif isinstance(gradHState, torch.Tensor):
            queryOmegas, referenceOmegas = gradHState, gradHState
        else:
            raise ValueError("Invalid type for gradHState: {}. Must be either GradHState, a tuple of two tensors, or a single tensor.".format(type(gradHState)))
        gradHState = GradHState(queryOmegas=queryOmegas, referenceOmegas=referenceOmegas if referenceOmegas is not None else queryOmegas)

    if renormalizationState is not None and not isinstance(renormalizationState, RenormalizationState):
        if isinstance(renormalizationState, torch.Tensor):
            renormalizationState = RenormalizationState(renormalizationMatrices=renormalizationState)
        else:
            raise ValueError("Invalid type for renormalizationState: {}. Must be either RenormalizationState or torch.Tensor.".format(type(renormalizationState)))

    operation = operationProperties.operation

    with record_function(f"warpSPH - Operation"):
        # Density is special-cased first (same ordering as before the call-graph
        # cleanup): it has no queryValues/referenceValues and no correction paths, so it
        # shouldn't flow through the queryValues/preScatteredQuantities validation below.
        if operation == WarpOperation.Density:
            return _computeSPHDensity_stateBackend(
                queryParticles, referenceParticles, domain,
                operationProperties.supportMode, operationProperties.kernel, operationProperties.operationMode,
                adjacency,
            )
        elif operation == WarpOperation.Covariance:
            return _computeSPHCovariance_stateBackend(
                queryParticles, operationProperties, domain,
                queryVolumes=queryVolumes, referenceVolumes=referenceVolumes,
                adjacency=adjacency,
                referenceParticles=referenceParticles,
                crkState=crkState,
                gradHState=gradHState,
                renormalizationState=renormalizationState,
                returnNumNeighbors=covarianceReturnNumNeighbors,
            )

        if queryValues is None and referenceValues is None:
            if preScatteredQuantities is None:
                raise ValueError("If queryValues and referenceValues are not provided, then pre-scattered quantities must be provided for the SPH operation.")
        if queryValues is not None and referenceValues is not None and preScatteredQuantities is not None:
            raise ValueError("Pre-scattered quantities should not be provided if queryValues and referenceValues are already provided, as they are redundant in this case.")
        if preScatteredQuantities is not None and operationProperties.gradientMode != GradientScheme.Naive:
            raise ValueError("Pre-scattered quantities only support the naive scheme as they are meant to provide pre-computed neighbor-level quantities for custom kernels that may not be compatible with the standard gradient schemes. If using pre-scattered quantities, the gradientMode must be set to Naive.")
        if preScatteredQuantities is not None:
            raise NotImplementedError(
                "Pre-scattered quantities are no longer supported by any SPH operator: no "
                "caller in this repo relies on them and they interacted badly with the "
                "state-aware autograd bridge. Pass queryValues/referenceValues instead."
            )

        if crkState is not None and (crkState.gradA is None or crkState.gradB is None) and operation in (WarpOperation.Gradient, WarpOperation.Divergence, WarpOperation.Curl):
            raise ValueError("CRK gradient correction A and B tensors must be provided if useCRK is True and the operation is Gradient, Divergence, or Curl.")

        if operation == WarpOperation.Interpolate:
            if referenceValues is None:
                raise ValueError("referenceValues must be provided for the interpolation computation.")
            return _computeSPHInterpolant_stateBackend(
                queryParticles, referenceParticles, domain, operationProperties.supportMode, operationProperties.kernel, operationProperties.operationMode,
                adjacency, referenceValues,
                queryVolumes=queryVolumes, referenceVolumes=referenceVolumes,
                crkState=crkState,
            )
        elif operation == WarpOperation.Gradient:
            if queryValues is None or referenceValues is None:
                raise ValueError("queryValues and referenceValues must be provided for the gradient computation.")
            return _computeSPHGradient_stateBackend(
                queryParticles, referenceParticles, domain, operationProperties.supportMode, operationProperties.kernel, operationProperties.gradientMode, operationProperties.operationMode,
                adjacency, queryValues, referenceValues,
                queryVolumes=queryVolumes, referenceVolumes=referenceVolumes,
                crkState=crkState, gradHState=gradHState, renormalizationState=renormalizationState,
            )
        elif operation == WarpOperation.Divergence:
            if queryValues is None or referenceValues is None:
                raise ValueError("queryValues and referenceValues must be provided for the divergence computation.")
            return _computeSPHDivergence_stateBackend(
                queryParticles, referenceParticles, domain, operationProperties.supportMode, operationProperties.kernel, operationProperties.gradientMode, operationProperties.operationMode,
                adjacency, queryValues, referenceValues,
                consistentDivergence=consistentDivergence, dotMode=operationProperties.divergenceDotMode,
                queryVolumes=queryVolumes, referenceVolumes=referenceVolumes,
                crkState=crkState, gradHState=gradHState, renormalizationState=renormalizationState,
            )
        elif operation == WarpOperation.Curl:
            if queryValues is None or referenceValues is None:
                raise ValueError("queryValues and referenceValues must be provided for the curl computation.")
            return _computeSPHCurl_stateBackend(
                queryParticles, referenceParticles, domain, operationProperties.supportMode, operationProperties.kernel, operationProperties.gradientMode, operationProperties.operationMode,
                adjacency, queryValues, referenceValues,
                queryVolumes=queryVolumes, referenceVolumes=referenceVolumes,
                crkState=crkState, gradHState=gradHState, renormalizationState=renormalizationState,
            )
        elif operation == WarpOperation.Laplacian:
            if queryValues is None or referenceValues is None:
                raise ValueError("queryValues and referenceValues must be provided for the laplacian computation.")
            return _computeSPHLaplacian_stateBackend(
                queryParticles, referenceParticles, domain, operationProperties.supportMode, operationProperties.kernel, operationProperties.gradientMode, operationProperties.laplacianMode, operationProperties.positiveDivergence, operationProperties.operationMode,
                adjacency, queryValues, referenceValues,
                queryVolumes=queryVolumes, referenceVolumes=referenceVolumes,
                crkState=crkState, gradHState=gradHState, renormalizationState=renormalizationState,
            )
        else:
            raise ValueError("Unsupported SPH operation: {}".format(operation))


# Operators `warpOperationJVP`'s value-tangent (formerly "Tier 1") JVP path can
# serve: exactly the ones with a `queryValues`/`referenceValues` input. Density
# has none (it reads `queryParticles.masses` directly) and Covariance takes
# volumes, not values -- routing either through `warpOperation` with a tangent
# standing in for `queryValues` would silently ignore the tangent and hand back
# the *primal* result instead of raising, so both are excluded rather than left
# to fail quietly. The geometry-tangent path (Phase 4) extends Density's
# mass-tangent path separately.
_VALUE_JVP_OPERATIONS = (
    WarpOperation.Interpolate, WarpOperation.Gradient, WarpOperation.Divergence,
    WarpOperation.Curl, WarpOperation.Laplacian,
)

# Operators the geometry-tangent (formerly "Tier 2") JVP path
# (`warpier_tier2_operators_plan.md`) covers at all -- Density (position/
# support/mass tangent, no queryValues/referenceValues) plus the five
# value-having operators (position/support tangent with frozen fi/fj values).
# Covariance is not in this set: no geometry-tangent formula was ever derived
# for it, so it always falls through to the generic "not in geometry-JVP
# scope" NotImplementedError below, same as before this plan.
_GEOMETRY_JVP_OPERATIONS = (
    WarpOperation.Density, WarpOperation.Interpolate, WarpOperation.Gradient,
    WarpOperation.Divergence, WarpOperation.Curl, WarpOperation.Laplacian,
)

# Operators the geometry JVP's CRK tangent support (`warpier_tier2_correction_jvp_plan.md`
# phases (c)/(e)) covers -- Gradient landed in phase (c); Divergence/Curl/
# Laplacian(Brookshaw only, enforced separately below) landed in phase (e).
_CRK_GEOMETRY_JVP_OPERATIONS = (
    WarpOperation.Gradient, WarpOperation.Divergence, WarpOperation.Curl, WarpOperation.Laplacian,
)

# Populated incrementally as `warpier_tier2_operators_plan.md`'s steps 2-7
# land each operator's `computeSPH<Op>GeometryJVP`. An operator in
# `_GEOMETRY_JVP_OPERATIONS` but not yet a key here still raises
# NotImplementedError (the plan's own gate: `test_otherOperators_geometryJVP_still_raise`
# gets repointed at whichever operator is still pending as each one lands).
_GEOMETRY_JVP_DISPATCH = {
    WarpOperation.Interpolate: computeSPHInterpolateGeometryJVP,
    WarpOperation.Gradient: computeSPHGradientGeometryJVP,
    WarpOperation.Divergence: computeSPHDivergenceGeometryJVP,
    WarpOperation.Curl: computeSPHCurlGeometryJVP,
    # Brookshaw/Naive (warpier_tier2_operators_plan.md Steps 7/8) and Dot/Default
    # (remaining-work plan's follow-up, resolved 2026-08-20) -- computeSPHLaplacianGeometryJVP
    # dispatches between all four by laplacianMode itself. positiveDivergence is
    # still rejected below regardless of laplacianMode.
    WarpOperation.Laplacian: computeSPHLaplacianGeometryJVP,
}


def warpOperationJVP(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    tangentQueryValues: Optional[torch.Tensor] = None, tangentReferenceValues: Optional[torch.Tensor] = None,
    queryTangentState: Optional[ParticleTangentState] = None, referenceTangentState: Optional[ParticleTangentState] = None,
    queryValues: Optional[torch.Tensor] = None, referenceValues: Optional[torch.Tensor] = None,
    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    tangentQueryVolumes: Optional[torch.Tensor] = None, tangentReferenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyListWarp, CompactHashMap]] = None, # if none a datastructure is created, same as warpOperation
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    crkTangentState: Optional[CRKTangentState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor, RenormalizationState]] = None,
    renormalizationTangentState: Optional[RenormalizationTangentState] = None,
    consistentDivergence: bool = False,
):
    """The JVP entry point `warpier_forward_mode_plan.md` Phase 2 promotes
    the value-tangent path into, and Phase 4 / `warpier_tier2_operators_plan.md`
    extend with the geometry-tangent path. The contract: **supply any subset
    of value tangents (`tangentQueryValues`/`tangentReferenceValues`) and
    geometry tangents (`queryTangentState`/`referenceTangentState`, each a
    `ParticleTangentState` bundling positions/supports/masses/densities);
    the return value is their sum** -- the
    operator's full JVP in whatever combined direction you asked for. The
    value/geometry split below is this function's own internal bookkeeping
    (which piece was derived separately), not something a caller needs to
    know to use it correctly.

    * **Value JVP** (value tangents): an operator is exactly linear and
      homogeneous in `queryValues`/`referenceValues`, so its JVP w.r.t. them
      is the same operator relaunched on the tangent arrays in place of the
      value arrays -- verified by `scripts/spike_forward_mode_tier1.py` and
      gated by `tests/operations/test_forward_mode_value_jvp.py`.
    * **Geometry JVP** (position/support/mass/density tangents): the kernel
      is genuinely nonlinear in these, so each operator needs a hand-derived
      JVP (`warpier_adjoint.md`). Density's position/support tangent plus a
      reference-side mass tangent
      (`coreOperations.wp_densityJVP.computeSPHDensityGeometryJVP`, gated by
      `tests/operations/test_forward_mode_geometry_jvp_density.py`) was Phase
      4's scope. `warpier_tier2_operators_plan.md` extended this to the five
      value-having operators' own position/support tangent, each computed
      with `queryValues`/`referenceValues` held at their **primal** value
      (`fi`/`fj` frozen) -- landed one at a time in `_GEOMETRY_JVP_DISPATCH`;
      whichever haven't landed yet still raise `NotImplementedError`.
      Density's density tangent does not exist (Density has no density
      input) and never will. `adjacency` accepts an `AdjacencyList`, a
      `CompactHashMap`, or `None` (a `CompactHashMap` is then built the same
      way `warpOperation` builds one) -- the CSR-ported kernels
      (`warpier_tier2_jvp_csr_backend_plan.md`) traverse either
      representation directly. `referenceVolumes`/`tangentReferenceVolumes`
      (`warpier_tier2_correction_jvp_plan.md` phase b) supply apparent-volume
      support and its tangent for the five value-having operators, matching
      `warpOperation`'s own `useVolume` correction exactly -- a
      pass-through, not a re-derivation, since apparent volume is a direct
      tensor substitution for `mass_j/density_j`. `queryVolumes`/
      `tangentQueryVolumes` are not supported: no derived formula ever reads
      a query-side apparent volume, only the reference side. `crkState`/
      `crkTangentState` (`warpier_tier2_correction_jvp_plan.md` phases
      (c)/(e)) supply CRK correction and its tangent for **Gradient/
      Divergence/Curl/Laplacian(`LaplacianScheme.Brookshaw` only)** --
      Laplacian's Naive/Dot/Default schemes raise `NotImplementedError`
      (out of scope, no derivation exists). `crkTangentState` may be
      omitted (treated as an all-zero tangent) to hold the CRK correction
      itself frozen while only the geometry moves.
      `renormalizationState`/`renormalizationTangentState`
      (`warpier_tier2_correction_jvp_plan.md` phase (d)) supply gradient-
      renormalization correction and its tangent -- **Gradient only so far**;
      Divergence/Curl/Laplacian raise `NotImplementedError` until phase (f).
      `renormalizationTangentState` may be omitted (treated as an all-zero
      tangent) the same way `crkTangentState` may -- `renorm.py`'s
      `computeRenormalizationMatricesJVP` is the usual way to derive a real
      one from the same geometry tangents supplied here. CRK and
      renormalization applied simultaneously is not supported by either
      phase (c) or (d) (each is independently derived/wired; the combination
      is a documented fast follow-up, not required by this plan).
      `gradHState` remains unsupported for every geometry-JVP operator (no
      consumer exists).
    * **Both at once** (`warpier_tier2_combined_jvp_plan.md`): for the five
      value-having operators, supplying a value tangent alongside a geometry
      tangent computes both of the above -- the geometry JVP with values
      frozen at primal, the value JVP via `warpOperation` relaunched on the
      tangent value arrays with geometry frozen at primal -- and returns
      their sum. This is not an approximation: every operator here is
      exactly linear in `queryValues`/`referenceValues` with a geometry-
      dependent (but value-independent) coefficient, so the cross term in
      the total differential is exactly zero and the two already-correct
      pieces are simply additive (verified against
      `torch.autograd.functional.jacobian` differentiating w.r.t. every
      input simultaneously, see the plan doc). Density has no value input,
      so this combination doesn't apply to it -- supplying a value tangent
      for Density still raises `NotImplementedError`.
    """
    if renormalizationState is not None and not isinstance(renormalizationState, RenormalizationState):
        if isinstance(renormalizationState, torch.Tensor):
            renormalizationState = RenormalizationState(renormalizationMatrices=renormalizationState)
        else:
            raise ValueError("Invalid type for renormalizationState: {}. Must be either RenormalizationState or torch.Tensor.".format(type(renormalizationState)))

    geometryTangentArgs = {
        "queryTangentState.positions": queryTangentState.positions if queryTangentState is not None else None,
        "referenceTangentState.positions": referenceTangentState.positions if referenceTangentState is not None else None,
        "queryTangentState.supports": queryTangentState.supports if queryTangentState is not None else None,
        "referenceTangentState.supports": referenceTangentState.supports if referenceTangentState is not None else None,
        "queryTangentState.masses": queryTangentState.masses if queryTangentState is not None else None,
        "referenceTangentState.masses": referenceTangentState.masses if referenceTangentState is not None else None,
        "queryTangentState.densities": queryTangentState.densities if queryTangentState is not None else None,
        "referenceTangentState.densities": referenceTangentState.densities if referenceTangentState is not None else None,
    }
    providedGeometryTangents = [name for name, value in geometryTangentArgs.items() if value is not None]
    # Local aliases so the rest of this function's scope-boundary checks and
    # dispatch-argument assembly below stay unchanged field-for-field --
    # only the two dataclass call sites (Density, the five value-having
    # operators) need to know about the bundled ParticleTangentState shape.
    tangentQueryPositions = queryTangentState.positions if queryTangentState is not None else None
    tangentQuerySupports = queryTangentState.supports if queryTangentState is not None else None
    tangentQueryMasses = queryTangentState.masses if queryTangentState is not None else None
    tangentQueryDensities = queryTangentState.densities if queryTangentState is not None else None
    tangentReferencePositions = referenceTangentState.positions if referenceTangentState is not None else None
    tangentReferenceSupports = referenceTangentState.supports if referenceTangentState is not None else None
    tangentReferenceMasses = referenceTangentState.masses if referenceTangentState is not None else None
    tangentReferenceDensities = referenceTangentState.densities if referenceTangentState is not None else None

    if providedGeometryTangents:
        if operationProperties.operation not in _GEOMETRY_JVP_OPERATIONS:
            raise NotImplementedError(
                f"warpOperationJVP: geometry JVP tangent argument(s) {providedGeometryTangents} are not "
                f"implemented for {operationProperties.operation} yet -- "
                "warpier_forward_mode_plan.md Phase 4 / warpier_tier2_operators_plan.md "
                "cover only Density and the five value-having operators."
            )

        if adjacency is None:
            # Same default the primal path builds (`autograd.arg_extract.extractStateInfo`
            # Section 4) -- symmetry with `warpOperation`, which has never required a
            # caller to pass adjacency explicitly. Safe now that the CSR-ported geometry
            # JVP kernels traverse a `CompactHashMap` directly
            # (`warpier_tier2_jvp_csr_backend_plan.md`), not just an `AdjacencyList`.
            referencePositionsForHash = referenceParticles.positions if referenceParticles is not None else queryParticles.positions
            referenceSupportsForHash = referenceParticles.supports if referenceParticles is not None else queryParticles.supports
            adjacency = buildCompactHashMap(
                queryParticles.positions, referencePositionsForHash,
                queryParticles.supports, referenceSupportsForHash,
                periodicity=domain.periodic,
                domainDescription=domain,
                mode=SupportScheme.SuperSymmetric,
            )

        if operationProperties.operation is WarpOperation.Density:
            # Left literally unmoved/unedited from before this dispatch table existed
            # (warpier_tier2_operators_plan.md Step 1: zero tolerance for behavior drift here).
            isDensityGeometryOnlyCase = (
                tangentQueryValues is None and tangentReferenceValues is None
                and tangentQueryMasses is None and tangentQueryDensities is None and tangentReferenceDensities is None
            )
            if not isDensityGeometryOnlyCase:
                raise NotImplementedError(
                    f"warpOperationJVP: geometry JVP tangent argument(s) {providedGeometryTangents} are not "
                    f"implemented for {operationProperties.operation} yet -- "
                    "warpier_forward_mode_plan.md Phase 4 implements only Density's "
                    "position/support/(reference-side) mass tangent so far."
                )
            nQuery = queryParticles.positions.shape[0]
            zeroPositions = lambda n: torch.zeros((n, domain.dim), device=queryParticles.positions.device, dtype=queryParticles.positions.dtype)
            return computeSPHDensityGeometryJVP(
                queryParticles, domain, operationProperties.kernel, operationProperties.supportMode, adjacency,
                queryTangentState=ParticleTangentState(
                    positions=tangentQueryPositions if tangentQueryPositions is not None else zeroPositions(nQuery),
                    supports=tangentQuerySupports,
                    masses=tangentQueryMasses,
                ),
                referenceParticles=referenceParticles,
                referenceTangentState=ParticleTangentState(
                    positions=tangentReferencePositions,
                    supports=tangentReferenceSupports,
                    masses=tangentReferenceMasses,
                ),
            )

        # The five value-having operators: scope boundaries enforced centrally here
        # (warpier_tier2_operators_plan.md's "Scope boundaries" section) so individual
        # wp_<op>JVP.py files stay focused on the math, mirroring wp_densityJVP.py
        # (which has no internal validation of its own).
        if gradHState is not None:
            raise NotImplementedError(
                "warpOperationJVP: geometry JVP does not support gradHState -- grad-h coupling is "
                "out of scope entirely (no consumer exists)."
            )
        if renormalizationState is not None and operationProperties.operation is not WarpOperation.Gradient:
            raise NotImplementedError(
                "warpOperationJVP: geometry JVP renormalization tangent support is implemented for "
                "Gradient only so far (warpier_tier2_correction_jvp_plan.md phase (d)); "
                "Divergence/Curl/Laplacian land in phase (f)."
            )
        if renormalizationTangentState is not None and renormalizationState is None:
            raise ValueError(
                "warpOperationJVP: renormalizationTangentState requires renormalizationState -- "
                "there is no renormalization correction tangent to take without a renormalization "
                "correction to perturb."
            )
        if crkState is not None and operationProperties.operation not in _CRK_GEOMETRY_JVP_OPERATIONS:
            raise NotImplementedError(
                "warpOperationJVP: geometry JVP CRK tangent support is implemented for "
                "Gradient/Divergence/Curl/Laplacian(Brookshaw) only "
                "(warpier_tier2_correction_jvp_plan.md phases (c)/(e))."
            )
        if (
            crkState is not None
            and operationProperties.operation is WarpOperation.Laplacian
            and operationProperties.laplacianMode is not LaplacianScheme.Brookshaw
        ):
            raise NotImplementedError(
                "warpOperationJVP: geometry JVP CRK tangent support for Laplacian is implemented "
                "for LaplacianScheme.Brookshaw only (warpier_tier2_correction_jvp_plan.md phase "
                "(e)) -- Naive/Dot/Default stay out of scope."
            )
        if crkTangentState is not None and crkState is None:
            raise ValueError(
                "warpOperationJVP: crkTangentState requires crkState -- there is no CRK "
                "correction tangent to take without a CRK correction to perturb."
            )
        if crkState is not None and renormalizationState is not None:
            raise NotImplementedError(
                "warpOperationJVP: geometry JVP does not support CRK and renormalization applied "
                "simultaneously -- each is independently supported (phases (c)/(d)); the combination "
                "is a documented fast follow-up, not yet implemented."
            )
        if queryVolumes is not None or tangentQueryVolumes is not None:
            raise NotImplementedError(
                "warpOperationJVP: geometry JVP does not support queryVolumes/tangentQueryVolumes "
                "-- no derived formula ever reads correctionData.queryVolumes[i], only "
                ".referenceVolumes[j] (`warpier_tier2_correction_jvp_plan.md` phase b)."
            )
        if tangentReferenceVolumes is not None and referenceVolumes is None:
            raise ValueError(
                "warpOperationJVP: tangentReferenceVolumes requires referenceVolumes -- there is "
                "no apparent-volume tangent to take without an apparent-volume primal value to "
                "perturb."
            )
        if tangentQueryMasses is not None:
            raise NotImplementedError(
                "warpOperationJVP: geometry JVP does not support tangentQueryMasses for "
                "value-having operators -- no derived formula has an m_i term."
            )
        if operationProperties.operation is WarpOperation.Divergence and (
            operationProperties.divergenceDotMode or consistentDivergence
        ):
            raise NotImplementedError(
                "warpOperationJVP: geometry JVP Divergence does not support divergenceDotMode "
                "or consistentDivergence -- neither is in the derived math."
            )
        if operationProperties.operation is WarpOperation.Curl and domain.dim != 2:
            raise NotImplementedError(
                "warpOperationJVP: geometry JVP Curl is only implemented for domain.dim == 2 "
                "-- 1D and 3D are both undecided by the spike."
            )
        if operationProperties.operation is WarpOperation.Laplacian and operationProperties.positiveDivergence:
            raise NotImplementedError(
                "warpOperationJVP: geometry JVP Laplacian does not support "
                "positiveDivergence -- positiveDotProduct's extra term isn't in any "
                "of the four derived laplacianMode formulas."
            )

        dispatchFn = _GEOMETRY_JVP_DISPATCH.get(operationProperties.operation)
        if dispatchFn is None:
            raise NotImplementedError(
                f"warpOperationJVP: geometry JVP tangent argument(s) {providedGeometryTangents} are not "
                f"implemented for {operationProperties.operation} yet -- "
                "warpier_tier2_operators_plan.md hasn't landed this operator."
            )
        nQuery = queryParticles.positions.shape[0]
        zeroPositions = lambda n: torch.zeros((n, domain.dim), device=queryParticles.positions.device, dtype=queryParticles.positions.dtype)
        # Not every value-having operator's computeSPH<Op>GeometryJVP formula
        # actually uses every field of the bundled ParticleTangentState (e.g.
        # Interpolate has no query-side density term at all) -- each op's own
        # computeSPH<Op>GeometryJVP silently drops the fields its formula
        # doesn't need (mirroring wp_densityJVP.py, no unused parameters),
        # same as before this dataclass bundled the loose kwargs.
        queryTangentStateForDispatch = ParticleTangentState(
            positions=tangentQueryPositions if tangentQueryPositions is not None else zeroPositions(nQuery),
            supports=tangentQuerySupports,
            masses=tangentQueryMasses,
            densities=tangentQueryDensities,
        )
        dispatchKwargs = dict(
            referenceParticles=referenceParticles,
            referenceTangentState=ParticleTangentState(
                positions=tangentReferencePositions,
                supports=tangentReferenceSupports,
                masses=tangentReferenceMasses,
                densities=tangentReferenceDensities,
            ),
            queryValues=queryValues,
            referenceValues=referenceValues,
            referenceVolumes=referenceVolumes,
            tangentReferenceVolumes=tangentReferenceVolumes,
        )
        # Laplacian's q_ij reuses Gradient's own B/dB (warpier_adjoint.md Tier 2.2
        # finding 2), so it needs gradientMode too, not just laplacianMode.
        if operationProperties.operation in (WarpOperation.Gradient, WarpOperation.Divergence,
                                              WarpOperation.Curl, WarpOperation.Laplacian):
            dispatchKwargs["gradientMode"] = operationProperties.gradientMode
        if operationProperties.operation is WarpOperation.Laplacian:
            dispatchKwargs["laplacianMode"] = operationProperties.laplacianMode
        # Naive/Dot/Default's own computeSPHLaplacian<Scheme>GeometryJVP take no
        # crkState/crkTangentState parameter at all (out of scope, enforced above) --
        # only pass these through for the operators/schemes that actually accept them,
        # rather than relying on every callee to silently ignore an unsupported kwarg.
        if operationProperties.operation in (WarpOperation.Gradient, WarpOperation.Divergence, WarpOperation.Curl) or (
            operationProperties.operation is WarpOperation.Laplacian
            and operationProperties.laplacianMode is LaplacianScheme.Brookshaw
        ):
            dispatchKwargs["crkState"] = crkState
            dispatchKwargs["crkTangentState"] = crkTangentState
        if operationProperties.operation is WarpOperation.Gradient:
            dispatchKwargs["renormalizationState"] = renormalizationState
            dispatchKwargs["renormalizationTangentState"] = renormalizationTangentState
        geometryResult = dispatchFn(
            queryParticles, domain, operationProperties.kernel, operationProperties.supportMode, adjacency,
            queryTangentStateForDispatch,
            **dispatchKwargs,
        )

        if tangentQueryValues is None and tangentReferenceValues is None:
            return geometryResult

        # Combined value+geometry JVP: `warpier_tier2_combined_jvp_plan.md` --
        # every operator in scope here is exactly linear in
        # queryValues/referenceValues (`warpier_adjoint.md`'s opening premise),
        # so the total JVP along a path where both the field values *and* the
        # geometry move has no missing cross term -- it's exactly the geometry
        # JVP contribution above (geometry tangent, values frozen at primal)
        # plus the value JVP contribution below (value tangent, geometry frozen
        # at primal). Verified against `torch.autograd.functional.jacobian`
        # differentiating w.r.t. every input at once, see the plan doc.
        valueResult = warpOperation(
            queryParticles, operationProperties, domain,
            queryValues=tangentQueryValues, referenceValues=tangentReferenceValues,
            queryVolumes=queryVolumes, referenceVolumes=referenceVolumes,
            adjacency=adjacency, referenceParticles=referenceParticles,
            crkState=crkState, gradHState=gradHState, renormalizationState=renormalizationState,
            consistentDivergence=consistentDivergence,
        )
        return geometryResult + valueResult

    if operationProperties.operation not in _VALUE_JVP_OPERATIONS:
        raise NotImplementedError(
            f"warpOperationJVP: value JVP is only defined for operators with a "
            f"queryValues/referenceValues input ({[op.name for op in _VALUE_JVP_OPERATIONS]}); "
            f"{operationProperties.operation} has none."
        )

    return warpOperation(
        queryParticles, operationProperties, domain,
        queryValues=tangentQueryValues, referenceValues=tangentReferenceValues,
        queryVolumes=queryVolumes, referenceVolumes=referenceVolumes,
        adjacency=adjacency, referenceParticles=referenceParticles,
        crkState=crkState, gradHState=gradHState, renormalizationState=renormalizationState,
        consistentDivergence=consistentDivergence,
    )


def warpOperationHVP(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    tangentQueryPositions: torch.Tensor,
    adjacency: Optional[Union[AdjacencyListWarp, CompactHashMap]] = None,
    referenceParticles: Optional[ParticleState] = None,
    tangentReferencePositions: Optional[torch.Tensor] = None,
):
    """The Hessian-vector-product entry point `warpier_forward_mode_plan.md`
    Phase 4 Step 3 adds alongside `warpOperationJVP`: `Hess(op)_i @ v`,
    obtained by differentiating `warpOperationJVP`'s own position tangent
    once more in the same direction `v` -- "a JVP of that JVP".

    Only Density's position Hessian is implemented so far
    (`coreOperations.wp_densityHVP.computeSPHDensityPositionHVP`), matching
    `warpOperationJVP`'s own current geometry-JVP scope. It is a **separate entry
    point**, not a `warpOperationJVP` option, because generic torch-level
    composition (`torch.func.jvp` twice, or nested
    `torch.autograd.forward_ad`) does not work for a warp-kernel-backed
    function in this codebase -- tried first, see `wp_densityHVP.py`'s
    module docstring for what actually happens (an immediate `RuntimeError`
    from one path, a silently dropped tangent from the other). This is the
    hand-written "small explicit second-order helper" the plan flagged as
    the fallback.
    """
    if operationProperties.operation is not WarpOperation.Density:
        raise NotImplementedError(
            f"warpOperationHVP: only Density's position Hessian is implemented "
            f"so far ({operationProperties.operation} is not); "
            "warpier_forward_mode_plan.md Phase 4 Step 3."
        )
    if not isinstance(adjacency, AdjacencyList):
        raise NotImplementedError(
            "warpOperationHVP: Density's Hessian-vector product needs the "
            f"torch-facing AdjacencyList (.i/.j neighbor pairs, what "
            f"buildVerletList returns), not {type(adjacency)} -- grid/"
            "CompactHashMap traversal is not implemented here."
        )
    return computeSPHDensityPositionHVP(
        queryParticles, domain, operationProperties.kernel, operationProperties.supportMode, adjacency,
        tangentQueryPositions=tangentQueryPositions,
        referenceParticles=referenceParticles,
        tangentReferencePositions=tangentReferencePositions,
    )


def sphOperation_warp(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,
    queryDensities, referenceDensities,
    queryValues : Optional[torch.Tensor], referenceValues : Optional[torch.Tensor],
    domain: DomainDescription,
    adjacency: Optional[Union[AdjacencyListWarp, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    operation: WarpOperation = WarpOperation.Interpolate,
    kernel: KernelFunctions = KernelFunctions.Wendland4,
    supportMode: SupportScheme = SupportScheme.Gather,
    gradientMode: GradientScheme = GradientScheme.Naive,
    laplacianMode: LaplacianScheme = LaplacianScheme.Default,
    operationMode: OperationDirection = OperationDirection.AllToAll,
    positiveDivergence: bool = False,
    consistentDivergence: bool = False,
    divergenceDotMode: bool = False,
    preScatteredQuantities: Optional[torch.Tensor] = None,
    *,
    queryKinds: torch.Tensor, referenceKinds: torch.Tensor,

    useGradientRenormalization: bool = False, renormalizationMatrices: Optional[torch.Tensor] = None,
    useGradHTerms: bool = False, queryOmegas: Optional[torch.Tensor] = None, referenceOmegas: Optional[torch.Tensor] = None,
    useVolume: bool = False, queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    useCRK: bool = False, crk_A: Optional[torch.Tensor] = None, crk_B: Optional[torch.Tensor] = None, crk_gradA: Optional[torch.Tensor] = None, crk_gradB: Optional[torch.Tensor] = None
):
    """Flat-tensor "manual" entry point. Assembles the same state objects
    ``warpOperation`` takes (``ParticleState``, ``OperationProperties``, and -- only if
    the corresponding ``useX`` flag is set -- ``CRKState``/``GradHState``/
    ``RenormalizationState``) and calls it, rather than re-deriving dispatch logic here.
    The ``useX`` booleans below are flat-API-only sanity checks: they catch a manual
    caller setting e.g. ``useCRK=True`` without actually passing ``crk_A``/``crk_B``, a
    mistake that can't happen through the state-object API since CRK is "on" precisely
    when a ``CRKState`` is passed.

    ``queryKinds``/``referenceKinds`` are required keyword-only arguments (not
    ``Optional``, no default): ``ParticleState.kinds`` is a required field --
    see warpier_fields.md Section 2.5 -- so this flat entry point can no
    longer default them to ``None`` either.
    """
    if useGradientRenormalization and renormalizationMatrices is None:
        raise ValueError("Renormalization matrices must be provided if useGradientRenormalization is True.")
    if useGradHTerms and (queryOmegas is None or referenceOmegas is None):
        raise ValueError("Omegas must be provided if useGradHTerms is True.")
    if useVolume and (queryVolumes is None or referenceVolumes is None):
        raise ValueError("Volumes must be provided if useVolume is True.")
    if useCRK and (crk_A is None or crk_B is None):
        raise ValueError("CRK correction A and B tensors must be provided if useCRK is True.")

    queryParticles = ParticleState(positions=queryPositions, supports=querySupports, masses=queryMasses, densities=queryDensities, kinds=queryKinds)
    referenceParticles = ParticleState(positions=referencePositions, supports=referenceSupports, masses=referenceMasses, densities=referenceDensities, kinds=referenceKinds)

    operationProperties = OperationProperties(
        kernel=kernel,
        operation=operation,
        gradientMode=gradientMode,
        laplacianMode=laplacianMode,
        positiveDivergence=positiveDivergence,
        supportMode=supportMode,
        operationMode=operationMode,
        divergenceDotMode=divergenceDotMode,
    )

    crkState = CRKState(A=crk_A, B=crk_B, gradA=crk_gradA, gradB=crk_gradB) if useCRK else None
    gradHState = GradHState(queryOmegas=queryOmegas, referenceOmegas=referenceOmegas) if useGradHTerms else None
    renormalizationState = RenormalizationState(renormalizationMatrices=renormalizationMatrices) if useGradientRenormalization else None

    return warpOperation(
        queryParticles, operationProperties, domain,
        queryValues=queryValues, referenceValues=referenceValues,
        queryVolumes=queryVolumes if useVolume else None, referenceVolumes=referenceVolumes if useVolume else None,
        adjacency=adjacency,
        referenceParticles=referenceParticles,
        preScatteredQuantities=preScatteredQuantities,
        crkState=crkState,
        gradHState=gradHState,
        renormalizationState=renormalizationState,
        consistentDivergence=consistentDivergence,
    )


__all__ = [
    "sphOperation_warp",
    "warpOperation",
    "warpOperationJVP",
    "warpOperationHVP",
]