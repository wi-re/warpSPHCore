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


# Operators `warpOperationJVP` (Tier 1) can serve: exactly the ones with a
# `queryValues`/`referenceValues` input. Density has none (it reads
# `queryParticles.masses` directly) and Covariance takes volumes, not values --
# routing either through `warpOperation` with a tangent standing in for
# `queryValues` would silently ignore the tangent and hand back the *primal*
# result instead of raising, so both are excluded rather than left to fail
# quietly. Tier 2 (Phase 4) extends Density's mass-tangent path separately.
_TIER1_JVP_OPERATIONS = (
    WarpOperation.Interpolate, WarpOperation.Gradient, WarpOperation.Divergence,
    WarpOperation.Curl, WarpOperation.Laplacian,
)


def warpOperationJVP(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    tangentQueryValues: Optional[torch.Tensor] = None, tangentReferenceValues: Optional[torch.Tensor] = None,
    tangentQueryPositions: Optional[torch.Tensor] = None, tangentReferencePositions: Optional[torch.Tensor] = None,
    tangentQuerySupports: Optional[torch.Tensor] = None, tangentReferenceSupports: Optional[torch.Tensor] = None,
    tangentQueryMasses: Optional[torch.Tensor] = None, tangentReferenceMasses: Optional[torch.Tensor] = None,
    tangentQueryDensities: Optional[torch.Tensor] = None, tangentReferenceDensities: Optional[torch.Tensor] = None,
    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyListWarp, CompactHashMap]] = None,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor, RenormalizationState]] = None,
    consistentDivergence: bool = False,
):
    """The JVP entry point `warpier_forward_mode_plan.md` Phase 2 promotes
    Tier 1 into, and Phase 4 begins extending for Tier 2:

    * **Tier 1** (value tangents): an operator is exactly linear and
      homogeneous in `queryValues`/`referenceValues`, so its JVP w.r.t. them
      is the same operator relaunched on the tangent arrays in place of the
      value arrays -- verified by `scripts/spike_forward_mode_tier1.py` and
      gated by `tests/operations/test_forward_mode_tier1.py`.
    * **Tier 2** (position/support/mass/density tangents): the kernel is
      genuinely nonlinear in these, so each operator needs a hand-derived
      JVP (`warpier_adjoint.md`). Only **Density's position/support tangent
      plus a reference-side mass tangent** is implemented so far
      (`coreOperations.wp_densityJVP.computeSPHDensityPositionJVP`, gated by
      `tests/operations/test_forward_mode_tier2_density.py`) -- every other
      Tier-2 combination (the other five operators; Density's density
      tangent, which does not exist since Density has no density input)
      raises `NotImplementedError` naming Phase 4.
    """
    tier2Args = {
        "tangentQueryPositions": tangentQueryPositions, "tangentReferencePositions": tangentReferencePositions,
        "tangentQuerySupports": tangentQuerySupports, "tangentReferenceSupports": tangentReferenceSupports,
        "tangentQueryMasses": tangentQueryMasses, "tangentReferenceMasses": tangentReferenceMasses,
        "tangentQueryDensities": tangentQueryDensities, "tangentReferenceDensities": tangentReferenceDensities,
    }
    providedTier2 = [name for name, value in tier2Args.items() if value is not None]

    if providedTier2:
        isDensityPositionSupportCase = (
            operationProperties.operation is WarpOperation.Density
            and tangentQueryValues is None and tangentReferenceValues is None
            and tangentQueryMasses is None and tangentQueryDensities is None and tangentReferenceDensities is None
        )
        if not isDensityPositionSupportCase:
            raise NotImplementedError(
                f"warpOperationJVP: Tier-2 tangent argument(s) {providedTier2} are not "
                f"implemented for {operationProperties.operation} yet -- "
                "warpier_forward_mode_plan.md Phase 4 implements only Density's "
                "position/support/(reference-side) mass tangent so far."
            )
        if not isinstance(adjacency, AdjacencyList):
            raise NotImplementedError(
                "warpOperationJVP: Density's Tier-2 JVP needs the torch-facing "
                f"AdjacencyList (.i/.j neighbor pairs, what buildVerletList returns), "
                f"not {type(adjacency)} -- grid/CompactHashMap traversal is not "
                "implemented for Tier 2 (warpier_forward_mode_plan.md Phase 4)."
            )
        nQuery = queryParticles.positions.shape[0]
        zeroPositions = lambda n: torch.zeros((n, domain.dim), device=queryParticles.positions.device, dtype=queryParticles.positions.dtype)
        return computeSPHDensityPositionJVP(
            queryParticles, domain, operationProperties.kernel, operationProperties.supportMode, adjacency,
            tangentQueryPositions=tangentQueryPositions if tangentQueryPositions is not None else zeroPositions(nQuery),
            referenceParticles=referenceParticles,
            tangentReferencePositions=tangentReferencePositions,
            tangentQuerySupports=tangentQuerySupports,
            tangentReferenceSupports=tangentReferenceSupports,
            tangentReferenceMasses=tangentReferenceMasses,
        )

    if operationProperties.operation not in _TIER1_JVP_OPERATIONS:
        raise NotImplementedError(
            f"warpOperationJVP: Tier 1 is only defined for operators with a "
            f"queryValues/referenceValues input ({[op.name for op in _TIER1_JVP_OPERATIONS]}); "
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
]