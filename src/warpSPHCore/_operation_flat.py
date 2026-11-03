"""Shared flat-tensor adapter; all execution goes through the semantic API."""
from __future__ import annotations
from typing import Optional,Union
import torch
from ._models import *
from .enumTypes import *

def sphOperation_warp(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,
    queryDensities, referenceDensities,
    queryValues : Optional[torch.Tensor], referenceValues : Optional[torch.Tensor],
    domain: DomainDescription,
    adjacency: Optional[object] = None, # if none a datastructure is created for EVERY operation!,
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

    from .operations import warpOperation
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
