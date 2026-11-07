"""Test-only corrected covariance JVP, separate from the public Warp API.

Retains wp_covarianceJVP's literal matrix equation and canonical state launcher;
uses the already qualified Warp CRK-gradient/renormalization primitives. Native
extensions are checked here, by an independent smooth Torch equation, and by
90-digit coefficient oracles. Coordinate-gradient identities provide an extra
smooth check but differ at singular IEEE boundaries (0*Inf intermediates).
"""
from typing import Any, Optional
import torch
import warp as wp
from warpSPHCore.type_config import *
from warpSPHCore.dataTypes import *
from warpSPHCore.enumTypes import *
from warpSPHCore.math import zero_like_warp, computeDistanceVec, matmul
from warpSPHCore.crk import computeKernelGradientCRKJVP
from warpSPHCore.util import getParticleCorrectionData_i, getParticleCorrectionTangentData_i
from warpSPHCore.kernels.kernelJVP import sphKernelGradientJVP
from warpSPHCore.radiusSearch.grid_util import getIndexRange
from warpSPHCore.util import _get_warp_matrix_dtype, checkDirectionality_i, checkDirectionality_j, getParticleData
from warpSPHCore.coreOperations._jvpCommon import (
    gradientWeightsJVP as _gradientWeightsJVP,
    launchGeometryJVP as _launchGeometryJVP,
)
__all__ = ['computeCovarianceGeometryJVP']
@wp.func
def computeCovarianceJVP_Func_i(
    i: wp.int32, dim: wp.int32,
    iPtcl: Any, iTangentPtcl: Any,
    referenceState: Any, referenceTangentState: Any,
    domainState: domainData,
    kernelProperties: kernelState,
    beginIndex: offset_t, numIndices: wp.int32, offsetArray: wp.array(dtype = index_t), # type: ignore
    correctionData: Any, correctionTangentData: Any,
    outputValue: Any, # type: ignore
):
    out = zero_like_warp(outputValue)
    for neighborIndex in range(numIndices):
        jj = beginIndex + offset_t(neighborIndex)
        j = wp.int32(offsetArray[jj])
        jPtcl = getParticleData(referenceState, j)
        if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
            if not checkDirectionality_j(jPtcl.kind, kernelProperties.operationMode):
                continue
        jTangentPtcl = getParticleData(referenceTangentState, j)
        _unusedA, Vj, _unusedDA, dVj = _gradientWeightsJVP(
            jPtcl.mass, iPtcl.density, jPtcl.density,
            jTangentPtcl.mass, iTangentPtcl.density, jTangentPtcl.density,
            wp.static(GradientScheme.Naive.value),
            correctionData.useVolume, correctionData.referenceVolumes[j], correctionTangentData.referenceVolumes[j],
        )
        coefficients = getParticleCorrectionData_i(correctionData, i)
        directions = getParticleCorrectionTangentData_i(correctionData, correctionTangentData, i)
        G, dG = computeKernelGradientCRKJVP(
            iPtcl.position, jPtcl.position, iPtcl.support, jPtcl.support,
            iTangentPtcl.position, jTangentPtcl.position, iTangentPtcl.support, jTangentPtcl.support,
            kernelProperties, domainState, correctionData.useCRK,
            coefficients.A, coefficients.B, coefficients.gradA, coefficients.gradB,
            directions.A, directions.B, directions.gradA, directions.gradB)
        if correctionData.useGradientRenormalization:
            dG = matmul(directions.renormalizationMatrix, G) + matmul(coefficients.renormalizationMatrix, dG)
            G = matmul(coefficients.renormalizationMatrix, G)
        fij = -computeDistanceVec(iPtcl.position, jPtcl.position, domainState)
        dfij = jTangentPtcl.position - iTangentPtcl.position
        out += dVj * wp.outer(fij, G) + Vj * wp.outer(dfij, G) + Vj * wp.outer(fij, dG)
    return out
@wp.func
def computeCovarianceJVP_Func_Adjacency(
    i: wp.int32, dim: wp.int32,
    queryState: Any, referenceState: Any,
    queryTangentState: Any, referenceTangentState: Any,
    correctionData: Any, correctionTangentData: Any,
    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,
    kernelProperties: kernelState,
    outputValue: Any, # type: ignore
):
    iPtcl = getParticleData(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(iPtcl.kind, kernelProperties.operationMode):
            return zero_like_warp(outputValue)
    iTangentPtcl = getParticleData(queryTangentState, i)
    out = zero_like_warp(outputValue)
    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRange(i, o, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue
        out += computeCovarianceJVP_Func_i(
            i, dim,
            iPtcl, iTangentPtcl,
            referenceState, referenceTangentState,
            domainState, kernelProperties,
            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            correctionData, correctionTangentData,
            outputValue,
        )
    return out
@wp.kernel
def computeCovarianceJVP_Kernel(
    queryState: Any,
    referenceState: Any,
    queryTangentState: Any,
    referenceTangentState: Any,
    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, correctionTangentData: Any,
    kernelProperties: kernelState,
    outputValues: wp.array(dtype = Any) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return
    outputValues[i] = computeCovarianceJVP_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState,
        queryTangentState, referenceTangentState,
        correctionData, correctionTangentData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        zero_like_warp(outputValues[i]),
    )
def computeCovarianceGeometryJVP(
    queryParticles: ParticleState,
    domain: DomainDescription,
    kernel: KernelFunctions,
    supportMode: SupportScheme,
    adjacency: 'AdjacencyList | CompactHashMap',
    queryTangentState: ParticleTangentState,
    referenceParticles: Optional[ParticleState] = None,
    referenceTangentState: Optional[ParticleTangentState] = None,
    referenceVolumes: Optional[torch.Tensor] = None,
    tangentReferenceVolumes: Optional[torch.Tensor] = None,
    operationMode: OperationDirection = OperationDirection.AllToAll,
    crkState=None, crkTangentState=None, renormalizationState=None, renormalizationTangentState=None,
) -> torch.Tensor:
    referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
    dim = domain.dim
    device, dtype = queryParticles.positions.device, queryParticles.positions.dtype
    nQuery = queryParticles.positions.shape[0]
    nRef = referenceParticles.positions.shape[0]
    zerosVec = lambda n: torch.zeros((n, dim), device=device, dtype=dtype)
    zerosScalar = lambda n: torch.zeros(n, device=device, dtype=dtype)
    queryTangentState = ParticleTangentState(
        positions=queryTangentState.positions,
        supports=queryTangentState.supports if queryTangentState.supports is not None else zerosScalar(nQuery),
        masses=zerosScalar(nQuery),
        densities=queryTangentState.densities if queryTangentState.densities is not None else zerosScalar(nQuery),
    )
    if referenceTangentState is None:
        referenceTangentState = ParticleTangentState(
            positions=zerosVec(nRef), supports=zerosScalar(nRef), masses=zerosScalar(nRef), densities=zerosScalar(nRef),
        )
    else:
        referenceTangentState = ParticleTangentState(
            positions=referenceTangentState.positions if referenceTangentState.positions is not None else zerosVec(nRef),
            supports=referenceTangentState.supports if referenceTangentState.supports is not None else zerosScalar(nRef),
            masses=referenceTangentState.masses if referenceTangentState.masses is not None else zerosScalar(nRef),
            densities=referenceTangentState.densities if referenceTangentState.densities is not None else zerosScalar(nRef),
        )
    return _launchGeometryJVP(
        computeCovarianceJVP_Kernel,
        domain, kernel, supportMode, adjacency,
        queryParticles.positions, queryParticles.supports, queryParticles.masses,
        referenceParticles.positions, referenceParticles.supports, referenceParticles.masses,
        queryTangentState, referenceTangentState,
        queryKinds=queryParticles.kinds, referenceKinds=referenceParticles.kinds,
        operationMode=operationMode,
        outputShape=nQuery,
        outputDtype=_get_warp_matrix_dtype(dim, dim, dtype),
        queryDensities=queryParticles.densities,
        referenceDensities=referenceParticles.densities,
        referenceVolumes=referenceVolumes,
        tangentReferenceVolumes=tangentReferenceVolumes,
        crkState=crkState, crkTangentState=crkTangentState,
        renormalizationState=renormalizationState, renormalizationTangentState=renormalizationTangentState,
    )
