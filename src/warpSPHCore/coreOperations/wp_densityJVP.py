"""Geometry-tangent JVP of the Density operator
(`warpier_forward_mode_plan.md` Phase 4, `warpier_adjoint.md` Tier 2.1):
`dDensity_i = sum_j [ dm_j * W_ij + m_j * dW_ij ]`, `dW_ij` from
`kernels.kernelJVP.sphKernelJVP`.

CSR (per-query-particle) launch shape (`warpier_tier2_jvp_csr_backend_plan.md`):
one warp thread per query particle `i`, looping `i`'s own neighbors via the
canonical structured kernel ABI (`beginIndex`/`numIndices`/`offsetArray`,
same shape `wp_density.py`'s `computeSPHDensity_Func_i`/`_Func_Adjacency`/
`_Kernel` use), with a tangent counterpart threaded alongside every primal
argument. Also supports grid (`CompactHashMap`) traversal via the same
`_Func_Adjacency` dispatch every primal operator already uses -- a side
effect of reusing that pattern. Replaced the original pair-indexed (COO,
one thread per adjacency pair, `torch.index_add_`-based) implementation once
proven numerically equivalent to float32 round-off (same plan, Step 1); see
git history around 2026-08-20 for that implementation and its own
equivalence tests if reference is ever needed.

Scope: position and support tangents on both query and reference roles, plus
a reference-side mass tangent (`dm_j`) -- the terms Tier 2.1's formula
actually has. No query-side mass tangent (`Density_i` has no `m_i` term) and
no density tangent (Density has no `queryValues`/`referenceValues`/density
input at all). Field-value tangents are the value JVP's `warpOperationJVP`
path, not this.
"""

from typing import Any, Optional
import torch
import warp as wp

from ..type_config import *
from ..dataTypes import *
from ..enumTypes import *
from ..math import zero_like_warp
from ..kernels.kernelJVP import sphKernelJVP
from ..radiusSearch.grid_util import getIndexRange
from ..util import allocateTorchWarp
from ..util import checkDirectionality_i, checkDirectionality_j, getParticleData, getParticleCorrectionData_i
from ._jvpCommon import (
    buildParticleSoA as _buildParticleSoA,
    buildDomainState as _buildDomainState,
    buildKernelState as _buildKernelState,
    buildAdjacencyOrGridState as _buildAdjacencyOrGridState,
    buildNullCorrectionData as _buildNullCorrectionData,
)

__all__ = ['computeSPHDensityGeometryJVP']


@wp.func
def computeSPHDensityJVP_Func_i(
    i: wp.int32, dim: wp.int32,
    iPtcl: Any, iTangentPtcl: Any,
    referenceState: Any, referenceTangentState: Any,

    domainState: domainData,
    kernelProperties: kernelState,

    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype = wp.int64), # type: ignore

    iCorrectionData: Any, correctionData: Any,

    outputValue: Any, # type: ignore
):
    out = zero_like_warp(outputValue)
    for neighborIndex in range(numIndices):
        jj = beginIndex + neighborIndex
        j = wp.int32(offsetArray[jj])
        jPtcl = getParticleData(referenceState, j)
        if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
            if not checkDirectionality_j(jPtcl.kind, kernelProperties.operationMode):
                continue
        ##########################################################
        #   The core particle-particle interaction starts here   #
        ##########################################################
        jTangentPtcl = getParticleData(referenceTangentState, j)

        W, dW = sphKernelJVP(
            iPtcl.position, jPtcl.position, iPtcl.support, jPtcl.support,
            iTangentPtcl.position, jTangentPtcl.position, iTangentPtcl.support, jTangentPtcl.support,
            kernelProperties, domainState,
        )

        out += jTangentPtcl.mass * W + jPtcl.mass * dW

    return out


@wp.func
def computeSPHDensityJVP_Func_Adjacency(
    i: wp.int32, dim: wp.int32,
    queryState: Any, referenceState: Any,
    queryTangentState: Any, referenceTangentState: Any,
    correctionData: Any,
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
    iCorrectionData = getParticleCorrectionData_i(correctionData, i)

    out = zero_like_warp(outputValue)
    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRange(i, o, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue

        out += computeSPHDensityJVP_Func_i(
            i, dim,
            iPtcl, iTangentPtcl,
            referenceState, referenceTangentState,

            domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,

            iCorrectionData, correctionData,

            outputValue,
        )
    return out


@wp.kernel
def computeSPHDensityJVP_Kernel(
    queryState: Any,
    referenceState: Any,
    queryTangentState: Any,
    referenceTangentState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,

    kernelProperties: kernelState,
    # Do not change the parameters above -- canonical structured kernel ABI, see warpier_core.md

    # The last parameter is always the output array and should not be changed
    outputValues: wp.array(dtype = Any) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeSPHDensityJVP_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState,
        queryTangentState, referenceTangentState,
        correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,

        zero_like_warp(outputValues[i]),
    )


def computeSPHDensityGeometryJVP(
    queryParticles: ParticleState,
    domain: DomainDescription,
    kernel: KernelFunctions,
    supportMode: SupportScheme,
    adjacency: 'AdjacencyList | CompactHashMap',
    tangentQueryPositions: torch.Tensor,
    referenceParticles: Optional[ParticleState] = None,
    tangentReferencePositions: Optional[torch.Tensor] = None,
    tangentQuerySupports: Optional[torch.Tensor] = None,
    tangentReferenceSupports: Optional[torch.Tensor] = None,
    tangentReferenceMasses: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """`dDensity_i`, shape `[numParticles]`. `adjacency` is the torch-facing
    `AdjacencyList` (`.i`/`.j` flat neighbor pairs -- what `buildVerletList`
    returns and what every `warpOperation` call in this codebase already
    threads through) or a `CompactHashMap`, not the warp-side struct
    `launchOperator` builds internally.

    Unlike the five value-having operators' `computeSPH<Op>GeometryJVP`,
    this is never a **partial** contribution: Density has no
    `queryValues`/`referenceValues` input at all (it reads
    `queryParticles.masses`/densities directly), so there is no value-JVP
    piece to add on top -- this function's return value already is
    Density's full JVP (`warpier_tier2_combined_jvp_plan.md`).
    """
    referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
    dim = domain.dim
    device, dtype = queryParticles.positions.device, queryParticles.positions.dtype
    nQuery = queryParticles.positions.shape[0]
    nRef = referenceParticles.positions.shape[0]

    zerosVec = lambda n: torch.zeros((n, dim), device=device, dtype=dtype)
    zerosScalar = lambda n: torch.zeros(n, device=device, dtype=dtype)

    tangentReferencePositions = tangentReferencePositions if tangentReferencePositions is not None else zerosVec(nRef)
    tangentQuerySupports = tangentQuerySupports if tangentQuerySupports is not None else zerosScalar(nQuery)
    tangentReferenceSupports = tangentReferenceSupports if tangentReferenceSupports is not None else zerosScalar(nRef)
    tangentReferenceMasses = tangentReferenceMasses if tangentReferenceMasses is not None else zerosScalar(nRef)

    queryState = _buildParticleSoA(dim, queryParticles.positions, queryParticles.supports, queryParticles.masses)
    referenceState = _buildParticleSoA(dim, referenceParticles.positions, referenceParticles.supports, referenceParticles.masses)
    queryTangentState = _buildParticleSoA(dim, tangentQueryPositions, tangentQuerySupports, zerosScalar(nQuery))
    referenceTangentState = _buildParticleSoA(dim, tangentReferencePositions, tangentReferenceSupports, tangentReferenceMasses)
    domainState = _buildDomainState(domain)
    kernelProperties = _buildKernelState(kernel, supportMode)
    correctionData = _buildNullCorrectionData(dim, device)

    useAdjacency, adjacencyState, gridState, _numOffsets = _buildAdjacencyOrGridState(adjacency, domain)

    massDtype = queryState.masses.dtype
    warpDevice = queryState.positions.device
    dDensity_t, dDensity_w = allocateTorchWarp(nQuery, massDtype, warpDevice)

    wp.launch(
        computeSPHDensityJVP_Kernel,
        dim=nQuery,
        inputs=[
            queryState, referenceState,
            queryTangentState, referenceTangentState,
            domainState,
            useAdjacency, adjacencyState, gridState,
            correctionData,
            kernelProperties,
            dDensity_w,
        ],
        device=warpDevice,
    )
    return dDensity_t
