"""JVP counterpart to `crk_volume.py`'s apparent-volume estimate
(`warpier_tier2_correction_jvp_plan.md` phase (c), Stage 1):
`V_i = 1/wsum_i`, `wsum_i = Sum_j W_ij`, always `SupportScheme.Gather`
(matching `_computeCRKVolume_stateBackend`'s own hardcode) -- so
`dwsum_i = Sum_j dW_ij` (Tier 2.1's single-h/Gather-mode kernel-value JVP,
`kernels.kernelJVP.sphKernelJVP`) and `dV_i = -dwsum_i/wsum_i^2 =
-dwsum_i*V_i^2`. Operator-agnostic (produces `dV_i` regardless of which
value-having operator eventually consumes it via `correctionData.useVolume`),
so it is built once here for reuse by phase (e)'s Divergence/Curl/Laplacian
extension, exactly like `crk_moments_jvp.py`.

Ported from `scripts/spike_forward_mode_tier2_crk.py`'s
`assembled_apparent_volume_jvp` (validated there to float64 round-off against
`_computeCRKVolume_stateBackend`'s own reverse-mode Jacobian).

Mirrors `computeCRKVolume_Func_i`/`_Func_Adjacency`/`_Kernel`'s dual-path
(adjacency/grid) structure field-for-field, with one deliberate deviation:
`computeCRKVolumeJVP_Func_Adjacency` returns the raw accumulated `dwsum`,
NOT `dV = -dwsum*V^2`. This mirrors `computeCRKVolume_Func_Adjacency`'s own
documented reverse-mode fix (a *dynamic* for-loop that accumulates into a
local via `+=` and then feeds that local into a nonlinear post-loop op, all
inside the same `@wp.func`, produces NaN/wrong adjoints under Warp -- see
that function's own docstring) -- the quotient-rule step is applied one
level up, in `computeCRKVolumeJVP_Kernel`, outside the function containing
the loop.
"""

from typing import Any, Optional
import torch
import warp as wp

from ..type_config import *
from ..dataTypes import *
from ..enumTypes import *
from ..radiusSearch.grid_util import getIndexRange
from ..kernels.kernelJVP import sphKernelJVP
from ..util import checkDirectionality_i, checkDirectionality_j, getParticleData, getParticleCorrectionData_i

__all__ = ['computeCRKVolumeGeometryJVP']


@wp.func
def computeCRKVolumeJVP_Func_i(
    i: wp.int32, dim: wp.int32,
    iPtcl: Any, iTangentPtcl: Any,
    referenceState: Any, referenceTangentState: Any,

    domainState: domainData,
    kernelProperties: kernelState,

    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype = wp.int64), # type: ignore

    iCorrectionData: Any, correctionData: Any,
):
    dwsum = scalar_t(0.0)
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

        _, dW = sphKernelJVP(
            iPtcl.position, jPtcl.position, iPtcl.support, jPtcl.support,
            iTangentPtcl.position, jTangentPtcl.position, iTangentPtcl.support, jTangentPtcl.support,
            kernelProperties, domainState,
        )
        dwsum += dW

    return dwsum


@wp.func
def computeCRKVolumeJVP_Func_Adjacency(
    i: wp.int32, dim: wp.int32,
    queryState: Any, referenceState: Any,
    queryTangentState: Any, referenceTangentState: Any,
    correctionData: Any,
    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,
    kernelProperties: kernelState,
):
    # Returns (dwsum, masked) rather than the reciprocal's own tangent -- see
    # this module's docstring / computeCRKVolume_Func_Adjacency's own
    # docstring (crk_volume.py) for why the nonlinear quotient-rule step must
    # live outside this dynamic-loop function.
    iPtcl = getParticleData(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(iPtcl.kind, kernelProperties.operationMode):
            return scalar_t(0.0), True
    iTangentPtcl = getParticleData(queryTangentState, i)
    iCorrectionData = getParticleCorrectionData_i(correctionData, i)

    dwsum = scalar_t(0.0)
    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRange(i, o, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue

        dwsum += computeCRKVolumeJVP_Func_i(
            i, dim,
            iPtcl, iTangentPtcl,
            referenceState, referenceTangentState,

            domainState, kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,

            iCorrectionData, correctionData,
        )

    return dwsum, False


@wp.kernel
def computeCRKVolumeJVP_Kernel(
    queryState: Any,
    referenceState: Any,
    queryTangentState: Any,
    referenceTangentState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, correctionTangentData: Any,
    # correctionTangentData is unused here (Stage 1 has no correction of its own) --
    # kept for canonical-ABI parity with every other kernel _jvpCommon.launchGeometryJVP drives.

    kernelProperties: kernelState,
    # Do not change the parameters above -- canonical structured JVP kernel ABI, see
    # coreOperations/_jvpCommon.py's launchGeometryJVP docstring.

    apparentVolumes: wp.array(dtype = scalar_t), # type: ignore -- Stage 1's own PRIMAL output (extraTensors[0])

    # The last parameter is always the output array and should not be changed
    outputValues: wp.array(dtype = scalar_t) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    dwsum, masked = computeCRKVolumeJVP_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState, queryTangentState, referenceTangentState,
        correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
    )
    # dV_i = -dwsum_i/wsum_i^2 = -dwsum_i*V_i^2 -- applied here, outside
    # computeCRKVolumeJVP_Func_Adjacency's dynamic loop; see this module's docstring.
    if masked:
        outputValues[i] = scalar_t(0.0)
    else:
        V = apparentVolumes[i]
        outputValues[i] = -dwsum * V * V


def computeCRKVolumeGeometryJVP(
    queryParticles: ParticleState,
    domain: DomainDescription,
    kernel: KernelFunctions,
    adjacency: 'AdjacencyList | CompactHashMap',
    apparentVolumes: torch.Tensor,
    queryTangentState: ParticleTangentState,
    referenceParticles: Optional[ParticleState] = None,
    referenceTangentState: Optional[ParticleTangentState] = None,
) -> torch.Tensor:
    """`dV_i`, shape `[numParticles]` -- the JVP counterpart to
    `_computeCRKVolume_stateBackend`. Always `SupportScheme.Gather`, matching
    that function's own hardcode (`crk_wrapper.py`'s `volumeProperties`).
    `apparentVolumes` is that function's own primal output `V_i = 1/wsum_i`,
    needed for the quotient rule (see `computeCRKVolumeJVP_Kernel`).
    """
    # Deferred import: `coreOperations` imports `crk` (`wp_gradient.py`'s
    # `from ..crk import computeKernelGradientCRK`), not the reverse -- a
    # module-level import here would be circular.
    from ..coreOperations._jvpCommon import launchGeometryJVP

    referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
    device, dtype = queryParticles.positions.device, queryParticles.positions.dtype
    dim = domain.dim
    nQuery = queryParticles.positions.shape[0]
    nRef = referenceParticles.positions.shape[0]

    zerosVec = lambda n: torch.zeros((n, dim), device=device, dtype=dtype)
    zerosScalar = lambda n: torch.zeros(n, device=device, dtype=dtype)

    # No mass/density term anywhere in Stage 1's formula (crk_volume.py has no
    # useVolume/mass/density reference at all) -- zero those tangents
    # unconditionally, mirroring computeSPHGradientGeometryJVP's own
    # tangentQueryMasses handling.
    queryTangentState = ParticleTangentState(
        positions=queryTangentState.positions,
        supports=queryTangentState.supports if queryTangentState.supports is not None else zerosScalar(nQuery),
        masses=zerosScalar(nQuery),
        densities=zerosScalar(nQuery),
    )
    if referenceTangentState is None:
        referenceTangentState = ParticleTangentState(
            positions=zerosVec(nRef), supports=zerosScalar(nRef), masses=zerosScalar(nRef), densities=zerosScalar(nRef),
        )
    else:
        referenceTangentState = ParticleTangentState(
            positions=referenceTangentState.positions if referenceTangentState.positions is not None else zerosVec(nRef),
            supports=referenceTangentState.supports if referenceTangentState.supports is not None else zerosScalar(nRef),
            masses=zerosScalar(nRef),
            densities=zerosScalar(nRef),
        )

    return launchGeometryJVP(
        computeCRKVolumeJVP_Kernel,
        domain, kernel, SupportScheme.Gather, adjacency,
        queryParticles.positions, queryParticles.supports, queryParticles.masses,
        referenceParticles.positions, referenceParticles.supports, referenceParticles.masses,
        queryTangentState, referenceTangentState,
        outputShape=nQuery,
        outputDtype=scalar_t,
        queryDensities=queryParticles.densities,
        referenceDensities=referenceParticles.densities,
        extraTensors=(apparentVolumes,),
    )
