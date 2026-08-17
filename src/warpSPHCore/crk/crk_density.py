import warp as wp
from warp.types import vector, matrix
from typing import Any, Optional, Union
import torch
from ..profiling import record_function

from ..type_config import *
from ..autograd import *

from ..dataTypes import *

from ..radiusSearch.grid_util import checkOffset, getIndexRange
from ..math import *
from ..kernels import *
from ..util import *

from ..enumTypes import *

from .kernel import computeKernelCRK

# Unified CRK-density kernel: same dual-path design as every migrated operator (see
# warpier_core.md's "Working Prototype -> Production" section) -- one wp.func/wp.kernel
# pair drives both neighbor-list ("adjacency") and compact-hash-grid traversal.
# Computes a CRK-corrected density estimate (mDensity/vol1, a Shepard-style ratio) used
# by computeCRKFactors as a diagnostic/consistency companion to A/B/gradA/gradB -- not
# the general SPH Density operator (coreOperations/wp_density.py).


@wp.func
def computeCRKDensity_Func_i(
    i: wp.int32, dim: wp.int32,
    # SPH properties for the query point (indexed by i)
    iPtcl: Any, # WarpParticle_1/2/3, picked by dimensionality
    # SPH properties for the reference set (indexed by j in the neighbor loop)
    referenceState: Any, # particleDataSoA_1/2/3, picked by dimensionality

    # Domain and kernel parameters
    domainState: domainData,
    kernelProperties: kernelState,

    # Neighbor range within offsetArray to iterate; offsetArray is either the adjacency
    # neighbor list or the grid's sorted particle index, depending on the caller.
    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype = wp.int64), # type: ignore

    # Optional correction terms
    iCorrectionData: Any, # ParticleCorrectionData_1/2/3, picked by dimensionality
    correctionData: Any, # correctionData_1/2/3
    # End of the canonical structured kernel ABI prefix; the rest of the arguments are specific to this operator.
):
    mDensity = scalar_t(0.0)
    vol1 = scalar_t(0.0)

    # mode/useCRK are hardcoded here (not kernelProperties.supportMode/a useCRK flag) --
    # pre-existing behavior carried over unchanged from the neighbor-list-only kernel
    # this replaces; not touched by this traversal-style migration.
    crkKernelProperties = kernelState()
    crkKernelProperties.kernelFunction = kernelProperties.kernelFunction
    crkKernelProperties.supportMode = wp.uint32(12)

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

        # xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
        _, Vj = getVolume_j(correctionData, j)

        w_ij = computeKernelCRK(
            iPtcl.position, jPtcl.position,
            iPtcl.support, jPtcl.support,
            crkKernelProperties, domainState,
            True, iCorrectionData.A, iCorrectionData.B
        )

        mDensity += jPtcl.mass * Vj * w_ij
        vol1 += Vj * Vj * w_ij

    return mDensity, vol1


@wp.func
def computeCRKDensity_Func_Adjacency(
    i: wp.int32, dim: wp.int32,
    # SPH properties for the points and the corrections
    queryState: Any, referenceState: Any, correctionData: Any,
    # Domain properties 
    domainState: domainData,
    # Adjacency / grid traversal properties
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,
    # Kernel properties, e.g., kernel type, support scheme, gradient scheme, etc.
    kernelProperties: kernelState,
    # end of the canonical structured kernel ABI prefix; the rest of the arguments are specific to this operator.
):
    # Returns (mDensity, vol1, masked) rather than the final mDensity/vol1 ratio --
    # Warp's adjoint for a dynamic for-loop (numOffsets is a runtime value) that
    # accumulates into locals via += and then feeds them into a nonlinear post-loop op
    # (division), all inside the same @wp.func, produces NaN gradients here -- same
    # issue as computeCRKVolume_Func_Adjacency, see its docstring comment and
    # scripts/debug_crk_backward.py for the minimal repro. The ratio is applied one
    # level up, in computeCRKDensity_Kernel, outside the function that contains the loop.
    iPtcl = getParticleData(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(iPtcl.kind, kernelProperties.operationMode):
            return scalar_t(0.0), scalar_t(0.0), True
    iCorrectionData = getParticleCorrectionData_i(correctionData, i)

    mDensity = scalar_t(0.0)
    vol1 = scalar_t(0.0)
    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRange(i, o, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue

        d_mDensity, d_vol1 = computeCRKDensity_Func_i(
            i, dim,
            iPtcl,
            referenceState, domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,

            iCorrectionData, correctionData,
            # end of the canonical structured kernel ABI prefix; the rest of the arguments are specific to this operator.
        )
        mDensity += d_mDensity
        vol1 += d_vol1

    return mDensity, vol1, False


@wp.kernel
def computeCRKDensity_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,

    kernelProperties: kernelState,
    # Do not change the parameters above -- this is the canonical structured kernel ABI
    # (see warpier_core.md, Phase 1 / Step 1); other operators share this argument prefix.

    # The last parameter is always the output array and should not be changed
    outputValues: wp.array(dtype = scalar_t) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    mDensity, vol1, masked = computeCRKDensity_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        # The parameters above are default parameters and shold not be changed
    )
    # The ratio is applied here, outside computeCRKDensity_Func_Adjacency's dynamic
    # loop -- see that function's docstring comment for why.
    if masked:
        outputValues[i] = scalar_t(0.0)
    else:
        outputValues[i] = mDensity / vol1


def _computeCRKDensity_stateBackend(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,

    crkState: CRKState,
    queryVolumes: torch.Tensor, referenceVolumes: torch.Tensor,
    adjacency: Optional[Union[AdjacencyListWarp, CompactHashMap]] = None,
    referenceParticles: Optional[ParticleState] = None,
) -> torch.Tensor:
    """Computes the CRK-corrected consistency density estimate for every query
    particle, using the CRK terms (A/B) and apparent volumes already solved for by
    computeCRKMoments/computeCRKTermsWarp.
    """
    with record_function("warpSPH[CRKDensity]"):
        with record_function("warpSPH[CRKDensity] - Preprocessing"):
            outputSize = queryParticles.positions.shape[0]

        with record_function("warpSPH[CRKDensity] - Kernel Execution"):
            result = warpWrapper2(
                launcher=launch_kernel,
                kernel=computeCRKDensity_Kernel,
                outputSizes=outputSize,
                outputDtypes=scalar_t,
                defaultStateArguments=(
                    queryParticles, operationProperties, domain,
                    queryVolumes, referenceVolumes,
                    adjacency,
                    referenceParticles,
                    crkState, None, None,
                ),
                additionalArguments=(),
            )

    return result
