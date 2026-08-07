import warp as wp
from warp.types import vector, matrix
from typing import Any, Optional, Union
import torch
from torch.profiler import record_function

from ..type_config import *
from ..autograd import *

from ..dataTypes import *

from ..radiusSearch.grid_util import checkOffset, getIndexRange
from ..math import *
from ..kernels import *
from ..util import *

from ..enumTypes import *

# Unified CRK-volume kernel: same dual-path design as every migrated operator (see
# warpier_core.md's "Working Prototype -> Production" section) -- one wp.func/wp.kernel
# pair drives both neighbor-list ("adjacency") and compact-hash-grid traversal.
# Computes the apparent volume estimate 1/sum_j(W_ij) used as the "volume" input to
# CRK moments/density -- no correction terms (CRK/grad-h/renorm) apply here, since this
# *is* one of the raw inputs those corrections are built from.


@wp.func
def computeCRKVolume_Func_i(
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
    out = scalar_t(0.0)
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

        x_ij = computeDistanceVec(iPtcl.position, jPtcl.position, domainState)
        w_ij = sphKernel_ij(x_ij, iPtcl.support, jPtcl.support, kernelProperties, domainState)

        out += w_ij

    return out


@wp.func
def computeCRKVolume_Func_Adjacency(
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
    # Returns (wsum, masked) rather than the final 1/wsum reciprocal -- Warp's adjoint
    # for a *dynamic* for-loop (numOffsets is a runtime value, not a compile-time
    # constant) that accumulates into a local via += and then feeds that local into a
    # nonlinear post-loop op (division), all inside the same @wp.func, produces NaN
    # gradients here (confirmed via a minimal standalone repro against just this
    # function -- see scripts/debug_crk_backward.py). Every other migrated operator's
    # _Func_Adjacency avoids this because it returns the loop-accumulated value
    # directly with no further transform. The reciprocal is applied one level up, in
    # computeCRKVolume_Kernel, outside the function that contains the loop -- verified
    # to fix the NaN gradient (same data, same case, clean backward once moved).
    iPtcl = getParticleData(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(iPtcl.kind, kernelProperties.operationMode):
            return scalar_t(0.0), True
    iCorrectionData = getParticleCorrectionData_i(correctionData, i)

    wsum = scalar_t(0.0)
    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRange(i, o, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue

        wsum += computeCRKVolume_Func_i(
            i, dim,
            iPtcl,
            referenceState, domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,

            iCorrectionData, correctionData,
        )

    return wsum, False


@wp.kernel
def computeCRKVolume_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, # unused (no corrections apply to the volume estimate itself) -- kept for ABI consistency, see coreOperations/wp_density.py for the same pattern

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

    wsum, masked = computeCRKVolume_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        # The parameters above are default parameters and shold not be changed
    )
    # The reciprocal is applied here, outside computeCRKVolume_Func_Adjacency's dynamic
    # loop -- see that function's docstring comment for why.
    if masked:
        outputValues[i] = scalar_t(0.0)
    else:
        outputValues[i] = scalar_t(1.0) / wsum


def _computeCRKVolume_stateBackend(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,

    adjacency: Optional[Union[AdjacencyListWarp, CompactHashMap]] = None,
    referenceParticles: Optional[ParticleState] = None,
) -> torch.Tensor:
    """Computes the CRK apparent-volume estimate V_i = 1 / sum_j W_ij for every query
    particle. This is the raw volume estimate only, used as an input to computeCRKMoments
    and _computeCRKDensity_stateBackend (crk_moments.py / crk_density.py) -- it applies
    no corrections of its own.
    """
    with record_function("warpSPH[CRKVolume]"):
        with record_function("warpSPH[CRKVolume] - Preprocessing"):
            outputSize = queryParticles.positions.shape[0]

        with record_function("warpSPH[CRKVolume] - Kernel Execution"):
            result = warpWrapper2(
                launcher=launch_kernel,
                kernel=computeCRKVolume_Kernel,
                outputSizes=outputSize,
                outputDtypes=scalar_t,
                defaultStateArguments=(
                    queryParticles, operationProperties, domain,
                    None, None,
                    adjacency,
                    referenceParticles,
                    None, None, None,
                ),
                additionalArguments=(),
            )

    return result
