import warp as wp
from typing import NamedTuple, Union, Tuple, List, Optional, Any
from warp.types import vector, matrix
import torch
from ..profiling import record_function

from ..type_config import *
from ..autograd import *

from ..dataTypes import *

from ..radiusSearch.grid_util import checkOffset, getIndexRange
from ..math import *
from ..kernels import *
from ..util import *
from torch.profiler import profile, record_function, ProfilerActivity

from ..enumTypes import *
from ..autograd import *

from ..crk import computeKernelCRK, computeKernelGradientCRK

# Unified Gradient kernel: a single wp.func/wp.kernel pair drives both neighbor-list
# ("adjacency") and compact-hash-grid traversal. The two only differ in how
# (beginIndex, numIndices, offsetArray) are produced -- see
# computeSPHGradientTensor_Func_Adjacency below -- so the per-neighbor physics is
# written exactly once. This replaced a former split between an adjacency-only flat-args
# kernel in this file and a separate grid-only kernel with duplicated physics in the
# now-deleted operations_grid/ package. See warpier_core.md's "Working Prototype" section
# for the design this follows
# (prototyped in the repo-root wp_grad.py / warp_gradient.ipynb).
#
# For matrices we need to implement the logic manually using outer products, since Warp
# does not support rank-2 field types natively. The output is stored as a flattened
# vector and reshaped back into its tensor shape on the Python side.


@wp.func
def computeSPHGradientTensor_Func_i(
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

    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    fi: Any, referenceValues: wp.array(dtype = Any), # type: ignore

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

        # Explicit if/else, not a ternary: confirmed (2026-08-21,
        # `warpier_tier2_correction_jvp_plan.md` phase b's spike) that this exact "different
        # arrays per branch" ternary DOES silently zero d(output)/d(referenceVolumes) under
        # the installed warp 1.16.0, contradicting `docs/lessons_learned.md`'s earlier
        # "confirmed non-issue" note for this pattern -- see that doc's updated entry.
        if correctionData.useVolume:
            apparentVolume = correctionData.referenceVolumes[j]
        else:
            apparentVolume = jPtcl.mass / jPtcl.density

        # Explicit if/else, not a ternary: both branches read referenceValues[j], and a
        # ternary assigned to a local where both branches index the *same* array silently
        # zeroes that array's adjoint (compiles fine, runs the correct branch, wrong
        # gradient) -- see docs/lessons_learned.md, the bug that broke Interpolate this way.
        # CONFIRMED FIXED upstream in warp-lang 1.17.0.dev3 (warp_dev env) as of 2026-08-11
        # -- see scripts/repro_ternary_adjoint_zeroing.py. Not yet on PyPI; once a 1.17+
        # release lands and pyproject.toml's warp-lang floor is bumped to it, this can go
        # back to the ternary form (see git history around 2026-08-11 for the tested diff).
        if correctionData.useGradHTerms:
            fj = referenceValues[j] / correctionData.referenceOmegas[j]
        else:
            fj = referenceValues[j]

        kernelGradient = computeKernelGradientCRK(
            iPtcl.position, jPtcl.position,
            iPtcl.support, jPtcl.support,
            kernelProperties, domainState,
            correctionData.useCRK, iCorrectionData.A, iCorrectionData.B, iCorrectionData.gradA, iCorrectionData.gradB
        )

        if correctionData.useGradientRenormalization:
            kernelGradient = matmul(iCorrectionData.renormalizationMatrix, kernelGradient)

        if kernelProperties.gradientMode == wp.static(GradientScheme.Naive.value): # Naive
            out += outerTensorProduct(fj * apparentVolume, kernelGradient, out, numDims, flatInputShape, flatOutputShape)
        elif kernelProperties.gradientMode == wp.static(GradientScheme.Symmetric.value): # Symmetric
            out += outerTensorProduct(jPtcl.mass * iPtcl.density * (fi / iPow(iPtcl.density,2) + fj / iPow(jPtcl.density,2)), kernelGradient, out, numDims, flatInputShape, flatOutputShape)
        elif kernelProperties.gradientMode == wp.static(GradientScheme.Difference.value): # Difference
            out += outerTensorProduct((fj - fi) * apparentVolume, kernelGradient, out, numDims, flatInputShape, flatOutputShape)
        elif kernelProperties.gradientMode == wp.static(GradientScheme.Summation.value): # Summation
            out += outerTensorProduct((fj + fi) * apparentVolume, kernelGradient, out, numDims, flatInputShape, flatOutputShape)

    return out


@wp.func
def computeSPHGradientTensor_Func_Adjacency(
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


    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    queryValue: Any, referenceValues: Any, # type: ignore
    

    outputValue: Any, # type: ignore
):
    iPtcl = getParticleData(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(iPtcl.kind, kernelProperties.operationMode):
            return zero_like_warp(outputValue)

    iCorrectionData = getParticleCorrectionData_i(correctionData, i)

    fi = queryValue[i]

    out = zero_like_warp(outputValue)
    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRange(i, o, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue

        out += computeSPHGradientTensor_Func_i(
            i, dim,
            iPtcl,
            referenceState, domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,

            iCorrectionData, correctionData,
            # end of the canonical structured kernel ABI prefix; the rest of the arguments are specific to this operator.

            numDims, flatInputShape, flatOutputShape,
            fi, referenceValues,

            outputValue,
        )
    return out


@wp.kernel
def computeSPHGradientTensor_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,

    kernelProperties: kernelState,
    # Do not change the parameters above -- this is the canonical structured kernel ABI
    # (see warpier_core.md, Phase 1 / Step 1); other operators share this argument prefix.

    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    queryValues: Any, referenceValues: Any, # type: ignore

    # The last parameter is always the output array and should not be changed
    outputValues: wp.array(dtype = Any) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeSPHGradientTensor_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        # The parameters above are default parameters and shold not be changed
        
        numDims, flatInputShape, flatOutputShape,
        queryValues, referenceValues,

        # zero_like_warp on the *array itself* only has overloads for output
        # lengths 1-3 (see wp_util.py) and silently falls back to a broken
        # generic `type(array)()` for anything longer -- vector/matrix-valued
        # fields flatten to longer outputs than that. Index into the array
        # first so this generalizes to any flatOutputShape, matching the
        # pattern the previous (adjacency-only) kernel used.
        zero_like_warp(outputValues[i]),
    )


def _gradientOutputDtype(ctx, extras):
    # Read the already-computed flatOutputShape extra rather than
    # re-deriving it here: it depends on the *unflattened* input shape
    # (queryValues.shape[1:]), which is only known in the caller's
    # preprocessing block, not recoverable from the flattened tensor alone.
    return _get_warp_vector_dtype(int(extras["flatOutputShape"]), extras["queryValuesFlat"].dtype)


_GRADIENT_SPEC = OperatorSpec(
    kernel=computeSPHGradientTensor_Kernel,
    outputs=(OutputSpec(dtype=_gradientOutputDtype),),
    extras=(
        ExtraSpec("numDims", ExtraKind.SCALAR),
        ExtraSpec("flatInputShape", ExtraKind.SCALAR),
        ExtraSpec("flatOutputShape", ExtraKind.SCALAR),
        ExtraSpec("queryValuesFlat", ExtraKind.TENSOR),
        ExtraSpec("referenceValuesFlat", ExtraKind.TENSOR),
    ),
)


def _computeSPHGradient_stateBackend(
    queryParticles: ParticleState,
    referenceParticles: ParticleState,
    domain: DomainDescription,
    mode: SupportScheme,
    kernel: KernelFunctions,
    gradientMode: GradientScheme,
    operationMode: OperationDirection,
    adjacency, # AdjacencyList | CompactHashMap | None -- both traversal modes go through this one path
    queryValues: torch.Tensor, referenceValues: torch.Tensor,
    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[GradHState] = None,
    renormalizationState: Optional[RenormalizationState] = None,
):
    with record_function("warpSPH[Gradient]"):
        with record_function("warpSPH[Gradient] - Preprocessing"):
            queryPositions = queryParticles.positions
            outputSize = queryPositions.shape[0]

            inputShape = queryValues.shape[1:]
            flatInputShape = 1
            for d in inputShape:
                flatInputShape *= d

            outputShape = inputShape + (queryPositions.shape[1],) # add an extra dimension for the gradient
            flatOutputShape = 1
            for d in outputShape:
                flatOutputShape *= d
            numDims = len(inputShape)

            operationProperties = OperationProperties(
                kernel=kernel,
                operation=WarpOperation.Gradient,
                gradientMode=gradientMode,
                supportMode=mode,
                operationMode=operationMode,
            )

        with record_function("warpSPH[Gradient] - Kernel Execution"):
            ctx = SPHContext(
                query=queryParticles,
                properties=operationProperties,
                domain=domain,
                adjacency=adjacency,
                reference=referenceParticles,
                corrections=Corrections(
                    volumes=(queryVolumes, referenceVolumes),
                    crk=crkState, gradH=gradHState, renorm=renormalizationState,
                ),
            )
            result = launchOperator(
                _GRADIENT_SPEC, ctx,
                numDims=wp.int32(numDims),
                flatInputShape=wp.int32(flatInputShape),
                flatOutputShape=wp.int32(flatOutputShape),
                queryValuesFlat=queryValues.view(-1, flatInputShape),
                referenceValuesFlat=referenceValues.view(-1, flatInputShape),
            )

    return result.view(outputSize, *outputShape)
