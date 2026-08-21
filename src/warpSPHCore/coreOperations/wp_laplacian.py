import warp as wp
from typing import NamedTuple, Union, Tuple, List, Optional, Any
from warp.types import vector, matrix
import torch
from ..profiling import record_function

from ..type_config import *
from ..autograd import *

from ..dataTypes import *

from ..radiusSearch.grid_util import getIndexRange, checkOffset
from ..math import *
from ..kernels import *
from ..util import *

from ..enumTypes import *
from ..autograd import *

from ..crk import computeKernelCRK, computeKernelGradientCRK
from ..util.support import computePairwiseSupport


@wp.func
def computeSPHLaplacianTensor_Func_i(
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
    flatInputShape: wp.int32, flatOutputShape: wp.int32,

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

        q_ij = zero_like_warp(fi)

        # The Brookshaw/Dot/Default laplacian schemes below all take the form
        # Sum_j K_ij * q_ij where K_ij = -2*(kernelGradient . n_ij)/r_ij. This
        # is only a consistent estimator of the Laplacian if q_ij vanishes
        # identically whenever f is spatially constant (fi == fj everywhere) --
        # otherwise the sum picks up an uncancelled O(1/h^2) residual from the
        # kernel's own second-moment scaling that grows without bound as the
        # resolution increases (confirmed empirically: the residual scales as
        # 1/h^2, i.e. it roughly quadruples every time h halves). Difference's
        # q_ij = (fj - fi) * V_j satisfies this by construction. The other
        # three GradientScheme variants were defined for the Gradient/
        # Divergence/Curl operators, where this constraint doesn't apply, and
        # reusing them here unmodified (or merely sign-flipped) inherits that
        # same divergent residual. Naive and Summation are simply Difference
        # with the (fj - fi) differencing replaced by fj and (fj + fi)
        # respectively; substituting fi for fj in each (the value the term
        # must reduce to for a constant field) and subtracting that self-term
        # collapses both back to exactly Difference's form. Symmetric's own
        # weighting is density-asymmetric, so the same self-term subtraction
        # instead yields a distinct (but equally consistent) density-weighted
        # variant of the same (fj - fi) family.
        if kernelProperties.gradientMode == wp.static(GradientScheme.Naive.value): # Naive
            q_ij = (fj - fi) * apparentVolume
        elif kernelProperties.gradientMode == wp.static(GradientScheme.Symmetric.value): # Symmetric
            q_ij = (fj - fi) * jPtcl.mass * iPtcl.density / iPow(jPtcl.density, 2)
        elif kernelProperties.gradientMode == wp.static(GradientScheme.Difference.value): # Difference
            q_ij = (fj - fi) * apparentVolume
        elif kernelProperties.gradientMode == wp.static(GradientScheme.Summation.value): # Summation
            q_ij = (fj - fi) * apparentVolume

        h_ij = computePairwiseSupport(iPtcl.support, jPtcl.support, kernelProperties.supportMode)
        x_ij = computeDistanceVec(iPtcl.position, jPtcl.position, domainState)
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))

        eps = scalar_t(1e-8)
        n_ij = x_ij / (r_ij + eps * h_ij)

        laplacian_contribution = zero_like_warp(outputValue)

        if kernelProperties.laplacianMode == wp.static(LaplacianScheme.Naive.value): # Naive
            laplacian_contribution = q_ij * sphKernelLaplacian(iPtcl.position, jPtcl.position, iPtcl.support, jPtcl.support, kernelProperties, domainState)
        elif kernelProperties.laplacianMode == wp.static(LaplacianScheme.Brookshaw.value): # Brookshaw
            # Explicit r_ij > 0 guard, not relied on falling out of dot(kernelGradient, n_ij)
            # naturally: at an exact self-pair (r_ij == 0) n_ij == 0 by construction, so the
            # *forward* contribution is always exactly 0 here regardless of kernelGradient --
            # but with CRK enabled, correctGradientCRK's own r=0 value is generically nonzero
            # (its W_ij*Bi/W_ij*gradAi terms don't vanish at x_ij=0 the way the plain kernel
            # gradient does), and Warp's reverse-mode through "nonzero vector dotted against an
            # exactly-zero n_ij" at that exact point produces a wrong adjoint (confirmed via
            # gradcheck on a from-scratch minimal repro and on this kernel directly -- see
            # warpier_tier2_correction_jvp_plan.md phase (e)'s own "Done" writeup). Skipping the
            # self-pair term outright sidesteps the bad adjoint with zero change to any forward
            # value (CRK or not).
            if r_ij > scalar_t(0.0):
                laplacian_contribution = -scalar_t(2.0) * q_ij * wp.dot(kernelGradient, n_ij) / (r_ij + eps * h_ij)
        elif kernelProperties.laplacianMode == wp.static(LaplacianScheme.Dot.value): # Dot
            # Same explicit r_ij > 0 guard and same reason as Brookshaw's above --
            # computeLaplacianDot2's own F_ab weighting is bit-for-bit Brookshaw's P
            # (dot(kernelGradient,n_ij)/D_ij), so it has the identical self-pair
            # reverse-mode-adjoint hazard once CRK makes kernelGradient nonzero at
            # x_ij == 0 (confirmed via gradcheck failing on the self-pair diagonal
            # Jacobian entries specifically, same as Brookshaw's own repro --
            # `warpier_tier2_correction_jvp_plan.md` follow-up, 2026-08-21). The true
            # contribution at r_ij == 0 is always exactly 0 (CRK or not, since n_ij == 0
            # there by construction), so this changes no forward value anywhere.
            if r_ij > scalar_t(0.0):
                laplacian_contribution = computeLaplacianDot2(q_ij, n_ij, kernelGradient, r_ij, h_ij, flatInputShape, dim)
        elif kernelProperties.laplacianMode == wp.static(LaplacianScheme.Default.value): # Default
            # Same guard, same reason -- computeDotLaplacian's own n_ij2 = n_ij/D2_ij
            # divides by a *second* regularized distance on top of n_ij's own division,
            # so it inherits the identical self-pair hazard, one quotient-rule level
            # deeper (confirmed via gradcheck, same as Dot above).
            if r_ij > scalar_t(0.0):
                laplacian_contribution = computeDotLaplacian(q_ij, n_ij, kernelGradient, r_ij, h_ij, flatInputShape, dim)

        if kernelProperties.positiveDivergenceMode:
            out += positiveDotProduct(x_ij, q_ij, laplacian_contribution, dim)
        else:
            out += laplacian_contribution

    return out


@wp.func
def computeSPHLaplacianTensor_Func_Adjacency(
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
    if correctionData.useGradHTerms:
        fi = queryValue[i] / iCorrectionData.omega

    out = zero_like_warp(outputValue)
    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRange(i, o, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue

        out += computeSPHLaplacianTensor_Func_i(
            i, dim,
            iPtcl,
            referenceState, domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,

            iCorrectionData, correctionData,
            # end of the canonical structured kernel ABI prefix; the rest of the arguments are specific to this operator.
            flatInputShape, flatOutputShape,

            fi, referenceValues,

            outputValue,
        )
    return out


@wp.kernel
def computeSPHLaplacianTensor_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    kernelProperties: kernelState,
    # Do not change the parameters above -- canonical structured kernel ABI, see warpier_core.md

    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    queryValues: Any, referenceValues: Any, # type: ignore

    # The last parameter is always the output array and should not be changed
    outputValues: wp.array(dtype = Any) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeSPHLaplacianTensor_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        numDims, flatInputShape, flatOutputShape,
        queryValues, referenceValues,

        zero_like_warp(outputValues[i]),
    )


def _laplacianOutputDtype(ctx, extras):
    return _get_warp_vector_dtype(int(extras["flatOutputShape"]), extras["queryValuesFlat"].dtype)


_LAPLACIAN_SPEC = OperatorSpec(
    kernel=computeSPHLaplacianTensor_Kernel,
    outputs=(OutputSpec(dtype=_laplacianOutputDtype),),
    extras=(
        ExtraSpec("numDims", ExtraKind.SCALAR),
        ExtraSpec("flatInputShape", ExtraKind.SCALAR),
        ExtraSpec("flatOutputShape", ExtraKind.SCALAR),
        ExtraSpec("queryValuesFlat", ExtraKind.TENSOR),
        ExtraSpec("referenceValuesFlat", ExtraKind.TENSOR),
    ),
)


def _computeSPHLaplacian_stateBackend(
    queryParticles: ParticleState,
    referenceParticles: ParticleState,
    domain: DomainDescription,
    mode: SupportScheme,
    kernel: KernelFunctions,
    gradientMode: GradientScheme,
    laplacianMode: LaplacianScheme,
    positiveDivergence: bool,
    operationMode: OperationDirection,
    adjacency, # AdjacencyList | CompactHashMap | None
    queryValues: torch.Tensor, referenceValues: torch.Tensor,
    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[GradHState] = None,
    renormalizationState: Optional[RenormalizationState] = None,
):
    with record_function("warpSPH[Laplacian]"):
        with record_function("warpSPH[Laplacian] - Preprocessing"):
            queryPositions = queryParticles.positions
            outputSize = queryPositions.shape[0]

            inputShape = queryValues.shape[1:]
            flatInputShape = 1
            for d in inputShape:
                flatInputShape *= d

            spatialDim = queryPositions.shape[1]
            if laplacianMode == LaplacianScheme.Dot and spatialDim > 1 and flatInputShape % spatialDim != 0:
                raise ValueError(
                    f"LaplacianScheme.Dot's computeLaplacianDot2 assumes the field's flattened size is a multiple "
                    f"of the spatial dimension ({spatialDim}) -- it indexes q_ij[block*dim + k] for k in range(dim), "
                    f"which reads out of bounds for a field whose flattened size ({flatInputShape}) isn't a multiple "
                    f"of dim. A plain scalar field (flatInputShape=1) in a {spatialDim}D domain hits this. Use "
                    f"LaplacianScheme.Naive, Brookshaw, or Default for scalar fields instead."
                )

            # Laplacian of a scalar field is a scalar field, of a vector field a vector
            # field -- same shape as the input, just flattened for the warp kernel.
            outputShape = inputShape
            flatOutputShape = 1
            for d in outputShape:
                flatOutputShape *= d
            numDims = len(inputShape)

            operationProperties = OperationProperties(
                kernel=kernel,
                operation=WarpOperation.Laplacian,
                gradientMode=gradientMode,
                laplacianMode=laplacianMode,
                positiveDivergence=positiveDivergence,
                supportMode=mode,
                operationMode=operationMode,
            )

        with record_function("warpSPH[Laplacian] - Kernel Execution"):
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
                _LAPLACIAN_SPEC, ctx,
                numDims=wp.int32(numDims),
                flatInputShape=wp.int32(flatInputShape),
                flatOutputShape=wp.int32(flatOutputShape),
                queryValuesFlat=queryValues.view(-1, flatInputShape),
                referenceValuesFlat=referenceValues.view(-1, flatInputShape),
            )

    return result.view(outputSize, *outputShape)
