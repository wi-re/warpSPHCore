import warp as wp
from warp.types import vector, matrix
from typing import Any, Optional, Union
import torch
from torch.profiler import record_function

from ..utils.wp_autograd import *

from ..radiusSearch.radius_util import AdjacencyList, DomainDescription, PointCloud
from ..radiusSearch.wp_compactHash import CompactHashMap
from ..radiusSearch.grid_util import checkOffset
from ..mathutil.wp_math import *
from ..kernels.wp_kernel import *
from ..utils.wp_util import (
    checkDirectionality_i, checkDirectionality_j,
    zero_like_warp, _get_warp_vector_dtype,
)

from ..enumTypes import *
from ..warp_state import (
    domainData, adjacencyData, gridData,
    getParticle, getL_i, getGradH_i, getVolume_i, getCRK_i,
)
from ..warp_state_util import warpWrapper2
from ..state import ParticleState, OperationProperties, CRKState, GradHState, RenormalizationState

# Unified Laplacian kernel: same design as the unified Gradient/Divergence/Curl kernels
# (see warpier_core.md's "Working Prototype -> Production" section). Laplacian shares
# Gradient's correction-path machinery (CRK, grad-h, volume, renormalization) and reuses
# GradientScheme to pick how the neighbor difference q_ij is formed (all four variants
# collapse to a (fj - fi)-based difference here -- see the comment below), then combines
# q_ij with the kernel gradient via one of `computeDotLaplacian`/`computeLaplacianDot2`/
# a direct second-derivative kernel evaluation, selected by LaplacianScheme. Unlike
# Gradient/Divergence/Curl, `positiveDivergence` is a real (non-decorative) part of the
# canonical ABI here.


@wp.func
def computeDotLaplacian(
    q_ij: vector(dtype = scalar_t, length=Any), n_ij: vector(dtype = scalar_t, length=Any), kernelGradient: vector(dtype = scalar_t, length=1), r_ij: scalar_t, h_ij: scalar_t, inputLength: wp.int32, dim: wp.int32 # type: ignore
):
    n_ij2 = n_ij / (r_ij + scalar_t(1e-12) * h_ij)
    dotx = q_ij * n_ij2[0]
    fkq = dotx * kernelGradient[0]
    result = type(q_ij)(-scalar_t(2.0) * fkq)
    return result

@wp.func
def computeDotLaplacian(
    q_ij: vector(dtype = scalar_t, length=Any), n_ij: vector(dtype = scalar_t, length=Any), kernelGradient: vector(dtype = scalar_t, length=2), r_ij: scalar_t, h_ij: scalar_t, inputLength: wp.int32, dim: wp.int32 # type: ignore
):
    n_ij2 = n_ij / (r_ij + scalar_t(1e-12) * h_ij)
    dotx = q_ij * n_ij2[0]
    doty = q_ij * n_ij2[1]
    output = type(q_ij)(scalar_t(0.0))
    for i in range(inputLength):
        output[i] += dotx[i] * kernelGradient[0] + doty[i] * kernelGradient[1]
    return -scalar_t(2.0) * output

@wp.func
def computeDotLaplacian(
    q_ij: vector(dtype = scalar_t, length=Any), n_ij: vector(dtype = scalar_t, length=Any), kernelGradient: vector(dtype = scalar_t, length=3), r_ij: scalar_t, h_ij: scalar_t, inputLength: wp.int32, dim: wp.int32 # type: ignore
):
    n_ij2 = n_ij / (r_ij + scalar_t(1e-12) * h_ij)
    dotx = q_ij * n_ij2[0]
    doty = q_ij * n_ij2[1]
    dotz = q_ij * n_ij2[2]
    output = type(q_ij)(scalar_t(0.0))
    for i in range(inputLength):
        output[i] += dotx[i] * kernelGradient[0] + doty[i] * kernelGradient[1] + dotz[i] * kernelGradient[2]
    return -scalar_t(2.0) * output

@wp.func
def computeLaplacianDot2(
    q_ij: vector(dtype = scalar_t, length=Any), n_ij: vector(dtype = scalar_t, length=Any), kernelGradient: vector(dtype = scalar_t, length=Any), r_ij: scalar_t, h_ij: scalar_t, inputLength: wp.int32, dim: wp.int32 # type: ignore
):
    # DJ Price Smoothed particle hydrodynamics and magnetohydrodynamics page 778 (eq 96)
    # in https://www.sciencedirect.com/science/article/pii/S0021999110006753
    r_eps = r_ij + scalar_t(1e-8) * h_ij
    F_ab = wp.dot(n_ij, kernelGradient) / r_eps # this is a scalar

    output = type(q_ij)(scalar_t(0.0))
    for i in range(inputLength):
        # q_ij has internal shape [..., dim]; compute the dot product across the trailing
        # dim of q_ij and n_ij, then multiply by n_ij again for each output component.
        d = i % dim                  # component within trailing dim
        b = i // dim                 # block index over leading dims
        base = b * dim               # start of this block in flattened storage

        proj = scalar_t(scalar_t(0.0))
        for k in range(dim):
            proj += q_ij[base + k] * n_ij[k]

        left = scalar_t(dim + 2) * proj * n_ij[d]
        output[i] += -left * F_ab

    for i in range(inputLength):
        rightTerm = - q_ij[i] * F_ab
        output[i] += -rightTerm

    return output


@wp.func
def positiveDotProduct(
    x_ij: vector(dtype = scalar_t, length=Any), # type: ignore
    fq_ij: vector(dtype = scalar_t, length=Any), # type: ignore
    f_ij: vector(dtype = scalar_t, length=Any), # type: ignore
    dim: wp.int32
):
    dot = scalar_t(scalar_t(0.0))
    for d in range(dim):
        dot += x_ij[d] * fq_ij[d]

    result = type(f_ij)(scalar_t(0.0))
    if dot >= scalar_t(0.0):
        for d in range(dim):
            result[d] = f_ij[d]
    return result


@wp.func
def computeSPHLaplacianTensor_Func_i(
    i: wp.int32, dim: wp.int32, numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,

    xi: vector(dtype = scalar_t, length=Any), hi: scalar_t, mi: scalar_t, rhoi: scalar_t, # type: ignore

    referenceState: Any, # particleDataSoA_1/2/3

    domainState: domainData,
    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32, laplacianMode_int: wp.int32, positiveDivergence_int: wp.int32, divergenceMode_int: wp.int32,

    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype = wp.int64), # type: ignore

    opInt: wp.int32, ki: wp.int32, referenceKinds: wp.array(dtype = wp.int32), # type: ignore

    useGradientRenormalization: wp.bool, Li: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    useGradHTerms: wp.bool, omega_i: scalar_t, referenceOmegas: wp.array(dtype = scalar_t), # type: ignore
    useVolume: bool, Vi: scalar_t, referenceVolumes: wp.array(dtype = scalar_t), # type: ignore
    useCRK: bool, Ai: scalar_t, Bi: vector(length=Any, dtype=scalar_t), gradAi: vector(length=Any, dtype=scalar_t), gradBi: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    correctionData: Any,

    fi: Any, referenceValues: wp.array(dtype = Any), # type: ignore

    outputValue: Any, # type: ignore
):
    out = zero_like_warp(outputValue)
    for neighborIndex in range(numIndices):
        jj = beginIndex + neighborIndex
        j = wp.int32(offsetArray[jj])
        if opInt != 0:
            if not checkDirectionality_j(referenceKinds[j], opInt):
                continue
        ##########################################################
        #   The core particle-particle interaction starts here   #
        ##########################################################

        xj, hj, mj, rhoj, kj = getParticle(referenceState, j)

        apparentVolume = mj / rhoj if not useVolume else referenceVolumes[j]

        # Explicit if/else, not a ternary: see docs/lessons_learned.md and the note in
        # computeSPHGradientTensor_Func_i (wp_gradient.py) -- both branches here would
        # read referenceValues[j], which silently zeroes that array's adjoint if written
        # as a ternary.
        fj = referenceValues[j]
        if useGradHTerms:
            fj = referenceValues[j] / referenceOmegas[j]

        kernelGradient = computeKernelGradientCRK(
            xi, xj,
            hi, hj,
            kernel_int, mode_uint, domainState.periodicity, domainState.domainMin, domainState.domainMax,
            useCRK, Ai, Bi, gradAi, gradBi
        )

        if useGradientRenormalization:
            kernelGradient = matmul(Li, kernelGradient)

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
        if gradientMode_int == wp.static(GradientScheme.Naive.value): # Naive
            q_ij = (fj - fi) * apparentVolume
        elif gradientMode_int == wp.static(GradientScheme.Symmetric.value): # Symmetric
            q_ij = (fj - fi) * mj * rhoi / iPow(rhoj, 2)
        elif gradientMode_int == wp.static(GradientScheme.Difference.value): # Difference
            q_ij = (fj - fi) * apparentVolume
        elif gradientMode_int == wp.static(GradientScheme.Summation.value): # Summation
            q_ij = (fj - fi) * apparentVolume

        h_ij = computePairwiseSupport(hi, hj, mode_uint)
        x_ij = computeDistanceVec(xi, xj, domainState.periodicity, domainState.domainMin, domainState.domainMax)
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))

        eps = scalar_t(1e-8)
        n_ij = x_ij / (r_ij + eps * h_ij)

        laplacian_contribution = zero_like_warp(outputValue)

        if laplacianMode_int == wp.static(LaplacianScheme.Naive.value): # Naive
            laplacian_contribution = q_ij * sphKernelLaplacian(xi, xj, hi, hj, kernel_int, mode_uint, domainState.periodicity, domainState.domainMin, domainState.domainMax)
        elif laplacianMode_int == wp.static(LaplacianScheme.Brookshaw.value): # Brookshaw
            laplacian_contribution = -scalar_t(2.0) * q_ij * wp.dot(kernelGradient, n_ij) / (r_ij + eps * h_ij)
        elif laplacianMode_int == wp.static(LaplacianScheme.Dot.value): # Dot
            laplacian_contribution = computeLaplacianDot2(q_ij, n_ij, kernelGradient, r_ij, h_ij, flatInputShape, dim)
        elif laplacianMode_int == wp.static(LaplacianScheme.Default.value): # Default
            laplacian_contribution = computeDotLaplacian(q_ij, n_ij, kernelGradient, r_ij, h_ij, flatInputShape, dim)

        if positiveDivergence_int != 0:
            out += positiveDotProduct(x_ij, q_ij, laplacian_contribution, dim)
        else:
            out += laplacian_contribution

    return out


@wp.func
def computeSPHLaplacianTensor_Func_Adjacency(
    i: wp.int32, dim: wp.int32,

    queryState: Any, referenceState: Any, correctionData: Any,

    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,

    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32, laplacianMode_int: wp.int32, positiveDivergence_int: wp.int32, divergenceMode_int: wp.int32, opInt: wp.int32,

    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    queryValue: Any, referenceValues: Any, # type: ignore

    outputValue: Any, # type: ignore
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if opInt != 0:
        if not checkDirectionality_i(ki, opInt):
            return zero_like_warp(outputValue)

    useGradientRenormalization, Li = getL_i(correctionData, i)
    useGradHTerms, omega_i = getGradH_i(correctionData, i)
    useVolume, Vi = getVolume_i(correctionData, i)
    useCRK, Ai, Bi, gradA_i, gradB_i = getCRK_i(correctionData, i)

    fi = queryValue[i]
    if useGradHTerms:
        fi = queryValue[i] / omega_i

    out = zero_like_warp(outputValue)
    for o in range(numOffsets):
        beginIndex = wp.int32(0)
        numIndices = wp.int32(0)
        if useAdjacency:
            beginIndex = adjacencyState.neighborOffsets[i]
            numIndices = adjacencyState.numNeighbors[i]
        else:
            beginIndex, numIndices = checkOffset(
                i, queryState.positions, gridState.numCells, gridState.D,
                o, gridState.cellOffsets, gridState.hashTable, gridState.cellTable,
                domainState.periodicity, gridState.qMin, gridState.qMax, gridState.hCell
            )
            if beginIndex < 0:
                continue

        out += computeSPHLaplacianTensor_Func_i(
            i, dim, numDims, flatInputShape, flatOutputShape,
            xi, hi, mi, rhoi,
            referenceState, domainState,
            mode_uint, kernel_int, gradientMode_int, laplacianMode_int, positiveDivergence_int, divergenceMode_int,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            opInt, ki, referenceState.kinds,

            useGradientRenormalization, Li,
            useGradHTerms, omega_i, correctionData.referenceOmegas,
            useVolume, Vi, correctionData.referenceVolumes,
            useCRK, Ai, Bi, gradA_i, gradB_i,
            correctionData,

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

    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32, laplacianMode_int: wp.int32, positiveDivergence_int: wp.int32, divergenceMode_int: wp.int32, opInt: wp.int32,
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
        mode_uint, kernel_int, gradientMode_int, laplacianMode_int, positiveDivergence_int, divergenceMode_int, opInt,
        numDims, flatInputShape, flatOutputShape,
        queryValues, referenceValues,

        zero_like_warp(outputValues[i]),
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

            outputDtype = _get_warp_vector_dtype(flatOutputShape, queryValues.dtype)

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
            result = warpWrapper2(
                launcher=launch_kernel,
                kernel=computeSPHLaplacianTensor_Kernel,
                outputSizes=outputSize,
                outputDtypes=outputDtype,
                defaultStateArguments=(
                    queryParticles, operationProperties, domain,
                    queryVolumes, referenceVolumes,
                    adjacency,
                    referenceParticles,
                    crkState,
                    gradHState,
                    renormalizationState,
                ),
                additionalArguments=(
                    wp.int32(numDims), wp.int32(flatInputShape), wp.int32(flatOutputShape),
                    queryValues.view(-1, flatInputShape), referenceValues.view(-1, flatInputShape),
                ),
            )

    return result.view(outputSize, *outputShape)
