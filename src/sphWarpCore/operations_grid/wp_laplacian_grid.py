import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any, Optional
import torch
from ..utils.wp_autograd import *


from ..radiusSearch.radius_util import AdjacencyList, AdjacencyListWarp, DomainDescription, PointCloud
from ..mathutil.wp_math import *
from ..kernels.wp_kernel import *
from ..utils.wp_util import checkDirectionality_i, checkDirectionality_j
from torch.profiler import profile, record_function, ProfilerActivity

from ..enumTypes import *
from ..radiusSearch.wp_compactHash import CompactHashMap, computeZOrderIndex64, hashGridVec3i
from .grid_util import checkOffset


@wp.func
def computeDotLaplacian(
    q_ij: vector(dtype = scalar_t, length=Any), n_ij: vector(dtype = scalar_t, length=Any), kernelGradient: vector(dtype = scalar_t, length=1), r_ij: scalar_t, h_ij: scalar_t, inputLength: wp.int32, dim: wp.int32 # type: ignore
):
    n_ij2 = n_ij / (r_ij + 1e-12 * h_ij)

    dotx = q_ij * n_ij2[0]

    # dot = wp.vec3f(dotx, doty, dotz)
    fkq = dotx * kernelGradient[0]

    result = type(q_ij)(-scalar_t(2.0) * fkq)
    return result

@wp.func
def computeDotLaplacian(
    q_ij: vector(dtype = scalar_t, length=Any), n_ij: vector(dtype = scalar_t, length=Any), kernelGradient: vector(dtype = scalar_t, length=2), r_ij: scalar_t, h_ij: scalar_t, inputLength: wp.int32, dim: wp.int32 # type: ignore
):
    n_ij2 = n_ij / (r_ij + 1e-12 * h_ij)

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
    n_ij2 = n_ij / (r_ij + 1e-12 * h_ij)

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
    # # DJ Price Smoothed particle hydrodynamics and magnetohydrodynamics page 778 (eq 96) in https://www.sciencedirect.com/science/article/pii/S0021999110006753

    # r_eps = r_ij + 1e-8 * h_i
    # n_ij = x_ij / r_eps.view(-1,1)
    # F_ab = torch.einsum('nd, nd -> n', n_ij, gradW_ij) / r_eps

    # leftTerm = (x_ij.shape[1] + 2) * torch.einsum('n..., nd -> n...d', torch.einsum('n...d, nd -> n...', fq, n_ij), n_ij)
    # rightTerm = - fq

    # fkq = -(leftTerm + rightTerm) * F_ab.view(-1,1)

    # dot = torch.einsum('n...d, nd -> n...d', fq, x_ij / (r_ij + 1e-8 * h_i).view(-1,1)**2)

    r_eps = r_ij + 1e-8 * h_ij
    F_ab = wp.dot(n_ij, kernelGradient) / r_eps # this is a scalar
    leading_dim = wp.int32(inputLength // dim) # this is the number of leading dimensions in q_ij before the last dimension of size dim
    
    output = type(q_ij)(scalar_t(0.0))
    for i in range(inputLength):
        # In this case q_ij is some quantity of internal shape [..., dim] so we compute the dot product across the trailing dimension of q_ij and n_ij, and then multiply by n_ij again to get the contribution for each component of the output. This is equivalent to the left term in Price's equation where we have a double dot product between n_ij and q_ij, but we compute it in a way that allows for q_ij to have an arbitrary number of leading dimensions as long as the last dimension has size dim.
        # leftTerm = scalar_t(dim + 2) * q_ij[i] * n_ij[i %dim] * n_ij[i%dim] * F_ab
        # fkq = -(leftTerm + rightTerm*scalar_t(0.0))
        d = i % dim                  # component within trailing dim
        b = i // dim                 # block index over leading dims
        base = b * dim               # start of this block in flattened storage
        
        proj = scalar_t(scalar_t(0.0))
        for k in range(dim):
            proj += q_ij[base + k] * n_ij[k]

        left = scalar_t(dim + 2) * proj * n_ij[d]
        output[i] += -left * F_ab


        # for k in range(dim):
            # output[i] += -scalar_t(dim + 2) * q_ij[k * leading_dim] * n_ij[k] * n_ij[i % dim] * F_ab
        # output[i] += leftTerm

    for i in range(inputLength):
        rightTerm = - q_ij[i]* F_ab
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
def computeSPHLaplacianTensor_grid_Func(
    # General Shape Parameters and indices
    i : wp.int32, dim: wp.int32, numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,

    # SPH properties for the query set (indexed by i)
    queryPositions: wp.array(dtype=vector(dtype = scalar_t, length=Any)), querySupports: wp.array(dtype = scalar_t), queryMasses: wp.array(dtype = scalar_t), queryDensities: wp.array(dtype = scalar_t), queryValues: wp.array(dtype = vector(dtype = scalar_t, length=Any)), # type: ignore

    # SPH properties for the reference set (indexed by j in the neighbor loop)
    referencePositions : wp.array(dtype=vector(length=Any, dtype = scalar_t)), referenceSupports : wp.array(dtype = scalar_t), referenceMasses: wp.array(dtype = scalar_t), referenceDensities: wp.array(dtype = scalar_t), referenceValues: wp.array(dtype = vector(dtype = scalar_t, length=Any)), # type: ignore
    
    # Domain and kernel parameters
    periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = scalar_t), domainMax : wp.array(dtype = scalar_t), # type: ignore
    mode_uint: wp.uint32, kernel_int: wp.int32, 
    
    # Operation specific parameters
    gradientMode_int: wp.int32, # type: ignore
    laplacianMode_int: wp.int32, # type: ignore
    positiveDivergence: wp.bool, # type: ignore
    
    # Neighbor list data, pre accessed to avoid gradient issues with dynamic for loops
    cellStartIndex: wp.int32, cellParticleCount: wp.int32, 
    sortIndex: wp.array(dtype = wp.int64), # type: ignore
    
    # Indicates if the input quantities have already been scattered to the neighbor level 
    preScatteredQuantities: wp. bool,
    
    # Operation Mode for masking certain kinds of interactions, e.g. for directional operations
    opInt: wp.int32, queryKinds : wp.array(dtype = wp.int32), referenceKinds : wp.array(dtype = wp.int32), # type: ignore

    # Optional Correction Terms:
    # Gradient renormalization matrices for each query point, used for correcting the kernel gradient based on the local particle distribution.
    useGradientRenormalization: wp.bool, queryRenormalizationMatrices: wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)), # type: ignore
    # Grad-h correction terms for each query and reference point, used for correcting the kernel gradient based on the local particle distribution and smoothing length variations.
    useGradHTerms: wp.bool, queryOmegas: wp.array(dtype = scalar_t), referenceOmegas: wp.array(dtype = scalar_t),  # type: ignore
    # Whether to use actual volume (mass/density) or apparent volume for the gradient computation, and the corresponding volumes if needed.
    useVolume: wp.bool, queryVolumes: wp.array(dtype = scalar_t), referenceVolumes: wp.array(dtype = scalar_t), # type: ignore
    # Whether to use CRK kernel correction for the computation, and the corresponding correction terms if needed.
    useCRK: wp.bool, queryA: wp.array(dtype = scalar_t), queryB: wp.array(dtype = vector(length=Any, dtype=scalar_t)), queryGradA: wp.array(dtype=vector(length=Any, dtype=scalar_t)), queryGradB: wp.array(dtype=matrix(shape=(Any, Any), dtype=scalar_t)), # type: ignore
    
    # Dummy value to allow allocation
    outputValue: Any # type: ignore
):
    if opInt != 0:
        if not checkDirectionality_i(queryKinds[i], opInt):
            return outputValue * scalar_t(0.0)
    # Unpack query point properties
    xi      = queryPositions[i]
    hi      = querySupports[i]
    # mi      = queryMasses[i] # Generally not needed
    rhoi    = queryDensities[i]
    fi      = queryValues[i]
    # Unpack optional correction terms
    if useGradHTerms:
        fi  = queryValues[i] / queryOmegas[i]
    Li      = queryRenormalizationMatrices[i] if useGradientRenormalization else type(queryRenormalizationMatrices[0])()*scalar_t(0.0)
    Ai      = queryA[i] if useCRK else type(queryA[0])(scalar_t(0.0))
    Bi      = queryB[i] if useCRK else type(queryB[0])(scalar_t(0.0))
    gradA_i = queryGradA[i] if useCRK else type(queryGradA[0])(scalar_t(0.0))
    gradB_i = queryGradB[i] if useCRK else type(queryGradB[0])()*scalar_t(0.0)
    
    # Initialize the output value
    out     = type(outputValue)(scalar_t(0.0))
        
    # Loop over neighbors to compute the gradient contribution from each neighbor    
    for neighborIndex in range(cellParticleCount):
        jj = cellStartIndex + neighborIndex
        j = wp.int32(sortIndex[jj])
        if opInt != 0:
            if not checkDirectionality_j(referenceKinds[j], opInt):
                continue
        ##########################################################
        #   The core particle-particle interaction starts here   #
        ##########################################################

        xj = referencePositions[j]
        mj = referenceMasses[j]
        rhoj = referenceDensities[j]
        apparentVolume = mj / rhoj if not useVolume else referenceVolumes[j]
        fj = type(fi)(scalar_t(0.0))
        if preScatteredQuantities:
            if useGradHTerms:
                fj = referenceValues[jj] / referenceOmegas[j]
            else:
                fj = referenceValues[jj]
        else:
            if useGradHTerms:
                fj = referenceValues[j] / referenceOmegas[j]
            else:
                fj = referenceValues[j]
        hj = referenceSupports[j]
        
        kernelGradient = computeKernelGradientCRK(
            xi, referencePositions[j], 
            hi, referenceSupports[j],
            kernel_int, mode_uint, periodicity, domainMin, domainMax,
            useCRK, Ai, Bi, gradA_i, gradB_i
        )
        
        if useGradientRenormalization:
            kernelGradient = matmul(Li, kernelGradient)
            
        q_ij = type(fi)(scalar_t(0.0))

        # See wp_laplacian.py::computeSPHLaplacianTensor_Func for the derivation:
        # the Brookshaw/Dot/Default laplacian schemes require q_ij to vanish for a
        # spatially constant field, or the sum picks up an O(1/h^2) residual that
        # diverges with resolution. Naive and Summation collapse to Difference's
        # (fj - fi) * V_j once corrected; Symmetric's density-asymmetric weighting
        # yields a distinct but equally consistent (fj - fi) variant.
        if gradientMode_int == wp.static(GradientScheme.Naive.value): # Naive
            q_ij = (fj - fi) * apparentVolume
        elif gradientMode_int == wp.static(GradientScheme.Symmetric.value): # Symmetric
            q_ij = (fj - fi) * mj * rhoi / iPow(rhoj, 2)
        elif gradientMode_int == wp.static(GradientScheme.Difference.value): # Difference
            q_ij = (fj - fi) * apparentVolume
        elif gradientMode_int == wp.static(GradientScheme.Summation.value): # Summation
            q_ij = (fj - fi) * apparentVolume

        h_ij = computePairwiseSupport(hi, hj, mode_uint)
        x_ij = computeDistanceVec(xi, xj, periodicity, domainMin, domainMax)
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))

        eps = 1e-8
        n_ij = x_ij / (r_ij + eps * h_ij)

        laplacian_contribution = type(outputValue)(scalar_t(0.0))

        if laplacianMode_int == wp.static(LaplacianScheme.Naive.value): # Naive
            laplacian_contribution = q_ij * sphKernelLaplacian(xi, referencePositions[j], hi, referenceSupports[j], kernel_int, mode_uint, periodicity, domainMin, domainMax)
        elif laplacianMode_int == wp.static(LaplacianScheme.Brookshaw.value): # Brookshaw
            laplacian_contribution = -scalar_t(2.0) * q_ij * wp.dot(kernelGradient, n_ij) / (r_ij + eps * h_ij)
        elif laplacianMode_int == wp.static(LaplacianScheme.Dot.value): # Dot
            laplacian_contribution = computeLaplacianDot2(q_ij, n_ij, kernelGradient, r_ij, h_ij, flatInputShape, dim)
        elif laplacianMode_int == wp.static(LaplacianScheme.Default.value): # Default
            laplacian_contribution = computeDotLaplacian(q_ij, n_ij, kernelGradient, r_ij, h_ij, flatInputShape, dim)

        if positiveDivergence:
            out += positiveDotProduct(x_ij, q_ij, laplacian_contribution, dim)
        else:
            out += laplacian_contribution

    return out

@wp.kernel
def computeSPHLaplacianTensor_grid_Kernel(
    queryPositions : wp.array(dtype = vector(length=Any, dtype=scalar_t)), referencePositions : wp.array(dtype=vector(length=Any, dtype=scalar_t)), # type: ignore
    querySupports : wp.array(dtype = scalar_t), referenceSupports : wp.array(dtype = scalar_t), # type: ignore
    queryMasses: wp.array(dtype = scalar_t), referenceMasses: wp.array(dtype = scalar_t),  # type: ignore
    queryDensities: wp.array(dtype = scalar_t), referenceDensities: wp.array(dtype = scalar_t), # type: ignore
    queryValues: wp.array(dtype =vector(dtype = scalar_t, length=Any)), referenceValues: wp.array(dtype = vector(dtype = scalar_t, length=Any)), # type: ignore
    
    domainMin : wp.array(dtype = scalar_t), domainMax : wp.array(dtype = scalar_t), periodicity : wp.array(dtype = wp.bool), # type: ignore
    
    mode_uint: wp.uint32, kernel_int : wp.int32, gradientMode_int: wp.int32, laplacianMode_int: wp.int32, positiveDivergence: wp.bool,
    sortIndex : wp.array(dtype = wp.int64), # type: ignore
    
    qMin: wp.array(dtype=scalar_t),  # shape [D] # type: ignore
    qMax: wp.array(dtype=scalar_t),  # shape [D] # type: ignore
    hCell: scalar_t,

    numCells: wp.array(dtype=wp.int32),  # shape [D] # type: ignore
    hashTable: wp.array(dtype=vector(length = 2, dtype = wp.int32)),  # shape [hashMapLength,2] # type: ignore
    cellTable: wp.array(dtype=vector(length = 3, dtype = wp.int64)),  # shape [C,3] with [cellIndex, cellStart, cellCount] # type: ignore
    D: int,
    numOffsets: int,
    cellOffsets: wp.array(dtype=vector(length=3, dtype=wp.int32)), # shape [numOffsets, 3] # type: ignore
    
    
    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    opInt: wp.int32, queryKinds : wp.array(dtype = wp.int32), referenceKinds : wp.array(dtype = wp.int32), # type: ignore

    useGradientRenormalization: wp.bool, queryRenormalizationMatrices: wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)),# type: ignore
    useGradHTerms: wp.bool, queryOmegas: wp.array(dtype = scalar_t), referenceOmegas: wp.array(dtype = scalar_t),  # type: ignore
    useVolume: wp.bool, queryVolumes: wp.array(dtype = scalar_t), referenceVolumes: wp.array(dtype = scalar_t), # type: ignore
    useCRK: wp.bool, crk_A: wp.array(dtype = scalar_t), crk_B: wp.array(dtype = vector(length=Any, dtype=scalar_t)), crk_gradA: wp.array(dtype = vector(length=Any, dtype=scalar_t)), crk_gradB: wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)), # type: ignore
        
    outputValues : wp.array(dtype = Any) # type: ignore
):                                                                                    
    i = wp.tid()
    if i >= queryPositions.shape[0]:
        return
    
    out_value = type(outputValues[0])() * scalar_t(0.0)

    for o in range(numOffsets):
        cellStartIndex, cellParticleCount = checkOffset(
            i, queryPositions, numCells, D, 
            o, cellOffsets, hashTable, cellTable,
            periodicity, qMin, qMax, hCell
        )
        if cellStartIndex < 0:
            continue

        out_value += computeSPHLaplacianTensor_grid_Func(
        i, get_dim(queryPositions), numDims, flatInputShape, flatOutputShape,

        queryPositions, querySupports, queryMasses, queryDensities, queryValues,
        referencePositions, referenceSupports, referenceMasses, referenceDensities, referenceValues,
        
        periodicity, domainMin, domainMax, 
        mode_uint, kernel_int, gradientMode_int, laplacianMode_int, positiveDivergence,

        cellStartIndex, cellParticleCount, sortIndex,

        False,
        
        opInt, queryKinds, referenceKinds,

        useGradientRenormalization, queryRenormalizationMatrices, 
        useGradHTerms, queryOmegas, referenceOmegas, 
        useVolume, queryVolumes, referenceVolumes,
        useCRK, crk_A, crk_B, crk_gradA, crk_gradB,

            type(outputValues[0])()
        )
    outputValues[i] = out_value
    
from ..enumTypes import *

def computeSPHLaplacian_grid_warpBackend(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,
    queryDensities, referenceDensities,
    queryValues, referenceValues,
    queryKinds, referenceKinds,
    domain: DomainDescription,
    mode: SupportScheme,
    kernel: KernelFunctions,    
    gradientMode: GradientScheme,
    laplacianMode: LaplacianScheme,
    positiveDivergence: bool,
    operationMode: OperationDirection,
    datastructure: CompactHashMap,
    scatteredQuantities: Optional[torch.Tensor] = None,
    
    useGradientRenormalization: bool = False, renormalizationMatrices: Optional[torch.Tensor] = None,
    useGradHTerms: bool = False, queryOmegas: Optional[torch.Tensor] = None, referenceOmegas: Optional[torch.Tensor] = None,
    useVolume: bool = False, queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    useCRK: bool = False, crk_A: Optional[torch.Tensor] = None, crk_B: Optional[torch.Tensor] = None, crk_gradA: Optional[torch.Tensor] = None, crk_gradB: Optional[torch.Tensor] = None
):
    with record_function("warpSPH[Laplacian]"):
        with record_function("warpSPH[Laplacian] - Preprocessing"):
            domainMin = domain.min
            domainMax = domain.max
            periodicity = domain.periodic

            mode_uint = supportSchemeToUint(mode)
            kernel_int = kernel.value
            gradientMode_int = gradientMode.value
            laplacianMode_int = laplacianMode.value
            positiveDivergence = wp.bool(positiveDivergence)
            opInt = wp.int32(operationMode.value)


            preScatteredQuantities = False
            if queryValues is None and referenceValues is None:
                if scatteredQuantities is None:
                    raise ValueError("If queryValues and referenceValues are not provided, then pre-scattered quantities must be provided for the laplacian computation.")
                preScatteredQuantities = True
                qV = scatteredQuantities
                rV = scatteredQuantities
            else:
                qV = queryValues
                rV = referenceValues

            # Warp kernels only support rank-1 (vector) and rank-2 (matrix) field types.
            outputSize = (queryPositions.shape[0])

            inputShape = qV.shape[1:]
            flatInputShape = 1
            for dim in inputShape:
                flatInputShape *= dim
                
            # For the output shape we keep the same shape as the input as the laplacian of a scalar field is still a scalar field, and the laplacian of a vector field is still a vector field. We just need to make sure to flatten the inner dimensions for the warp kernel.
            outputShape = inputShape

            flatOutputShape = 1
            for dim in outputShape:
                flatOutputShape *= dim
            numDims = len(inputShape)
            
        with record_function("warpSPH[Laplacian] - Kernel Execution"):
            D = queryPositions.shape[1]
            warp_result = warpWrapper(
                launch_kernel, computeSPHLaplacianTensor_grid_Kernel, outputSize, vector(length=flatOutputShape, dtype = scalar_t),
                queryPositions, referencePositions,
                querySupports, referenceSupports,
                queryMasses, referenceMasses,
                queryDensities, referenceDensities,
                qV.view(-1, flatInputShape), rV.view(-1, flatInputShape),
                domainMin, domainMax, periodicity,
                mode_uint, kernel_int, gradientMode_int, laplacianMode_int, positiveDivergence,
                datastructure.sortIndex,
                datastructure.qMin, datastructure.qMax, datastructure.hCell,
                datastructure.numCells, datastructure.hashTable, datastructure.sortedCellTable, D,
                datastructure.numOffsets, datastructure.cellOffsets,
                wp.int32(numDims), wp.int32(flatInputShape), wp.int32(flatOutputShape),
                opInt, queryKinds, referenceKinds,
                
                wp.bool(useGradientRenormalization), renormalizationMatrices,
                wp.bool(useGradHTerms), queryOmegas, referenceOmegas,                
                wp.bool(useVolume), queryVolumes, referenceVolumes,
                wp.bool(useCRK), crk_A, crk_B, crk_gradA, crk_gradB
            )
    # print(f"computeSPHLaplacian_warpBackend: inputShape={inputShape}, flatInputShape={flatInputShape}, outputShape={outputShape}, flatOutputShape={flatOutputShape}, numDims={numDims}")


    return warp_result.view(queryPositions.shape[0], *outputShape) # reshape back to original shape with new gradient dimension