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
# In dot mode we compute torch.einsum('nd..., nd -> n...', q, k) 
# Otherwise we compute torch.einsum('n...d, nd -> n...', q, k)
# the inputs are the flattened versions of the original tensors, i.e.,
# if q initially was of shape [n, d, d, d] it is now [n, d^3]
# The output is always of shape [n, d^(N-1)] where N is the rank of the input tensor q. So if q was originally [n, d, d, d] the output will be [n, d^2]
# The kernelGradient is always of shape [n, d]
# this also allows us to overload the function based on d! We can then use the numDims parameter to do the correct indexing inside the kernel without having to write separate kernels for different dimensions.
@wp.func
def divergenceProduct(
    fij: vector(dtype = wp.float32, length=Any),  # type: ignore
    kernelGradient: vector(dtype = wp.float32, length=3), # type: ignore
    output: vector(dtype = wp.float32, length=Any), # type: ignore
    outputElements: wp.int32, dotMode: wp.bool
):
    res = type(output)(0.0)
    # res is now of shape [d^(N-1)] where N is the rank of the original input tensor.
    # we need to do a product between fij which is of shape [d^N] and kernelGradient which is of shape [d] to get an output of shape [d^(N-1)]
    
    dim = wp.int32(3) # hardcoded as this is the overload for 3D.
    
    if dotMode:
        for i in range(outputElements):
            for d in range(dim):
                res[i] += fij[i * dim + d] * kernelGradient[d]
    else:
        for i in range(outputElements):
            for d in range(dim):
                res[i] += fij[i + d * outputElements] * kernelGradient[d]
                
    return res

@wp.func
def divergenceProduct(
    fij: vector(dtype = wp.float32, length=Any),  # type: ignore
    kernelGradient: vector(dtype = wp.float32, length=2), # type: ignore
    output: vector(dtype = wp.float32, length=Any), # type: ignore
    outputElements: wp.int32, dotMode: wp.bool
):
    res = type(output)(0.0)
    # res is now of shape [d^(N-1)] where N is the rank of the original input tensor.
    # we need to do a product between fij which is of shape [d^N] and kernelGradient which is of shape [d] to get an output of shape [d^(N-1)]
    
    dim = wp.int32(2) # hardcoded as this is the overload for 2D.
    # wp.printf("divergenceProduct: dim=%d, outputElements=%d, inputElements=%d\n", dim, outputElements, dim * outputElements)
    
    if dotMode:
        for i in range(outputElements):
            for d in range(dim):
                res[i] += fij[i * dim + d] * kernelGradient[d]
    else:
        for i in range(outputElements):
            for d in range(dim):
                res[i] += fij[i + d * outputElements] * kernelGradient[d]
                
    return res

@wp.func
def divergenceProduct(
    fij: vector(dtype = wp.float32, length=Any),  # type: ignore
    kernelGradient: vector(dtype = wp.float32, length=1), # type: ignore
    output: vector(dtype = wp.float32, length=Any), # type: ignore
    outputElements: wp.int32, dotMode: wp.bool
):
    res = type(output)(0.0)
    # in 1D the divergence product is just a simple multiplication
    res[0] = fij[0] * kernelGradient[0]
    return res 

@wp.func
def computeSPHDivergenceTensor_grid_Func(
    # General Shape Parameters and indices
    i : wp.int32, dim: wp.int32, numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,

    # SPH properties for the query set (indexed by i)
    queryPositions: wp.array(dtype=vector(dtype = wp.float32, length=Any)), querySupports: wp.array(dtype = wp.float32), queryMasses: wp.array(dtype = wp.float32), queryDensities: wp.array(dtype = wp.float32), queryValues: wp.array(dtype = vector(dtype = wp.float32, length=Any)), # type: ignore

    # SPH properties for the reference set (indexed by j in the neighbor loop)
    referencePositions : wp.array(dtype=vector(length=Any, dtype = wp.float32)), referenceSupports : wp.array(dtype = wp.float32), referenceMasses: wp.array(dtype = wp.float32), referenceDensities: wp.array(dtype = wp.float32), referenceValues: wp.array(dtype = vector(dtype = wp.float32, length=Any)), # type: ignore
    
    # Domain and kernel parameters
    periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), # type: ignore
    mode_uint: wp.uint32, kernel_int: wp.int32, 
    
    # Operation specific parameters
    gradientMode_int: wp.int32, # type: ignore
    
    # Neighbor list data, pre accessed to avoid gradient issues with dynamic for loops
    cellStartIndex: wp.int32, cellParticleCount: wp.int32, 
    sortIndex: wp.array(dtype = wp.int64), # type: ignore
    
    # Indicates if the input quantities have already been scattered to the neighbor level 
    preScatteredQuantities: wp. bool,
    
    # Operation Mode for masking certain kinds of interactions, e.g. for directional operations
    opInt: wp.int32, queryKinds : wp.array(dtype = wp.int32), referenceKinds : wp.array(dtype = wp.int32), # type: ignore

    # Optional Correction Terms:
    # Gradient renormalization matrices for each query point, used for correcting the kernel gradient based on the local particle distribution.
    useGradientRenormalization: wp.bool, queryRenormalizationMatrices: wp.array(dtype = matrix(shape=(Any, Any), dtype=wp.float32)), # type: ignore
    # Grad-h correction terms for each query and reference point, used for correcting the kernel gradient based on the local particle distribution and smoothing length variations.
    useGradHTerms: wp.bool, queryOmegas: wp.array(dtype = wp.float32), referenceOmegas: wp.array(dtype = wp.float32),  # type: ignore
    # Whether to use actual volume (mass/density) or apparent volume for the gradient computation, and the corresponding volumes if needed.
    useVolume: wp.bool, queryVolumes: wp.array(dtype = wp.float32), referenceVolumes: wp.array(dtype = wp.float32), # type: ignore
    # Whether to use CRK kernel correction for the computation, and the corresponding correction terms if needed.
    useCRK: wp.bool, queryA: wp.array(dtype = wp.float32), queryB: wp.array(dtype = vector(length=Any, dtype=wp.float32)), queryGradA: wp.array(dtype=vector(length=Any, dtype=wp.float32)), queryGradB: wp.array(dtype=matrix(shape=(Any, Any), dtype=wp.float32)), # type: ignore

    # kernel specific parameters 
    consistentDivergence: wp.bool, dotMode: wp.bool, # type: ignore
    
    # Dummy value to allow allocation
    outputValue: vector(length=Any, dtype=wp.float32) # type: ignore
):
    if opInt != 0:
        if not checkDirectionality_i(queryKinds[i], opInt):
            return outputValue * 0.0
    # Unpack query point properties
    xi      = queryPositions[i]
    hi      = querySupports[i]
    # mi      = queryMasses[i] # Generally not needed
    rhoi    = queryDensities[i]
    fi      = queryValues[i]
    # Unpack optional correction terms
    if useGradHTerms:
        fi  = queryValues[i] / queryOmegas[i]
    Li      = queryRenormalizationMatrices[i] if useGradientRenormalization else type(queryRenormalizationMatrices[0])()*0.0
    Ai      = queryA[i] if useCRK else type(queryA[0])(0.0)
    Bi      = queryB[i] if useCRK else type(queryB[0])(0.0)
    gradA_i = queryGradA[i] if useCRK else type(queryGradA[0])(0.0)
    gradB_i = queryGradB[i] if useCRK else type(queryGradB[0])()*0.0
    
    # Initialize the output value
    out     = type(outputValue)(0.0)

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
                
        mj = referenceMasses[j]
        rhoj = referenceDensities[j]
        apparentVolume = mj / rhoj if not useVolume else referenceVolumes[j]
        if consistentDivergence:
            apparentVolume = mj / rhoi if not useVolume else referenceVolumes[j] * rhoj / rhoi
        fj = type(fi)(0.0)
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
        
        kernelGradient = computeKernelGradientCRK(
            xi, referencePositions[j], 
            hi, referenceSupports[j],
            kernel_int, mode_uint, periodicity, domainMin, domainMax,
            useCRK, Ai, Bi, gradA_i, gradB_i
        )
        
        if useGradientRenormalization:
            kernelGradient = matmul(Li, kernelGradient)
            
        if gradientMode_int == wp.static(GradientScheme.Naive.value): # Naive
            out += divergenceProduct(fj * apparentVolume, kernelGradient, outputValue, flatOutputShape, dotMode)
        elif gradientMode_int == wp.static(GradientScheme.Symmetric.value): # Symmetric
            out += divergenceProduct(mj * rhoi * (fi / iPow(rhoi,2) + fj / iPow(rhoj,2)) * apparentVolume, kernelGradient, outputValue, flatOutputShape, dotMode)
        elif gradientMode_int == wp.static(GradientScheme.Difference.value): # Difference
            out += divergenceProduct((fj - fi) * apparentVolume, kernelGradient, outputValue, flatOutputShape, dotMode)
        elif gradientMode_int == wp.static(GradientScheme.Summation.value): # Summation
            out += divergenceProduct((fj + fi) * apparentVolume, kernelGradient, outputValue, flatOutputShape, dotMode)
            
    return out

@wp.kernel
def computeSPHDivergenceTensor_grid_Kernel(
    queryPositions : wp.array(dtype = vector(length=Any, dtype=wp.float32)), referencePositions : wp.array(dtype=vector(length=Any, dtype=wp.float32)), # type: ignore
    querySupports : wp.array(dtype = wp.float32), referenceSupports : wp.array(dtype = wp.float32), # type: ignore
    queryMasses: wp.array(dtype = wp.float32), referenceMasses: wp.array(dtype = wp.float32),  # type: ignore
    queryDensities: wp.array(dtype = wp.float32), referenceDensities: wp.array(dtype = wp.float32), # type: ignore
    queryValues: wp.array(dtype = vector(dtype = wp.float32, length=Any)), referenceValues: wp.array(dtype = vector(dtype = wp.float32, length=Any)), # type: ignore
    
    domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), periodicity : wp.array(dtype = wp.bool), # type: ignore
    
    mode_uint: wp.uint32, kernel_int : wp.int32, gradientMode_int: wp.int32, consistentDivergence: wp.bool,
    
    sortIndex : wp.array(dtype = wp.int64), # type: ignore
    
    qMin: wp.array(dtype=wp.float32),  # shape [D] # type: ignore
    qMax: wp.array(dtype=wp.float32),  # shape [D] # type: ignore
    hCell: float,

    numCells: wp.array(dtype=wp.int32),  # shape [D] # type: ignore
    hashTable: wp.array(dtype=vector(length = 2, dtype = wp.int32)),  # shape [hashMapLength,2] # type: ignore
    cellTable: wp.array(dtype=vector(length = 3, dtype = wp.int64)),  # shape [C,3] with [cellIndex, cellStart, cellCount] # type: ignore
    D: int,
    numOffsets: int,
    cellOffsets: wp.array(dtype=vector(length=3, dtype=wp.int32)), # shape [numOffsets, 3] # type: ignore

    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32, dotMode: wp.bool,
    opInt: wp.int32, queryKinds : wp.array(dtype = wp.int32), referenceKinds : wp.array(dtype = wp.int32), # type: ignore

    useGradientRenormalization: wp.bool, queryRenormalizationMatrices: wp.array(dtype = matrix(shape=(Any, Any), dtype=wp.float32)),# type: ignore
    useGradHTerms: wp.bool, queryOmegas: wp.array(dtype = wp.float32), referenceOmegas: wp.array(dtype = wp.float32),  # type: ignore
    useVolume: bool, queryVolumes: wp.array(dtype = wp.float32), referenceVolumes: wp.array(dtype = wp.float32), # type: ignore
    useCRK: bool, crk_A: wp.array(dtype = wp.float32), crk_B: wp.array(dtype = vector(length=Any, dtype=wp.float32)), crk_gradA: wp.array(dtype = vector(length=Any, dtype=wp.float32)), crk_gradB: wp.array(dtype = matrix(shape=(Any, Any), dtype=wp.float32)), # type: ignore
    
    outputValues : wp.array(dtype = vector(length = Any, dtype = wp.float32)) # type: ignore
):                                                                                    
    i = wp.tid()
    if i >= queryPositions.shape[0]:
        return
    
    out_value = type(outputValues[0])() * 0.0

    for o in range(numOffsets):
        cellStartIndex, cellParticleCount = checkOffset(
            i, queryPositions, numCells, D, 
            o, cellOffsets, hashTable, cellTable,
            periodicity, qMin, qMax, hCell
        )
        if cellStartIndex < 0:
            continue

        out_value += computeSPHDivergenceTensor_grid_Func(
        i, get_dim(queryPositions), numDims, flatInputShape, flatOutputShape,

        queryPositions, querySupports, queryMasses, queryDensities, queryValues,
        referencePositions, referenceSupports, referenceMasses, referenceDensities, referenceValues,
        
        periodicity, domainMin, domainMax, 
        mode_uint, kernel_int, gradientMode_int,

        cellStartIndex, cellParticleCount, sortIndex,

        False,
        
        opInt, queryKinds, referenceKinds,

        useGradientRenormalization, queryRenormalizationMatrices, 
        useGradHTerms, queryOmegas, referenceOmegas, 
        useVolume, queryVolumes, referenceVolumes,
        useCRK, crk_A, crk_B, crk_gradA, crk_gradB,

        consistentDivergence, dotMode,

            type(outputValues[0])()
        )
    outputValues[i] = out_value
    
from ..enumTypes import *

def computeSPHDivergence_grid_warpBackend(
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
    operationMode: OperationDirection,
    datastructure: CompactHashMap,
    consistentDivergence: bool = False,
    dotMode: bool = False, # if true compute the divergence based on torch.einsum('nd..., nd -> n...', q, k) instead of the normal div torch.einsum('n...d, nd -> n...', q, k)
    scatteredQuantities: Optional[torch.Tensor] = None,
    
    useGradientRenormalization: bool = False, renormalizationMatrices: Optional[torch.Tensor] = None,
    useGradHTerms: bool = False, queryOmegas: Optional[torch.Tensor] = None, referenceOmegas: Optional[torch.Tensor] = None,
    useVolume: bool = False, queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    useCRK: bool = False, crk_A: Optional[torch.Tensor] = None, crk_B: Optional[torch.Tensor] = None, crk_gradA: Optional[torch.Tensor] = None, crk_gradB: Optional[torch.Tensor] = None
):
    with record_function("warpSPH[Divergence]"):
        with record_function("warpSPH[Divergence] - Preprocessing"):
            domainMin = domain.min
            domainMax = domain.max
            periodicity = domain.periodic

            mode_uint = supportSchemeToUint(mode)
            kernel_int = kernel.value
            gradientMode_int = gradientMode.value
            opInt = wp.int32(operationMode.value)

            preScatteredQuantities = False
            if queryValues is None and referenceValues is None:
                if scatteredQuantities is None:
                    raise ValueError("If queryValues and referenceValues are not provided, then pre-scattered quantities must be provided for the divergence computation.")
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
                
            outputShape = inputShape[1:] if dotMode else inputShape[:-1]
            flatOutputShape = 1
            for dim in outputShape:
                flatOutputShape *= dim
            numDims = len(inputShape)
            
    # print(f"computeSPHDivergenceTensor_warpBackend: inputShape={inputShape}, flatInputShape={flatInputShape}, outputShape={outputShape}, flatOutputShape={flatOutputShape}, numDims={numDims}")
        with record_function("warpSPH[Divergence] - Kernel Execution"):
            D = queryPositions.shape[1]
            warp_result = warpWrapper(
                launch_kernel, computeSPHDivergenceTensor_grid_Kernel, outputSize, vector(length=flatOutputShape, dtype = wp.float32),
                queryPositions, referencePositions,
                querySupports, referenceSupports,
                queryMasses, referenceMasses,
                queryDensities, referenceDensities,
                qV.view(-1, flatInputShape), rV.view(-1, flatInputShape),
                domainMin, domainMax, periodicity,
                mode_uint, kernel_int, gradientMode_int, wp.bool(consistentDivergence),
                datastructure.sortIndex,
                datastructure.qMin, datastructure.qMax, datastructure.hCell,
                datastructure.numCells, datastructure.hashTable, datastructure.sortedCellTable, D,
                datastructure.numOffsets, datastructure.cellOffsets,
                wp.int32(numDims), wp.int32(flatInputShape), wp.int32(flatOutputShape), wp.bool(dotMode),
                opInt, queryKinds, referenceKinds,
                
                wp.bool(useGradientRenormalization), renormalizationMatrices,
                wp.bool(useGradHTerms), queryOmegas, referenceOmegas,                
                wp.bool(useVolume), queryVolumes, referenceVolumes,
                wp.bool(useCRK), crk_A, crk_B, crk_gradA, crk_gradB
            )

    return warp_result.view(queryPositions.shape[0], *outputShape) # reshape back to original shape with new gradient dimension