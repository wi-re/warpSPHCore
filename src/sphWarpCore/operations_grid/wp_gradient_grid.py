import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any
import torch
from ..utils.wp_autograd import *


from ..radiusSearch.radius_util import AdjacencyList, AdjacencyListWarp, DomainDescription, PointCloud
from ..mathutil.wp_math import *
from ..kernels.wp_kernel import *
from ..utils.wp_util import getCachedDummyTensor, checkDirectionality_i, checkDirectionality_j
from torch.profiler import profile, record_function, ProfilerActivity



from ..enumTypes import *
from ..radiusSearch.wp_compactHash import CompactHashMap, computeZOrderIndex64, hashGridVec3i
from .grid_util import checkOffset
@wp.func
def computeSPHGradientTensor_grid_Func(
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
    
    # Dummy value to allow allocation
    outputValue: Any # type: ignore
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
        
        mj   = referenceMasses[j]
        rhoj = referenceDensities[j]
        apparentVolume = mj / rhoj if not useVolume else referenceVolumes[j]

        fj   = type(fi)(0.0)
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
        
        if gradientMode_int == 1: # Naive
            out += outerTensorProduct(fj * apparentVolume, kernelGradient, out, numDims, flatInputShape, flatOutputShape)
        elif gradientMode_int == 2: # Symmetric
            out += outerTensorProduct(mj * rhoi * (fi / iPow(rhoi,2) + fj / iPow(rhoj,2)) * apparentVolume, kernelGradient, out, numDims, flatInputShape, flatOutputShape)
        elif gradientMode_int == 3: # Difference
            out += outerTensorProduct((fj - fi) * apparentVolume, kernelGradient, out, numDims, flatInputShape, flatOutputShape)
        elif gradientMode_int == 4: # Summation
            out += outerTensorProduct((fj + fi) * apparentVolume, kernelGradient, out, numDims, flatInputShape, flatOutputShape)
            
    return out

@wp.kernel
def computeSPHGradientTensor_grid_Kernel(
    queryPositions : wp.array(dtype = vector(length=Any, dtype=wp.float32)), referencePositions : wp.array(dtype=vector(length=Any, dtype=wp.float32)), # type: ignore
    querySupports : wp.array(dtype = wp.float32), referenceSupports : wp.array(dtype = wp.float32), # type: ignore
    queryMasses: wp.array(dtype = wp.float32), referenceMasses: wp.array(dtype = wp.float32),  # type: ignore
    queryDensities: wp.array(dtype = wp.float32), referenceDensities: wp.array(dtype = wp.float32), # type: ignore
    queryValues: wp.array(dtype =vector(dtype = wp.float32, length=Any)), referenceValues: wp.array(dtype = vector(dtype = wp.float32, length=Any)), # type: ignore
    
    domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), periodicity : wp.array(dtype = wp.bool), # type: ignore
    
    mode_uint: wp.uint32, kernel_int : wp.int32, gradientMode_int: wp.int32,
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
    
    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    opInt: wp.int32, queryKinds : wp.array(dtype = wp.int32), referenceKinds : wp.array(dtype = wp.int32), # type: ignore

    useGradientRenormalization: wp.bool, queryRenormalizationMatrices: wp.array(dtype = matrix(shape=(Any, Any), dtype=wp.float32)),# type: ignore
    useGradHTerms: wp.bool, queryOmegas: wp.array(dtype = wp.float32), referenceOmegas: wp.array(dtype = wp.float32),  # type: ignore
    useVolume: wp.bool, queryVolumes: wp.array(dtype = wp.float32), referenceVolumes: wp.array(dtype = wp.float32), # type: ignore
    useCRK: wp.bool, crk_A: wp.array(dtype = wp.float32), crk_B: wp.array(dtype = vector(length=Any, dtype=wp.float32)), crk_gradA: wp.array(dtype = vector(length=Any, dtype=wp.float32)), crk_gradB: wp.array(dtype = matrix(shape=(Any, Any), dtype=wp.float32)), # type: ignore
    
    outputValues : wp.array(dtype = Any) # type: ignore
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

        out_value += computeSPHGradientTensor_grid_Func(
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

            type(outputValues[0])()
        )
    outputValues[i] = out_value
    


from ..enumTypes import *
from typing import Optional

def computeSPHGradient_grid_warpBackend(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,
    queryDensities, referenceDensities,
    queryValues : Optional[torch.Tensor], referenceValues : Optional[torch.Tensor],
    queryKinds, referenceKinds,
    domain: DomainDescription,
    mode: SupportScheme,
    kernel: KernelFunctions,    
    gradientMode: GradientScheme,
    operationMode: OperationDirection,
    datastructure: CompactHashMap,
    scatteredQuantities: Optional[torch.Tensor] = None,

    useGradientRenormalization: bool = False, renormalizationMatrices: Optional[torch.Tensor] = None,
    useGradHTerms: bool = False, queryOmegas: Optional[torch.Tensor] = None, referenceOmegas: Optional[torch.Tensor] = None,
    useVolume: bool = False, queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    useCRK: bool = False, crk_A: Optional[torch.Tensor] = None, crk_B: Optional[torch.Tensor] = None, crk_gradA: Optional[torch.Tensor] = None, crk_gradB: Optional[torch.Tensor] = None
):
    with record_function("warpSPH[Gradient]"):
        with record_function("warpSPH[Gradient] - Preprocessing"):
            domainMin = domain.min
            domainMax = domain.max
            periodicity = domain.periodic

            mode_uint = supportSchemeToUint(mode)
            kernel_int = kernel.value
            gradientMode_int = gradientMode.value
            opInt = wp.int32(operationMode.value)

            preScatteredQuantities = False # Indicates if the input quantities have already been scattered to the neighbor level (e.g. mass/density products), which can save some redundant computations if they are needed for multiple operations. This can also help with some custom kernels where we want to pre-compute certain quantities at the neighbor level on the Python side and pass them in as additional fields to avoid redundant computations in the kernel. 
            if queryValues is None and referenceValues is None:
                if scatteredQuantities is None:
                    raise ValueError("If queryValues and referenceValues are not provided, then pre-scattered quantities must be provided for the gradient computation.")
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

            outputShape = inputShape + (queryPositions.shape[1],) # add an extra dimension for the gradient
            flatOutputShape = 1
            for dim in outputShape:
                flatOutputShape *= dim
            # Warp kernels only support rank-1 (vector) and rank-2 (matrix) field types.
            numDims = len(inputShape)

            D = queryPositions.shape[1]

        with record_function("warpSPH[Gradient] - Kernel Execution"):
            D = queryPositions.shape[1]
            warp_result = warpWrapper(
                launch_kernel, computeSPHGradientTensor_grid_Kernel, outputSize, vector(length=flatOutputShape, dtype = wp.float32),
                queryPositions, referencePositions,
                querySupports, referenceSupports,
                queryMasses, referenceMasses,
                queryDensities, referenceDensities,
                qV.view(-1, flatInputShape), rV.view(-1, flatInputShape),
                domainMin, domainMax, periodicity,
                mode_uint, kernel_int, gradientMode_int,
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

    return warp_result.view(queryPositions.shape[0], *outputShape) # reshape back to original shape with new gradient dimension