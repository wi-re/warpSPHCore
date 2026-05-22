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


@wp.func
def computeSPHGradientTensor_Func(
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
    
    # Neighbor list data, pre accessed to avoid gradient issues with dynamic for loops
    neighborList: wp.array(dtype = wp.int64), # type: ignore
    neighborOffset : wp.int32, numNeighs: wp.int32, 
    
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
    useCRK: wp.bool, 
    queryA: wp.array(dtype = scalar_t), queryB: wp.array(dtype = vector(length=Any, dtype=scalar_t)), queryGradA: wp.array(dtype=vector(length=Any, dtype=scalar_t)), queryGradB: wp.array(dtype=matrix(shape=(Any, Any), dtype=scalar_t)), # type: ignore

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
    for neighborIndex in range(numNeighs):
        jj = neighborOffset + neighborIndex
        j  = wp.int32(neighborList[jj])
        if opInt != 0:
            if not checkDirectionality_j(referenceKinds[j], opInt):
                continue
        ##########################################################
        #   The core particle-particle interaction starts here   #
        ##########################################################
        
        mj   = referenceMasses[j]
        rhoj = referenceDensities[j]
        apparentVolume = mj / rhoj if not useVolume else referenceVolumes[j]

        fj   = type(fi)(scalar_t(0.0))
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

        # Fix for scatter based CRK, should be generalized!
        Aj = queryA[j] if useCRK else type(queryA[0])(scalar_t(0.0))
        Bj = queryB[j] if useCRK else type(queryB[0])(scalar_t(0.0))
        gradA_j = queryGradA[j] if useCRK else type(queryGradA[0])(scalar_t(0.0))
        gradB_j = queryGradB[j] if useCRK else type(queryGradB[0])()*scalar_t(0.0)
        xj = referencePositions[j]
        hj = referenceSupports[j]

        # kernelGradient = -computeKernelGradientCRK(
        #     xj, xi,
        #     hj, hi,
        #     kernel_int, wp.uint32(12), # scatter mode for gradW
        #     periodicity, domainMin, domainMax,
        #     True, Aj, Bj, gradA_j, gradB_j
        # )

        kernelGradient = computeKernelGradientCRK(
            xi, referencePositions[j], 
            hi, referenceSupports[j],
            kernel_int, mode_uint, periodicity, domainMin, domainMax,
            useCRK, Ai, Bi, gradA_i, gradB_i
        )

        if useGradientRenormalization:
            kernelGradient = matmul(Li, kernelGradient)
        
        # fqj = (fj * apparentVolume) if gradientMode_int != 2 else (mj * rhoi * fj)
        # fqi = (fi * apparentVolume) if gradientMode_int != 2 else (mj * rhoi * fi)

        if gradientMode_int == wp.static(GradientScheme.Naive.value): # Naive
            out += outerTensorProduct(fj * apparentVolume, kernelGradient, out, numDims, flatInputShape, flatOutputShape)
        elif gradientMode_int == wp.static(GradientScheme.Symmetric.value): # Symmetric
            out += outerTensorProduct(mj * rhoi * (fi / iPow(rhoi,2) + fj / iPow(rhoj,2)), kernelGradient, out, numDims, flatInputShape, flatOutputShape)
        elif gradientMode_int == wp.static(GradientScheme.Difference.value): # Difference
            out += outerTensorProduct((fj - fi) * apparentVolume, kernelGradient, out, numDims, flatInputShape, flatOutputShape)
        elif gradientMode_int == wp.static(GradientScheme.Summation.value): # Summation
            out += outerTensorProduct((fj + fi) * apparentVolume, kernelGradient, out, numDims, flatInputShape, flatOutputShape)
            
    return out

@wp.kernel
def computeSPHGradientTensor_Kernel(
    queryPositions : wp.array(dtype = vector(length=Any, dtype=scalar_t)), referencePositions : wp.array(dtype=vector(length=Any, dtype=scalar_t)), # type: ignore
    querySupports : wp.array(dtype = scalar_t), referenceSupports : wp.array(dtype = scalar_t), # type: ignore
    queryMasses: wp.array(dtype = scalar_t), referenceMasses: wp.array(dtype = scalar_t),  # type: ignore
    queryDensities: wp.array(dtype = scalar_t), referenceDensities: wp.array(dtype = scalar_t), # type: ignore
    queryValues: wp.array(dtype =vector(dtype = scalar_t, length=Any)), referenceValues: wp.array(dtype = vector(dtype = scalar_t, length=Any)), # type: ignore
    
    domainMin : wp.array(dtype = scalar_t), domainMax : wp.array(dtype = scalar_t), periodicity : wp.array(dtype = wp.bool), # type: ignore
    
    mode_uint: wp.uint32, kernel_int : wp.int32, gradientMode_int: wp.int32,
    neighborList: wp.array(dtype = wp.int64), neighborListRowOffsets: wp.array(dtype = wp.int32), numNeighbors: wp.array(dtype = wp.int32), # type: ignore
    
    preScatteredQuantities: wp. bool,
    
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
    
    outputValues[i] = computeSPHGradientTensor_Func(
        i, get_dim(queryPositions), numDims, flatInputShape, flatOutputShape,

        queryPositions, querySupports, queryMasses, queryDensities, queryValues,
        referencePositions, referenceSupports, referenceMasses, referenceDensities, referenceValues,
        
        periodicity, domainMin, domainMax, 
        mode_uint, kernel_int, gradientMode_int,

        neighborList, neighborListRowOffsets[i], numNeighbors[i], 

        preScatteredQuantities,
        
        opInt, queryKinds, referenceKinds,

        useGradientRenormalization, queryRenormalizationMatrices, 
        useGradHTerms, queryOmegas, referenceOmegas, 
        useVolume, queryVolumes, referenceVolumes,
        useCRK, crk_A, crk_B, crk_gradA, crk_gradB,

        type(outputValues[i])(scalar_t(0.0))
    )
    


from ..enumTypes import *
from typing import Optional

def computeSPHGradient_warpBackend(
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
    adjacency: AdjacencyListWarp,
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
            # print(f"Mode uint: {mode_uint}")
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
            warp_result = warpWrapper(
                launch_kernel, computeSPHGradientTensor_Kernel, outputSize, vector(length=flatOutputShape, dtype = scalar_t),
                queryPositions, referencePositions,
                querySupports, referenceSupports,
                queryMasses, referenceMasses,
                queryDensities, referenceDensities,
                qV.view(-1, flatInputShape), rV.view(-1, flatInputShape),
                domainMin, domainMax, periodicity,
                mode_uint, kernel_int, gradientMode_int,
                adjacency.j, adjacency.edgeOffsets, adjacency.numNeighbors, preScatteredQuantities,
                wp.int32(numDims), wp.int32(flatInputShape), wp.int32(flatOutputShape),
                opInt, queryKinds, referenceKinds,

                wp.bool(useGradientRenormalization), renormalizationMatrices,
                wp.bool(useGradHTerms), queryOmegas, referenceOmegas,                
                wp.bool(useVolume), queryVolumes, referenceVolumes,
                wp.bool(useCRK), crk_A, crk_B, crk_gradA, crk_gradB
            )

    return warp_result.view(queryPositions.shape[0], *outputShape) # reshape back to original shape with new gradient dimension