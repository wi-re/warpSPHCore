import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any
import torch
from ..utils.wp_autograd import *
from ..radiusSearch.radius_util import convertModeToUint

from ..radiusSearch.radius_util import AdjacencyList, AdjacencyListWarp, DomainDescription, PointCloud
from ..mathutil.wp_math import *
from ..kernels.wp_kernel import *
from ..utils.wp_util import getCachedDummyTensor, checkDirectionality_i, checkDirectionality_j
from torch.profiler import profile, record_function, ProfilerActivity

# For matrices we need to implement the logic manually using outer products, since Warp does not support rank-2 field types natively. The output is stored as a flattened vector and reshaped on the Python side.
@wp.func
def computeSPHGradientTensor_Func(
    i : wp.int32,
    xi: vector(dtype = wp.float32, length=Any), hi : wp.float32, mi: wp.float32, rhoi: wp.float32, fi : vector(dtype = wp.float32, length=Any), # type: ignore
    
    positions : wp.array(dtype=vector(length=Any, dtype = wp.float32)), supports : wp.array(dtype = wp.float32), masses: wp.array(dtype = wp.float32), densities: wp.array(dtype = wp.float32), values: wp.array(dtype = vector(dtype = wp.float32, length=Any)), # type: ignore
    
    periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), # type: ignore
    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32, # type: ignore
    
    neighborList: wp.array(dtype = wp.int64), # type: ignore
    neighborOffset : wp.int32, numNeighs: wp.int32, preScatteredQuantities: wp. bool,
    
    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    opInt: wp.int32, referenceKinds : wp.array(dtype = wp.int32), # type: ignore

    useGradientRenormalization: wp.bool, L: wp.array(dtype = matrix(shape=(Any, Any), dtype=wp.float32)), # type: ignore
    useGradHTerms: wp.bool, referenceOmegas: wp.array(dtype = wp.float32),  # type: ignore
    useVolume: bool, referenceVolumes: wp.array(dtype = wp.float32), # type: ignore
    useCRK: bool, Ai: wp.float32, Bi: vector(length=Any, dtype=wp.float32), gradA_i: vector(length=Any, dtype=wp.float32), gradB_i: matrix(shape=(Any, Any), dtype=wp.float32), # type: ignore
    
    outputValue: vector(length=Any, dtype=wp.float32) # type: ignore
):
    dim = wp.int32(xi.length)
    grad_f_interpolated = type(outputValue)(0.0)

    Li = type(L[0])()
    if useGradientRenormalization:
        Li = L[i]
    
    for neighborIndex in range(numNeighs):
        jj = neighborOffset + neighborIndex
        j = wp.int32(neighborList[jj])
        if opInt != 0:
            if not checkDirectionality_j(referenceKinds[j], opInt):
                continue
        
        mj = masses[j]
        rhoj = densities[j]
        apparentVolume = mj / rhoj if not useVolume else referenceVolumes[j]
        fj = type(fi)(0.0)
        if preScatteredQuantities:
            if useGradHTerms:
                fj = values[jj] / referenceOmegas[j]
            else:
                fj = values[jj]
        else:
            if useGradHTerms:
                fj = values[j] / referenceOmegas[j]
            else:
                fj = values[j]


        kernelGradient = computeKernelGradientCRK(
            xi, positions[j], 
            hi, supports[j],
            kernel_int, mode_uint, periodicity, domainMin, domainMax,
            useCRK, Ai, Bi, gradA_i, gradB_i
        )
        # x_ij = computeDistanceVec(xi, positions[j], periodicity, domainMin, domainMax)
        # kernelGradient = sphKernelGradient_ij(x_ij, hi, supports[j], kernel_int, mode_uint, periodicity, domainMin, domainMax)

        # if useCRK:
        #     W_ij = sphKernel_ij(x_ij, hi, supports[j], kernel_int, mode_uint, periodicity, domainMin, domainMax)
        #     kernelGradient = correctGradientCRK(
        #         W_ij,
        #         kernelGradient, 
        #         x_ij, 
        #         Ai, Bi, gradA_i, gradB_i, 
        #         get_dim(xi))

        if useGradientRenormalization:
            kernelGradient = matmul(Li, kernelGradient)
        

        if gradientMode_int == 1: # Naive
            grad_f_interpolated += outerTensorProduct(fj * apparentVolume, kernelGradient, grad_f_interpolated, numDims, flatInputShape, flatOutputShape)
        elif gradientMode_int == 2: # Symmetric
            grad_f_interpolated += outerTensorProduct(mj * rhoi * (fi / iPow(rhoi,2) + fj / iPow(rhoj,2)) * apparentVolume, kernelGradient, grad_f_interpolated, numDims, flatInputShape, flatOutputShape)
        elif gradientMode_int == 3: # Difference
            grad_f_interpolated += outerTensorProduct((fj - fi) * apparentVolume, kernelGradient, grad_f_interpolated, numDims, flatInputShape, flatOutputShape)
        elif gradientMode_int == 4: # Summation
            grad_f_interpolated += outerTensorProduct((fj + fi) * apparentVolume, kernelGradient, grad_f_interpolated, numDims, flatInputShape, flatOutputShape)
            
    return grad_f_interpolated

@wp.kernel
def computeSPHGradientTensor_Kernel(
    queryPositions : wp.array(dtype = vector(length=Any, dtype=wp.float32)), referencePositions : wp.array(dtype=vector(length=Any, dtype=wp.float32)), # type: ignore
    querySupports : wp.array(dtype = wp.float32), referenceSupports : wp.array(dtype = wp.float32), # type: ignore
    queryMasses: wp.array(dtype = wp.float32), referenceMasses: wp.array(dtype = wp.float32),  # type: ignore
    queryDensities: wp.array(dtype = wp.float32), referenceDensities: wp.array(dtype = wp.float32), # type: ignore
    queryValues: wp.array(dtype =vector(dtype = wp.float32, length=Any)), referenceValues: wp.array(dtype = vector(dtype = wp.float32, length=Any)), # type: ignore
    
    domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), periodicity : wp.array(dtype = wp.bool), # type: ignore
    
    mode_uint: wp.uint32, kernel_int : wp.int32, gradientMode_int: wp.int32,
    neighborList: wp.array(dtype = wp.int64), neighborListRowOffsets: wp.array(dtype = wp.int32), numNeighbors: wp.array(dtype = wp.int32), preScatteredQuantities: wp. bool,# type: ignore
    
    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    opInt: wp.int32, queryKinds : wp.array(dtype = wp.int32), referenceKinds : wp.array(dtype = wp.int32), # type: ignore

    useGradientRenormalization: wp.bool, L: wp.array(dtype = matrix(shape=(Any, Any), dtype=wp.float32)),# type: ignore
    useGradHTerms: wp.bool, queryOmegas: wp.array(dtype = wp.float32), referenceOmegas: wp.array(dtype = wp.float32),  # type: ignore
    useVolume: bool, queryVolumes: wp.array(dtype = wp.float32), referenceVolumes: wp.array(dtype = wp.float32), # type: ignore
    useCRK: bool, crk_A: wp.array(dtype = wp.float32), crk_B: wp.array(dtype = vector(length=Any, dtype=wp.float32)), crk_gradA: wp.array(dtype = vector(length=Any, dtype=wp.float32)), crk_gradB: wp.array(dtype = matrix(shape=(Any, Any), dtype=wp.float32)), # type: ignore
    
    outputValues : wp.array(dtype = vector(length=Any, dtype = wp.float32)) # type: ignore
):                                                                                    
    i = wp.tid()
    if i >= queryPositions.shape[0]:
        return
    if opInt != 0:
        if not checkDirectionality_i(queryKinds[i], opInt):
            return
    
    xi = queryPositions[i]
    hi = querySupports[i]
    mi = queryMasses[i]
    rhoi = queryDensities[i]
    fi = queryValues[i]
    if useGradHTerms:
        fi = queryValues[i] / queryOmegas[i]

    Ai = crk_A[i] if useCRK else type(crk_A[0])(0.0)
    Bi = crk_B[i] if useCRK else type(crk_B[0])(0.0)
    gradA_i = crk_gradA[i] if useCRK else type(crk_gradA[0])(0.0)
    gradB_i = crk_gradB[i] if useCRK else type(crk_gradB[0])()*0.0
    
    outputValues[i] = computeSPHGradientTensor_Func(
        i,
        xi, hi, mi, rhoi, fi, 
        referencePositions, referenceSupports, referenceMasses, referenceDensities, referenceValues,
        
        periodicity, domainMin, domainMax, 
        mode_uint, kernel_int, gradientMode_int,
        neighborList, neighborListRowOffsets[i], numNeighbors[i], 
        preScatteredQuantities,
        numDims, flatInputShape, flatOutputShape,
        opInt, referenceKinds,

        useGradientRenormalization, L, 
        useGradHTerms, referenceOmegas, 
        useVolume, referenceVolumes,
        useCRK, Ai, Bi, gradA_i, gradB_i,

        type(outputValues[i])(0.0))
    


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

            mode_uint = convertModeToUint(mode.name)
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
                launch_kernel, computeSPHGradientTensor_Kernel, outputSize, vector(length=flatOutputShape, dtype = wp.float32),
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