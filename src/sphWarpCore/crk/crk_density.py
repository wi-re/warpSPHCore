import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any
import torch
from sphWarpCore.utils.wp_autograd import *

from sphWarpCore.radiusSearch.radius_util import AdjacencyList, AdjacencyListWarp, DomainDescription, PointCloud
from sphWarpCore.mathutil.wp_math import *
from sphWarpCore.kernels.wp_kernel import *
from sphWarpCore.utils.wp_util import getCachedDummyTensor, checkDirectionality_i, checkDirectionality_j
from torch.profiler import profile, record_function, ProfilerActivity
from sphWarpCore.enumTypes import *
from sphWarpCore.utils.arg_check import *
from typing import Optional

@wp.func
def computeCRKDensity_Func(
    # General Shape Parameters and indices
    i : wp.int32, dim: wp.int32, 

    # SPH properties for the query set (indexed by i)
    queryPositions: wp.array(dtype=vector(dtype = wp.float32, length=Any)), querySupports: wp.array(dtype = wp.float32), queryMasses: wp.array(dtype = wp.float32), # type: ignore

    # SPH properties for the reference set (indexed by j in the neighbor loop)
    referencePositions : wp.array(dtype=vector(length=Any, dtype = wp.float32)), referenceSupports : wp.array(dtype = wp.float32), referenceMasses: wp.array(dtype = wp.float32), # type: ignore
    
    # Domain and kernel parameters
    periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), # type: ignore
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
    useGradientRenormalization: wp.bool, queryRenormalizationMatrices: wp.array(dtype = matrix(shape=(Any, Any), dtype=wp.float32)), # type: ignore
    # Grad-h correction terms for each query and reference point, used for correcting the kernel gradient based on the local particle distribution and smoothing length variations.
    useGradHTerms: wp.bool, queryOmegas: wp.array(dtype = wp.float32), referenceOmegas: wp.array(dtype = wp.float32),  # type: ignore
    # Whether to use actual volume (mass/density) or apparent volume for the gradient computation, and the corresponding volumes if needed.
    useVolume: bool, queryVolumes: wp.array(dtype = wp.float32), referenceVolumes: wp.array(dtype = wp.float32), # type: ignore
    # Whether to use CRK kernel correction for the computation, and the corresponding correction terms if needed.
    useCRK: bool, queryA: wp.array(dtype = wp.float32), queryB: wp.array(dtype = vector(length=Any, dtype=wp.float32)), queryGradA: wp.array(dtype=vector(length=Any, dtype=wp.float32)), queryGradB: wp.array(dtype=matrix(shape=(Any, Any), dtype=wp.float32)), # type: ignore
    
    # CRKDensity function parameters begin here

    # Dummy value to allow allocation
    outputValue: Any # type: ignore
):
    if opInt != 0:
        if not checkDirectionality_i(queryKinds[i], opInt):
            return outputValue * 0.0
    # Unpack query point properties
    xi      = queryPositions[i]
    hi      = querySupports[i]
    mi      = queryMasses[i] # Generally not needed
    # Unpack optional correction terms
    Ai      = queryA[i] if useCRK else type(queryA[0])(0.0)
    Bi      = queryB[i] if useCRK else type(queryB[0])(0.0)
    gradA_i = queryGradA[i] if useCRK else type(queryGradA[0])(0.0)
    gradB_i = queryGradB[i] if useCRK else type(queryGradB[0])()*0.0
    
    # Initialize the output value
    outA     = wp.float32(0.0)
    outB = wp.float32(0.0)
    
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
        Vj = referenceVolumes[j]

        w_ij = computeKernelCRK(
            xi, referencePositions[j], 
            hi, referenceSupports[j], 
            kernel_int, mode_uint, periodicity, domainMin, domainMax,
            useCRK, Ai, Bi
        )

        # termA = scatter_sum(m_i * V_j * W_ij, i, dim = 0, dim_size=particles.positions.shape[0])
        # termB = scatter_sum(V_j * V_j * W_ij, i, dim = 0, dim_size=particles.positions.shape[0])


        outA += mi * Vj * w_ij
        outB += Vj * Vj * w_ij

    return outA/outB

@wp.kernel
def computeCRKDensity_Kernel(
    queryPositions : wp.array(dtype = vector(length=Any, dtype=wp.float32)), referencePositions : wp.array(dtype=vector(length=Any, dtype=wp.float32)), # type: ignore
    querySupports : wp.array(dtype = wp.float32), referenceSupports : wp.array(dtype = wp.float32), # type: ignore
    queryMasses: wp.array(dtype = wp.float32), referenceMasses: wp.array(dtype = wp.float32),  # type: ignore
    
    domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), periodicity : wp.array(dtype = wp.bool), # type: ignore
    
    mode_uint: wp.uint32, kernel_int : wp.int32, gradientMode_int: wp.int32,
    neighborList: wp.array(dtype = wp.int64), neighborListRowOffsets: wp.array(dtype = wp.int32), numNeighbors: wp.array(dtype = wp.int32), # type: ignore
    
    preScatteredQuantities: wp. bool,
    
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
    
    outputValues[i] = computeCRKDensity_Func(
        i, get_dim(queryPositions), 

        queryPositions, querySupports, queryMasses,
        referencePositions, referenceSupports, referenceMasses, 
        
        periodicity, domainMin, domainMax, 
        mode_uint, kernel_int, gradientMode_int,

        neighborList, neighborListRowOffsets[i], numNeighbors[i], 

        preScatteredQuantities,
        
        opInt, queryKinds, referenceKinds,

        useGradientRenormalization, queryRenormalizationMatrices, 
        useGradHTerms, queryOmegas, referenceOmegas, 
        useVolume, queryVolumes, referenceVolumes,
        useCRK, crk_A, crk_B, crk_gradA, crk_gradB,

        type(outputValues[i])(0.0)
    )
    
def computeCRKDensityWarp(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,
    domain: DomainDescription,
    supportMode: SupportScheme,
    kernel: KernelFunctions,    
    operationMode: OperationDirection,
    adjacency: AdjacencyListWarp,

    scatteredQuantities: Optional[torch.Tensor] = None,
    
    queryKinds: Optional[torch.Tensor] = None, referenceKinds: Optional[torch.Tensor] = None,
    useGradientRenormalization: bool = False, renormalizationMatrices: Optional[torch.Tensor] = None,
    useGradHTerms: bool = False, queryOmegas: Optional[torch.Tensor] = None, referenceOmegas: Optional[torch.Tensor] = None,
    useVolume: bool = False, queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    useCRK: bool = False, crk_A: Optional[torch.Tensor] = None, crk_B: Optional[torch.Tensor] = None, crk_gradA: Optional[torch.Tensor] = None, crk_gradB: Optional[torch.Tensor] = None
):
    with record_function("warpSPH[CRKDensity]"):
        with record_function("warpSPH[CRKDensity] - Preprocessing"):
            # Preprocessing and input validation
            domainMin = domain.min
            domainMax = domain.max
            periodicity = domain.periodic

            mode_uint = supportSchemeToUint(supportMode)
            kernel_int = kernel.value
            gradientMode_int = 0
            opInt = wp.int32(operationMode.value)

            device = queryPositions.device
            dim = queryPositions.shape[1]

            qK, rK = checkKinds(operationMode, device, queryKinds, referenceKinds)
            renormalizationMatrices_ = checkInputRenormalization(dim, device, useGradientRenormalization, renormalizationMatrices)
            queryOmegas_, referenceOmegas_ = checkInputGradHTerms(dim, device, useGradHTerms, queryOmegas, referenceOmegas)
            queryVolumes_, referenceVolumes_ = checkInputVolume(dim, device, useVolume, queryVolumes, referenceVolumes)
            crk_A_, crk_B_, crk_gradA_, crk_gradB_ = checkInputCRK(dim, device, useCRK, crk_A, crk_B, crk_gradA, crk_gradB)

            # Warp kernels only support rank-1 (vector) and rank-2 (matrix) field types.
            outputSize = queryPositions.shape[0]
            inputShape = queryPositions.shape[1:]

        with record_function("warpSPH[CRKDensity] - Kernel Execution"):
            warp_result = warpWrapper(
                launch_kernel, computeCRKDensity_Kernel, outputSize, wp.float32,
                queryPositions, referencePositions,
                querySupports, referenceSupports,
                queryMasses, referenceMasses,

                domainMin, domainMax, periodicity,
                mode_uint, kernel_int, gradientMode_int,
                adjacency.j, adjacency.edgeOffsets, adjacency.numNeighbors, 
                
                False,

                opInt, qK, rK,

                wp.bool(useGradientRenormalization), renormalizationMatrices_,
                wp.bool(useGradHTerms), queryOmegas_, referenceOmegas_,
                wp.bool(useVolume), queryVolumes_, referenceVolumes_,
                wp.bool(useCRK), crk_A_, crk_B_, crk_gradA_, crk_gradB_
            )

    return warp_result
