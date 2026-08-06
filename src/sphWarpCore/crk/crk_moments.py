import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any
import torch
from sphWarpCore.autograd import *

from ..dataTypes import *
from sphWarpCore.math import *
from sphWarpCore.kernels import *
from ..util import *
from torch.profiler import profile, record_function, ProfilerActivity
from sphWarpCore.enumTypes import *
from sphWarpCore.autograd.arg_check import *
from typing import Optional
from ..types import *


@wp.func
def get_eye(m: matrix(shape=(Any, Any), dtype=scalar_t)):
    result = type(m)() * scalar_t(0.0)
    for d in range(get_dim(m)):
        result[d, d] = scalar_t(1.0)
    return result

@wp.func
def delta(a: wp.int32, b: wp.int32):
    return scalar_t(1.0) if a == b else scalar_t(0.0)

@wp.func
def computeCRKMoments_Func(
    # General Shape Parameters and indices
    i : wp.int32, dim: wp.int32, 

    # SPH properties for the query set (indexed by i)
    queryPositions: wp.array(dtype=vector(dtype = scalar_t, length=Any)), querySupports: wp.array(dtype = scalar_t), queryMasses: wp.array(dtype = scalar_t), # type: ignore

    # SPH properties for the reference set (indexed by j in the neighbor loop)
    referencePositions : wp.array(dtype=vector(length=Any, dtype = scalar_t)), referenceSupports : wp.array(dtype = scalar_t), referenceMasses: wp.array(dtype = scalar_t),  # type: ignore
    
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
    useVolume: bool, queryVolumes: wp.array(dtype = scalar_t), referenceVolumes: wp.array(dtype = scalar_t), # type: ignore
    # Whether to use CRK kernel correction for the computation, and the corresponding correction terms if needed.
    useCRK: bool, queryA: wp.array(dtype = scalar_t), queryB: wp.array(dtype = vector(length=Any, dtype=scalar_t)), queryGradA: wp.array(dtype=vector(length=Any, dtype=scalar_t)), queryGradB: wp.array(dtype=matrix(shape=(Any, Any), dtype=scalar_t)), # type: ignore
    
    # CRKMoments function parameters begin here

    # Dummy value to allow allocation
    output_m_0 : scalar_t, # type: ignore
    output_m_1 : vector(length=Any, dtype=scalar_t), # type: ignore
    output_m_2 :matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    output_dm_0dgamma : vector(length=Any, dtype=scalar_t), # type: ignore
    output_dm_1dgamma : matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    output_dm_2dgamma : vector(length=Any, dtype=scalar_t) # type: ignore (flattened to avoid issues with warp's handling of rank
):
    if opInt != 0:
        if not checkDirectionality_i(queryKinds[i], opInt):
            return output_m_0, output_m_1, output_m_2, output_dm_0dgamma, output_dm_1dgamma, output_dm_2dgamma
    # Unpack query point properties
    xi      = queryPositions[i]
    hi      = querySupports[i]
    # mi      = queryMasses[i] # Generally not needed
    # Unpack optional correction terms
    
    # Initialize the output value
    m_0 = type(output_m_0)(scalar_t(0.0))
    m_1 = type(output_m_1)(scalar_t(0.0))
    m_2 = type(output_m_2)() * scalar_t(0.0)

    dm_0dgamma = type(output_dm_0dgamma)(scalar_t(0.0))
    dm_1dgamma = type(output_dm_1dgamma)() * scalar_t(0.0)
    dm_2dgamma = type(output_dm_2dgamma)(scalar_t(0.0)) 

    eye = get_eye(m_2)
    
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

        xj = referencePositions[j]
        hj = referenceSupports[j]

        V_j  = referenceVolumes[j]
        x_ij = computeDistanceVec(xi, xj, periodicity, domainMin, domainMax)
        w_ij = sphKernel_ij(x_ij, hi, hj, kernel_int, mode_uint, periodicity, domainMin, domainMax)
        gradw_ij = sphKernelGradient_ij(x_ij, hi, hj, kernel_int, mode_uint, periodicity, domainMin, domainMax)

            
        m_0 += V_j * w_ij
        m_1 += x_ij * (V_j * w_ij)
        m_2 += wp.outer(x_ij, x_ij) * (V_j * w_ij)

        dm_0dgamma += V_j * gradw_ij
        dm_1dgamma += V_j * (wp.outer(x_ij, gradw_ij) + w_ij * eye)
            
        for alpha in range(dim):
            for beta in range(dim):
                for gamma in range(dim):
                    gradTerm = x_ij[alpha] * x_ij[beta] * gradw_ij[gamma]
                    deltaA = x_ij[alpha] * delta(beta, gamma)
                    deltaB = delta(alpha, gamma) * x_ij[beta]
                    kernelTerm = w_ij * (deltaA + deltaB)
                    dm_2dgamma[gamma * dim * dim + alpha * dim + beta] += V_j * (gradTerm + kernelTerm)


    return m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma

@wp.kernel
def computeCRKMoments_Kernel(
    queryPositions : wp.array(dtype = vector(length=Any, dtype=scalar_t)), referencePositions : wp.array(dtype=vector(length=Any, dtype=scalar_t)), # type: ignore
    querySupports : wp.array(dtype = scalar_t), referenceSupports : wp.array(dtype = scalar_t), # type: ignore
    queryMasses: wp.array(dtype = scalar_t), referenceMasses: wp.array(dtype = scalar_t),  # type: ignore

    domainMin : wp.array(dtype = scalar_t), domainMax : wp.array(dtype = scalar_t), periodicity : wp.array(dtype = wp.bool), # type: ignore
    
    mode_uint: wp.uint32, kernel_int : wp.int32, gradientMode_int: wp.int32,
    neighborList: wp.array(dtype = wp.int64), neighborListRowOffsets: wp.array(dtype = wp.int32), numNeighbors: wp.array(dtype = wp.int32), # type: ignore
    
    preScatteredQuantities: wp. bool,
    
    opInt: wp.int32, queryKinds : wp.array(dtype = wp.int32), referenceKinds : wp.array(dtype = wp.int32), # type: ignore

    useGradientRenormalization: wp.bool, queryRenormalizationMatrices: wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)),# type: ignore
    useGradHTerms: wp.bool, queryOmegas: wp.array(dtype = scalar_t), referenceOmegas: wp.array(dtype = scalar_t),  # type: ignore
    useVolume: wp.bool, queryVolumes: wp.array(dtype = scalar_t), referenceVolumes: wp.array(dtype = scalar_t), # type: ignore
    useCRK: wp.bool, crk_A: wp.array(dtype = scalar_t), crk_B: wp.array(dtype = vector(length=Any, dtype=scalar_t)), crk_gradA: wp.array(dtype = vector(length=Any, dtype=scalar_t)), crk_gradB: wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)), # type: ignore
    
    output_m_0 : wp.array(dtype = scalar_t), # type: ignore
    output_m_1 : wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    output_m_2 : wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)), # type: ignore
    output_dm_0dgamma : wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    output_dm_1dgamma : wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)), # type: ignore
    output_dm_2dgamma : wp.array(dtype = vector(length=Any, dtype=scalar_t)) # type: ignore (flattened to avoid issues with warp's handling of rank-3 tensors) # type: ignore
):                                                                                    
    i = wp.tid()
    if i >= queryPositions.shape[0]:
        return
    
    m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma = computeCRKMoments_Func(
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

        type(output_m_0[0])(scalar_t(0.0)), # Dummy output value to allow allocation, will be overwritten with actual output
        type(output_m_1[0])(scalar_t(0.0)), # Dummy output value to allow allocation, will be overwritten with actual output
        type(output_m_2[0])(scalar_t(0.0)), # Dummy output value to allow allocation, will be overwritten with actual output
        type(output_dm_0dgamma[0])(scalar_t(0.0)), # Dummy output value to allow allocation, will be overwritten with actual output
        type(output_dm_1dgamma[0])(scalar_t(0.0)), # Dummy output value to allow allocation, will be overwritten with actual output
        type(output_dm_2dgamma[0])(scalar_t(0.0))  # Dummy output value to allow allocation, will be overwritten with actual output
    )

    output_m_0[i] = m_0
    output_m_1[i] = m_1
    output_m_2[i] = m_2
    output_dm_0dgamma[i] = dm_0dgamma
    output_dm_1dgamma[i] = dm_1dgamma
    output_dm_2dgamma[i] = dm_2dgamma

def computeCRKMomentsWarp(
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
    with record_function("warpSPH[CRKMoments]"):
        with record_function("warpSPH[CRKMoments] - Preprocessing"):
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

        with record_function("warpSPH[CRKMoments] - Kernel Execution"):
            m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma = warpWrapper(
                launch_kernel, computeCRKMoments_Kernel, outputSize, (
                    scalar_t, # kernel moment 0
                    vector(length=dim, dtype=scalar_t), # kernel moment 1
                    matrix(shape=(dim, dim), dtype=scalar_t), # kernel moment 2
                    vector(length=dim, dtype=scalar_t), # kernel moment 0 gradient
                    matrix(shape=(dim, dim), dtype=scalar_t), # kernel moment 1 gradient
                    vector(length=dim**3, dtype=scalar_t) # kernel moment 2 gradient (flattened to avoid issues with warp's handling of rank-3 tensors)                    
                    ), # type: ignore
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


    return m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma.view(-1, dim, dim, dim)
