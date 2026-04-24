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
def computeSPHCovariance_Func(
    # General Shape Parameters and indices
    i : wp.int32, dim: wp.int32, numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,

    # SPH properties for the query set (indexed by i)
    queryPositions: wp.array(dtype=vector(dtype = wp.float32, length=Any)), querySupports: wp.array(dtype = wp.float32), queryMasses: wp.array(dtype = wp.float32), queryDensities: wp.array(dtype = wp.float32), # type: ignore

    # SPH properties for the reference set (indexed by j in the neighbor loop)
    referencePositions : wp.array(dtype=vector(length=Any, dtype = wp.float32)), referenceSupports : wp.array(dtype = wp.float32), referenceMasses: wp.array(dtype = wp.float32), referenceDensities: wp.array(dtype = wp.float32), # type: ignore
    
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
    # Whether to use actual volume (mass/density) or apparent volume for the gradient computation, and the corresponding volumes if needed.
    useVolume: bool, queryVolumes: wp.array(dtype = wp.float32), referenceVolumes: wp.array(dtype = wp.float32), # type: ignore
    # Whether to use CRK kernel correction for the computation, and the corresponding correction terms if needed.
    useCRK: bool, queryA: wp.array(dtype = wp.float32), queryB: wp.array(dtype = vector(length=Any, dtype=wp.float32)), queryGradA: wp.array(dtype=vector(length=Any, dtype=wp.float32)), queryGradB: wp.array(dtype=matrix(shape=(Any, Any), dtype=wp.float32)), # type: ignore
    
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
    # Unpack optional correction terms
    Li      = queryRenormalizationMatrices[i] if useGradientRenormalization else type(queryRenormalizationMatrices[0])()*0.0
    Ai      = queryA[i] if useCRK else type(queryA[0])(0.0)
    Bi      = queryB[i] if useCRK else type(queryB[0])(0.0)
    gradA_i = queryGradA[i] if useCRK else type(queryGradA[0])(0.0)
    gradB_i = queryGradB[i] if useCRK else type(queryGradB[0])()*0.0
    
    # Initialize the output value
    out     = type(outputValue)(0.0)
    
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
        
        mj = referenceMasses[j]
        rhoj = referenceDensities[j]
        apparentVolume = mj / rhoj if not useVolume else referenceVolumes[j]

        fij = -computeDistanceVec(xi, referencePositions[j], periodicity, domainMin, domainMax)
        
        kernelGradient = computeKernelGradientCRK(
            xi, referencePositions[j], 
            hi, referenceSupports[j],
            kernel_int, mode_uint, periodicity, domainMin, domainMax,
            useCRK, Ai, Bi, gradA_i, gradB_i
        )

        if useGradientRenormalization:
            kernelGradient = matmul(Li, kernelGradient)        

        grad_f_interpolated += outerTensorProduct(fij * apparentVolume, kernelGradient, grad_f_interpolated, numDims, flatInputShape, flatOutputShape)
            
    return grad_f_interpolated

@wp.kernel
def computeSPHCovariance_Kernel(
    queryPositions : wp.array(dtype = vector(length=Any, dtype=wp.float32)), referencePositions : wp.array(dtype=vector(length=Any, dtype=wp.float32)), # type: ignore
    querySupports : wp.array(dtype = wp.float32), referenceSupports : wp.array(dtype = wp.float32), # type: ignore
    queryMasses: wp.array(dtype = wp.float32), referenceMasses: wp.array(dtype = wp.float32),  # type: ignore
    queryDensities: wp.array(dtype = wp.float32), referenceDensities: wp.array(dtype = wp.float32), # type: ignore
    
    domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), periodicity : wp.array(dtype = wp.bool), # type: ignore
    
    mode_uint: wp.uint32, kernel_int : wp.int32, gradientMode_int: wp.int32,
    neighborList: wp.array(dtype = wp.int64), neighborListRowOffsets: wp.array(dtype = wp.int32), numNeighbors: wp.array(dtype = wp.int32), # type: ignore
    
    
    numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    opInt: wp.int32, queryKinds : wp.array(dtype = wp.int32), referenceKinds : wp.array(dtype = wp.int32), # type: ignore

    useGradientRenormalization: wp.bool, queryRenormalizationMatrices: wp.array(dtype = matrix(shape=(Any, Any), dtype=wp.float32)),# type: ignore
    useVolume: wp.bool, queryVolumes: wp.array(dtype = wp.float32), referenceVolumes: wp.array(dtype = wp.float32), # type: ignore
    useCRK: wp.bool, crk_A: wp.array(dtype = wp.float32), crk_B: wp.array(dtype = vector(length=Any, dtype=wp.float32)), crk_gradA: wp.array(dtype = vector(length=Any, dtype=wp.float32)), crk_gradB: wp.array(dtype = matrix(shape=(Any, Any), dtype=wp.float32)), # type: ignore
    
    outputValues : wp.array(dtype = vector(length=Any, dtype = wp.float32)) # type: ignore
):                                                                                    
    i = wp.tid()
    if i >= queryPositions.shape[0]:
        return
    
    outputValues[i] = computeSPHCovariance_Func(
        i, get_dim(queryPositions), numDims, flatInputShape, flatOutputShape,

        queryPositions, querySupports, queryMasses, queryDensities,
        referencePositions, referenceSupports, referenceMasses, referenceDensities,
        
        periodicity, domainMin, domainMax, 
        mode_uint, kernel_int, gradientMode_int,

        neighborList, neighborListRowOffsets[i], numNeighbors[i], 
        
        opInt, queryKinds, referenceKinds,

        useGradientRenormalization, queryRenormalizationMatrices, 
        useVolume, queryVolumes, referenceVolumes,
        useCRK, crk_A, crk_B, crk_gradA, crk_gradB,

        type(outputValues[i])(0.0)
    )
    


from ..enumTypes import *
from typing import Optional

def computeSPHCovariance_warpBackend(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,
    queryDensities, referenceDensities,
    queryKinds, referenceKinds,
    domain: DomainDescription,
    mode: SupportScheme,
    kernel: KernelFunctions,    
    operationMode: OperationDirection,
    adjacency: AdjacencyListWarp,
    
    useGradientRenormalization: bool = False, renormalizationMatrices: Optional[torch.Tensor] = None,
    useVolume: bool = False, queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    useCRK: bool = False, crk_A: Optional[torch.Tensor] = None, crk_B: Optional[torch.Tensor] = None, crk_gradA: Optional[torch.Tensor] = None, crk_gradB: Optional[torch.Tensor] = None
):
    with record_function("warpSPH[Covariance]"):
        with record_function("warpSPH[Covariance] - Preprocessing"):
            # Preprocessing and input validation
            domainMin = domain.min
            domainMax = domain.max
            periodicity = domain.periodic

            mode_uint = convertModeToUint(mode.name)
            kernel_int = kernel.value
            gradientMode_int = 0
            opInt = wp.int32(operationMode.value)

            preScatteredQuantities = False # Indicates if the input quantities have already been scattered to the neighbor level (e.g. mass/density products), which can save some redundant computations if they are needed for multiple operations. This can also help with some custom kernels where we want to pre-compute certain quantities at the neighbor level on the Python side and pass them in as additional fields to avoid redundant computations in the kernel. 

            # Warp kernels only support rank-1 (vector) and rank-2 (matrix) field types.
            outputSize = (queryPositions.shape[0])

            inputShape = queryPositions.shape[1:]
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
            L = renormalizationMatrices if renormalizationMatrices is not None else getCachedDummyTensor((1, D, D), dtype=torch.float32, device=queryPositions.device)
            renormalize = renormalizationMatrices is not None

        with record_function("warpSPH[Covariance] - Kernel Execution"):
            warp_result = warpWrapper(
                launch_kernel, computeSPHCovariance_Kernel, outputSize, vector(length=flatOutputShape, dtype = wp.float32),
                queryPositions, referencePositions,
                querySupports, referenceSupports,
                queryMasses, referenceMasses,
                queryDensities, referenceDensities,
                domainMin, domainMax, periodicity,
                mode_uint, kernel_int, gradientMode_int,
                adjacency.j, adjacency.edgeOffsets, adjacency.numNeighbors, preScatteredQuantities,
                wp.int32(numDims), wp.int32(flatInputShape), wp.int32(flatOutputShape),
                opInt, queryKinds, referenceKinds,

                wp.bool(useGradientRenormalization), renormalizationMatrices,
                wp.bool(useVolume), queryVolumes, referenceVolumes,
                wp.bool(useCRK), crk_A, crk_B, crk_gradA, crk_gradB
            )

    return warp_result.view(queryPositions.shape[0], *outputShape) # reshape back to original shape with new gradient dimension


from ..math import outerTensorProduct

import torch
from torch.profiler import record_function

# @torch.jit.script
def pinv2x2(M):
    with record_function('Pseudo Inverse 2x2'):
        a = M[:,0,0]
        b = M[:,0,1]
        c = M[:,1,0]
        d = M[:,1,1]

        theta = 0.5 * torch.atan2(2 * a * c + 2 * b * d, a**2 + b**2 - c**2 - d**2)
        cosTheta = torch.cos(theta)
        sinTheta = torch.sin(theta)
        U = torch.zeros_like(M)
        U[:,0,0] = cosTheta
        U[:,0,1] = - sinTheta
        U[:,1,0] = sinTheta
        U[:,1,1] = cosTheta

        S1 = a**2 + b**2 + c**2 + d**2
        S2 = torch.sqrt((a**2 + b**2 - c**2 - d**2)**2 + 4* (a * c + b *d)**2)

        o1 = torch.sqrt((S1 + S2) / 2)
        o2 = torch.sqrt(torch.clamp(S1 - S2, min = 1e-9) / 2)

        phi = 0.5 * torch.atan2(2 * a * b + 2 * c * d, a**2 - b**2 + c**2 - d**2)
        cosPhi = torch.cos(phi)
        sinPhi = torch.sin(phi)
        s11 = torch.sign((a * cosTheta + c * sinTheta) * cosPhi + ( b * cosTheta + d * sinTheta) * sinPhi)
        s22 = torch.sign((a * sinTheta - c * cosTheta) * sinPhi + (-b * sinTheta + d * cosTheta) * cosPhi)

        # s11 = torch.sign(o1)
        # s22 = torch.sign(o2)

        V = torch.zeros_like(M)
        V[:,0,0] = cosPhi * s11
        V[:,0,1] = - sinPhi * s22
        V[:,1,0] = sinPhi * s11
        V[:,1,1] = cosPhi * s22

        eigVals = torch.vstack((o1, o2)).mT
        eigVals[torch.abs(eigVals[:,1]) > torch.abs(eigVals[:,0]),:] = torch.flip(eigVals[torch.abs(eigVals[:,1]) > torch.abs(eigVals[:,0]),:],[1])

        # S = torch.diag_embed(eigVals, dim1 = 2, dim2 = 1)

        o1_1 = torch.zeros_like(o1)
        o2_1 = torch.zeros_like(o2)

        o1_1[torch.abs(eigVals[:,0]) > 1e-7] = 1 / eigVals[torch.abs(eigVals[:,0]) > 1e-7, 0] 
        o2_1[torch.abs(eigVals[:,1]) > 1e-7] = 1 / eigVals[torch.abs(eigVals[:,1]) > 1e-7, 1] 
        o = torch.vstack((o1_1, o2_1))
        S_1 = torch.diag_embed(o.mT, dim1 = 2, dim2 = 1)
        
        inv = torch.matmul(torch.matmul(V, S_1), U.mT)
        return inv, eigVals


def computeRenormalizationMatrices_(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,
    queryDensities, referenceDensities,
    queryKinds, referenceKinds,
    domain: DomainDescription,
    mode: SupportScheme,
    kernel: KernelFunctions,    
    operationMode: OperationDirection,
    adjacency: AdjacencyListWarp,
    
    useGradientRenormalization: bool = False, renormalizationMatrices: Optional[torch.Tensor] = None,
    useVolume: bool = False, queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    useCRK: bool = False, crk_A: Optional[torch.Tensor] = None, crk_B: Optional[torch.Tensor] = None, crk_gradA: Optional[torch.Tensor] = None, crk_gradB: Optional[torch.Tensor] = None
):
    
    C = computeSPHCovariance_warpBackend(
        queryPositions, referencePositions, 
        querySupports, referenceSupports, 
        queryMasses, referenceMasses, 
        queryDensities, referenceDensities, 
        domain = domain, adjacency = adjacency, 
        mode = mode, kernel = kernel, renormalizationMatrices = None)

    num_nbrs = adjacency.numNeighbors
    dtype = C.dtype

    dim = queryPositions.shape[1]
    dtype = C.dtype
    device = queryPositions.device

    C[num_nbrs < 4,:,:] = torch.eye(dim, dtype = dtype, device = device)[None,:,:]

    if queryPositions.shape[1] == 2:
        L, eigVals = pinv2x2(C)
    else:
        L = torch.linalg.pinv(C)
        eigVals = torch.linalg.eigvals(C).real

        if queryPositions.shape[1] == 3:
            eigVals[torch.abs(eigVals[:,1]) > torch.abs(eigVals[:,0]),:] = torch.flip(eigVals[torch.abs(eigVals[:,1]) > torch.abs(eigVals[:,0]),:],[1])
            eigVals[torch.abs(eigVals[:,2]) > torch.abs(eigVals[:,1]),:] = torch.flip(eigVals[torch.abs(eigVals[:,2]) > torch.abs(eigVals[:,1]),:],[1])
            eigVals[torch.abs(eigVals[:,2]) > torch.abs(eigVals[:,0]),:] = torch.flip(eigVals[torch.abs(eigVals[:,2]) > torch.abs(eigVals[:,0]),:],[1])

    return C, eigVals, L

from ..state import *
from typing import Union
from ..radius import CompactHashMap

def computeRenormalizationMatrices(
  queryParticles: ParticleState,
  domain: DomainDescription,
  kernel: KernelFunctions,
  mode: SupportScheme = SupportScheme.Gather,
  operationMode: OperationDirection = OperationDirection.AllToAll,
  adjacency: Optional[Union[AdjacencyListWarp, CompactHashMap]] = None,   
  referenceState: Optional[ParticleState] = None,   
  returnEigVals: bool = True
):
    if referenceState is None:
        referenceState = queryParticles

    if adjacency is None or isinstance(adjacency, CompactHashMap):
        raise NotImplementedError("Adjacency list must be provided for Renormalization matrices computation. Building a compact hash map and using it as adjacency is not currently supported for this operation.")
    
    C, eigVals, L = computeRenormalizationMatrices_(
        queryParticles.positions, referenceState.positions, 
        queryParticles.supports, referenceState.supports, 
        queryParticles.masses, referenceState.masses, 
        queryParticles.densities, 
        referenceState.densities, 
        queryParticles.kinds if queryParticles.kinds is not None else getCachedDummyTensor((queryParticles.masses.shape[0],), dtype=torch.int32, device=queryParticles.masses.device),
        referenceState.kinds if referenceState.kinds is not None else getCachedDummyTensor((referenceState.masses.shape[0],), dtype=torch.int32, device=referenceState.masses.device),
         domain = domain, adjacency = adjacency,
         mode = mode, kernel = kernel, operationMode = operationMode
    )

    if returnEigVals:
        return C, eigVals, RenormalizationState(renormalizationMatrices = L)
    else:
        return RenormalizationState(renormalizationMatrices = L)