import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any
import torch

from sphWarpCore.operations_grid.grid_util import checkOffset
from ..state import GradHState, RenormalizationState
from ..utils.wp_autograd import *


from ..radiusSearch.radius_util import AdjacencyList, AdjacencyListWarp, DomainDescription, PointCloud
from ..mathutil.wp_math import *
from ..kernels.wp_kernel import *
from ..utils.wp_util import _get_warp_matrix_dtype, getCachedDummyTensor, checkDirectionality_i, checkDirectionality_j
from torch.profiler import profile, record_function, ProfilerActivity

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from sphWarpCore import *


# For matrices we need to implement the logic manually using outer products, since Warp does not support rank-2 field types natively. The output is stored as a flattened vector and reshaped on the Python side.

@wp.func
def computeCovariance_Func_i(
    # General Shape Parameters and indices
    i : wp.int32,  dim: wp.int32, 

    # SPH properties for the query set (indexed by i)
    xi: vector(dtype = scalar_t, length=Any), hi: scalar_t, mi: scalar_t, rhoi: scalar_t, # type: ignore

    # SPH properties for the reference set (indexed by j in the neighbor loop)
    referenceState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.

    # Domain and kernel parameters
    # periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = scalar_t), domainMax : wp.array(dtype = scalar_t), # type: ignore
    domainState: domainData,
    mode_uint: wp.uint32, kernel_int: wp.int32, 
    
    # Operation specific parameters
    gradientMode_int: wp.int32, # type: ignore
            
    beginIndex: wp.int32, # type: ignore
    numIndices: wp.int32, # type: ignore
    offsetArray: wp.array(dtype = wp.int64), # type: ignore

    # Operation Mode for masking certain kinds of interactions, e.g. for directional operations
    opInt: wp.int32, ki : wp.int32, referenceKinds : wp.array(dtype = wp.int32), # type: ignore

    # Optional Correction Terms:
    # Gradient renormalization matrices for each query point, used for correcting the kernel gradient based on the local particle distribution.
    useGradientRenormalization: wp.bool, Li: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    # Grad-h correction terms for each query and reference point, used for correcting the kernel gradient based on the local particle distribution and smoothing length variations.
    useGradHTerms: wp.bool, omega_i: scalar_t, referenceOmegas: wp.array(dtype = scalar_t),  # type: ignore
    # Whether to use actual volume (mass/density) or apparent volume for the gradient computation, and the corresponding volumes if needed.
    useVolume: bool, Vi: scalar_t, referenceVolumes: wp.array(dtype = scalar_t), # type: ignore
    # Whether to use CRK kernel correction for the computation, and the corresponding correction terms if needed.
    useCRK: bool, Ai: scalar_t, Bi: vector(length=Any, dtype=scalar_t), gradAi: vector(length=Any, dtype=scalar_t), gradBi: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    correctionData: Any, # correctionData_1 or correctionData_2 or correctionData_3, containing all the optional correction terms and their usage flags

    # Dummy value to allow allocation
    outputValue: Any, # type: ignore
):
    # Initialize the output value
    out     = zero_like_warp(outputValue)
    # # Loop over neighbors to compute the gradient contribution from each neighbor    
    for neighborIndex in range(numIndices):
        jj = beginIndex + neighborIndex
        j  = wp.int32(offsetArray[jj])
        if opInt != 0:
            if not checkDirectionality_j(referenceKinds[j], opInt):
                return out * scalar_t(0.0)
        ##########################################################
        #   The core particle-particle interaction starts here   #
        ##########################################################
        
        xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
        apparentVolume = mj / rhoj if not useVolume else referenceVolumes[j]

        fij = -computeDistanceVec(xi, xj, domainState.periodicity, domainState.domainMin, domainState.domainMax)
        
        kernelGradient = computeKernelGradientCRK(
            xi, xj, 
            hi, hj,
            kernel_int, mode_uint, domainState.periodicity, domainState.domainMin, domainState.domainMax,
            useCRK, Ai, Bi, gradAi, gradBi
        )

        if useGradientRenormalization:
            kernelGradient = matmul(Li, kernelGradient)        

        out += wp.outer(fij * apparentVolume, kernelGradient)
            
    return out


@wp.func
def computeCovariance_Func_Adjacency(
    i : wp.int32, dim: wp.int32, 

    queryState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.
    referenceState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.
    correctionData: Any, # correctionData_1 or correctionData_2 or correctionData_3, containing all the optional correction terms and their usage flags

    domainState: domainData,
    useAdjacency: wp.bool,
    adjacencyState: adjacencyData,
    gridState: gridData,
    numOffsets: wp.int32,

    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32, opInt: wp.int32, 
    
    outputValue : Any, # type: ignore
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if opInt != 0:
        if not checkDirectionality_i(ki, opInt):
            return zero_like_warp(outputValue)
        
    useGradientRenormalization, Li = getL_i(correctionData, i)
    useGradHTerms, omega_i = getGradH_i(correctionData, i)
    useVolume, Vi = getVolume_i(correctionData, i)
    useCRK, Ai, Bi, gradA_i, gradB_i = getCRK_i(correctionData, i)

    out = type(outputValue)() * scalar_t(0.0)
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
        
        out += computeCovariance_Func_i(
            i, dim, 
            xi, hi, mi, rhoi,
            referenceState, domainState,
            mode_uint, kernel_int, gradientMode_int,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            opInt, ki, referenceState.kinds,

            useGradientRenormalization, Li,
            useGradHTerms, omega_i, correctionData.referenceOmegas,
            useVolume, Vi , correctionData.referenceVolumes,
            useCRK, Ai, Bi, gradA_i, gradB_i,
            correctionData,
            

            outputValue,
        )
    return out

@wp.kernel
def computeCovariance_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    
    mode_uint: wp.uint32, kernel_int : wp.int32, gradientMode_int: wp.int32, opInt: wp.int32,
    # Do not change the parameters above
    
    # The last parameter is always the output array and should not be changed
    outputValues : wp.array(dtype = Any) # type: ignore
):                                                                                    
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeCovariance_Func_Adjacency(
        i, domainState.dim, 
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        mode_uint, kernel_int, gradientMode_int,  opInt, #queryKinds, referenceKinds,
        # The parameters above are default parameters and shold not be changed

        zero_like_warp(outputValues),
    )

from ..enumTypes import *
from typing import Optional



from ..math import outerTensorProduct

import torch
from torch.profiler import record_function



@torch.compile
def pinv2x2(M):
    with record_function('Pseudo Inverse 2x2'):
        a = M[:,0,0]
        b = M[:,0,1]
        c = M[:,1,0]
        d = M[:,1,1]

        theta = (0.5) * torch.atan2(2 * a * c + 2 * b * d, a**2 + b**2 - c**2 - d**2)
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

        phi = (0.5) * torch.atan2(2 * a * b + 2 * c * d, a**2 - b**2 + c**2 - d**2)
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

from warp.types import *

@wp.func
def matmul2(
    A: matrix(shape=(2,2), dtype=scalar_t),
    B: matrix(shape=(2,2), dtype=scalar_t)
):
    out = zero_like_warp(A)
    out[0,0] = A[0,0] * B[0,0] + A[0,1] * B[1,0]
    out[0,1] = A[0,0] * B[0,1] + A[0,1] * B[1,1]
    out[1,0] = A[1,0] * B[0,0] + A[1,1] * B[1,0]
    out[1,1] = A[1,0] * B[0,1] + A[1,1] * B[1,1]
    return out

@wp.kernel
def pinv2x2_warp(
    C: wp.array(dtype=matrix(shape=(2,2), dtype=scalar_t)), # type: ignore
    L: wp.array(dtype=matrix(shape=(2,2), dtype=scalar_t)), # type: ignore
    EV: wp.array(dtype=vector(length=2, dtype=scalar_t)), # type: ignore
    num_nbrs: wp.array(dtype=wp.int32)  # type: ignore
):
    i = wp.tid()
    a = C[i][0,0]
    b = C[i][0,1]
    c = C[i][1,0]
    d = C[i][1,1]

    if num_nbrs[i] < 4:
        L[i][0,0] = 1.0
        L[i][0,1] = 0.0
        L[i][1,0] = 0.0
        L[i][1,1] = 1.0
        EV[i][0] = 1.0
        EV[i][1] = 1.0
        return

    theta = (0.5) * wp.atan2(2.0 * a * c + 2.0 * b * d, a*a + b*b - c*c - d*d)
    cosTheta = wp.cos(theta)
    sinTheta = wp.sin(theta)
    U = zero_like_warp(C)
    U[0,0] = cosTheta
    U[0,1] = - sinTheta
    U[1,0] = sinTheta
    U[1,1] = cosTheta

    S1 = a*a + b*b + c*c + d*d
    S2 = wp.sqrt((a*a + b*b - c*c - d*d)**2.0 + 4.0* (a * c + b *d)**2.0)

    o1 = wp.sqrt((S1 + S2) / 2.0)
    o2 = wp.sqrt(wp.clamp(S1 - S2, low = 1e-9, high = 1e9) / 2.0)

    phi = (0.5) * wp.atan2(2.0 * a * b + 2.0 * c * d, a*a - b*b + c*c - d*d)
    cosPhi = wp.cos(phi)
    sinPhi = wp.sin(phi)
    s11 = wp.sign((a * cosTheta + c * sinTheta) * cosPhi + ( b * cosTheta + d * sinTheta) * sinPhi)
    s22 = wp.sign((a * sinTheta - c * cosTheta) * sinPhi + (-b * sinTheta + d * cosTheta) * cosPhi)

    V = zero_like_warp(C)
    V[0,0] = cosPhi * s11
    V[0,1] = - sinPhi * s22
    V[1,0] = sinPhi * s11
    V[1,1] = cosPhi * s22

    eigVals = zero_like_warp(EV)
    eigVals[0] = o1
    eigVals[1] = o2 

    EV[i] = eigVals
    

        # o1_1[torch.abs(eigVals[:,0]) > 1e-7] = 1 / eigVals[torch.abs(eigVals[:,0]) > 1e-7, 0] 
        # o2_1[torch.abs(eigVals[:,1]) > 1e-7] = 1 / eigVals[torch.abs(eigVals[:,1]) > 1e-7, 1] 
    o1_1 = 0.0 if wp.abs(eigVals[0]) <= 1e-7 else 1.0 / eigVals[0]
    o2_1 = 0.0 if wp.abs(eigVals[1]) <= 1e-7 else 1.0 / eigVals[1]

    S_1 = zero_like_warp(C)
    S_1[0,0] = o1_1
    S_1[1,1] = o2_1

    L[i] = matmul2(matmul2(V, S_1), wp.transpose(U))

def pinv2x2_warpBackend(
    C: torch.Tensor,
    num_nbrs: torch.Tensor
):
    mat_warp = castTorchToWarpAsBuiltins(C)
    inv = torch.empty_like(C)
    evs = torch.empty((C.shape[0], 2), device = C.device, dtype = C.dtype)

    inv_warp = castTorchToWarpAsBuiltins(inv)
    evs_warp = castTorchToWarpAsBuiltins(evs)
    nnbrs = castTorchToWarpAsBuiltins(num_nbrs)
    wp.launch(kernel=pinv2x2_warp, dim=mat_warp.shape[0], inputs=[mat_warp, inv_warp, evs_warp, nnbrs])
    return inv, evs

# invs, evs = pinv2x2_warpBackend(randomMat)

from ..warp_state_util import warpWrapper2

def computeRenormalizationMatrices_(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    
    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
):
    with record_function("[warpSPH] - Renorm - Compute Covariance"):
        outputSize  = queryParticles.positions.shape[0]
        outputDtype = _get_warp_matrix_dtype(queryParticles.positions.shape[1], queryParticles.positions.shape[1], queryParticles.positions.dtype)

        C = warpWrapper2(
            launcher = launch_kernel,
            kernel   = computeCovariance_Kernel,
            outputSizes  = outputSize,
            outputDtypes = outputDtype,
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
            ),
        )
    with record_function("[warpSPH] - Renorm - Covariance Postprocess"):
        num_nbrs = adjacency.numNeighbors
        dtype = C.dtype

        queryPositions = queryParticles.positions
        dim = queryPositions.shape[1]
        dtype = C.dtype
        device = queryPositions.device

        # C[num_nbrs < 4,:,:] = torch.eye(dim, dtype = dtype, device = device)[None,:,:]

    with record_function("[warpSPH] - Renorm - Pseudo Inverse"):
        if queryPositions.shape[1] == 2:
            L, eigVals = pinv2x2_warpBackend(C, num_nbrs)
        else:
            L = torch.linalg.pinv(C)
            eigVals = torch.linalg.eigvals(C).real

            # print(f"Renormalization matrices computed. C shape: {C.shape}, L shape: {L.shape}, eigVals shape: {eigVals.shape}")

            if queryPositions.shape[1] == 3:
                eigVals[torch.abs(eigVals[:,1]) > torch.abs(eigVals[:,0]),:] = torch.flip(eigVals[torch.abs(eigVals[:,1]) > torch.abs(eigVals[:,0]),:],[1])
                eigVals[torch.abs(eigVals[:,2]) > torch.abs(eigVals[:,1]),:] = torch.flip(eigVals[torch.abs(eigVals[:,2]) > torch.abs(eigVals[:,1]),:],[1])
                eigVals[torch.abs(eigVals[:,2]) > torch.abs(eigVals[:,0]),:] = torch.flip(eigVals[torch.abs(eigVals[:,2]) > torch.abs(eigVals[:,0]),:],[1])
            elif queryPositions.shape[1] == 2:
                eigVals[torch.abs(eigVals[:,1]) > torch.abs(eigVals[:,0]),:] = torch.flip(eigVals[torch.abs(eigVals[:,1]) > torch.abs(eigVals[:,0]),:],[1]) 

    return C, eigVals, L

from ..state import *
from typing import Union
from ..radius import CompactHashMap

def computeRenormalizationMatrices(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    
    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
  returnEigVals: bool = True
):
    with record_function("[warpSPH] - computeRenormalizationMatrices"):
        if adjacency is None or isinstance(adjacency, CompactHashMap):
            raise NotImplementedError("Adjacency list must be provided for Renormalization matrices computation. Building a compact hash map and using it as adjacency is not currently supported for this operation.")
        
        C, eigVals, L = computeRenormalizationMatrices_(
            queryParticles, operationProperties, domain,
            queryVolumes = queryVolumes, referenceVolumes = referenceVolumes,
            adjacency = adjacency,
            referenceParticles = referenceParticles,
            crkState = crkState,
            gradHState = gradHState,
            renormalizationState = renormalizationState
        )

        if returnEigVals:
            return C, eigVals, RenormalizationState(renormalizationMatrices = L)
        else:
            return RenormalizationState(renormalizationMatrices = L)