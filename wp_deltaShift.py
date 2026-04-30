import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from sphWarpCore import *


@wp.func
def computeDeltaShift_Func_i(
    # General Shape Parameters and indices
    i : wp.int32,  dim: wp.int32, 

    # SPH properties for the query set (indexed by i)
    xi: vector(dtype = wp.float32, length=Any), hi: wp.float32, mi: wp.float32, rhoi: wp.float32, # type: ignore

    # SPH properties for the reference set (indexed by j in the neighbor loop)
    referenceState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.

    # Domain and kernel parameters
    # periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), # type: ignore
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
    useGradientRenormalization: wp.bool, Li: matrix(shape=(Any, Any), dtype=wp.float32), # type: ignore
    # Grad-h correction terms for each query and reference point, used for correcting the kernel gradient based on the local particle distribution and smoothing length variations.
    useGradHTerms: wp.bool, omega_i: wp.float32, referenceOmegas: wp.array(dtype = wp.float32),  # type: ignore
    # Whether to use actual volume (mass/density) or apparent volume for the gradient computation, and the corresponding volumes if needed.
    useVolume: bool, Vi: wp.float32, referenceVolumes: wp.array(dtype = wp.float32), # type: ignore
    # Whether to use CRK kernel correction for the computation, and the corresponding correction terms if needed.
    useCRK: bool, Ai: wp.float32, Bi: vector(length=Any, dtype=wp.float32), gradAi: vector(length=Any, dtype=wp.float32), gradBi: matrix(shape=(Any, Any), dtype=wp.float32), # type: ignore
    correctionData: Any, # correctionData_1 or correctionData_2 or correctionData_3, containing all the optional correction terms and their usage flags

    # Dummy value to allow allocation
    outputValue: Any, # type: ignore

    # DeltaShift function parameters begin here
    R: float, n: int, CFL: float, computeMach: bool, c_max: float,
    rho0: float, dx: float, 

):
    # Initialize the output value
    out     = zero_like_warp(outputValue)
    
    # # Loop over neighbors to compute the gradient contribution from each neighbor    
    for neighborIndex in range(numIndices):
        jj = beginIndex + neighborIndex
        j  = wp.int32(offsetArray[jj])
        if opInt != 0:
            if not checkDirectionality_j(referenceKinds[j], opInt):
                return out * 0.0
        ##########################################################
        #   The core particle-particle interaction starts here   #
        ##########################################################
        
        xj, hj, mj, rhoj, kj = getParticle(referenceState, j)

        apparentVolume = mj / rhoj if not useVolume else referenceVolumes[j]
        # DeltaShift Functionality begins here        
        w_ij = computeKernelCRK(
            xi, xj, 
            hi, hj, 
            kernel_int, mode_uint, domainState.periodicity, domainState.domainMin, domainState.domainMax,
            useCRK, Ai, Bi
        )
        gradw_ij = computeKernelGradientCRK(
            xi, xj, 
            hi, hj,
            kernel_int, mode_uint, domainState.periodicity, domainState.domainMin, domainState.domainMax,
            useCRK, Ai, Bi, gradAi, gradBi
        )
        # Alternative crk correction scheme for reference
        # useCRK, Aj, Bj, gradAj, gradBj = getCRK_i(correctionData, i)
        # gradw_ji = computeKernelGradientCRK(
        #     xj, xi, 
        #     hj, hi,
        #     kernel_int, mode_uint, domainState.periodicity, domainState.domainMin, domainState.domainMax,
        #     useCRK, Aj, Bj, gradAj, gradBj
        # )
        # gradw_ij = (gradw_ij - gradw_ji) * 0.5
        if useGradientRenormalization:
            gradw_ij = matmul(Li, gradw_ij)

        ### GENERIC CODE STOPS HERE ###

        dx = wp.pow(mj / rho0, 1.0 / wp.float32(dim))
        
        dx_ = dx / eval_kernelScale(kernel_int, dim)
        
        x_ij = computeDistanceVec(xi, xj, domainState.periodicity, domainState.domainMin, domainState.domainMax)
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))

        hij = computePairwiseSupport(hi, hj, mode_uint)
        q = dx_ / hij
        W_0 = eval_k(q, dim, kernel_int) * eval_C_d(dim, kernel_int) / iPow(hij, dim)
        k = w_ij / W_0

        term = (1.0 + R * wp.pow(k, wp.float32(n)))
        densityTerm = mj / (rhoi + rhoj)

        phi_ij = 1.0        
        scalarTerm = term * densityTerm * phi_ij

        shiftAmount = scalarTerm * gradw_ij

        Ma = wp.float32(0.1)
        if computeMach:
            Ma = c_max
        h2 = (hi / eval_kernelScale(kernel_int, dim) * 2.0)
        shiftScaling = -CFL * Ma * h2 * h2
        
        out += shiftScaling * shiftAmount
    return out

from sphWarpCore.operations_grid.grid_util import checkOffset

@wp.func
def computeDeltaShift_Func_Adjacency(
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

    R: float, n: int, CFL: float, computeMach: bool, c_max: float,
    rho0: float, dx: float,
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if opInt != 0:
        if not checkDirectionality_i(ki, opInt):
            return zero_like_warp(outputValue)
        
    useGradientRenormalization, Li = getL_i(correctionData, i)
    useGradHTerms, omega_i = getGradH_i(correctionData, i)
    useVolume, Vi = getVolume_i(correctionData, i)
    useCRK, Ai, Bi, gradA_i, gradB_i = getCRK_i(correctionData, i)

    out = type(outputValue)() * 0.0
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
        
        out += computeDeltaShift_Func_i(
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

            # DeltaShift function parameters
            R, n, CFL, computeMach, c_max,
            rho0, dx,
        )
    return out



@wp.kernel
def computeDeltaShift_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    
    mode_uint: wp.uint32, kernel_int : wp.int32, gradientMode_int: wp.int32, opInt: wp.int32,
    # Do not change the parameters above

    R: wp.float32, n: wp.int32, CFL: wp.float32, computeMach: wp.bool, c_max: wp.float32,
    rho0: wp.float32, dx: wp.float32,
    
    # The last parameter is always the output array and should not be changed
    outputValues : wp.array(dtype = Any) # type: ignore
):                                                                                    
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeDeltaShift_Func_Adjacency(
        i, domainState.dim, 
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        mode_uint, kernel_int, gradientMode_int,  opInt, #queryKinds, referenceKinds,
        # The parameters above are default parameters and shold not be changed

        zero_like_warp(outputValues),

        R, n, CFL, computeMach, c_max,
        rho0, dx,
    )

def computeDeltaShiftWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,

    CFL: float, computeMach: bool, c_max: float,
    rho0: float, dx: float, 

    R: float = 0.25, n: int = 4,
    
    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
):
    with record_function("warpSPH[CRKVolume]"):
        outputSize  = queryParticles.positions.shape[0]
        outputDtype = castTorchToWarpAsBuiltins(queryParticles.positions).dtype

        return warpWrapper2(
            launcher = launch_kernel,
            kernel   = computeDeltaShift_Kernel,
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
                R, n, CFL, computeMach, c_max,
                rho0, dx,
            ),
        )

