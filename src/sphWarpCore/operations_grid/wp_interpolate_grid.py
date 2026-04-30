import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any, Optional
import torch
from ..utils.wp_autograd import *


from ..radiusSearch.radius_util import AdjacencyList, AdjacencyListWarp, DomainDescription, PointCloud
from ..mathutil.wp_math import *
from ..kernels.wp_kernel import *
from torch.profiler import profile, record_function, ProfilerActivity
from ..utils.wp_util import checkDirectionality_i, checkDirectionality_j

from ..enumTypes import *
from ..radiusSearch.wp_compactHash import CompactHashMap, computeZOrderIndex64, hashGridVec3i
from .grid_util import checkOffset

@wp.func
def computeSPHInterpolation_grid_Func(
    # General Shape Parameters and indices
    i : wp.int32, dim: wp.int32, 

    # SPH properties for the query set (indexed by i)
    queryPositions: wp.array(dtype=vector(dtype = wp.float32, length=Any)), querySupports: wp.array(dtype = wp.float32), queryMasses: wp.array(dtype = wp.float32), queryDensities: wp.array(dtype = wp.float32), queryValues: wp.array(dtype = Any), # type: ignore

    # SPH properties for the reference set (indexed by j in the neighbor loop)
    referencePositions : wp.array(dtype=vector(length=Any, dtype = wp.float32)), referenceSupports : wp.array(dtype = wp.float32), referenceMasses: wp.array(dtype = wp.float32), referenceDensities: wp.array(dtype = wp.float32), referenceValues: wp.array(dtype = Any), # type: ignore
    
    # Domain and kernel parameters
    periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), # type: ignore
    mode_uint: wp.uint32, kernel_int: wp.int32, 
    
    # Neighbor list data, pre accessed to avoid gradient issues with dynamic for loops
    cellStartIndex: wp.int32, cellParticleCount: wp.int32, 
    sortIndex: wp.array(dtype = wp.int64), # type: ignore
        
    # Operation Mode for masking certain kinds of interactions, e.g. for directional operations
    opInt: wp.int32, queryKinds : wp.array(dtype = wp.int32), referenceKinds : wp.array(dtype = wp.int32), # type: ignore

    # Optional Correction Terms:
    # Whether to use actual volume (mass/density) or apparent volume for the gradient computation, and the corresponding volumes if needed.
    useVolume: wp.bool, queryVolumes: wp.array(dtype = wp.float32), referenceVolumes: wp.array(dtype = wp.float32), # type: ignore
    # Whether to use CRK kernel correction for the computation, and the corresponding correction terms if needed.
    useCRK: wp.bool, queryA: wp.array(dtype = wp.float32), queryB: wp.array(dtype = vector(length=Any, dtype=wp.float32)), # type: ignore
    
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
    Ai      = queryA[i] if useCRK else type(queryA[0])(0.0)
    Bi      = queryB[i] if useCRK else type(queryB[0])(0.0)
    
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

        fv = referenceValues[j]

        vj = referenceMasses[j] / referenceDensities[j] if not useVolume else referenceVolumes[j]

        w_ij = computeKernelCRK(
            xi, referencePositions[j], 
            hi, referenceSupports[j], 
            kernel_int, mode_uint, periodicity, domainMin, domainMax,
            useCRK, Ai, Bi
        )

        out += fv * vj * w_ij
            
    return out

@wp.kernel
def computeSPHInterpolation_Kernel(
    queryPositions : wp.array(dtype = vector(length=Any, dtype=wp.float32)), referencePositions : wp.array(dtype=vector(length=Any, dtype=wp.float32)), # type: ignore
    querySupports : wp.array(dtype = wp.float32), referenceSupports : wp.array(dtype = wp.float32), # type: ignore
    queryMasses: wp.array(dtype = wp.float32), referenceMasses: wp.array(dtype = wp.float32),  # type: ignore
    queryDensities: wp.array(dtype = wp.float32), referenceDensities: wp.array(dtype = wp.float32), # type: ignore
    queryValues: wp.array(dtype = Any), referenceValues: wp.array(dtype = Any), # type: ignore
    
    domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), periodicity : wp.array(dtype = wp.bool), # type: ignore
    
    mode_uint: wp.uint32, kernel_int : wp.int32,
    
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

    opInt: wp.int32, queryKinds : wp.array(dtype = wp.int32), referenceKinds : wp.array(dtype = wp.int32), # type: ignore
    
    useVolume: wp.bool, queryVolumes: wp.array(dtype = wp.float32), referenceVolumes: wp.array(dtype = wp.float32), # type: ignore
    useCRK: wp.bool, crk_A: wp.array(dtype = wp.float32), crk_B: wp.array(dtype = vector(length=Any, dtype=wp.float32)), # type: ignore
    
    outputValues : wp.array(dtype = Any) # type: ignore
):                                                                    
    i = wp.tid()
    if i >= queryPositions.shape[0]:
        return
    
    out_value = type(outputValues[0])(0.0)

    for o in range(numOffsets):
        cellStartIndex, cellParticleCount = checkOffset(
            i, queryPositions, numCells, D, 
            o, cellOffsets, hashTable, cellTable,
            periodicity, qMin, qMax, hCell
        )
        if cellStartIndex < 0:
            continue

        out_value += computeSPHInterpolation_grid_Func(
            i, D,
            queryPositions, querySupports, queryMasses, queryDensities, queryValues,
            referencePositions, referenceSupports, referenceMasses, referenceDensities, referenceValues,
            periodicity, domainMin, domainMax,
            mode_uint, kernel_int,
            cellStartIndex, cellParticleCount, sortIndex,
            opInt, queryKinds, referenceKinds,
            useVolume, queryVolumes, referenceVolumes,
            useCRK, crk_A, crk_B,
            type(outputValues[0])(0.0)
        )
    outputValues[i] = out_value
    
    

def computeSPHInterpolant_grid_warpBackend(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,
    queryDensities, referenceDensities,
    queryValues, referenceValues,
    queryKinds, referenceKinds,
    domain: DomainDescription,
    mode: SupportScheme,
    kernel: KernelFunctions,    
    operationMode: OperationDirection,
    datastructure: CompactHashMap,
    useVolume: bool = False, queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    useCRK: bool = False, crk_A: Optional[torch.Tensor] = None, crk_B: Optional[torch.Tensor] = None,
):
    with record_function("warpSPH[Interpolation]"):
        with record_function("warpSPH[Interpolation] - Preprocessing"):
            domainMin = domain.min
            domainMax = domain.max
            periodicity = domain.periodic

            preScatteredQuantities = False
            qV = queryValues
            rV = referenceValues

            # Warp kernels only support rank-1 (vector) and rank-2 (matrix) field types.
            # For higher-rank inputs (e.g. shape (n, p, m, d)) we flatten the field
            # dimensions to a single vector dimension, interpolate, then restore the
            # original shape.  Rank <= 3 inputs (scalar / vector / matrix per particle)
            # pass through unchanged.
            field_shape = qV.shape[1:]   # all dims after the particle batch dim
            needs_flatten = qV.dim() > 3
            if needs_flatten:
                flat_len = qV[0].numel()
                qV = qV.reshape(qV.shape[0], flat_len).contiguous()
                rV = rV.reshape(rV.shape[0], flat_len).contiguous()

            
            modeUint = wp.uint32(mode.value)
            kernelInt = wp.int32(kernel.value)
            outputShape = qV.shape[0]
            opInt = wp.int32(operationMode.value)
            
            wpValues = castTorchToWarpAsBuiltins(qV)
        with record_function("warpSPH[Interpolation] - Kernel Launch"):
            D = queryPositions.shape[1]
            warp_interpolation = warpWrapper(
                launch_kernel, computeSPHInterpolation_Kernel, outputShape, wpValues.dtype, 
                queryPositions, referencePositions,
                querySupports, referenceSupports,
                queryMasses, referenceMasses,
                queryDensities, referenceDensities,
                qV, rV,
                
                domainMin, domainMax, periodicity,
                modeUint,
                kernelInt,
                
                datastructure.sortIndex,
                datastructure.qMin, datastructure.qMax, datastructure.hCell,
                datastructure.numCells, datastructure.hashTable, datastructure.sortedCellTable, D,
                datastructure.numOffsets, datastructure.cellOffsets,
            
                opInt, queryKinds, referenceKinds,

                wp.bool(useVolume), queryVolumes, referenceVolumes,
                wp.bool(useCRK), crk_A, crk_B,
            )

            if needs_flatten:
                warp_interpolation = warp_interpolation.reshape(
                    warp_interpolation.shape[0], *field_shape
                )

    return warp_interpolation