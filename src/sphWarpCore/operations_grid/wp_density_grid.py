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
def computeSPHDensity_grid_Func(
    # General Shape Parameters and indices
    i : wp.int32, dim: wp.int32, 

    # SPH properties for the query set (indexed by i)
    queryPositions: wp.array(dtype=vector(dtype = wp.float32, length=Any)), querySupports: wp.array(dtype = wp.float32), queryMasses: wp.array(dtype = wp.float32), # type: ignore

    # SPH properties for the reference set (indexed by j in the neighbor loop)
    referencePositions : wp.array(dtype=vector(length=Any, dtype = wp.float32)), referenceSupports : wp.array(dtype = wp.float32), referenceMasses: wp.array(dtype = wp.float32), # type: ignore
    
    # Domain and kernel parameters
    periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), # type: ignore
    mode_uint: wp.uint32, kernel_int: wp.int32, 
    
    # Neighbor list data, pre accessed to avoid gradient issues with dynamic for loops
    cellStartIndex: wp.int32, cellParticleCount: wp.int32, 
    sortIndex: wp.array(dtype = wp.int64), # type: ignore
        
    # Operation Mode for masking certain kinds of interactions, e.g. for directional operations
    opInt: wp.int32, queryKinds : wp.array(dtype = wp.int32), referenceKinds : wp.array(dtype = wp.int32), # type: ignore

    # Optional Correction Terms:
):
    if opInt != 0:
        if not checkDirectionality_i(queryKinds[i], opInt):
            return wp.float32(0.0)
    # Unpack query point properties
    xi      = queryPositions[i]
    hi      = querySupports[i]
    # mi      = queryMasses[i] # Generally not needed
    
    # Initialize the output value
    out = wp.float32(0.0)
    
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

        out += referenceMasses[j] * sphKernel(xi, referencePositions[j], hi, referenceSupports[j], kernel_int, mode_uint, periodicity, domainMin, domainMax) 
            
    return out

@wp.kernel
def computeSPHDensity_grid_Kernel(
    queryPositions : wp.array(dtype = vector(length=Any, dtype=wp.float32)), referencePositions : wp.array(dtype=vector(length=Any, dtype=wp.float32)), # type: ignore
    querySupports : wp.array(dtype = wp.float32), referenceSupports : wp.array(dtype = wp.float32), # type: ignore
    queryMasses: wp.array(dtype = wp.float32), referenceMasses: wp.array(dtype = wp.float32),  # type: ignore
    
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
    
    outputValues : wp.array(dtype = Any) # type: ignore
):                                                                                    
    i = wp.tid()
    if i >= queryPositions.shape[0]:
        return
    
    out_value = wp.float32(0.0)

    for o in range(numOffsets):
        cellStartIndex, cellParticleCount = checkOffset(
            i, queryPositions, numCells, D, 
            o, cellOffsets, hashTable, cellTable,
            periodicity, qMin, qMax, hCell
        )
        if cellStartIndex < 0:
            continue

        out_value +=  computeSPHDensity_grid_Func(
        i, get_dim(queryPositions), 

        queryPositions, querySupports, queryMasses, 
        referencePositions, referenceSupports, referenceMasses, 

        periodicity, domainMin, domainMax, 
        mode_uint, kernel_int,

        cellStartIndex, cellParticleCount, sortIndex,
        
        
        opInt, queryKinds, referenceKinds,
    )
    outputValues[i] = out_value
    
    

from ..enumTypes import *

def computeSPHDensity_grid_warpBackend(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,
    queryKinds, referenceKinds,
    domain: DomainDescription,
    mode: SupportScheme,
    kernel: KernelFunctions,    
    operationMode: OperationDirection,
    datastructure: CompactHashMap,
):
    with record_function("warpSPH[Density]"):
        with record_function("warpSPH[Density] - Preprocessing"):
            domainMin = domain.min
            domainMax = domain.max
            periodicity = domain.periodic

            modeUint = wp.uint32(mode.value)
            kernelInt = wp.int32(kernel.value)
            outputShape = queryPositions.shape[0]
            opInt = wp.int32(operationMode.value)
            
            wpValues = castTorchToWarpAsBuiltins(queryMasses)
        with record_function("warpSPH[Density] - Kernel Launch"):

            D = queryPositions.shape[1]
            warp_interpolation = warpWrapper(
                launch_kernel, computeSPHDensity_grid_Kernel, outputShape, wpValues.dtype, 
                queryPositions, referencePositions,
                querySupports, referenceSupports,
                queryMasses, referenceMasses,
                domainMin, domainMax, periodicity,
                modeUint,
                kernelInt,
                
                datastructure.sortIndex,
                datastructure.qMin, datastructure.qMax, datastructure.hCell,
                datastructure.numCells, datastructure.hashTable, datastructure.sortedCellTable, D,
                datastructure.numOffsets, datastructure.cellOffsets,
                opInt, queryKinds, referenceKinds,
            )

    return warp_interpolation