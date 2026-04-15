import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any, Optional
import torch
from ..utils.wp_autograd import *
from ..radiusSearch.radius_util import convertModeToUint

from ..radiusSearch.radius_util import AdjacencyList, AdjacencyListWarp, DomainDescription, PointCloud
from ..mathutil.wp_math import *
from ..kernels.wp_kernel import *
from torch.profiler import profile, record_function, ProfilerActivity
from ..utils.wp_util import checkDirectionality_i, checkDirectionality_j


@wp.func
def computeSPHDensity_Func(
    xi: vector(dtype = wp.float32, length=Any), hi : wp.float32, mi: wp.float32, # type: ignore
    positions : wp.array(dtype=vector(length=Any, dtype = wp.float32)), supports : wp.array(dtype = wp.float32), masses: wp.array(dtype = wp.float32),  # type: ignore
    
    periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), # type: ignore
    mode_uint: wp.uint32, kernel_int: wp.int32,
    
    neighborList: wp.array(dtype = wp.int64), # type: ignore
    neighborOffset : wp.int32, numNeighs: wp.int32, 
    
    opInt: wp.int32, referenceKinds : wp.array(dtype = wp.int32) # type: ignore
):
    f_interpolated = type(mi)(0.0)
    
    for neighborIndex in range(numNeighs):
        jj = neighborOffset + neighborIndex
        j = wp.int32(neighborList[jj])

        if opInt != 0:
            if not checkDirectionality_j(referenceKinds[j], opInt):
                continue

        f_interpolated += masses[j] * sphKernel(xi, positions[j], hi, supports[j], kernel_int, mode_uint, periodicity, domainMin, domainMax) 
            
    return f_interpolated

@wp.kernel
def computeSPHDensity_Kernel(
    queryPositions : wp.array(dtype = vector(length=Any, dtype=wp.float32)), referencePositions : wp.array(dtype=vector(length=Any, dtype=wp.float32)), # type: ignore
    querySupports : wp.array(dtype = wp.float32), referenceSupports : wp.array(dtype = wp.float32), # type: ignore
    queryMasses: wp.array(dtype = wp.float32), referenceMasses: wp.array(dtype = wp.float32),  # type: ignore
    domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), periodicity : wp.array(dtype = wp.bool), # type: ignore
    
    mode_uint: wp.uint32, kernel_int : wp.int32,
    neighborList: wp.array(dtype = wp.int64), neighborListRowOffsets: wp.array(dtype = wp.int32), numNeighbors: wp.array(dtype = wp.int32),  # type: ignore
    opInt: wp.int32, queryKinds : wp.array(dtype = wp.int32), referenceKinds : wp.array(dtype = wp.int32), # type: ignore
    
    outputValues : wp.array(dtype = Any) # type: ignore
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
    
    neighborOffset = neighborListRowOffsets[i]
    numNeighs = numNeighbors[i]
    
    outputValues[i] = computeSPHDensity_Func(
        xi, hi, mi,
        referencePositions, referenceSupports, referenceMasses,
        periodicity, domainMin, domainMax, mode_uint, kernel_int,
        neighborList, neighborOffset, wp.int32(numNeighs),
        opInt, referenceKinds
    )
    
    

from ..enumTypes import *

def computeSPHDensity_warpBackend(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,
    queryKinds, referenceKinds,
    domain: DomainDescription,
    mode: SupportScheme,
    kernel: KernelFunctions,    
    operationMode: OperationDirection,
    adjacency,
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

            warp_interpolation = warpWrapper(
                launch_kernel, computeSPHDensity_Kernel, outputShape, wpValues.dtype, 
                queryPositions, referencePositions,
                querySupports, referenceSupports,
                queryMasses, referenceMasses,
                domainMin, domainMax, periodicity,
                modeUint,
                kernelInt,
                
                adjacency.j, adjacency.edgeOffsets, adjacency.numNeighbors,
                opInt, queryKinds, referenceKinds,
            )

    return warp_interpolation