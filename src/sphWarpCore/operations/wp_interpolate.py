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
def computeSPHInterpolation_Func(
    xi: vector(dtype = wp.float32, length=Any), hi : wp.float32, mi: wp.float32, rhoi: wp.float32, # type: ignore
    positions : wp.array(dtype=vector(length=Any, dtype = wp.float32)), supports : wp.array(dtype = wp.float32), masses: wp.array(dtype = wp.float32), densities: wp.array(dtype = wp.float32), # type: ignore
    
    periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), # type: ignore
    mode_uint: wp.uint32, kernel_int: wp.int32,
    
    neighborList: wp.array(dtype = wp.int64), # type: ignore
    neighborOffset : wp.int32, numNeighs: wp.int32, preScatteredQuantities: wp.bool,
    
    fi: Any,
    fieldValues: wp.array(dtype = Any), # type: ignore
    opInt: wp.int32, referenceKinds : wp.array(dtype = wp.int32), # type: ignore
    useApparentVolume: wp.bool, referenceVolumes: wp.array(dtype = wp.float32), # type: ignore
    crkCorrection: wp.bool, Ai: wp.float32, Bi: vector(length=Any, dtype=wp.float32), # type: ignore
):
    f_interpolated = type(fi)(0.0)
    
    for neighborIndex in range(numNeighs):
        jj = neighborOffset + neighborIndex
        j = wp.int32(neighborList[jj])
        if opInt != 0:
            if not checkDirectionality_j(referenceKinds[j], opInt):
                continue

        fv = fieldValues[jj] if preScatteredQuantities else fieldValues[j]

        vj = masses[j] / densities[j] if not useApparentVolume else referenceVolumes[j]

        w_ij = computeKernelCRK(
            xi, positions[j], 
            hi, supports[j], 
            kernel_int, mode_uint, periodicity, domainMin, domainMax,
            crkCorrection, Ai, Bi
        )

        # w_ij = sphKernel(xi, positions[j], hi, supports[j], kernel_int, mode_uint, periodicity, domainMin, domainMax)
        # if crkCorrection:
        #     xij = computeDistanceVec(xi, positions[j], periodicity, domainMin, domainMax)
        #     prod = Ai * (1.0 + wp.dot(Bi, xij)) * w_ij
        #     w_ij = prod

        f_interpolated += fv * vj * w_ij
            
    return f_interpolated


@wp.kernel
def computeSPHInterpolation_Kernel(
    queryPositions : wp.array(dtype = vector(length=Any, dtype=wp.float32)), referencePositions : wp.array(dtype=vector(length=Any, dtype=wp.float32)), # type: ignore
    querySupports : wp.array(dtype = wp.float32), referenceSupports : wp.array(dtype = wp.float32), # type: ignore
    queryMasses: wp.array(dtype = wp.float32), referenceMasses: wp.array(dtype = wp.float32),  # type: ignore
    queryDensities: wp.array(dtype = wp.float32), referenceDensities: wp.array(dtype = wp.float32), # type: ignore
    queryValues: wp.array(dtype = Any), referenceValues: wp.array(dtype = Any), # type: ignore
    
    domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), periodicity : wp.array(dtype = wp.bool), # type: ignore
    
    mode_uint: wp.uint32, kernel_int : wp.int32,
    neighborList: wp.array(dtype = wp.int64), neighborListRowOffsets: wp.array(dtype = wp.int32), numNeighbors: wp.array(dtype = wp.int32), preScatteredQuantities: wp.bool,  # type: ignore

    opInt: wp.int32, queryKinds : wp.array(dtype = wp.int32), referenceKinds : wp.array(dtype = wp.int32), # type: ignore
    
    
    useApparentVolume: bool, queryVolumes: wp.array(dtype = wp.float32), referenceVolumes: wp.array(dtype = wp.float32), # type: ignore
    crkCorrection: bool, crk_A: wp.array(dtype = wp.float32), crk_B: wp.array(dtype = vector(length=Any, dtype=wp.float32)), # type: ignore
    
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
    rhoi = queryDensities[i]
    fi = queryValues[i]
    
    neighborOffset = neighborListRowOffsets[i]
    numNeighs = numNeighbors[i]

    Ai = crk_A[i] if crkCorrection else type(crk_A[0])(0.0)
    Bi = crk_B[i] if crkCorrection else type(crk_B[0])(0.0)
    
    outputValues[i] = computeSPHInterpolation_Func(
        xi, hi, mi, rhoi,
        referencePositions, referenceSupports, referenceMasses, referenceDensities,
        periodicity, domainMin, domainMax, mode_uint, kernel_int,
        neighborList, neighborOffset, wp.int32(numNeighs), preScatteredQuantities,
        fi, referenceValues, opInt, referenceKinds,
        useApparentVolume, referenceVolumes,
        crkCorrection, Ai, Bi,
    )
    
    

from ..enumTypes import *

def computeSPHInterpolant_warpBackend(
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
    adjacency,
    scatteredQuantities: Optional[torch.Tensor] = None,
    useVolume: bool = False, queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    useCRK: bool = False, crk_A: Optional[torch.Tensor] = None, crk_B: Optional[torch.Tensor] = None,
):
    with record_function("warpSPH[Interpolation]"):
        with record_function("warpSPH[Interpolation] - Preprocessing"):
            domainMin = domain.min
            domainMax = domain.max
            periodicity = domain.periodic

            preScatteredQuantities = False
            if queryValues is None and referenceValues is None:
                if scatteredQuantities is None:
                    raise ValueError("If queryValues and referenceValues are not provided, then pre-scattered quantities must be provided for the interpolation computation.")
                preScatteredQuantities = True
                qV = scatteredQuantities
                rV = scatteredQuantities
            else:
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
                
                adjacency.j, adjacency.edgeOffsets, adjacency.numNeighbors, wp.bool(preScatteredQuantities),
                opInt, queryKinds, referenceKinds,

                wp.bool(useVolume), queryVolumes, referenceVolumes,
                wp.bool(useCRK), crk_A, crk_B,
            )

            if needs_flatten:
                warp_interpolation = warp_interpolation.reshape(
                    warp_interpolation.shape[0], *field_shape
                )

    return warp_interpolation