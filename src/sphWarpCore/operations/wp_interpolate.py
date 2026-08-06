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


@wp.func
def computeSPHInterpolation_Func(
    # General Shape Parameters and indices
    i : wp.int32, dim: wp.int32, 

    # SPH properties for the query set (indexed by i)
    queryPositions: wp.array(dtype=vector(dtype = scalar_t, length=Any)), querySupports: wp.array(dtype = scalar_t), queryMasses: wp.array(dtype = scalar_t), queryDensities: wp.array(dtype = scalar_t), queryValues: wp.array(dtype = Any), # type: ignore

    # SPH properties for the reference set (indexed by j in the neighbor loop)
    referencePositions : wp.array(dtype=vector(length=Any, dtype = scalar_t)), referenceSupports : wp.array(dtype = scalar_t), referenceMasses: wp.array(dtype = scalar_t), referenceDensities: wp.array(dtype = scalar_t), referenceValues: wp.array(dtype = Any), # type: ignore
    
    # Domain and kernel parameters
    periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = scalar_t), domainMax : wp.array(dtype = scalar_t), # type: ignore
    mode_uint: wp.uint32, kernel_int: wp.int32, 
    
    # Neighbor list data, pre accessed to avoid gradient issues with dynamic for loops
    neighborList: wp.array(dtype = wp.int64), # type: ignore
    neighborOffset : wp.int32, numNeighs: wp.int32, 
    
    # Indicates if the input quantities have already been scattered to the neighbor level 
    preScatteredQuantities: wp. bool,
    
    # Operation Mode for masking certain kinds of interactions, e.g. for directional operations
    opInt: wp.int32, queryKinds : wp.array(dtype = wp.int32), referenceKinds : wp.array(dtype = wp.int32), # type: ignore

    # Optional Correction Terms:
    # Whether to use actual volume (mass/density) or apparent volume for the gradient computation, and the corresponding volumes if needed.
    useVolume: wp.bool, queryVolumes: wp.array(dtype = scalar_t), referenceVolumes: wp.array(dtype = scalar_t), # type: ignore
    # Whether to use CRK kernel correction for the computation, and the corresponding correction terms if needed.
    useCRK: wp.bool, queryA: wp.array(dtype = scalar_t), queryB: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    
    # Dummy value to allow allocation
    outputValue: Any # type: ignore
):
    if opInt != 0:
        if not checkDirectionality_i(queryKinds[i], opInt):
            return outputValue * scalar_t(0.0)
    # Unpack query point properties
    xi      = queryPositions[i]
    hi      = querySupports[i]
    # mi      = queryMasses[i] # Generally not needed
    rhoi    = queryDensities[i]
    fi      = queryValues[i]
    # Unpack optional correction terms
    Ai      = queryA[i] if useCRK else type(queryA[0])(scalar_t(0.0))
    Bi      = queryB[i] if useCRK else type(queryB[0])(scalar_t(0.0))
    
    # Initialize the output value
    out     = type(outputValue)(scalar_t(0.0))
    
    # Loop over neighbors to compute the gradient contribution from each neighbor    
    for neighborIndex in range(numNeighs):
        jj = neighborOffset + neighborIndex
        j = wp.int32(neighborList[jj])
        if opInt != 0:
            if not checkDirectionality_j(referenceKinds[j], opInt):
                continue
        ##########################################################
        #   The core particle-particle interaction starts here   #
        ##########################################################

        # A ternary expression here (`fv = a if cond else b`) compiles fine but
        # silently produces a zero adjoint for referenceValues -- confirmed via
        # a minimal warp-lang repro isolating ternary-vs-if/else array reads on
        # both concrete and generic (Any) dtypes; every other operator in this
        # codebase already uses the if/else block form for this exact
        # preScatteredQuantities branch. See warpier_core.md.
        if preScatteredQuantities:
            fv = referenceValues[jj]
        else:
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
    queryPositions : wp.array(dtype = vector(length=Any, dtype=scalar_t)), referencePositions : wp.array(dtype=vector(length=Any, dtype=scalar_t)), # type: ignore
    querySupports : wp.array(dtype = scalar_t), referenceSupports : wp.array(dtype = scalar_t), # type: ignore
    queryMasses: wp.array(dtype = scalar_t), referenceMasses: wp.array(dtype = scalar_t),  # type: ignore
    queryDensities: wp.array(dtype = scalar_t), referenceDensities: wp.array(dtype = scalar_t), # type: ignore
    queryValues: wp.array(dtype = Any), referenceValues: wp.array(dtype = Any), # type: ignore
    
    domainMin : wp.array(dtype = scalar_t), domainMax : wp.array(dtype = scalar_t), periodicity : wp.array(dtype = wp.bool), # type: ignore
    
    mode_uint: wp.uint32, kernel_int : wp.int32,
    neighborList: wp.array(dtype = wp.int64), neighborListRowOffsets: wp.array(dtype = wp.int32), numNeighbors: wp.array(dtype = wp.int32), # type: ignore
    
    preScatteredQuantities: wp.bool,  

    opInt: wp.int32, queryKinds : wp.array(dtype = wp.int32), referenceKinds : wp.array(dtype = wp.int32), # type: ignore
    
    useVolume: wp.bool, queryVolumes: wp.array(dtype = scalar_t), referenceVolumes: wp.array(dtype = scalar_t), # type: ignore
    useCRK: wp.bool, crk_A: wp.array(dtype = scalar_t), crk_B: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    
    outputValues : wp.array(dtype = Any) # type: ignore
):                                                                                    
    i = wp.tid()
    if i >= queryPositions.shape[0]:
        return
    
    outputValues[i] = computeSPHInterpolation_Func(
        i, get_dim(queryPositions), 

        queryPositions, querySupports, queryMasses, queryDensities, queryValues,
        referencePositions, referenceSupports, referenceMasses, referenceDensities, referenceValues,

        periodicity, domainMin, domainMax, 
        mode_uint, kernel_int,

        neighborList, neighborListRowOffsets[i], numNeighbors[i], 
        
        preScatteredQuantities,
        
        opInt, queryKinds, referenceKinds,
        
        useVolume, queryVolumes, referenceVolumes,
        useCRK, crk_A, crk_B,

        outputValues[i]
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