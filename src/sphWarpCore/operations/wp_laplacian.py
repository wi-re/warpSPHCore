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
from ..utils.wp_util import checkDirectionality_i, checkDirectionality_j
from torch.profiler import profile, record_function, ProfilerActivity


@wp.func
def computeDotLaplacian(
    q_ij: vector(dtype = wp.float32, length=Any), n_ij: vector(dtype = wp.float32, length=Any), kernelGradient: vector(dtype = wp.float32, length=1), r_ij: wp.float32, h_ij: wp.float32, inputLength: wp.int32, dim: wp.int32 # type: ignore
):
    n_ij2 = n_ij / (r_ij + 1e-12 * h_ij)

    dotx = q_ij * n_ij2[0]

    # dot = wp.vec3f(dotx, doty, dotz)
    fkq = dotx * kernelGradient[0]

    result = type(q_ij)(-2.0 * fkq)
    return result

@wp.func
def computeDotLaplacian(
    q_ij: vector(dtype = wp.float32, length=Any), n_ij: vector(dtype = wp.float32, length=Any), kernelGradient: vector(dtype = wp.float32, length=2), r_ij: wp.float32, h_ij: wp.float32, inputLength: wp.int32, dim: wp.int32 # type: ignore
):
    n_ij2 = n_ij / (r_ij + 1e-12 * h_ij)

    dotx = q_ij * n_ij2[0]
    doty = q_ij * n_ij2[1]

    output = type(q_ij)(0.0)
    for i in range(inputLength):
        output[i] += dotx[i] * kernelGradient[0] + doty[i] * kernelGradient[1]

    return -2.0 * output

@wp.func
def computeDotLaplacian(
    q_ij: vector(dtype = wp.float32, length=Any), n_ij: vector(dtype = wp.float32, length=Any), kernelGradient: vector(dtype = wp.float32, length=3), r_ij: wp.float32, h_ij: wp.float32, inputLength: wp.int32, dim: wp.int32 # type: ignore
):
    n_ij2 = n_ij / (r_ij + 1e-12 * h_ij)

    dotx = q_ij * n_ij2[0]
    doty = q_ij * n_ij2[1]
    dotz = q_ij * n_ij2[2]

    output = type(q_ij)(0.0)
    for i in range(inputLength):
        output[i] += dotx[i] * kernelGradient[0] + doty[i] * kernelGradient[1] + dotz[i] * kernelGradient[2]

    return -2.0 * output

@wp.func
def computeLaplacianDot2(
    q_ij: vector(dtype = wp.float32, length=Any), n_ij: vector(dtype = wp.float32, length=Any), kernelGradient: vector(dtype = wp.float32, length=Any), r_ij: wp.float32, h_ij: wp.float32, inputLength: wp.int32, dim: wp.int32 # type: ignore
):
    # # DJ Price Smoothed particle hydrodynamics and magnetohydrodynamics page 778 (eq 96) in https://www.sciencedirect.com/science/article/pii/S0021999110006753

    # r_eps = r_ij + 1e-8 * h_i
    # n_ij = x_ij / r_eps.view(-1,1)
    # F_ab = torch.einsum('nd, nd -> n', n_ij, gradW_ij) / r_eps

    # leftTerm = (x_ij.shape[1] + 2) * torch.einsum('n..., nd -> n...d', torch.einsum('n...d, nd -> n...', fq, n_ij), n_ij)
    # rightTerm = - fq

    # fkq = -(leftTerm + rightTerm) * F_ab.view(-1,1)

    # dot = torch.einsum('n...d, nd -> n...d', fq, x_ij / (r_ij + 1e-8 * h_i).view(-1,1)**2)

    r_eps = r_ij + 1e-8 * h_ij
    F_ab = wp.dot(n_ij, kernelGradient) / r_eps # this is a scalar
    leading_dim = wp.int32(inputLength // dim) # this is the number of leading dimensions in q_ij before the last dimension of size dim
    
    output = type(q_ij)(0.0)
    for i in range(inputLength):
        # In this case q_ij is some quantity of internal shape [..., dim] so we compute the dot product across the trailing dimension of q_ij and n_ij, and then multiply by n_ij again to get the contribution for each component of the output. This is equivalent to the left term in Price's equation where we have a double dot product between n_ij and q_ij, but we compute it in a way that allows for q_ij to have an arbitrary number of leading dimensions as long as the last dimension has size dim.
        # leftTerm = wp.float32(dim + 2) * q_ij[i] * n_ij[i %dim] * n_ij[i%dim] * F_ab
        # fkq = -(leftTerm + rightTerm*0.0)
        d = i % dim                  # component within trailing dim
        b = i // dim                 # block index over leading dims
        base = b * dim               # start of this block in flattened storage
        
        proj = wp.float32(0.0)
        for k in range(dim):
            proj += q_ij[base + k] * n_ij[k]

        left = wp.float32(dim + 2) * proj * n_ij[d]
        output[i] += -left * F_ab


        # for k in range(dim):
            # output[i] += -wp.float32(dim + 2) * q_ij[k * leading_dim] * n_ij[k] * n_ij[i % dim] * F_ab
        # output[i] += leftTerm

    for i in range(inputLength):
        rightTerm = - q_ij[i]* F_ab
        output[i] += -rightTerm

    return output

    
@wp.func
def positiveDotProduct(
    x_ij: vector(dtype = wp.float32, length=Any), # type: ignore
    fq_ij: vector(dtype = wp.float32, length=Any), # type: ignore
    f_ij: vector(dtype = wp.float32, length=Any), # type: ignore
    dim: wp.int32
):
    dot = wp.float32(0.0)
    for d in range(dim):
        dot += x_ij[d] * fq_ij[d]
    
    result = type(f_ij)(0.0)
    if dot >= 0.0:
        for d in range(dim):
            result[d] = f_ij[d]
    return result

@wp.func
def computeSPHLaplacianTensor_Func(
    xi: vector(dtype = wp.float32, length=Any), hi : wp.float32, mi: wp.float32, rhoi: wp.float32, fi : vector(dtype = wp.float32, length=Any), # type: ignore
    
    positions : wp.array(dtype=vector(length=Any, dtype = wp.float32)), supports : wp.array(dtype = wp.float32), masses: wp.array(dtype = wp.float32), densities: wp.array(dtype = wp.float32), values: wp.array(dtype = vector(dtype = wp.float32, length=Any)), # type: ignore
    
    periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), # type: ignore
    mode_uint: wp.uint32, kernel_int: wp.int32, gradientMode_int: wp.int32, laplacianMode_int: wp.int32, positiveDivergence: wp.bool,
    
    neighborList: wp.array(dtype = wp.int64), # type: ignore
    neighborOffset : wp.int32, numNeighs: wp.int32, preScatteredQuantities: wp.bool,
    dim: wp.int32, numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    opInt: wp.int32, referenceKinds : wp.array(dtype = wp.int32), # type: ignore
    
    outputValue: vector(dtype = wp.float32, length=Any) # type: ignore
):
    laplacian_result = type(outputValue)(0.0)
    
    for neighborIndex in range(numNeighs):
        jj = neighborOffset + neighborIndex
        j = wp.int32(neighborList[jj])
        if opInt != 0:
            if not checkDirectionality_j(referenceKinds[j], opInt):
                continue

        xj = positions[j]
        mj = masses[j]
        rhoj = densities[j]
        apparentVolume = mj / rhoj
        fj = type(fi)(0.0)
        if preScatteredQuantities:
            fj = values[jj]
        else:
            fj = values[j]
        hj = supports[j]
        
        kernelGradient = sphKernelGradient(xi, positions[j], hi, supports[j], kernel_int, mode_uint, periodicity, domainMin, domainMax)
        
        q_ij = type(fi)(0.0)

        if gradientMode_int == 1: # Naive
            q_ij = fj * apparentVolume
        elif gradientMode_int == 2: # Symmetric
            q_ij = mj * rhoi * (fi / iPow(rhoi,2) + fj / iPow(rhoj,2))
        elif gradientMode_int == 3: # Difference
            q_ij = (fj - fi) * apparentVolume
        elif gradientMode_int == 4: # Summation
            q_ij = (fj + fi) * apparentVolume

        h_ij = computePairwiseSupport(hi, hj, mode_uint)
        x_ij = computeDistanceVec(xi, xj, periodicity, domainMin, domainMax)
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))

        eps = 1e-8
        n_ij = x_ij / (r_ij + eps * h_ij)

        laplacian_contribution = type(outputValue)(0.0)

        if laplacianMode_int == 1: # Naive
            laplacian_contribution = q_ij * sphKernelLaplacian(xi, positions[j], hi, supports[j], kernel_int, mode_uint, periodicity, domainMin, domainMax)
        elif laplacianMode_int == 2: # Brookshaw
            laplacian_contribution = -2.0 * q_ij * wp.dot(kernelGradient, n_ij) / (r_ij + eps * h_ij)
        elif laplacianMode_int == 3: # Dot
            laplacian_contribution = computeLaplacianDot2(q_ij, n_ij, kernelGradient, r_ij, h_ij, flatInputShape, dim)
        elif laplacianMode_int == 4: # Default
            laplacian_contribution = computeDotLaplacian(q_ij, n_ij, kernelGradient, r_ij, h_ij, flatInputShape, dim)

        if positiveDivergence:
            laplacian_result += positiveDotProduct(x_ij, q_ij, laplacian_contribution, dim)
        else:
            laplacian_result += laplacian_contribution

    return laplacian_result

@wp.kernel
def computeSPHLaplacianTensor_Kernel(
    queryPositions : wp.array(dtype = vector(length=Any, dtype=wp.float32)), referencePositions : wp.array(dtype=vector(length=Any, dtype=wp.float32)), # type: ignore
    querySupports : wp.array(dtype = wp.float32), referenceSupports : wp.array(dtype = wp.float32), # type: ignore
    queryMasses: wp.array(dtype = wp.float32), referenceMasses: wp.array(dtype = wp.float32), # type: ignore
    queryDensities: wp.array(dtype = wp.float32), referenceDensities: wp.array(dtype = wp.float32),# type: ignore
    queryValues: wp.array(dtype = vector(dtype = wp.float32, length=Any)), referenceValues: wp.array(dtype = vector(dtype = wp.float32, length=Any)), # type: ignore
    
    domainMin : wp.array(dtype = wp.float32), domainMax : wp.array(dtype = wp.float32), periodicity : wp.array(dtype = wp.bool), # type: ignore
    
    mode_uint: wp.uint32, kernel_int : wp.int32, gradientMode_int: wp.int32, laplacianMode_int: wp.int32, positiveDivergence: wp.bool,
    neighborList: wp.array(dtype = wp.int64), neighborListRowOffsets: wp.array(dtype = wp.int32), numNeighbors: wp.array(dtype = wp.int32), preScatteredQuantities: wp.bool, # type: ignore
    
    dim: wp.int32, numDims: wp.int32, flatInputShape: wp.int32, flatOutputShape: wp.int32,
    opInt: wp.int32, queryKinds : wp.array(dtype = wp.int32), referenceKinds : wp.array(dtype = wp.int32), # type: ignore
    
    outputValues : wp.array(dtype = vector(length = Any, dtype = wp.float32)) # type: ignore
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
    
    outputValues[i] = computeSPHLaplacianTensor_Func(
        xi, hi, mi, rhoi, fi, 
        referencePositions, referenceSupports, referenceMasses, referenceDensities, referenceValues,
        
        periodicity, domainMin, domainMax, 
        mode_uint, kernel_int, gradientMode_int, laplacianMode_int, positiveDivergence,
        neighborList, neighborListRowOffsets[i], numNeighbors[i], 
        preScatteredQuantities,
        dim, numDims, flatInputShape, flatOutputShape,
        opInt, referenceKinds,
        type(outputValues[i])(0.0))
    
from ..enumTypes import *

def computeSPHLaplacian_warpBackend(
    queryPositions, referencePositions,
    querySupports, referenceSupports,
    queryMasses, referenceMasses,
    queryDensities, referenceDensities,
    queryValues, referenceValues,
    queryKinds, referenceKinds,
    domain: DomainDescription,
    mode: SupportScheme,
    kernel: KernelFunctions,    
    gradientMode: GradientScheme,
    laplacianMode: LaplacianScheme,
    positiveDivergence: bool,
    operationMode: OperationDirection,
    adjacency: AdjacencyListWarp,
    scatteredQuantities: Optional[torch.Tensor] = None,
):
    with record_function("warpSPH[Laplacian]"):
        with record_function("warpSPH[Laplacian] - Preprocessing"):
            domainMin = domain.min
            domainMax = domain.max
            periodicity = domain.periodic

            mode_uint = convertModeToUint(mode.name)
            kernel_int = kernel.value
            gradientMode_int = gradientMode.value
            laplacianMode_int = laplacianMode.value
            positiveDivergence = wp.bool(positiveDivergence)
            opInt = wp.int32(operationMode.value)


            preScatteredQuantities = False
            if queryValues is None and referenceValues is None:
                if scatteredQuantities is None:
                    raise ValueError("If queryValues and referenceValues are not provided, then pre-scattered quantities must be provided for the laplacian computation.")
                preScatteredQuantities = True
                qV = scatteredQuantities
                rV = scatteredQuantities
            else:
                qV = queryValues
                rV = referenceValues

            # Warp kernels only support rank-1 (vector) and rank-2 (matrix) field types.
            outputSize = (queryPositions.shape[0])

            inputShape = qV.shape[1:]
            flatInputShape = 1
            for dim in inputShape:
                flatInputShape *= dim
                
            # For the output shape we keep the same shape as the input as the laplacian of a scalar field is still a scalar field, and the laplacian of a vector field is still a vector field. We just need to make sure to flatten the inner dimensions for the warp kernel.
            outputShape = inputShape

            flatOutputShape = 1
            for dim in outputShape:
                flatOutputShape *= dim
            numDims = len(inputShape)
            
        with record_function("warpSPH[Laplacian] - Kernel Execution"):
            warp_result = warpWrapper(
                launch_kernel, computeSPHLaplacianTensor_Kernel, outputSize, vector(length=flatOutputShape, dtype = wp.float32),
                queryPositions, referencePositions,
                querySupports, referenceSupports,
                queryMasses, referenceMasses,
                queryDensities, referenceDensities,
                qV.view(-1, flatInputShape), rV.view(-1, flatInputShape),
                domainMin, domainMax, periodicity,
                mode_uint, kernel_int, gradientMode_int, laplacianMode_int, positiveDivergence,
                adjacency.j, adjacency.edgeOffsets, adjacency.numNeighbors, wp.bool(preScatteredQuantities),
                wp.int32(queryPositions.shape[1]), wp.int32(numDims), wp.int32(flatInputShape), wp.int32(flatOutputShape),
                opInt, queryKinds, referenceKinds,
            )
    # print(f"computeSPHLaplacian_warpBackend: inputShape={inputShape}, flatInputShape={flatInputShape}, outputShape={outputShape}, flatOutputShape={flatOutputShape}, numDims={numDims}")


    return warp_result.view(queryPositions.shape[0], *outputShape) # reshape back to original shape with new gradient dimension