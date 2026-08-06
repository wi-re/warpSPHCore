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
                continue# out * scalar_t(0.0)
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
    # Symmetric closed-form eigendecomposition -- see pinv2x2_warp below (the actual production
    # path) for the derivation and why the general 2x2-SVD formula this used to use is unstable for
    # near-isotropic covariance matrices. This function is currently unused (computeRenormalizationMatrices_
    # calls pinv2x2_warpBackend instead) but kept fixed in step with it since it operates on the same
    # symmetric covariance matrices.
    with record_function('Pseudo Inverse 2x2'):
        a = M[:,0,0]
        b = 0.5 * (M[:,0,1] + M[:,1,0])
        d = M[:,1,1]

        theta = 0.5 * torch.atan2(2 * b, a - d)
        cosTheta = torch.cos(theta)
        sinTheta = torch.sin(theta)
        v1 = torch.stack([cosTheta, sinTheta], -1)
        v2 = torch.stack([-sinTheta, cosTheta], -1)

        lam1 = a * cosTheta**2 + 2 * b * cosTheta * sinTheta + d * sinTheta**2
        lam2 = (a + d) - lam1

        swap = lam2.abs() > lam1.abs()
        big = torch.where(swap, lam2, lam1)
        small = torch.where(swap, lam1, lam2)
        bigV = torch.where(swap.unsqueeze(-1), v2, v1)
        smallV = torch.where(swap.unsqueeze(-1), v1, v2)

        eigVals = torch.stack([big, small], -1)

        rcond = 1e-6
        threshold = rcond * big.abs()
        big_inv = torch.where(big.abs() > 1e-12, 1 / big, torch.zeros_like(big))
        small_inv = torch.where(small.abs() > threshold, 1 / small, torch.zeros_like(small))

        inv = big_inv[:, None, None] * torch.einsum('ni,nj->nij', bigV, bigV) \
            + small_inv[:, None, None] * torch.einsum('ni,nj->nij', smallV, smallV)
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
    # C is a sum of V_j * x_ij (x) gradW_ij; for any isotropic kernel gradW_ij is parallel to x_ij,
    # so C is symmetric by construction (a sum of symmetric x_ij (x) x_ij terms). b/c below can still
    # differ at the floating-point-noise level depending on neighbor summation order -- symmetrize
    # rather than treat that noise as signal.
    b = scalar_t(0.5) * (C[i][0,1] + C[i][1,0])
    d = C[i][1,1]

    if num_nbrs[i] < 4:
        L[i][0,0] = scalar_t(1.0)
        L[i][0,1] = scalar_t(0.0)
        L[i][1,0] = scalar_t(0.0)
        L[i][1,1] = scalar_t(1.0)
        EV[i][0] = scalar_t(1.0)
        EV[i][1] = scalar_t(1.0)
        return

    # Closed-form symmetric 2x2 eigendecomposition: a single atan2 call. This replaces a general (and
    # for a symmetric input, unnecessary) 2x2 SVD that computed U's rotation angle and V's rotation
    # angle from two SEPARATE atan2 expressions. For a near-isotropic C (a~=d, b~=c~=0 -- the common
    # case for a locally regular/well-resolved particle neighborhood) both of those expressions'
    # denominators round to ~0, and because the two expressions round differently at the float-noise
    # level, the two angles could land on unrelated values instead of the (here) required theta==phi,
    # producing an inverse that was spuriously rotated by tens of degrees instead of staying diagonal
    # -- reproduced directly against production covariance matrices, see warpier_core.md. A symmetric
    # matrix only has one rotation angle in the first place, so computing it once removes the
    # possibility of the two desyncing.
    theta = (scalar_t(0.5)) * wp.atan2(scalar_t(2.0) * b, a - d)
    cosTheta = wp.cos(theta)
    sinTheta = wp.sin(theta)

    v1x = cosTheta
    v1y = sinTheta
    v2x = -sinTheta
    v2y = cosTheta

    lam1 = a * v1x * v1x + scalar_t(2.0) * b * v1x * v1y + d * v1y * v1y
    lam2 = (a + d) - lam1

    # order by magnitude, largest first, matching the "o1 >= o2" convention the rest of this function
    # (and its callers) assume. Eigenvalues are signed here, unlike the old singular-value convention.
    bigX = v1x
    bigY = v1y
    big = lam1
    smallX = v2x
    smallY = v2y
    small = lam2
    if wp.abs(lam2) > wp.abs(lam1):
        bigX = v2x
        bigY = v2y
        big = lam2
        smallX = v1x
        smallY = v1y
        small = lam1

    EV[i][0] = big
    EV[i][1] = small

    # Zeroing based on a fixed absolute epsilon lets thin/anisotropic neighborhoods (e.g. free-surface
    # fingers, near-collinear particle rows) through with a tiny but nonzero small eigenvalue, which
    # then gets inverted into a huge amplification factor. Use a cutoff relative to the largest
    # eigenvalue instead, matching the rcond convention torch.linalg.pinv uses for the 3D path.
    rcond = scalar_t(1.0e-6)
    threshold = rcond * wp.abs(big)
    big_inv = scalar_t(0.0) if wp.abs(big) <= scalar_t(1.0e-12) else scalar_t(1.0) / big
    small_inv = scalar_t(0.0) if wp.abs(small) <= threshold else scalar_t(1.0) / small

    L[i][0,0] = big_inv * bigX * bigX + small_inv * smallX * smallX
    L[i][0,1] = big_inv * bigX * bigY + small_inv * smallX * smallY
    L[i][1,0] = big_inv * bigY * bigX + small_inv * smallY * smallX
    L[i][1,1] = big_inv * bigY * bigY + small_inv * smallY * smallY

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
    wp.launch(kernel=pinv2x2_warp, dim=mat_warp.shape[0], inputs=[mat_warp, inv_warp, evs_warp, nnbrs], device=inv_warp.device)
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

        # Too few neighbors to trust the covariance matrix (e.g. free-surface fingers, isolated
        # particles): fall back to the identity so the pseudo-inverse doesn't amplify noise. The
        # 2D path re-checks this internally in pinv2x2_warp; applying it here as well covers 3D+.
        lowNbrMask = num_nbrs < dim + 2
        if torch.any(lowNbrMask):
            C = C.clone()
            C[lowNbrMask, :, :] = torch.eye(dim, dtype = dtype, device = device)[None, :, :]

    with record_function("[warpSPH] - Renorm - Pseudo Inverse"):
        if queryPositions.shape[1] == 2:
            L, eigVals = pinv2x2_warpBackend(C, num_nbrs)
            # L = torch.linalg.pinv(C)
        else:
            # rcond matches the relative eigenvalue cutoff used in the 2D path (pinv2x2_warp):
            # zero out directions that are near-singular relative to the dominant eigenvalue,
            # rather than only truly-zero ones, so anisotropic/thin neighborhoods don't produce
            # huge inverted eigenvalues.
            L = torch.linalg.pinv(C, rtol=1e-6)
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