"""Turns a covariance matrix (wp_covariance.py) into a gradient-renormalization matrix:
low-neighbor-count fallback to the identity, then a pseudo-inverse (pinv_warp, dispatching
on dimension -- see pinv/wrapper.py). Split out of wp_covariance.py so that file only has
to contain the covariance computation itself.

Supports all three traversal inputs (explicit AdjacencyList, explicit CompactHashMap,
adjacency=None) -- the low-neighbor-count fallback reads its neighbor count from the
Covariance kernel's own per-particle output (covarianceReturnNumNeighbors=True below,
computed identically under either traversal mode), not from adjacency.numNeighbors,
which is the only thing that would have actually required an AdjacencyList. Verified by
gradcheck_renorm_native.py (forward-value parity + torch.autograd.gradcheck across all
three traversal inputs) -- see warpier_core.md's "Renormalization Grid-Mode Coverage"
note for why this needed its own script rather than reusing gradcheck_covariance_native.py.
"""

import torch
from typing import Optional, Union, Tuple
from torch.profiler import record_function

from .enumTypes import WarpOperation

from .dataTypes import *

from .operations import warpOperation
from .pinv import pinv2x2_warpBackend, pinv_warp


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
        operationProperties.operation = WarpOperation.Covariance
        C, num_nbrs = warpOperation(
            queryParticles, operationProperties, domain,
            queryVolumes = queryVolumes, referenceVolumes = referenceVolumes,
            adjacency = adjacency,
            referenceParticles = referenceParticles,
            crkState = crkState,
            gradHState = gradHState,
            renormalizationState = renormalizationState,
            covarianceReturnNumNeighbors = True
        )
    with record_function("[warpSPH] - Renorm - Covariance Postprocess"):
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
        L, eigVals = pinv_warp(C, num_nbrs)
        # if queryPositions.shape[1] == 2:
        #     L, eigVals = pinv2x2_warpBackend(C, num_nbrs)
        #     # L = torch.linalg.pinv(C)
        # else:
        #     # rcond matches the relative eigenvalue cutoff used in the 2D path (pinv2x2_warp):
        #     # zero out directions that are near-singular relative to the dominant eigenvalue,
        #     # rather than only truly-zero ones, so anisotropic/thin neighborhoods don't produce
        #     # huge inverted eigenvalues.
        #     L = torch.linalg.pinv(C, rtol=1e-6)
        #     eigVals = torch.linalg.eigvals(C).real

        #     if queryPositions.shape[1] == 3:
        #         eigVals[torch.abs(eigVals[:,1]) > torch.abs(eigVals[:,0]),:] = torch.flip(eigVals[torch.abs(eigVals[:,1]) > torch.abs(eigVals[:,0]),:],[1])
        #         eigVals[torch.abs(eigVals[:,2]) > torch.abs(eigVals[:,1]),:] = torch.flip(eigVals[torch.abs(eigVals[:,2]) > torch.abs(eigVals[:,1]),:],[1])
        #         eigVals[torch.abs(eigVals[:,2]) > torch.abs(eigVals[:,0]),:] = torch.flip(eigVals[torch.abs(eigVals[:,2]) > torch.abs(eigVals[:,0]),:],[1])
        #     elif queryPositions.shape[1] == 2:
        #         eigVals[torch.abs(eigVals[:,1]) > torch.abs(eigVals[:,0]),:] = torch.flip(eigVals[torch.abs(eigVals[:,1]) > torch.abs(eigVals[:,0]),:],[1])

    return C, eigVals, L


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
        C, eigVals, L = computeRenormalizationMatrices_(
            queryParticles, operationProperties, domain,
            queryVolumes = queryVolumes, referenceVolumes = referenceVolumes,
            adjacency = adjacency,
            referenceParticles = referenceParticles,
            crkState = crkState,
            gradHState = gradHState,
            renormalizationState = renormalizationState,
        )

        if returnEigVals:
            return C, eigVals, RenormalizationState(renormalizationMatrices = L)
        else:
            return RenormalizationState(renormalizationMatrices = L)
