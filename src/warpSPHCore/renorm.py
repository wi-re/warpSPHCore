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
from dataclasses import replace
from typing import Optional, Union, Tuple
from .profiling import record_function

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
        # Covariance is this function's own internal step, not something the
        # caller selects -- so it is applied to a *copy* of the properties
        # rather than written back into the caller's object. This used to be
        # `operationProperties.operation = WarpOperation.Covariance`, an
        # in-place write to a dataclass the caller still owns: any caller that
        # reused its properties object afterwards silently got a Covariance
        # launch where it asked for a Gradient, and the (N, D, D) result of
        # that is plausible enough to go unnoticed. Every call site in both
        # repos today happens to construct a fresh OperationProperties inline
        # for this call, which is why it never bit -- but that also means
        # hoisting those constructions out of the hot path (warpier_fields.md
        # Section 3.5's suggested follow-up, which wants a reusable, hashable
        # properties object to key the StateBundle on) would have introduced
        # the bug rather than found it. Found by Step G's Tier-1 spike, which
        # reuses one properties object across the renorm call and the gradient
        # call that consumes its output.
        covarianceProperties = replace(operationProperties, operation=WarpOperation.Covariance)
        C, num_nbrs = warpOperation(
            queryParticles, covarianceProperties, domain,
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
        # Selected with torch.where rather than `if torch.any(mask): C[mask] = eye`.
        # That guard was two host readbacks per step in a real run (`torch.any` on a
        # device tensor in a Python `if`), and the masked assignment it guarded is
        # itself a synchronising op (a boolean-mask index_put_ reads the mask count on
        # the host), so the fast path paid a stall to *maybe* avoid a stall. `where`
        # has neither, needs no clone, and gives the same values and the same
        # gradients -- a low-neighbour row is replaced by a constant either way, so
        # nothing flows back through it. See Step H's sync census in
        # docs/regression/real_workload_bottleneck_audit.md.
        lowNbrMask = num_nbrs < dim + 2
        identity = torch.eye(dim, dtype = dtype, device = device)
        C = torch.where(lowNbrMask.view(-1, 1, 1), identity.unsqueeze(0), C)

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

__all__ = [
    "computeRenormalizationMatrices",
]