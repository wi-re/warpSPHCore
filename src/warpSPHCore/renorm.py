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
from .coreOperations.wp_covarianceJVP import computeCovarianceGeometryJVP


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


def computeRenormalizationMatricesJVP(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,

    queryTangentState: 'ParticleTangentState',

    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    tangentReferenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None,
    referenceParticles: Optional[ParticleState] = None,
    referenceTangentState: Optional['ParticleTangentState'] = None,
    returnEigVals: bool = True,
):
    """JVP counterpart to `computeRenormalizationMatrices`
    (`warpier_tier2_correction_jvp_plan.md` phase (d)): given
    `queryTangentState`/`referenceTangentState` (position/support/mass/
    density tangents), returns `dL_i`, obtained from `wp_covarianceJVP.py`'s
    raw `dC_i` by the same two steps `computeRenormalizationMatrices_` itself
    applies to the primal `C_i` -- the low-neighbor-count identity fallback's
    exact-zero tangent (`dC = where(lowNbrMask, 0, dC_raw)`), then the
    standard matrix-inverse-derivative identity `d(C^-1) = -C^-1(dC)C^-1`
    (`scripts/spike_forward_mode_tier2_renorm.py`, already validated to
    float64 round-off). `L` itself is consumed from this function's own
    primal `computeRenormalizationMatrices_` call rather than re-derived --
    same "consume an already-validated value" pattern the spike uses and
    `crk_wrapper.computeCRKFactorsJVP` uses for its own already-validated
    inputs.

    Like `computeCRKFactorsJVP`, this computes the correction (`L`) AND its
    tangent (`dL`) together -- a caller only supplies geometry tangents, not
    a pre-existing `RenormalizationState`. `crkState`/`gradHState` are not
    accepted here (unlike `computeRenormalizationMatrices`'s own signature):
    `wp_covarianceJVP.py`'s kernel is deliberately CRK/grad-h-free, matching
    `computeRenormalizationMatrices_`'s own internal covariance call under
    the "renorm alone first" scoping this plan phase settled on (CRK+renorm
    simultaneous is a fast follow-up, not required here).

    Returns `(C, eigVals, RenormalizationState, RenormalizationTangentState)`
    (or the two states only if `returnEigVals=False`), mirroring
    `computeRenormalizationMatrices`'s own return shape plus the tangent.
    """
    with record_function("[warpSPH] - computeRenormalizationMatricesJVP"):
        C, eigVals, L = computeRenormalizationMatrices_(
            queryParticles, operationProperties, domain,
            queryVolumes = queryVolumes, referenceVolumes = referenceVolumes,
            adjacency = adjacency,
            referenceParticles = referenceParticles,
        )

        with record_function("[warpSPH] - RenormJVP - Compute Covariance Neighbor Counts"):
            # Needed for the same low-neighbor-count mask computeRenormalizationMatrices_
            # itself applies to C -- not returned by that function, so recomputed here via
            # its own internal Covariance call (mirrors the spike's own
            # `assembled_renorm_jvp`, which likewise calls the primal Covariance operator
            # a second time purely for `num_nbrs`).
            covarianceProperties = replace(operationProperties, operation=WarpOperation.Covariance)
            _, num_nbrs = warpOperation(
                queryParticles, covarianceProperties, domain,
                queryVolumes = queryVolumes, referenceVolumes = referenceVolumes,
                adjacency = adjacency,
                referenceParticles = referenceParticles,
                covarianceReturnNumNeighbors = True,
            )

        with record_function("[warpSPH] - RenormJVP - Covariance JVP"):
            dC_raw = computeCovarianceGeometryJVP(
                queryParticles, domain, operationProperties.kernel, operationProperties.supportMode, adjacency,
                queryTangentState=queryTangentState,
                referenceParticles=referenceParticles, referenceTangentState=referenceTangentState,
                referenceVolumes=referenceVolumes, tangentReferenceVolumes=tangentReferenceVolumes,
            )

        with record_function("[warpSPH] - RenormJVP - Pseudo Inverse Derivative"):
            dim = queryParticles.positions.shape[1]
            lowNbrMask = (num_nbrs < dim + 2).view(-1, 1, 1)
            dC = torch.where(lowNbrMask, torch.zeros_like(dC_raw), dC_raw)
            dL = -torch.matmul(L, torch.matmul(dC, L))

        if returnEigVals:
            return C, eigVals, RenormalizationState(renormalizationMatrices=L), RenormalizationTangentState(renormalizationMatrices=dL)
        else:
            return RenormalizationState(renormalizationMatrices=L), RenormalizationTangentState(renormalizationMatrices=dL)


__all__ = [
    "computeRenormalizationMatrices",
    "computeRenormalizationMatricesJVP",
]