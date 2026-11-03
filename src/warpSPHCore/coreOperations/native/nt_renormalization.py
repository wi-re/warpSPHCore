"""Explicit geometry JVP adapter used by the shared renormalization scaffold."""
from dataclasses import replace
import torch
from torch.autograd import forward_ad
from ..._models import OperationProperties
from ...enumTypes import WarpOperation, OperationDirection


def _dual_state(state,tangent):
    if tangent is None:return state
    values={}
    for name in ('positions','supports','masses','densities'):
        primal,direction=getattr(state,name),getattr(tangent,name,None)
        if primal is not None and direction is not None:
            values[name]=forward_ad.make_dual(primal,direction)
    return replace(state,**values)


def computeCovarianceGeometryJVP(queryParticles,domain,kernel,supportMode,adjacency=None,*,
        operationMode=OperationDirection.AllToAll,queryTangentState=None,
        referenceParticles=None,referenceTangentState=None,referenceVolumes=None,
        tangentReferenceVolumes=None):
    from ...native.api import warpOperation
    with forward_ad.dual_level():
        query=_dual_state(queryParticles,queryTangentState)
        # Explicit JVP roles stay independent even for a self-neighborhood: an
        # omitted reference tangent means a fixed reference cloud, as in Warp.
        reference=_dual_state(referenceParticles if referenceParticles is not None else queryParticles,
                              referenceTangentState)
        volumes=referenceVolumes
        if volumes is not None and tangentReferenceVolumes is not None:
            volumes=forward_ad.make_dual(volumes,tangentReferenceVolumes)
        properties=OperationProperties(kernel,operation=WarpOperation.Covariance,
                                        supportMode=supportMode,operationMode=operationMode)
        covariance,_=warpOperation(query,properties,domain,adjacency=adjacency,referenceParticles=reference,
                                    referenceVolumes=volumes,covarianceReturnNumNeighbors=True)
        primal,tangent=forward_ad.unpack_dual(covariance)
        return tangent if tangent is not None else torch.zeros_like(primal)
