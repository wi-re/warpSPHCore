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
    from .nt_jvp import warpOperationJVP
    properties=OperationProperties(kernel,operation=WarpOperation.Covariance,
        supportMode=supportMode,operationMode=operationMode)
    return warpOperationJVP(queryParticles,properties,domain,
        queryTangentState=queryTangentState,referenceTangentState=referenceTangentState,
        referenceParticles=referenceParticles,adjacency=adjacency,
        referenceVolumes=referenceVolumes,tangentReferenceVolumes=tangentReferenceVolumes)
