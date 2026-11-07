"""Explicit CRK preparation products over the shared native pair derivatives."""
from dataclasses import replace
import torch
from ..._models import OperationProperties,CRKState,CRKTangentState
from ...enumTypes import SupportScheme,OperationDirection


def _product(kind,query,reference,domain,properties,adjacency,query_tangent,reference_tangent,
             volumes=None,volume_tangent=None):
    from ...radiusSearch.chunked import ChunkedAdjacency
    if isinstance(adjacency,ChunkedAdjacency):raise NotImplementedError('Row-chunk operators do not support AD')
    if kind==8:
        from .nt_volume import computeCRKVolume
        from .nt_volume_jvp import compute
        primal=computeCRKVolume(query,properties,domain,adjacency,reference)
        return primal,compute(query,reference,domain,properties,adjacency,primal,query_tangent,reference_tangent)
    if kind==9:
        from .nt_moments import computeCRKMoments
        from .nt_moments_jvp import compute
        primal=computeCRKMoments(query,properties,domain,None,volumes,adjacency,reference)
        return primal,compute(query,reference,domain,properties,adjacency,volumes,volume_tangent,
                              query_tangent,reference_tangent)
    raise ValueError('Unknown CRK preparation product')


def computeCRKVolumeGeometryJVP(queryParticles,domain,kernel,adjacency,apparentVolumes,
        queryTangentState,referenceParticles=None,referenceTangentState=None,
        operationMode=OperationDirection.AllToAll):
    properties=OperationProperties(kernel,supportMode=SupportScheme.Gather,operationMode=operationMode)
    from .nt_volume_jvp import compute
    return compute(queryParticles,referenceParticles,domain,properties,adjacency,
                   apparentVolumes,queryTangentState,referenceTangentState)


def computeCRKMomentsGeometryJVP(queryParticles,domain,kernel,adjacency,referenceVolumes,
        tangentReferenceVolumes,queryTangentState,referenceParticles=None,referenceTangentState=None,
        operationMode=OperationDirection.AllToAll):
    properties=OperationProperties(kernel,supportMode=SupportScheme.Scatter,operationMode=operationMode)
    return _product(9,queryParticles,referenceParticles,domain,properties,adjacency,
                    queryTangentState,referenceTangentState,referenceVolumes,tangentReferenceVolumes)[1]


def computeCRKFactorsJVP(queryParticles,domain,kernel,queryTangentState,
        operationMode=OperationDirection.AllToAll,adjacency=None,referenceState=None,
        referenceTangentState=None,*,referenceVolumeMode=None,referenceAdjacency=None):
    """Return query volumes, their product, factors and first-order products.

    A distinct reference population has its own reference-to-reference volume
    pass. Omitted role tangents are zero. Limiter and rank branches stay fixed.
    """
    from ..crk_terms import computeCRKTerms
    reference=queryParticles if referenceState is None else referenceState
    properties=OperationProperties(kernel,supportMode=SupportScheme.Gather,operationMode=operationMode)
    area,darea=_product(8,queryParticles,reference,domain,properties,adjacency,
                       queryTangentState,referenceTangentState)
    rarea,drarea=area,darea
    if reference is not queryParticles or referenceVolumeMode is not None:
        rarea,drarea=_product(8,reference,reference,domain,
            replace(properties,operationMode=operationMode if referenceVolumeMode is None else referenceVolumeMode),
            referenceAdjacency,referenceTangentState,referenceTangentState)
    elif referenceAdjacency is not None:
        raise ValueError('referenceAdjacency requires a distinct referenceState or referenceVolumeMode')
    moments,dmoments=_product(9,queryParticles,reference,domain,
        replace(properties,supportMode=SupportScheme.Scatter),adjacency,
        queryTangentState,referenceTangentState,rarea,drarea)
    def factors(*values):return computeCRKTerms(*values,moments[-1],queryParticles.supports)
    primal=factors(*moments[:-1])
    _,products=torch.autograd.functional.jvp(factors,moments[:-1],dmoments,create_graph=True)
    return area,darea,CRKState(*primal),CRKTangentState(*products)
