"""Explicit CRK preparation products over the shared native pair derivatives."""
from dataclasses import replace
from math import prod
import torch
from torch.autograd import forward_ad
from ..._models import OperationProperties,CRKState,CRKTangentState
from ...enumTypes import SupportScheme,OperationDirection
from .nt_preparation import prepare


def _product(kind,query,reference,domain,properties,adjacency,query_tangent,reference_tangent,
             volumes=None,volume_tangent=None):
    from ...native._differential import _check_geometry,apply_preparation,_FirstOrderTangent
    from ...native._build import build
    from ...radiusSearch.chunked import ChunkedAdjacency
    if isinstance(adjacency,ChunkedAdjacency):raise NotImplementedError('Row-chunk operators do not support AD')
    inputs,options,topology=prepare(query,reference,domain,properties,adjacency,volumes)
    inputs[4]=None  # These two passes have no mass dependence.
    flat=[*inputs,None,volumes,*([None]*7)]
    if any(x is not None and forward_ad.unpack_dual(x).tangent is not None for x in flat):
        raise NotImplementedError('Explicit CRK JVPs are first-order; dual-valued primals are unsupported')
    directions=[None]*len(flat)
    for state,slots in ((query_tangent,(0,2)),(reference_tangent,(1,3))):
        if state is not None:
            for name,slot in zip(('positions','supports'),slots):directions[slot]=getattr(state,name,None)
    directions[18]=volume_tangent
    directions=[x.contiguous() if x is not None else None for x in directions]
    if any(x is not None and forward_ad.unpack_dual(x).tangent is not None for x in directions):
        raise NotImplementedError('Explicit CRK JVPs are first-order; dual-valued directions are unsupported')
    config=(kind,*options,1,2,False,False,kind==9)
    _check_geometry(config,flat,directions)
    # Primal row sums keep their ordinary reverse graph. The explicit product
    # is first-order; the Torch factor solve must not imply a native Hessian.
    primal=apply_preparation(kind,flat[:17],options,topology,volumes=flat[18])
    with torch.no_grad():
        example=primal[0] if isinstance(primal,tuple) else primal
        product=build().corrected_jvp(flat,directions,example,*config[:8])
    dependencies=[x for x in (*flat,*directions) if x is not None and x.requires_grad]
    if torch.is_grad_enabled() and dependencies:product=_FirstOrderTangent.apply(product,*dependencies)
    if kind==8:return primal,product[:,0]
    columns=[];start=0
    for value in primal[:-1]:
        width=prod(value.shape[1:])
        columns.append(product[:,start:start+width].reshape(value.shape));start+=width
    return primal,tuple(columns)


def computeCRKVolumeGeometryJVP(queryParticles,domain,kernel,adjacency,apparentVolumes,
        queryTangentState,referenceParticles=None,referenceTangentState=None,
        operationMode=OperationDirection.AllToAll):
    properties=OperationProperties(kernel,supportMode=SupportScheme.Gather,operationMode=operationMode)
    return _product(8,queryParticles,referenceParticles,domain,properties,adjacency,
                    queryTangentState,referenceTangentState)[1]


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
    from ...native._differential import _FirstOrderTangent
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
    _,products=torch.autograd.functional.jvp(factors,moments[:-1],dmoments,create_graph=False)
    dependencies=[x for x in (*moments[:-1],*dmoments) if x.requires_grad]
    if torch.is_grad_enabled() and dependencies:
        products=tuple(_FirstOrderTangent.apply(x,*dependencies) for x in products)
    return area,darea,CRKState(*primal),CRKTangentState(*products)
