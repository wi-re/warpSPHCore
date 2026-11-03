"""Explicit first-order products with independent query/reference tangent roles."""
import torch
from torch.autograd import forward_ad
from ..._models import RenormalizationTangentState
from ...enumTypes import WarpOperation


def warpOperationJVP(queryParticles, operationProperties, domain,
    tangentQueryValues=None, tangentReferenceValues=None,
    queryTangentState=None, referenceTangentState=None,
    queryValues=None, referenceValues=None, queryVolumes=None, referenceVolumes=None,
    tangentQueryVolumes=None, tangentReferenceVolumes=None, adjacency=None,
    referenceParticles=None, crkState=None, crkTangentState=None, gradHState=None,
    renormalizationState=None, renormalizationTangentState=None, consistentDivergence=False):
    """Return the analytic first-order product on a fixed CSR topology.

    Omitted tangents are zero, including reference tangents in a self cloud.
    Unlike an implicit dual-valued call, this helper never aliases a query
    tangent into the reference role. It also works inside an existing dual level.
    Support selection, minimum-image winding and clipped branches stay fixed.
    Higher derivatives of this product are deliberately unsupported.
    """
    from .nt_corrected import _prepare
    from ...native._differential import _core_primal, _check_geometry, _FirstOrderTangent
    from ...native._options import laplacian_policy, normalization_factor
    from ...native._build import build
    from ...radiusSearch.chunked import ChunkedAdjacency
    if isinstance(adjacency,ChunkedAdjacency):
        raise NotImplementedError('Row-chunk operators do not support AD')
    properties=operationProperties
    reference=queryParticles if referenceParticles is None else referenceParticles
    referenceValues=queryValues if referenceValues is None else referenceValues
    operation=properties.operation
    if operation == WarpOperation.Density:
        if tangentQueryValues is not None or tangentReferenceValues is not None:
            raise ValueError('Density has no field-value tangent')
        queryValues=referenceValues=None
        queryVolumes=referenceVolumes=crkState=gradHState=renormalizationState=None
    elif operation == WarpOperation.Covariance:
        queryValues=referenceValues=None
        gradHState=None
    elif operation == WarpOperation.Interpolate:
        queryValues=gradHState=renormalizationState=None
    # Calibration is a constant geometry-independent output factor.
    from dataclasses import replace
    factor=normalization_factor(properties,domain.dim) if properties.calibrateNormalization else 1.
    properties=replace(properties,calibrateNormalization=False)
    inputs,corrections,_=_prepare(queryParticles,reference,domain,properties,adjacency,
        queryValues,referenceValues,queryVolumes,referenceVolumes,crkState,gradHState,
        renormalizationState,consistent_divergence=consistentDivergence,allow_forward=True)
    flat=[*inputs,*corrections]
    if any(t is not None and forward_ad.unpack_dual(t).tangent is not None for t in flat):
        raise NotImplementedError('Explicit native JVPs are first-order; dual-valued primals are unsupported')
    tangents=[None]*len(flat)
    for state,slots in ((queryTangentState,(0,2,None,16)),(referenceTangentState,(1,3,4,15))):
        if state is not None:
            for name,slot in zip(('positions','supports','masses','densities'),slots):
                if slot is not None:tangents[slot]=getattr(state,name,None)
    tangents[13],tangents[14]=tangentQueryValues,tangentReferenceValues
    tangents[17],tangents[18]=tangentQueryVolumes,tangentReferenceVolumes
    if referenceVolumes is None and tangentReferenceVolumes is None:tangents[18]=tangentQueryVolumes
    if crkTangentState is not None:
        state=crkTangentState[0] if isinstance(crkTangentState,tuple) else crkTangentState
        for slot,name in enumerate(('A','B','gradA','gradB'),19):tangents[slot]=getattr(state,name,None)
    if renormalizationTangentState is not None:
        tangents[24]=(renormalizationTangentState.renormalizationMatrices
            if isinstance(renormalizationTangentState,RenormalizationTangentState) else renormalizationTangentState)
    # Primal dispatch intentionally ignores unread options. Their tangents must
    # follow the same policy, without manufacturing dummy native pointers.
    tangents=[t.contiguous() if t is not None and p is not None else None for p,t in zip(flat,tangents)]
    if any(t is not None and forward_ad.unpack_dual(t).tangent is not None for t in tangents):
        raise NotImplementedError('Explicit native JVPs are first-order; dual-valued directions are unsupported')
    config=(operation.value,properties.supportMode.value,properties.operationMode.value,
        properties.kernel.value,properties.gradientMode.value,laplacian_policy(properties),
        properties.divergenceDotMode,consistentDivergence,False)
    _check_geometry(config,flat,tangents)
    with torch.no_grad():
        example=_core_primal(config,flat)
        result=build().corrected_jvp(flat,tangents,example,*config[:8])*factor
    dependencies=[t for t in (*flat,*tangents) if isinstance(t,torch.Tensor) and t.requires_grad]
    return _FirstOrderTangent.apply(result,*dependencies) if torch.is_grad_enabled() and dependencies else result
