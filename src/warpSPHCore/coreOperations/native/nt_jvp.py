"""Explicit products with independent query/reference tangent roles."""
import torch
from torch.autograd import forward_ad
from ..._models import RenormalizationTangentState
from ...enumTypes import WarpOperation, GradientScheme


def _active_correction_tangent(crk,renormalization):
    if isinstance(crk,tuple):crk=crk[0]
    if crk is not None and any(getattr(crk,name,None) is not None for name in ('A','B','gradA','gradB')):return True
    return renormalization is not None and (isinstance(renormalization,torch.Tensor) or getattr(renormalization,'renormalizationMatrices',None) is not None)


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
    Density, interpolation, gradient, divergence, curl and Laplacian support the reference first reverse
    of their explicit products.
    Other owners retain their individually documented derivative limits.
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
    if operation==WarpOperation.Density:
        from .nt_density_jvp import compute
        return compute(queryParticles,reference,domain,properties,adjacency,queryTangentState,referenceTangentState)*factor
    if operation in (WarpOperation.Interpolate,WarpOperation.Gradient,WarpOperation.Divergence,WarpOperation.Curl,WarpOperation.Laplacian) and not any(
            getattr(state,name,None) is not None for state in (queryTangentState,referenceTangentState)
            if state is not None for name in ('positions','supports','masses','densities')) and all(
            x is None for x in (tangentQueryVolumes,tangentReferenceVolumes)) and not _active_correction_tangent(crkTangentState,renormalizationTangentState) and (
            tangentReferenceValues is not None or tangentQueryValues is not None):
        # Warp's field-only route relaunches the ordinary field operator.
        # It retains bound/correction reverse, unlike its geometry launcher.
        from .wrapper import sphOperation
        from .._jvpFields import field_directions
        query_direction,reference_direction=field_directions(len(queryParticles.positions),len(reference.positions),
            queryValues,referenceValues,tangentQueryValues,tangentReferenceValues)
        return sphOperation(queryParticles,properties,domain,queryValues=query_direction,referenceValues=reference_direction,
            queryVolumes=queryVolumes,referenceVolumes=referenceVolumes,adjacency=adjacency,
            referenceParticles=reference,crkState=crkState,gradHState=gradHState,renormalizationState=renormalizationState,consistentDivergence=consistentDivergence,backend='native')*factor
    inputs,corrections,_=_prepare(queryParticles,reference,domain,properties,adjacency,
        queryValues,referenceValues,queryVolumes,referenceVolumes,crkState,gradHState,
        renormalizationState,consistent_divergence=consistentDivergence,allow_forward=True)
    flat=[*inputs,*corrections]
    if operation in (WarpOperation.Interpolate,WarpOperation.Gradient,WarpOperation.Divergence,WarpOperation.Curl,WarpOperation.Laplacian):
        for slot in (7,8):flat[slot]=forward_ad.unpack_dual(flat[slot]).primal.detach()
        if operation==WarpOperation.Interpolate:flat[16]=None
        # The reference launcher supplies zero defaults for absent, unused
        # density fields. Retain a provided density-direction tensor as its
        # state-aware zero-gradient dependency, even when that primal is absent.
        if operation in (WarpOperation.Gradient,WarpOperation.Divergence,WarpOperation.Curl,WarpOperation.Laplacian) and not (operation==WarpOperation.Divergence and consistentDivergence) and properties.gradientMode!=GradientScheme.Symmetric and flat[16] is None and getattr(queryTangentState,'densities',None) is not None:
            flat[16]=torch.zeros_like(flat[2])
        if flat[15] is None and flat[18] is not None and (operation==WarpOperation.Interpolate or properties.gradientMode!=GradientScheme.Symmetric) and not (operation==WarpOperation.Divergence and consistentDivergence) and getattr(referenceTangentState,'densities',None) is not None:
            flat[15]=torch.zeros_like(flat[3])
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
    if operation==WarpOperation.Interpolate:
        from .nt_interpolate_jvp import apply
        return apply(config,flat,tangents,queryParticles.masses)*factor
    if operation==WarpOperation.Gradient:
        from .nt_gradient_jvp import apply
        return apply(config,flat,tangents,queryParticles.masses)*factor
    if operation==WarpOperation.Divergence:
        from .nt_divergence_jvp import apply
        return apply(config,flat,tangents,queryParticles.masses)*factor
    if operation==WarpOperation.Curl:
        from .nt_curl_jvp import apply
        return apply(config,flat,tangents,queryParticles.masses)*factor
    if operation==WarpOperation.Laplacian:
        from .nt_laplacian_jvp import apply
        return apply(config,flat,tangents,queryParticles.masses)*factor
    with torch.no_grad():
        if operation == WarpOperation.Interpolate:
            from .nt_interpolate_backward import product, used_slots
            flat=[value.contiguous() if value is not None else None for value in flat]
            used=used_slots(flat)
            tangents=[t if slot in used else None for slot,t in enumerate(tangents)]
            result=product(config,flat,tangents)*factor
        elif operation == WarpOperation.Gradient:
            from .nt_gradient_backward import product, used_slots
            flat=[value.contiguous() if value is not None else None for value in flat]
            used=used_slots(flat,config[4])
            tangents=[t if slot in used else None for slot,t in enumerate(tangents)]
            result=product(config,flat,tangents)*factor
        elif operation == WarpOperation.Divergence:
            from .nt_divergence_backward import product, used_slots
            flat=[value.contiguous() if value is not None else None for value in flat]
            used=used_slots(flat,config[4],config[7])
            tangents=[t if slot in used else None for slot,t in enumerate(tangents)]
            result=product(config,flat,tangents)*factor
        elif operation == WarpOperation.Curl:
            from .nt_curl_backward import product, used_slots
            flat=[value.contiguous() if value is not None else None for value in flat]
            used=used_slots(flat,config[4])
            tangents=[t if slot in used else None for slot,t in enumerate(tangents)]
            result=product(config,flat,tangents)*factor
        elif operation == WarpOperation.Laplacian:
            from .nt_laplacian_backward import product, used_slots
            flat=[value.contiguous() if value is not None else None for value in flat]
            used=used_slots(flat,config[4])
            tangents=[t if slot in used else None for slot,t in enumerate(tangents)]
            result=product(config,flat,tangents)*factor
        elif operation == WarpOperation.Covariance:
            from .nt_covariance_backward import product, used_slots
            flat=[value.contiguous() if value is not None else None for value in flat]
            used=used_slots(flat)
            tangents=[t if slot in used else None for slot,t in enumerate(tangents)]
            result=product(config,flat,tangents)*factor
        else:
            example=_core_primal(config,flat)
            result=build().corrected_jvp(flat,tangents,example,*config[:8])*factor
    dependencies=[t for t in (*flat,*tangents) if isinstance(t,torch.Tensor) and t.requires_grad]
    return _FirstOrderTangent.apply(result,*dependencies) if torch.is_grad_enabled() and dependencies else result
