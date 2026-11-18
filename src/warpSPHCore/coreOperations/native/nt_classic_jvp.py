"""Ordinary forward-AD tangents retain the reference-supported first reverse."""
import importlib
import torch
from .._jvpFields import field_directions


def packed_operator_jvp(ctx, _config, _topology, *tangents):
    """Dispatch one packed operator block, including native-only directions."""
    inputs, example = ctx.saved_tensors[:-1], ctx.saved_tensors[-1]
    result = product(ctx.config, inputs, tangents)
    if result is not None:
        return result

    from ...native._build import build
    from ...native._differential import _FirstOrderTangent

    directions = [t.contiguous() if slot in ctx.used and t is not None else None
                  for slot, t in enumerate(tangents[:26])]
    with torch.no_grad():
        result = build().corrected_jvp(list(inputs[:26]), directions, example, *ctx.config[:8])
    dependencies = [value for value in (*inputs, *tangents)
                    if value is not None and value.requires_grad]
    if torch.is_grad_enabled() and dependencies:
        result = _FirstOrderTangent.apply(result, *dependencies)
    return result


def product(config,inputs,tangents):
    operation=config[0];width=17 if operation==6 else 26
    flat=list(inputs[:width]);directions=list(tangents[:width])
    # Existing native bounds/omega-direction extensions retain their original
    # first-order route. The public Warp JVP has no formula for these directions.
    extended=[k for k in (7,8,*((23,25) if width==26 else ())) if directions[k] is not None]
    if extended:
        needs_reverse=any(value is not None and value.requires_grad for value in (*inputs,*tangents))
        capturing=flat[0].is_cuda and torch.cuda.is_current_stream_capturing()
        if not needs_reverse or capturing or any(bool(directions[k].count_nonzero()) for k in extended):
            return None
        # The Warp bridge drops exactly-zero directions before dispatch.
        for k in extended:directions[k]=None
    name={1:'interpolate',2:'gradient',3:'divergence',4:'curl',5:'laplacian',6:'density',7:'covariance'}[operation]
    query_mass=inputs[width] if len(inputs)>width else None
    ignored=(*inputs[width+1:],*tangents[width:])
    field_only=operation not in (6,7) and any(directions[k] is not None for k in (13,14)) and not any(value is not None for k,value in enumerate(directions) if k not in (13,14))
    if operation==5 and field_only and config[5]&16:
        from .nt_laplacian_jvp import apply
        return apply(config,flat,directions,query_mass,ignored=ignored,bounds_ad=True,field_only=True)
    if field_only:
        # A pure field product is another ordinary operator. Its reverse keeps
        # the bound/correction couplings, as does the public Warp field route.
        flat[13],flat[14]=field_directions(len(flat[0]),len(flat[1]),flat[13],flat[14],directions[13],directions[14])
        owner=importlib.import_module('.nt_'+name+'_backward',__package__)
        from ...enumTypes import OperationDirection,KernelFunctions,SupportScheme,GradientScheme,LaplacianScheme,WarpOperation
        from ..._models import OperationProperties
        from ...native._options import POSITIVE_DIVERGENCE
        properties=OperationProperties(operation=WarpOperation(operation),supportMode=SupportScheme(config[1]),operationMode=OperationDirection(config[2]),kernel=KernelFunctions(config[3]),gradientMode=GradientScheme(config[4]),laplacianMode=LaplacianScheme(config[5]&~POSITIVE_DIVERGENCE),positiveDivergence=bool(config[5]&POSITIVE_DIVERGENCE),divergenceDotMode=config[6])
        options=dict(corrections=flat[17:],query_mass=query_mass,ignored=ignored)
        if operation==3:options['consistent']=config[7]
        return owner.apply(flat[:17],properties,**options)
    owner=importlib.import_module('.nt_'+name+'_jvp',__package__)
    # Retain unread ordinary density inputs as explicit zero dependencies.
    if operation==1:ignored=(*ignored,flat[16],directions[16])
    return owner.apply(config,flat,directions,query_mass,ignored=ignored)
