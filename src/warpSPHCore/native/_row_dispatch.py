"""Native launch records for the core's shared owned-row dispatcher."""
from dataclasses import dataclass
import torch
from ..autograd.operator_spec import (SPHContext,Corrections,OperatorSpec,OutputSpec,
                                      ExtraSpec,ExtraKind,RowRole,launchOperator)
from .._models import GradHState,RenormalizationState


@dataclass(frozen=True)
class Kernel:
    key: str
    function: object

    def launch_native(self,context,extras):
        return self.function(context,extras)


def output(trailing=(),dtype=None):
    def resolve(context,extras):
        shape = trailing(context,extras) if callable(trailing) else trailing
        return (context.query.positions.dtype if dtype is None else dtype,tuple(shape))
    return OutputSpec(dtype=resolve)


def spec(key,function,outputs,roles,*,queryOnlyCRK=True,rowChunked=True):
    extras = tuple(ExtraSpec(name,ExtraKind.SCALAR if role is None or role is RowRole.ROW_START else
                            ExtraKind.CHUNK_TENSOR if role is RowRole.CHUNK else ExtraKind.TENSOR,role)
                   for name,role in roles.items())
    return OperatorSpec(Kernel(key,function),tuple(outputs),extras,rowChunked=rowChunked,
                        queryOnlyCRK=queryOnlyCRK,blockDim=128)


def corrections(queryVolumes=None,referenceVolumes=None,crkState=None,gradHState=None,renormalizationState=None):
    if gradHState is not None and not isinstance(gradHState,GradHState):
        gradHState = GradHState(*gradHState) if isinstance(gradHState,tuple) else GradHState(gradHState)
    if renormalizationState is not None and not isinstance(renormalizationState,RenormalizationState):
        renormalizationState = RenormalizationState(renormalizationState)
    return Corrections((queryVolumes,referenceVolumes),crkState,gradHState,renormalizationState)


def standard(query,reference,domain,properties,adjacency,queryValues,referenceValues,
             queryVolumes,referenceVolumes,crkState,gradHState,renormalizationState,
             consistent,counts):
    from ..coreOperations.native.wrapper import sphOperation
    from ..enumTypes import WarpOperation as Op
    context = SPHContext(query,properties,domain,adjacency,reference,
                        corrections(queryVolumes,referenceVolumes,crkState,gradHState,renormalizationState))
    extras = {name:value for name,value in (('queryValues',queryValues),('referenceValues',referenceValues)) if value is not None}
    roles = {name:RowRole.QUERY if name=='queryValues' else RowRole.REFERENCE for name in extras}
    def shape(ctx,values):
        operation=ctx.properties.operation;dimension=ctx.domain.dim
        if operation is Op.Density:return ()
        if operation is Op.Covariance:return (dimension,dimension)
        field=values.get('referenceValues')
        dimensions=() if field is None else tuple(field.shape[1:])
        if operation is Op.Gradient:return (*dimensions,dimension)
        if operation is Op.Divergence:return dimensions[1:] if ctx.properties.divergenceDotMode else dimensions[:-1]
        if operation is Op.Curl:
            if dimension==1:return ()
            if dimension==2:return dimensions[1:] or (1,)
        return dimensions
    def launch(ctx,values):
        c=ctx.corrections
        return sphOperation(ctx.query,ctx.properties,ctx.domain,
            values.get('queryValues'),values.get('referenceValues'),*c.volumes,
            ctx.adjacency,ctx.reference,crkState=c.crk,gradHState=c.gradH,renormalizationState=c.renorm,
            consistentDivergence=consistent,covarianceReturnNumNeighbors=counts,backend='native')
    outputs=[output(shape)]
    if counts:outputs.append(output(dtype=torch.int32))
    declaration=spec(f'native.{properties.operation.name}.consistent{int(consistent)}.counts{int(counts)}',launch,outputs,roles)
    return launchOperator(declaration,context,**extras)


def preparation(kind,function,query,reference,domain,properties,adjacency,
                queryVolumes=None,referenceVolumes=None,crkState=None):
    context=SPHContext(query,properties,domain,adjacency,reference,
                       corrections(queryVolumes,referenceVolumes,crkState))
    if kind==9:
        dimension=domain.dim
        outputs=[output(shape) for shape in ((),(dimension,),(dimension,dimension),
                 (dimension,),(dimension,dimension),(dimension,dimension,dimension))]
        outputs.append(output(dtype=torch.int32))
    else:outputs=[output()]
    def launch(ctx,extras):
        if kind==8:return function(ctx.query,ctx.properties,ctx.domain,ctx.adjacency,ctx.reference)
        c=ctx.corrections
        if kind==9:return function(ctx.query,ctx.properties,ctx.domain,*c.volumes,ctx.adjacency,ctx.reference)
        state=c.crk[0] if isinstance(c.crk,tuple) else c.crk
        return function(ctx.query,ctx.properties,ctx.domain,state,*c.volumes,ctx.adjacency,ctx.reference)
    return launchOperator(spec(f'native.preparation.{kind}',launch,outputs,{}),context)
