"""Density's explicit position block, with fixed geometry and CSR/CSC products."""
import torch
from torch.autograd import forward_ad
from torch.autograd.function import once_differentiable
from ...enumTypes import OperationDirection,WarpOperation
from ..._models import OperationProperties


def _run(inputs,options,row_start=0,transpose=(),cotangent=None):
    from ...native._build import build
    return build().density_position_hvp(inputs,list(transpose),cotangent,*options,row_start)


class _PositionBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx,options,topology,row_start,*inputs):
        ctx.options,ctx.topology,ctx.row_start=options,topology,row_start
        ctx.token=topology.token()
        ctx.save_for_backward(*inputs)
        ctx.save_for_forward(*inputs)
        return _run(list(inputs),options,row_start)[0]

    @staticmethod
    @once_differentiable
    def backward(ctx,cotangent):
        if ctx.topology.token()!=ctx.token:raise RuntimeError('CSR topology changed before density HVP backwards')
        inputs=list(ctx.saved_tensors)
        if any(forward_ad.unpack_dual(t).tangent is not None for t in (*inputs,cotangent) if t is not None):
            raise NotImplementedError('Density position-block VJP is first-order')
        csc=ctx.topology.transpose()
        q,r,m=_run(inputs,ctx.options,ctx.row_start,(csc.edgeOffsets,csc.numNeighbors,csc.paddedJ),cotangent.contiguous())
        gradients=[None]*len(inputs)
        for slot,value in ((4,m),(13,q),(14,r)):
            if ctx.needs_input_grad[slot+3]:gradients[slot]=value
        return None,None,None,*gradients

    @staticmethod
    def jvp(ctx,_options,_topology,_row_start,*directions):
        from ...native._differential import _FirstOrderTangent
        inputs=list(ctx.saved_tensors)
        result=torch.zeros_like(inputs[13])
        for mass_direction in (False,True):
            slots=(4,) if mass_direction else (13,14)
            if not any(directions[slot] is not None for slot in slots):continue
            values=list(inputs)
            for slot in slots:
                values[slot]=directions[slot].contiguous() if directions[slot] is not None else None
            if not mass_direction and values[13] is None:values[13]=torch.zeros_like(inputs[13])
            result=result+_run(values,ctx.options,ctx.row_start)[0]
        dependencies=[t for t in (*inputs,*directions) if t is not None and t.requires_grad]
        return _FirstOrderTangent.apply(result,*dependencies) if torch.is_grad_enabled() and dependencies else result


def computeSPHDensityPositionHVP(queryParticles,domain,kernel,supportMode,adjacency,
        tangentQueryPositions,referenceParticles=None,tangentReferencePositions=None,
        operationMode=OperationDirection.AllToAll,*,_rowStart=0):
    """Sum m_j H_ij (v_i-v_j), excluding equal global particle IDs.

    This preserves the existing implicit-shifting block and regularized Hessian
    convention. Geometry/support/domain are fixed; first-order products in mass
    and the supplied vectors are supported. It is not general double backward.
    An omitted reference vector is zero, even for a self population.
    """
    from ...native._inputs import prepare_geometry,prepare_adjacency
    from ...native._checks import check_execution
    from ...native._topology import CSRTopology
    from ...radiusSearch.chunked import ChunkedAdjacency
    reference=queryParticles if referenceParticles is None else referenceParticles
    properties=OperationProperties(kernel,operation=WarpOperation.Density,
        supportMode=supportMode,operationMode=operationMode)
    geometry=prepare_geometry(queryParticles,reference,domain)
    check_execution(reference.masses,tangentQueryPositions,tangentReferencePositions,
                    allow_reverse=True,allow_forward=True)
    if isinstance(adjacency,ChunkedAdjacency):
        from ...autograd.operator_spec import SPHContext,RowRole,launchOperator
        from ...native._row_dispatch import spec,output
        extras={'queryVector':tangentQueryPositions,'rowStart':0}
        roles={'queryVector':RowRole.QUERY,'rowStart':RowRole.ROW_START}
        if tangentReferencePositions is not None:
            extras['referenceVector']=tangentReferencePositions;roles['referenceVector']=RowRole.REFERENCE
        def launch(ctx,e):
            return computeSPHDensityPositionHVP(ctx.query,ctx.domain,kernel,supportMode,ctx.adjacency,
                e['queryVector'],ctx.reference,e.get('referenceVector'),operationMode,_rowStart=e['rowStart'])
        declaration=spec('native.density.position_block',launch,[output((domain.dim,))],roles)
        return launchOperator(declaration,SPHContext(queryParticles,properties,domain,adjacency,reference),**extras)
    topology=adjacency if isinstance(adjacency,CSRTopology) else CSRTopology(
        prepare_adjacency(queryParticles,reference,domain,properties,adjacency))
    csr=topology.adjacency
    inputs=[*geometry[:4],reference.masses,queryParticles.kinds,reference.kinds,*geometry[4:],
        csr.edgeOffsets,csr.numNeighbors,csr.paddedJ,tangentQueryPositions,tangentReferencePositions,None,None]
    options=(supportMode.value,operationMode.value,kernel.value)
    active=any(t is not None and ((torch.is_grad_enabled() and t.requires_grad) or
        forward_ad.unpack_dual(t).tangent is not None) for t in inputs)
    if active:return _PositionBlock.apply(options,topology,_rowStart,*inputs)
    return _run(inputs,options,_rowStart)[0]


def warpOperationHVP(queryParticles,operationProperties,domain,tangentQueryPositions,
                     adjacency=None,referenceParticles=None,tangentReferencePositions=None):
    """Public adapter for the legacy density position block only."""
    if operationProperties.operation != WarpOperation.Density:
        raise NotImplementedError('warpOperationHVP supports only the density position block')
    return computeSPHDensityPositionHVP(queryParticles,domain,operationProperties.kernel,
        operationProperties.supportMode,adjacency,tangentQueryPositions,referenceParticles,
        tangentReferencePositions,operationProperties.operationMode)
