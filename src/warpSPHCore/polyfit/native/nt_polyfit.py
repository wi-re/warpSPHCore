"""Owned polynomial assembly, directional products and their first reverse."""
from dataclasses import dataclass, replace
import torch
from torch.autograd import forward_ad
from torch.autograd.function import once_differentiable
from ...native._inputs import prepare_adjacency, prepare_geometry, materialize_cotangent
from ...native._topology import CSRTopology
from ...native._options import normalization_factor
from ...enumTypes import SupportScheme, OperationDirection, KernelFunctions
from ..._models import OperationProperties, ParticleTangentState


@dataclass
class _Spec:
    inputs: list
    exponents: torch.Tensor
    entries: torch.Tensor
    topology: CSRTopology
    options: tuple

    def arguments(self, values):
        qx,rx,qh,rh,mass,density,volumes,field,lower,upper = values[:10]
        inputs = self.inputs.copy()
        for slot,value in ((0,qx),(1,rx),(2,qh),(3,rh),(4,mass),(7,lower),(8,upper),(15,density)):
            inputs[slot] = value.contiguous()
        return inputs,self.exponents,self.entries,volumes.contiguous(),field.contiguous()

    def metadata(self):
        return (self.exponents,self.entries,*(self.inputs[k] for k in (5,6,9,10,11,12)))


class _Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx,spec,*values):
        from ...native._build import build
        ctx.spec,ctx.token,ctx.size = spec,spec.topology.token(),len(values)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(*values,*spec.metadata())
        return build().polyfit_jvp(*spec.arguments(values),list(values[10:19]),*spec.options)

    @staticmethod
    @once_differentiable
    def backward(ctx,cotangent):
        from ...native._build import build
        values=ctx.saved_tensors[:ctx.size]
        if cotangent is None:return (None,)*(len(values)+1)
        if ctx.spec.topology.token()!=ctx.token:raise RuntimeError('CSR topology changed before native polynomial product reverse')
        csc=ctx.spec.topology.transpose()
        bars=build().polyfit_product_vjp(*ctx.spec.arguments(values),list(values[10:19]),
            materialize_cotangent(cotangent),[csc.edgeOffsets,csc.numNeighbors,csc.paddedJ],*ctx.spec.options)
        length=bars[8].sum(0);direction_length=bars[17].sum(0)
        gradients=[*bars[:8],-length,length,*bars[9:17],direction_length]
        return None,*(None if x is None or not ctx.needs_input_grad[k+1] else gradients[k] if k<19 else torch.zeros_like(x)
                      for k,x in enumerate(values))


class _Polynomial(torch.autograd.Function):
    @staticmethod
    def forward(ctx,spec,*values):
        from ...native._build import build
        output,counts=build().polyfit_forward(*spec.arguments(values),*spec.options)
        ctx.spec,ctx.token,ctx.size=spec,spec.topology.token(),len(values)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(*values,*spec.metadata());ctx.save_for_forward(*values,*spec.metadata())
        if counts is not None:ctx.mark_non_differentiable(counts)
        return output,counts

    @staticmethod
    @once_differentiable
    def backward(ctx,cotangent,_counts):
        from ...native._build import build
        values=ctx.saved_tensors[:ctx.size]
        if cotangent is None:return (None,)*(len(values)+1)
        if ctx.spec.topology.token()!=ctx.token:raise RuntimeError('CSR topology changed before native polynomial backwards')
        if any(x is not None and forward_ad.unpack_dual(x).tangent is not None for x in (*values,cotangent)):
            raise NotImplementedError('Ordinary polynomial reverse is first-order; forward-over-backwards has no registration')
        csc=ctx.spec.topology.transpose()
        bars=build().polyfit_vjp(*ctx.spec.arguments(values),materialize_cotangent(cotangent),
            [csc.edgeOffsets,csc.numNeighbors,csc.paddedJ],*ctx.spec.options)
        length=bars[8].sum(0);gradients=[*bars[:8],-length,length]
        return None,*(None if x is None or not ctx.needs_input_grad[k+1] else gradients[k] if k<10 else torch.zeros_like(x)
                      for k,x in enumerate(values))

    @staticmethod
    def jvp(ctx,_spec,*directions):
        values=ctx.saved_tensors[:ctx.size];lower,upper=directions[8:10]
        length=None if lower is None and upper is None else ((torch.zeros_like(values[9]) if upper is None else upper)
            -(torch.zeros_like(values[8]) if lower is None else lower))
        tangents=[None if t is None else t.contiguous() for t in (*directions[:8],length)]
        return _Product.apply(ctx.spec,*values[:10],*tangents,*values[10:],*directions[10:]),None


def _sum(query,properties,domain,exponents,values=None,*,subtractCenter=False,
         queryVolumes=None,referenceVolumes=None,adjacency=None,referenceParticles=None,
         sector=-1,numSectors=8,unitWeights=False,_tangents=None,_rowStart=0):
    if not isinstance(properties.kernel,KernelFunctions):raise TypeError('kernel must be a KernelFunctions')
    reference=query if referenceParticles is None else referenceParticles
    from ...radiusSearch.chunked import ChunkedAdjacency
    if isinstance(adjacency,ChunkedAdjacency):
        if _tangents is not None:raise NotImplementedError('Row-chunk polynomial geometry JVP is not supported')
        return _chunked(query,reference,properties,domain,exponents,values,queryVolumes,
                        referenceVolumes,adjacency,sector,numSectors,unitWeights,subtractCenter)
    if type(adjacency).__name__ in ('CompactHashMap','SearchIndex'):adjacency=None
    if _tangents is not None:
        domain=replace(domain,min=forward_ad.unpack_dual(domain.min).primal.detach(),
                       max=forward_ad.unpack_dual(domain.max).primal.detach())
    scale=1. if unitWeights else normalization_factor(properties,domain.dim)
    properties=replace(properties,calibrateNormalization=False,positiveDivergence=False)
    geometry=prepare_geometry(query,reference,domain,allow_reverse=True,allow_forward=True)
    topology=adjacency if isinstance(adjacency,CSRTopology) else None
    adjacency=prepare_adjacency(query,reference,domain,properties,adjacency)
    topology=topology or CSRTopology(adjacency)
    inputs=[*geometry[:4],reference.masses,query.kinds,reference.kinds,*geometry[4:],
            adjacency.edgeOffsets,adjacency.numNeighbors,adjacency.paddedJ,None,None,None,None]
    inputs=[forward_ad.unpack_dual(t).primal if isinstance(t,torch.Tensor) else t for t in inputs]
    entries=torch.tril_indices(len(exponents),len(exponents),device=exponents.device).T.int().contiguous()
    empty=query.positions.new_empty(0)
    volumes=referenceVolumes if referenceVolumes is not None else queryVolumes
    mass,density=reference.masses,reference.densities
    if mass is None:
        if not unitWeights and volumes is None:raise ValueError('Polynomial weights need source masses or explicit volumes')
        mass=reference.positions.new_ones(len(reference.positions))
    if density is None:
        if not unitWeights and volumes is None:raise ValueError('Polynomial weights need source densities or explicit volumes')
        density=torch.ones_like(mass)
    arguments=(query.positions,reference.positions,query.supports,reference.supports,mass,density,
               empty if volumes is None else volumes,empty if values is None else values,domain.min,domain.max)
    ignored=(query.masses,query.densities,queryVolumes)
    spec=_Spec(inputs,exponents.contiguous(),entries,topology,
        (properties.supportMode.value,properties.operationMode.value,properties.kernel.value,int(sector),
         int(numSectors),bool(unitWeights),bool(subtractCenter),int(_rowStart),float(scale)))
    if _tangents is not None:
        if any(t is not None and forward_ad.unpack_dual(t).tangent is not None for t in (*arguments,*_tangents,*ignored)):
            raise NotImplementedError('The reference explicit polynomial product has no second-forward registration')
        directions=[None if t is None else t.contiguous() for t in _tangents]
        return _Product.apply(spec,*arguments,*directions,*ignored),None
    return _Polynomial.apply(spec,*arguments,*ignored)


def _computePolyMoments_stateBackend(queryParticles,operationProperties,domain,exponents,**kwargs):
    return _sum(queryParticles,operationProperties,domain,exponents,**kwargs)


def _computePolyRHS_stateBackend(queryParticles,operationProperties,domain,exponents,values,subtractCenter,**kwargs):
    return _sum(queryParticles,operationProperties,domain,exponents,values,subtractCenter=subtractCenter,**kwargs)[0]


def _geometry_directions(tangent,query,field=None):
    def get(name):
        value=None if tangent is None else getattr(tangent,name,None)
        original=getattr(query,name)
        return (query.positions.new_zeros(len(query.positions)) if original is None else torch.zeros_like(original)) if value is None else value
    return [get('positions'),get('positions'),get('supports'),get('supports'),
            get('masses'),get('densities'),None,field,None]


def computePolyMomentsGeometryJVP(queryParticles,domain,kernel,adjacency,queryTangentState,exponents,
        supportMode=SupportScheme.Gather,sector=-1,numSectors=8,unitWeights=False,
        operationMode=OperationDirection.AllToAll):
    properties=OperationProperties(kernel,supportMode=supportMode,operationMode=operationMode)
    return _sum(queryParticles,properties,domain,exponents,adjacency=adjacency,
                sector=sector,numSectors=numSectors,unitWeights=unitWeights,
                _tangents=_geometry_directions(queryTangentState,queryParticles))[0]


def computePolyRHSGeometryJVP(queryParticles,domain,kernel,adjacency,queryTangentState,exponents,
        values,tangentValues,subtractCenter,supportMode=SupportScheme.Gather,sector=-1,numSectors=8,
        unitWeights=False,operationMode=OperationDirection.AllToAll):
    properties=OperationProperties(kernel,supportMode=supportMode,operationMode=operationMode)
    tangentValues=torch.zeros_like(values) if tangentValues is None else tangentValues
    return _sum(queryParticles,properties,domain,exponents,values,subtractCenter=subtractCenter,
                adjacency=adjacency,sector=sector,numSectors=numSectors,unitWeights=unitWeights,
                _tangents=_geometry_directions(queryTangentState,queryParticles,tangentValues))[0]


def _chunked(query,reference,properties,domain,exponents,values,queryVolumes,referenceVolumes,
             adjacency,sector,numSectors,unitWeights,subtract):
    from ...native._row_dispatch import spec,output,corrections,SPHContext,RowRole,launchOperator
    context=SPHContext(query,properties,domain,adjacency,reference,corrections(queryVolumes,referenceVolumes))
    extras=dict(exponents=exponents,sector=sector,numSectors=numSectors,unitWeights=unitWeights,
                subtractCenter=subtract,rowStart=0)
    roles=dict(exponents=RowRole.REPLICATED,sector=None,numSectors=None,unitWeights=None,
               subtractCenter=None,rowStart=RowRole.ROW_START)
    if values is None:
        width=len(exponents)*(len(exponents)+1)//2
        outputs=[output((width,)),output(dtype=torch.int32)]
    else:
        extras['values']=values;roles['values']=RowRole.REFERENCE
        outputs=[output(lambda ctx,e:(len(e['exponents'])*e['values'].shape[1],))]
    def launch(ctx,fields):
        result=_sum(ctx.query,ctx.properties,ctx.domain,fields['exponents'],fields.get('values'),
            subtractCenter=fields['subtractCenter'],queryVolumes=ctx.corrections.volumes[0],
            referenceVolumes=ctx.corrections.volumes[1],adjacency=ctx.adjacency,referenceParticles=ctx.reference,
            sector=fields['sector'],numSectors=fields['numSectors'],unitWeights=fields['unitWeights'],
            _rowStart=fields['rowStart'])
        return result if values is None else result[0]
    declaration=spec('native.polyfit.'+('moments' if values is None else 'rhs'),launch,outputs,roles)
    result=launchOperator(declaration,context,**extras)
    return result if values is None else (result,None)
