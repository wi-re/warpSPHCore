"""Polynomial neighbor sums with analytic CSR/CSC first-order derivatives."""
from dataclasses import dataclass
import torch
from torch.autograd import forward_ad
from torch.autograd.function import once_differentiable
from ...native._inputs import prepare_adjacency, prepare_geometry
from ...native._topology import CSRTopology
from ...enumTypes import SupportScheme, OperationDirection, KernelFunctions
from ..._models import OperationProperties


@dataclass
class _Spec:
    inputs: list
    exponents: torch.Tensor
    entries: torch.Tensor
    topology: CSRTopology
    options: tuple

    def arguments(self, values):
        qx,rx,qh,mass,density,volumes,field,lower,upper = values
        inputs = self.inputs.copy()
        for slot,value in ((0,qx),(1,rx),(2,qh),(4,mass),(7,lower),(8,upper),(15,density)):
            inputs[slot] = value.contiguous()
        return inputs,self.exponents,self.entries,volumes.contiguous(),field.contiguous()


class _Polynomial(torch.autograd.Function):
    @staticmethod
    def forward(ctx,spec,*values):
        from ...native._build import build
        output,counts = build().polyfit_forward(*spec.arguments(values),*spec.options)
        ctx.spec = spec
        metadata = (spec.exponents,spec.entries,*(spec.inputs[slot] for slot in (9,10,11,12)))
        ctx.save_for_backward(*values,*metadata)
        ctx.save_for_forward(*values,*metadata)
        if counts is not None: ctx.mark_non_differentiable(counts)
        return output,counts

    @staticmethod
    @once_differentiable
    def backward(ctx,cotangent,_counts):
        from ...native._build import build
        values = ctx.saved_tensors[:9]
        transpose = ctx.spec.topology.transpose()
        bars = build().polyfit_vjp(*ctx.spec.arguments(values),cotangent.contiguous(),
            [transpose.edgeOffsets,transpose.numNeighbors,transpose.paddedJ],*ctx.spec.options)
        length_bar = bars[7].sum(0)
        return None,*bars[:7],-length_bar,length_bar

    @staticmethod
    def jvp(ctx,_spec,*directions):
        from ...native._build import build
        values = ctx.saved_tensors[:9]
        directions = list(directions)
        lower,upper = directions[7:]
        length = None if lower is None and upper is None else (
            (torch.zeros_like(values[8]) if upper is None else upper)-
            (torch.zeros_like(values[7]) if lower is None else lower))
        tangents = [None if value is None else value.contiguous() for value in directions[:7]+[length]]
        tangent = build().polyfit_jvp(*ctx.spec.arguments(values),tangents,*ctx.spec.options)
        return tangent,None


def _sum(query,properties,domain,exponents,values=None,*,subtractCenter=False,
         queryVolumes=None,referenceVolumes=None,adjacency=None,referenceParticles=None,
         sector=-1,numSectors=8,unitWeights=False,_tangents=None,_rowStart=0):
    if not isinstance(properties.kernel,KernelFunctions):
        raise TypeError('kernel must be a KernelFunctions')
    if properties.supportMode != SupportScheme.Gather:
        raise NotImplementedError('Polynomial fits use Gather support')
    if properties.calibrateNormalization or properties.positiveDivergence:
        raise NotImplementedError('Polynomial fits do not apply operator calibration/clipping')
    reference = query if referenceParticles is None else referenceParticles
    from ...radiusSearch.chunked import ChunkedAdjacency
    if isinstance(adjacency,ChunkedAdjacency):
        if _tangents is not None:
            raise NotImplementedError('Row-chunk polynomial geometry JVP is not supported')
        return _chunked(query,reference,properties,domain,exponents,values,queryVolumes,
                        referenceVolumes,adjacency,sector,numSectors,unitWeights,subtractCenter)
    geometry = prepare_geometry(query,reference,domain,allow_reverse=True,allow_forward=True)
    topology = adjacency if isinstance(adjacency,CSRTopology) else None
    adjacency = prepare_adjacency(query,reference,domain,properties,adjacency)
    topology = topology or CSRTopology(adjacency)
    inputs = [*geometry[:4],reference.masses,query.kinds,reference.kinds,*geometry[4:],
              adjacency.edgeOffsets,adjacency.numNeighbors,adjacency.paddedJ,None,None,None,None]
    inputs = [forward_ad.unpack_dual(t).primal if isinstance(t,torch.Tensor) else t for t in inputs]
    entries = torch.tril_indices(len(exponents),len(exponents),device=exponents.device).T.int().contiguous()
    empty = query.positions.new_empty(0)
    density = reference.densities
    if density is None:
        if referenceVolumes is None: raise ValueError('Polynomial weights need source densities or explicit volumes')
        density = torch.ones_like(reference.masses)
    arguments = (query.positions,reference.positions,query.supports,reference.masses,density,
                 empty if referenceVolumes is None else referenceVolumes,
                 empty if values is None else values,domain.min,domain.max)
    if not unitWeights:
        from ...native._autograd import REVERSE_KERNELS
        def moving(value):
            return ((torch.is_grad_enabled() and value.requires_grad) or
                    forward_ad.unpack_dual(value).tangent is not None)
        geometry_ad = any(moving(arguments[slot]) for slot in (0,1,2,7,8))
        geometry_ad |= _tangents is not None and any(_tangents[slot] is not None for slot in (0,1,2,7))
        if geometry_ad and properties.kernel.value not in REVERSE_KERNELS:
            raise NotImplementedError('Polynomial geometry AD requires a qualified smooth radial kernel')
    spec = _Spec(inputs,exponents.contiguous(),entries,topology,
        (properties.operationMode.value,properties.kernel.value,int(sector),int(numSectors),bool(unitWeights),bool(subtractCenter),int(_rowStart)))
    if _tangents is not None:
        from ...native._build import build
        directions = [None if value is None else value.contiguous() for value in _tangents]
        return build().polyfit_jvp(*spec.arguments(arguments),directions,*spec.options),None
    return _Polynomial.apply(spec,*arguments)


def _computePolyMoments_stateBackend(queryParticles,operationProperties,domain,exponents,**kwargs):
    return _sum(queryParticles,operationProperties,domain,exponents,**kwargs)


def _computePolyRHS_stateBackend(queryParticles,operationProperties,domain,exponents,values,subtractCenter,**kwargs):
    return _sum(queryParticles,operationProperties,domain,exponents,values,
                subtractCenter=subtractCenter,**kwargs)[0]



def _geometry_directions(tangent,field=None):
    get = lambda name: None if tangent is None else getattr(tangent,name,None)
    return [get('positions'),get('positions'),get('supports'),get('masses'),
            get('densities'),None,field,None]


def computePolyMomentsGeometryJVP(queryParticles,domain,kernel,adjacency,queryTangentState,exponents,
        supportMode=SupportScheme.Gather,sector=-1,numSectors=8,unitWeights=False,
        operationMode=OperationDirection.AllToAll):
    properties = OperationProperties(kernel,supportMode=supportMode,operationMode=operationMode)
    return _sum(queryParticles,properties,domain,exponents,adjacency=adjacency,
                sector=sector,numSectors=numSectors,unitWeights=unitWeights,
                _tangents=_geometry_directions(queryTangentState))[0]


def computePolyRHSGeometryJVP(queryParticles,domain,kernel,adjacency,queryTangentState,exponents,
        values,tangentValues,subtractCenter,supportMode=SupportScheme.Gather,sector=-1,numSectors=8,
        unitWeights=False,operationMode=OperationDirection.AllToAll):
    properties = OperationProperties(kernel,supportMode=supportMode,operationMode=operationMode)
    return _sum(queryParticles,properties,domain,exponents,values,subtractCenter=subtractCenter,
                adjacency=adjacency,sector=sector,numSectors=numSectors,unitWeights=unitWeights,
                _tangents=_geometry_directions(queryTangentState,tangentValues))[0]


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
