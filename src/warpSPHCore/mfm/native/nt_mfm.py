"""Native meshless pass adapter; the geometry and closure controllers are shared."""
from dataclasses import dataclass
import torch
from ...native._inputs import prepare_adjacency, prepare_geometry
from ...native._checks import check_execution
from ...autograd.operator_spec import SPHContext,OperatorSpec,ExtraSpec,ExtraKind,RowRole,OutputSpec
from ...native._row_dispatch import output

_FIELDS = ('rho','vel','pres','gRho','gVel','gP','aRho','aVel','aP','vol','Einv','lam','condBad','beta','kappa','p','cellId')
_FLOATS = dict(gamma=1.4,dt=0.,psi1=.5,psi2=.25,power=1.,massLimit=.1,closureCap=0.,cfl=.2)
_INTS = dict(mode=0,order=2,timeCentred=1,centred=1,areaCap=0,retry=1,faceGuard=0,n0=1,n1=1,n2=1,p0=0,p1=0,p2=0,rowStart=0)


def _ctx(particles,props,domain,adjacency):
    return SPHContext(particles,props,domain,adjacency)


class PreparedLauncher:
    def __init__(self,context):
        from ...radiusSearch.chunked import ChunkedAdjacency
        self.context=context
        self._chunks=None
        if isinstance(context.adjacency,ChunkedAdjacency):
            from ...autograd.row_dispatch import chunkContexts,validateContext
            communication=context.adjacency.communication
            if communication is None:validateContext(context)
            else:communication.localPhase('prepared:validate',lambda:validateContext(context))
            self._adjacency=context.adjacency
            self._chunks=tuple((interval,PreparedLauncher(local)) for interval,local in chunkContexts(context))
            return
        query=context.query
        reference=query if context.reference is None else context.reference
        self.properties,domain=context.properties,context.domain
        check_execution(query.positions,reference.positions,query.supports,reference.supports,domain.min,domain.max)
        geometry=prepare_geometry(query,reference,domain)
        adjacency=prepare_adjacency(query,reference,domain,self.properties,context.adjacency)
        self.inputs=[*geometry[:4],reference.masses,query.kinds,reference.kinds,*geometry[4:],
                     adjacency.edgeOffsets,adjacency.numNeighbors,adjacency.paddedJ,None,None,None,None]

    def launch(self,spec,**extras):
        if self._chunks is not None:
            if self.context.adjacency is not self._adjacency:
                raise ValueError('Build a new prepared launcher after replacing its chunked adjacency')
            from ...autograd.row_dispatch import launchRowChunks
            return launchRowChunks(spec,self.context,extras,prepared=self._chunks)
        return self._launch(spec.kernel.task,extras)

    def _launch(self,task,extras):
        from ...native._build import build
        check_execution(*(value for value in extras.values() if isinstance(value,torch.Tensor)))
        extras = dict(extras)
        if 'deficient' in extras: extras['condBad'] = extras.pop('deficient')
        unknown = extras.keys()-set(_FIELDS)-_FLOATS.keys()-_INTS.keys()
        if unknown: raise TypeError(f'Unknown meshless arguments: {sorted(unknown)}')
        tensors = [None if extras.get(name) is None else extras[name].contiguous() for name in _FIELDS]
        floats = [float(extras.get(name,default)) for name,default in _FLOATS.items()]
        integers = [int(extras.get(name,default)) for name,default in _INTS.items()]
        result = build().meshless(self.inputs,tensors,floats,integers,int(task),self.properties.kernel.value)
        return result[0] if len(result)==1 else tuple(result)


@dataclass(frozen=True)
class _Kernel:
    task:int
    key:str

    def launch_native(self,context,extras):
        return PreparedLauncher(context)._launch(self.task,extras)


def _declaration(task,name,outputs,fields=(),scalars=(),pair=False):
    extras=[]
    for field in fields:
        role=RowRole.CHUNK if field=='kappa' else RowRole.REFERENCE
        extras.append(ExtraSpec(field,ExtraKind.CHUNK_TENSOR if field=='kappa' else ExtraKind.TENSOR,role))
    extras.extend(ExtraSpec(name,ExtraKind.SCALAR) for name in scalars)
    extras.append(ExtraSpec('rowStart',ExtraKind.SCALAR,RowRole.ROW_START))
    return OperatorSpec(_Kernel(task,'native.mfm.'+name),tuple(outputs),tuple(extras),
                        rowChunked=not pair,queryOnlyCRK=True,blockDim=128)


def _vector(context,extras):return (context.domain.dim,)

def _matrix(context,extras):return (context.domain.dim,context.domain.dim)

_MOMENTS_SPEC=_declaration(0,'moments',[output(),output(_matrix),output(dtype=torch.int32)])
_GRADIENTS_SPEC=_declaration(1,'gradients',[output(_vector),output(_matrix),output(_vector)],('rho','vel','pres','Einv','deficient'))
_LIMITER_SPEC=_declaration(2,'limiter',[output(),output(_vector),output()],('rho','vel','pres','gRho','gVel','gP','beta'))
_CLOSURE_RES_SPEC=_declaration(3,'closure_residual',[output(_vector),output()],('vol','Einv','condBad'),('centred','areaCap'))
_CLOSURE_W_SPEC=_declaration(4,'closure_weights',
    [OutputSpec(dtype=lambda ctx,e:(ctx.query.positions.dtype,()),shape=lambda ctx,e:ctx.adjacency.paddedJ.numel())],
    ('vol','Einv','condBad'),('centred','areaCap','power'),pair=True)
_CLOSURE_MV_SPEC=_declaration(5,'closure_matvec',[output(_vector),output()],('kappa','p'))
_MATVEC_SPEC=_CLOSURE_MV_SPEC
_COARSE_STENCIL_SPEC=_declaration(6,'coarse_stencil',[output(lambda ctx,e:(3**ctx.domain.dim,))],
    ('kappa','cellId'),('n0','n1','n2','p0','p1','p2'))
_FLUX_SPEC=_declaration(7,'flux',[output(),output(_vector),output()],
    ('rho','vel','pres','gRho','gVel','gP','aRho','aVel','aP','vol','Einv','lam','condBad'),
    ('gamma','dt','mode','order','psi1','psi2','power','timeCentred','centred','areaCap','retry','massLimit','faceGuard','closureCap'))
_TIMESTEP_SPEC=_declaration(8,'timestep',[output()],('rho','vel','pres'),('gamma','cfl'))


def _call(spec,particles,props,domain,adjacency,**extras):
    extras.setdefault('rowStart',0)
    return PreparedLauncher(_ctx(particles,props,domain,adjacency)).launch(spec,**extras)


def mfmMomentsNative(particles,props,domain,adjacency=None):
    return _call(_MOMENTS_SPEC,particles,props,domain,adjacency)


def mfmGradientsNative(particles,props,domain,adjacency,rho,vel,pres,Einv,deficient):
    return _call(_GRADIENTS_SPEC,particles,props,domain,adjacency,rho=rho,vel=vel,pres=pres,Einv=Einv,deficient=deficient.int())


def mfmLimiterNative(particles,props,domain,adjacency,rho,vel,pres,gRho,gVel,gP,beta):
    return _call(_LIMITER_SPEC,particles,props,domain,adjacency,rho=rho,vel=vel,pres=pres,gRho=gRho,gVel=gVel,gP=gP,beta=beta)


def mfmClosureMatvecNative(particles,props,domain,adjacency,kappa,p):
    return _call(_MATVEC_SPEC,particles,props,domain,adjacency,kappa=kappa,p=p)


# Internal shared-controller compatibility; numerical execution is native.
mfmClosureMatvecWarp = mfmClosureMatvecNative
