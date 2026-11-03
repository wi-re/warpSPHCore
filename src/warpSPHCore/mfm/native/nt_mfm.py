"""Native meshless pass adapter; the geometry and closure controllers are shared."""
import torch
from ...native._inputs import prepare_adjacency, prepare_geometry
from ...native._checks import check_execution

_FIELDS = ('rho','vel','pres','gRho','gVel','gP','aRho','aVel','aP','vol','Einv','lam','condBad','beta','kappa','p','cellId')
_FLOATS = dict(gamma=1.4,dt=0.,psi1=.5,psi2=.25,power=1.,massLimit=.1,closureCap=0.,cfl=.2)
_INTS = dict(mode=0,order=2,timeCentred=1,centred=1,areaCap=0,retry=1,faceGuard=0,n0=1,n1=1,n2=1,p0=0,p1=0,p2=0,rowStart=0)
_MOMENTS_SPEC,_GRADIENTS_SPEC,_LIMITER_SPEC,_CLOSURE_RES_SPEC,_CLOSURE_W_SPEC,_MATVEC_SPEC,_COARSE_STENCIL_SPEC,_FLUX_SPEC,_TIMESTEP_SPEC = range(9)


def _ctx(particles,props,domain,adjacency):
    return particles,props,domain,adjacency


class PreparedLauncher:
    def __init__(self,context):
        query,self.properties,domain,adjacency = context
        check_execution(query.positions,query.supports,domain.min,domain.max)
        geometry = prepare_geometry(query,query,domain)
        adjacency = prepare_adjacency(query,query,domain,self.properties,adjacency)
        self.inputs = [*geometry[:4],query.masses,query.kinds,query.kinds,*geometry[4:],
                       adjacency.edgeOffsets,adjacency.numNeighbors,adjacency.paddedJ,None,None,None,None]

    def launch(self,spec,**extras):
        from ...native._build import build
        check_execution(*(value for value in extras.values() if isinstance(value,torch.Tensor)))
        extras = dict(extras)
        if 'deficient' in extras: extras['condBad'] = extras.pop('deficient')
        unknown = extras.keys()-set(_FIELDS)-_FLOATS.keys()-_INTS.keys()
        if unknown: raise TypeError(f'Unknown meshless arguments: {sorted(unknown)}')
        tensors = [None if extras.get(name) is None else extras[name].contiguous() for name in _FIELDS]
        floats = [float(extras.get(name,default)) for name,default in _FLOATS.items()]
        integers = [int(extras.get(name,default)) for name,default in _INTS.items()]
        result = build().meshless(self.inputs,tensors,floats,integers,int(spec),self.properties.kernel.value)
        return result[0] if len(result)==1 else tuple(result)


def _call(spec,particles,props,domain,adjacency,**extras):
    return PreparedLauncher(_ctx(particles,props,domain,adjacency)).launch(spec,**extras)


def mfmMomentsNative(particles,props,domain,adjacency=None):
    return _call(_MOMENTS_SPEC,particles,props,domain,adjacency)


def mfmGradientsNative(particles,props,domain,adjacency,rho,vel,pres,Einv,deficient):
    return _call(_GRADIENTS_SPEC,particles,props,domain,adjacency,rho=rho,vel=vel,pres=pres,Einv=Einv,condBad=deficient.int())


def mfmLimiterNative(particles,props,domain,adjacency,rho,vel,pres,gRho,gVel,gP,beta):
    return _call(_LIMITER_SPEC,particles,props,domain,adjacency,rho=rho,vel=vel,pres=pres,gRho=gRho,gVel=gVel,gP=gP,beta=beta)


def mfmClosureMatvecNative(particles,props,domain,adjacency,kappa,p):
    return _call(_MATVEC_SPEC,particles,props,domain,adjacency,kappa=kappa,p=p)


# Internal shared-controller compatibility; numerical execution is native.
mfmClosureMatvecWarp = mfmClosureMatvecNative
