"""Density numerical scope versus Warp's taped reverse and explicit JVP paths."""
from dataclasses import replace
import pytest
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.type_config import get_torch_precision
from warpSPHCore.coreOperations.native.wrapper import computeDensity
from warpSPHCore.native.api import warpOperationJVP as native_jvp
from _support.density_contract import case


@pytest.fixture(scope='module',autouse=True)
def initialize_reference():
    wp.init()


def compare(values,functions,tolerance):
    actual=functions('native')(*values);expected=functions('warp')(*values)
    seed=actual.detach().cos().contiguous()
    torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance,equal_nan=True)
    got=torch.autograd.grad(actual,values,seed,allow_unused=True)
    want=torch.autograd.grad(expected,values,seed,allow_unused=True)
    for a,b,value in zip(got,want,values):
        torch.testing.assert_close(torch.zeros_like(value) if a is None else a,
            torch.zeros_like(value) if b is None else b,rtol=tolerance,atol=tolerance,equal_nan=True)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('dimension',[1,2,3])
def test_all_kernel_values_and_reverse_products(device,kernel,dimension):
    dtype=get_torch_precision();index=list(sph.KernelFunctions).index(kernel)
    q,r,d,p,g,v,states,functions=case(device,dimension,dtype,kernel=kernel,
                                   support=list(sph.SupportScheme)[index%6])
    tolerance=7e-4 if dtype==torch.float32 else 3e-8
    compare(v,functions,tolerance)
    # Explicit reference JVP rejects PartialSymmetric. Native extends that path.
    if p.supportMode!=sph.SupportScheme.PartialSymmetric:
        qi,rj,domain=states(tuple(x.detach() for x in v))
        t=tuple(.03*x.detach().cos() for x in v)
        options=dict(referenceParticles=rj,adjacency=g,
            queryTangentState=sph.ParticleTangentState(t[0],t[2],None,None),
            referenceTangentState=sph.ParticleTangentState(t[1],t[3],t[4],None))
        # The reference explicit JVP carries positions/supports/mass, but no
        # bound tangent. Its periodic branch remains fixed under these inputs.
        torch.testing.assert_close(native_jvp(qi,p,domain,**options),
            sph.warpOperationJVP(qi,p,domain,**options),rtol=tolerance,atol=tolerance,equal_nan=True)


@pytest.mark.parametrize('support',list(sph.SupportScheme))
@pytest.mark.parametrize('direction',list(sph.OperationDirection))
def test_support_and_direction_contracts(device,support,direction):
    q,r,d,p,g,v,states,functions=case(device,2,get_torch_precision(),support=support)
    p=replace(p,operationMode=direction)
    compare(v,lambda backend:functions(backend,p),7e-4 if v[0].dtype==torch.float32 else 3e-8)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('separation',['self','tiny','cutoff','interior'])
def test_origin_cutoff_and_support_ties(device,kernel,separation):
    dtype=get_torch_precision();delta={'self':0.,'tiny':1e-12 if dtype==torch.float32 else 1e-20,'cutoff':.8,'interior':.56}[separation]
    q,r,d,p,g,v,states,functions=case(device,2,dtype,kernel=kernel,support=sph.SupportScheme.PartialSymmetric)
    qx=v[0].new_tensor([[delta,0],[delta,0]]);rx=torch.zeros_like(qx)
    qh=v[2].new_full((2,),.8);mass=qh.new_ones(2)
    q=sph.ParticleState(qx,qh,mass,torch.zeros(2,device=device,dtype=torch.int32),mass)
    r=replace(q,positions=rx,supports=qh.clone())
    d=replace(d,periodic=torch.zeros(2,device=device,dtype=torch.bool))
    g=sph.AdjacencyList(None,torch.arange(2,device=device,dtype=sph.get_torch_index_dtype()),torch.ones(2,device=device,dtype=torch.int32),torch.arange(2,device=device,dtype=sph.get_torch_offset_dtype()),2,2)
    values=tuple(x.detach().clone().requires_grad_() for x in (qx,rx,qh,qh.clone(),mass,d.min,d.max))
    def run(backend):
        return lambda a,b,c,e,m,lo,hi:computeDensity(replace(q,positions=a,supports=c),
            replace(r,positions=b,supports=e,masses=m),replace(d,min=lo,max=hi),p,g,backend=backend)
    # Singular viscosity self values and adhesion cutoff adjoints are inherited
    # nonfinite reference behavior; compare locations, not a fabricated smooth limit.
    compare(values,run,3e-3 if dtype==torch.float32 else 2e-8)
    from warpSPHCore.native._inputs import prepare_operator_inputs
    from warpSPHCore.native._build import build
    aa,bb,cc,ee,mm,ll,hh=values
    packed=prepare_operator_inputs(replace(q,positions=aa,supports=cc),
        replace(r,positions=bb,supports=ee,masses=mm),replace(d,min=ll,max=hh),p,g)
    slots=(0,1,2,3,4,7,8);seed=aa.new_tensor([.2,-.6])
    raw=build().density_adjoint(packed,seed,sum(1<<slot for slot in slots),
        p.supportMode.value,p.operationMode.value,p.kernel.value,1,2,False)
    owned=torch.autograd.grad(run('native')(*values),values,seed)
    for slot,wanted in zip(slots,owned):
        torch.testing.assert_close(raw[slot],wanted,rtol=3e-3 if dtype==torch.float32 else 2e-8,
                                   atol=3e-3 if dtype==torch.float32 else 2e-8,equal_nan=True)
    from warpSPHCore.coreOperations.wp_densityJVP import computeSPHDensityGeometryJVP
    fixed=tuple(z.detach() for z in values)
    a,b,c,e,m,lo,hi=fixed
    qi=replace(q,positions=a,supports=c);rj=replace(r,positions=b,supports=e,masses=m)
    qt=sph.ParticleTangentState(torch.ones_like(a)*.02,c*.03,None,None)
    rt=sph.ParticleTangentState(torch.ones_like(b)*-.01,e*-.02,m*.04,None)
    reference=computeSPHDensityGeometryJVP(qi,d,kernel,p.supportMode,g,qt,rj,rt,
                                          operationMode=p.operationMode)
    actual=native_jvp(qi,p,d,queryTangentState=qt,referenceTangentState=rt,
                       referenceParticles=rj,adjacency=g)
    torch.testing.assert_close(actual,reference,rtol=3e-3 if dtype==torch.float32 else 2e-8,
                               atol=3e-3 if dtype==torch.float32 else 2e-8,equal_nan=True)


def test_reference_higher_transform_and_determinism_scope(device):
    q,r,d,p,g,v,states,functions=case(device,dtype=get_torch_precision())
    run=functions('warp')
    first=torch.autograd.grad(run(*v),v,torch.ones(2,device=device,dtype=v[0].dtype),create_graph=True)
    assert all(not z.requires_grad for z in first)
    with pytest.raises(RuntimeError,match='setup_context'):
        torch.func.vjp(lambda a:run(a,*v[1:]),v[0].detach())
    with pytest.raises(RuntimeError,match='setup_context'):
        torch.func.jvp(lambda a:run(a,*v[1:]),(v[0].detach(),),(torch.ones_like(v[0]),))
    previous=torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        compare(v,functions,7e-4 if v[0].dtype==torch.float32 else 3e-8)
    finally:torch.use_deterministic_algorithms(previous)


def test_calibrated_density_and_warp_grid_rebuild(device,monkeypatch):
    from warpSPHCore.radiusSearch.compactHash.buildHashmap import buildCompactHashMap
    dtype=get_torch_precision()
    q,r,d,p,g,v,states,functions=case(device,dtype=dtype)
    p=replace(p,calibrateNormalization=True,n_h=2.57)
    compare(v,lambda backend:functions(backend,p),7e-4 if dtype==torch.float32 else 3e-8)
    qi,rj,domain=states(v)
    # Topology builders see detached geometry, exactly as a simulation rebuild.
    grid=buildCompactHashMap(q.positions,r.positions,q.supports,r.supports,d.periodic,d,p.supportMode)
    expected=computeDensity(qi,rj,domain,p,grid,backend='warp')
    expected_grad=torch.autograd.grad(expected,v,torch.ones_like(expected),allow_unused=True)
    monkeypatch.setattr(wp,'launch',lambda *a,**k:pytest.fail('native density/grid rebuild launched Warp'))
    actual=computeDensity(qi,rj,domain,p,grid,backend='native')
    got=torch.autograd.grad(actual,v,torch.ones_like(actual),allow_unused=True)
    tolerance=7e-4 if dtype==torch.float32 else 3e-8
    torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance)
    for a,b,value in zip(got,expected_grad,v):
        torch.testing.assert_close(a,torch.zeros_like(value) if b is None else b,rtol=tolerance,atol=tolerance)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
def test_registered_density_adjoint_matches_owned_bridge(device,kernel):
    from warpSPHCore.native._inputs import prepare_operator_inputs
    from warpSPHCore.native._build import build
    index=list(sph.KernelFunctions).index(kernel)
    q,r,d,p,g,v,states,functions=case(device,1+index%3,get_torch_precision(),
        kernel=kernel,support=list(sph.SupportScheme)[index%6])
    q,r,d=states(v)
    inputs=prepare_operator_inputs(q,r,d,p,g)
    slots=(0,1,2,3,4,7,8)
    seed=v[0].new_tensor([.2,-.6])
    raw=build().density_adjoint(inputs,seed,sum(1<<slot for slot in slots),
        p.supportMode.value,p.operationMode.value,p.kernel.value,1,2,False)
    owned=torch.autograd.grad(functions('native')(*v),v,seed)
    tolerance=7e-4 if v[0].dtype==torch.float32 else 3e-8
    for slot,reference in zip(slots,owned):
        torch.testing.assert_close(raw[slot],reference,rtol=tolerance,atol=tolerance)
