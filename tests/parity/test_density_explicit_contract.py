"""Taped reverse of the full explicit density product versus Warp."""
from dataclasses import replace
import pytest
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.type_config import get_torch_precision
from _support.density_explicit_contract import case


@pytest.fixture(scope='module',autouse=True)
def reference_runtime():wp.init()


def compare(values,functions,tolerance):
    for weighted in (False,True):
        actual=functions('native')(*values);expected=functions('warp')(*values)
        torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance,equal_nan=True)
        seed=actual.new_tensor([.3,-.7]) if weighted else torch.ones_like(actual)
        inputs=[x for x in values if x.requires_grad]
        a=torch.autograd.grad(actual,inputs,seed,allow_unused=True);b=torch.autograd.grad(expected,inputs,seed,allow_unused=True)
        for slot,(x,y,v) in enumerate(zip(a,b,inputs)):
            torch.testing.assert_close(torch.zeros_like(v) if x is None else x,torch.zeros_like(v) if y is None else y,
                rtol=tolerance,atol=tolerance,equal_nan=True,msg=lambda msg:f'explicit reverse role {slot}: {msg}')


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('dimension',[1,2,3])
def test_all_kernel_explicit_product_reverse(device,kernel,dimension):
    i=list(sph.KernelFunctions).index(kernel)
    q,r,d,p,g,v,states,functions=case(device,dimension,get_torch_precision(),kernel=kernel,support=list(sph.SupportScheme)[i%6])
    compare(v,functions,3e-3 if v[0].dtype==torch.float32 else 5e-8)


@pytest.mark.parametrize('support',list(sph.SupportScheme))
@pytest.mark.parametrize('direction',list(sph.OperationDirection))
def test_support_direction_and_mass_couplings(device,support,direction):
    q,r,d,p,g,v,states,functions=case(device,dtype=get_torch_precision(),support=support)
    p=replace(p,operationMode=direction)
    compare(v,lambda backend:functions(backend,p),3e-3 if v[0].dtype==torch.float32 else 5e-8)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('support',[sph.SupportScheme.Gather,sph.SupportScheme.PartialSymmetric,sph.SupportScheme.SuperSymmetric])
@pytest.mark.parametrize('separation',['self','ultratiny','clamped','epsilon','near','cutoff','outside','interior'])
def test_boundary_explicit_reverse_and_support_ties(device,kernel,dimension,support,separation):
    from _support.density_explicit_contract import boundary_case
    dtype=get_torch_precision();eps=1e-7 if dtype==torch.float32 else 1e-15
    delta={'self':0.,'ultratiny':eps*1e-9,'clamped':eps*.85,'epsilon':eps,'near':eps*10,'cutoff':.8,'outside':.81,'interior':.56}[separation]
    values,functions=boundary_case(device,dimension,dtype,kernel,support,delta)
    compare(values,functions,4e-3 if dtype==torch.float32 else 3e-7)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('role',['support','mass'])
@pytest.mark.parametrize('separation',['self','clamped','cutoff','interior'])
def test_sparse_direction_roles_at_singular_pairs(device,kernel,role,separation):
    from _support.density_explicit_contract import boundary_case
    dtype=get_torch_precision();eps=1e-7 if dtype==torch.float32 else 1e-15
    delta={'self':0.,'clamped':eps*.85,'cutoff':.8,'interior':.56}[separation]
    values,functions=boundary_case(device,1,dtype,kernel,sph.SupportScheme.Gather,delta,role)
    compare(values,functions,4e-3 if dtype==torch.float32 else 3e-7)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
def test_reference_explicit_reverse_gradcheck(device,kernel):
    if get_torch_precision()!=torch.float64:pytest.skip('FP64 finite-difference qualification')
    index=list(sph.KernelFunctions).index(kernel)
    q,r,d,p,g,v,states,functions=case(device,1+index%3,kernel=kernel,support=list(sph.SupportScheme)[index%6])
    assert torch.autograd.gradcheck(functions('warp'),v,fast_mode=True,atol=5e-5,rtol=8e-4,check_undefined_grad=False)


@pytest.mark.parametrize('role',['query_mass','query_density','reference_density','lower','upper'])
def test_unused_state_and_frozen_domain_metadata(device,role):
    from warpSPHCore.native.api import warpOperationJVP as native
    from warpSPHCore.coreOperations.wp_densityJVP import computeSPHDensityGeometryJVP as reference
    q,r,d,p,g,v,states,functions=case(device,dtype=get_torch_precision());q,r,d=states(tuple(x.detach() for x in v[:10]))
    obj,attr=(q,'masses') if role=='query_mass' else (q,'densities') if role=='query_density' else (r,'densities') if role=='reference_density' else (d,'min' if role=='lower' else 'max')
    x=getattr(obj,attr).clone().requires_grad_()
    if role.startswith('query'):q=replace(q,**{attr:x})
    elif role=='reference_density':r=replace(r,**{attr:x})
    else:d=replace(d,**{attr:x})
    qt=sph.ParticleTangentState(torch.ones_like(q.positions)*.03,None,None,None)
    a=native(q,p,d,referenceParticles=r,adjacency=g,queryTangentState=qt)
    b=reference(q,d,p.kernel,p.supportMode,g,qt,r,operationMode=p.operationMode)
    assert a.requires_grad==b.requires_grad==(role=='query_mass')
    if role=='query_mass':
        for y in (a,b):
            bar=torch.autograd.grad(y.sum(),x,create_graph=True)[0]
            torch.testing.assert_close(bar,torch.zeros_like(x));assert not bar.requires_grad
