"""Position-block matrix/product scope versus Warp's fixed-Hessian helper."""
from dataclasses import replace
import pytest
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.type_config import get_torch_precision
from _support.density_block_contract import case


@pytest.fixture(scope='module',autouse=True)
def initialize_reference():wp.init()


def compare(v,actual,reference,tolerance):
    y=actual(*v);expected=reference(*v)
    torch.testing.assert_close(y,expected,rtol=tolerance,atol=tolerance,equal_nan=True)
    seed=torch.linspace(.2,.7,y.numel(),device=y.device,dtype=y.dtype).reshape_as(y)
    got=torch.autograd.grad(y,v,seed);wanted=torch.autograd.grad(expected,v,seed)
    for slot,(a,b) in enumerate(zip(got,wanted)):
        torch.testing.assert_close(a,b,rtol=tolerance,atol=tolerance,equal_nan=True,msg=lambda message:f'role {slot}: {message}')
    for evaluate in (actual,reference):
        scalar=torch.autograd.grad(evaluate(*v).sum(),v)
        if evaluate is actual:first=scalar
        else:
            for a,b in zip(first,scalar):torch.testing.assert_close(a,b,rtol=tolerance,atol=tolerance,equal_nan=True)
    directions=tuple(x.detach().cos()*.03 for x in v)
    for evaluate in (actual,reference):
        with torch.autograd.forward_ad.dual_level():
            dual=[torch.autograd.forward_ad.make_dual(x.detach(),t) for x,t in zip(v,directions)]
            product=torch.autograd.forward_ad.unpack_dual(evaluate(*dual)).tangent
        assert product is not None
        if evaluate is actual:got=product
        else:torch.testing.assert_close(got,product,rtol=tolerance,atol=tolerance,equal_nan=True)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('support',list(sph.SupportScheme))
def test_all_kernel_support_matrix_and_products(device,kernel,dimension,support):
    q,r,d,p,g,v,functions=case(device,dimension,get_torch_precision(),kernel=kernel,support=support)
    compare(v,functions('native'),functions('warp'),2e-3 if v[0].dtype==torch.float32 else 5e-8)


@pytest.mark.parametrize('direction',list(sph.OperationDirection))
@pytest.mark.parametrize('support',list(sph.SupportScheme))
def test_support_direction_and_self_id_contracts(device,direction,support):
    q,r,d,p,g,v,functions=case(device,dtype=get_torch_precision(),support=support)
    p=replace(p,operationMode=direction)
    compare(v,functions('native',p),functions('warp',p),2e-3 if v[0].dtype==torch.float32 else 5e-8)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('support',[sph.SupportScheme.Gather,sph.SupportScheme.KernelMeanSymmetric,sph.SupportScheme.SuperSymmetric])
@pytest.mark.parametrize('separation',['self','ultratiny','tiny','clamped','epsilon','near','cutoff','onesided','interior'])
def test_boundary_matrix_and_field_products(device,kernel,dimension,support,separation):
    from warpSPHCore.coreOperations.native.nt_density_hvp import computeSPHDensityPositionHVP as native
    from warpSPHCore.coreOperations.wp_densityHVP import computeSPHDensityPositionHVP as reference
    dtype=get_torch_precision();eps=1e-7 if dtype==torch.float32 else 1e-15
    delta={'self':0.,'ultratiny':eps*1e-9,'tiny':eps*1e-5,'clamped':eps*.85,'epsilon':eps,
        'near':eps*10,'cutoff':.8,'onesided':.95,'interior':.56}[separation]
    q,r,d,p,g,v,functions=case(device,dimension,dtype,kernel=kernel,support=support)
    positions=q.positions.new_zeros((2,dimension));positions[:,0]=delta
    q=replace(q,positions=positions,supports=positions.new_tensor([.8,1.03]))
    r=replace(r,positions=r.positions.new_zeros((3,dimension)),supports=r.supports.new_tensor([.87,.8,.93]))
    d=replace(d,periodic=torch.zeros(dimension,device=device,dtype=torch.bool))
    # Different global IDs remain live, even when their positions coincide.
    g=sph.AdjacencyList(None,torch.tensor([1,2],device=device,dtype=sph.get_torch_index_dtype()),
        torch.ones(2,device=device,dtype=torch.int32),torch.arange(2,device=device,dtype=sph.get_torch_offset_dtype()),2,3)
    def function(compute):
        return lambda m,a,b:compute(q,d,kernel,support,g,a,replace(r,masses=m),b,p.operationMode)
    compare(v,function(native),function(reference),6e-3 if dtype==torch.float32 else 5e-7)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
def test_higher_and_legacy_batched_products(device,kernel):
    index=list(sph.KernelFunctions).index(kernel)
    q,r,d,p,g,v,functions=case(device,1+index%3,get_torch_precision(),kernel=kernel,support=list(sph.SupportScheme)[index%6])
    tolerance=8e-3 if v[0].dtype==torch.float32 else 5e-7
    seed=functions('native')(*v).detach().cos()
    directions=tuple(x.detach().cos()*.03 for x in v)
    for evaluation_index,evaluate in enumerate((functions('native'),functions('warp'))):
        y=evaluate(*v);first=torch.autograd.grad((y.sin()*seed).sum(),v,create_graph=True)
        second=torch.autograd.grad(sum((a*b).sum() for a,b in zip(first,directions)),v,create_graph=True)
        third=torch.autograd.grad(sum((a*b).sum() for a,b in zip(second,directions)),v)
        y=evaluate(*v);batch=torch.stack((torch.ones_like(y),seed,seed.sin()))
        batched=torch.autograd.grad(y,v,batch,is_grads_batched=True,create_graph=True)
        if evaluation_index==0:got=(first,second,third,batched)
        else:
            for actual_level,reference_level in zip(got,(first,second,third,batched)):
                for a,b in zip(actual_level,reference_level):torch.testing.assert_close(a,b,rtol=tolerance,atol=tolerance)


@pytest.mark.parametrize('role',['query_positions','reference_positions','query_supports','reference_supports','minimum','maximum'])
def test_fixed_geometry_ad_metadata_matches_reference(device,role):
    from warpSPHCore.coreOperations.native.nt_density_hvp import computeSPHDensityPositionHVP as native
    from warpSPHCore.coreOperations.wp_densityHVP import computeSPHDensityPositionHVP as reference
    q,r,d,p,g,v,functions=case(device,dtype=get_torch_precision());v=tuple(x.detach() for x in v)
    if role.startswith('query_'):
        name=role.removeprefix('query_');q=replace(q,**{name:getattr(q,name).clone().requires_grad_()})
    elif role.startswith('reference_'):
        name=role.removeprefix('reference_');r=replace(r,**{name:getattr(r,name).clone().requires_grad_()})
    else:
        name='min' if role=='minimum' else 'max';d=replace(d,**{name:getattr(d,name).clone().requires_grad_()})
    actual=native(q,d,p.kernel,p.supportMode,g,v[1],replace(r,masses=v[0]),v[2],p.operationMode)
    expected=reference(q,d,p.kernel,p.supportMode,g,v[1],replace(r,masses=v[0]),v[2],p.operationMode)
    assert not actual.requires_grad and not expected.requires_grad
    tolerance=2e-3 if v[0].dtype==torch.float32 else 5e-8
    torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance)


@pytest.mark.parametrize('role',['query_positions','reference_positions','query_supports','reference_supports','minimum','maximum'])
def test_frozen_geometry_forward_dual_contract(device,role):
    from warpSPHCore.coreOperations.native.nt_density_hvp import computeSPHDensityPositionHVP as native
    from warpSPHCore.coreOperations.wp_densityHVP import computeSPHDensityPositionHVP as reference
    q,r,d,p,g,v,functions=case(device,dtype=get_torch_precision());v=tuple(x.detach() for x in v)
    with torch.autograd.forward_ad.dual_level():
        if role.startswith('query_'):
            name=role.removeprefix('query_');value=getattr(q,name)
            q=replace(q,**{name:torch.autograd.forward_ad.make_dual(value,torch.ones_like(value)*.03)})
        elif role.startswith('reference_'):
            name=role.removeprefix('reference_');value=getattr(r,name)
            r=replace(r,**{name:torch.autograd.forward_ad.make_dual(value,torch.ones_like(value)*.03)})
        else:
            name='min' if role=='minimum' else 'max';value=getattr(d,name)
            d=replace(d,**{name:torch.autograd.forward_ad.make_dual(value,torch.ones_like(value)*.03)})
        actual=native(q,d,p.kernel,p.supportMode,g,v[1],replace(r,masses=v[0]),v[2],p.operationMode)
        expected=reference(q,d,p.kernel,p.supportMode,g,v[1],replace(r,masses=v[0]),v[2],p.operationMode)
        a,da=torch.autograd.forward_ad.unpack_dual(actual);b,db=torch.autograd.forward_ad.unpack_dual(expected)
    assert da is None and db is None
    tolerance=2e-3 if v[0].dtype==torch.float32 else 5e-8
    torch.testing.assert_close(a,b,rtol=tolerance,atol=tolerance)


@pytest.mark.parametrize('transform',['vjp','jvp'])
def test_functional_transforms_remain_reference_unsupported(device,transform):
    q,r,d,p,g,v,functions=case(device,dtype=get_torch_precision())
    for backend in ('native','warp'):
        fn=lambda mass:functions(backend)(mass,*v[1:])
        with pytest.raises(RuntimeError,match='setup_context|storage'):
            if transform=='vjp':torch.func.vjp(fn,v[0])
            else:torch.func.jvp(fn,(v[0],),(torch.ones_like(v[0]),))


def test_sparse_strided_csr_metadata_and_batched_jvp_seeds(device):
    q,r,d,p,g,v,functions=case(device,dtype=get_torch_precision())
    index=sph.get_torch_index_dtype();offset=sph.get_torch_offset_dtype()
    def strided(values,dtype):
        x=torch.tensor(values,device=device,dtype=dtype);storage=torch.empty((len(values),2),device=device,dtype=dtype)
        storage[:,0]=x;return storage[:,0]
    g.paddedI=strided([-1,0,0,0,-1,-1],index)
    g.paddedJ=strided([-1,0,2,2,-1,-1],index)
    g.numNeighbors=strided([3,0],torch.int32);g.edgeOffsets=strided([1,5],offset)
    tolerance=3e-3 if v[0].dtype==torch.float32 else 5e-8
    compare(v,functions('native'),functions('warp'),tolerance)
    for backend in ('native','warp'):
        jacobian=torch.autograd.functional.jacobian(functions(backend),v,vectorize=True,strategy='forward-mode')
        if backend=='native':got=jacobian
        else:
            for a,b in zip(got,jacobian):torch.testing.assert_close(a,b,rtol=tolerance,atol=tolerance)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
def test_reference_reverse_forward_and_gradgradcheck(device,kernel):
    if get_torch_precision()!=torch.float64:pytest.skip('FP64 finite-difference qualification')
    index=list(sph.KernelFunctions).index(kernel)
    q,r,d,p,g,v,functions=case(device,1+index%3,torch.float64,kernel=kernel,support=list(sph.SupportScheme)[index%6])
    assert torch.autograd.gradcheck(functions('warp'),v,fast_mode=True,check_forward_ad=True,atol=5e-5,rtol=5e-4)
    assert torch.autograd.gradgradcheck(functions('warp'),v,fast_mode=True,atol=5e-5,rtol=5e-4)
