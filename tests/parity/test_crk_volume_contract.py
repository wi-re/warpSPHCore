"""Volume sums, taped reverse and explicit given-volume products versus Warp."""
from dataclasses import replace
import pytest
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.type_config import get_torch_precision
from _support.crk_volume_contract import case
from test_covariance_contract import compare


@pytest.fixture(scope='module',autouse=True)
def initialize_reference():wp.init()


def explicit_compare(q,r,d,p,g,supplied,tolerance):
    from warpSPHCore.crk.crk_volume_jvp import computeCRKVolumeGeometryJVP as reference
    from warpSPHCore.crk.native.nt_jvp import computeCRKVolumeGeometryJVP as native
    qt=sph.ParticleTangentState(q.positions.detach().cos()*.03,q.supports.detach()*.02,None,None)
    rt=sph.ParticleTangentState(r.positions.detach().cos()*-.02,r.supports.detach()*-.01,None,None)
    args=(q,d,p.kernel,g,supplied,qt,r,rt,p.operationMode)
    torch.testing.assert_close(native(*args),reference(*args),rtol=tolerance,atol=tolerance,equal_nan=True)
    # The explicit reference JVP itself has a taped reverse. Bounds are frozen
    # in that API; state, direction and supplied-volume leaves are tracked.
    values=tuple(x.detach().clone().requires_grad_() for x in
        (q.positions,r.positions,q.supports,r.supports,qt.positions,rt.positions,qt.supports,rt.supports,supplied))
    outputs=[];gradients=[]
    for fn in (native,reference):
        a,b,h,k,ta,tb,th,tk,V=values
        y=fn(replace(q,positions=a,supports=h),d,p.kernel,g,V,sph.ParticleTangentState(ta,th,None,None),
             replace(r,positions=b,supports=k),sph.ParticleTangentState(tb,tk,None,None),p.operationMode)
        outputs.append(y);gradients.append(torch.autograd.grad(y,values,torch.ones_like(y),allow_unused=True))
    for slot,(a,b,x) in enumerate(zip(*gradients,values)):
        torch.testing.assert_close(torch.zeros_like(x) if a is None else a,torch.zeros_like(x) if b is None else b,
            rtol=tolerance,atol=tolerance,equal_nan=True,msg=lambda msg:f'explicit JVP reverse role {slot}: {msg}')


@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
def test_all_kernel_values_reverse_and_explicit_products(device,dimension,kernel):
    index=list(sph.KernelFunctions).index(kernel)
    q,r,d,p,g,v,states,functions=case(device,dimension,get_torch_precision(),kernel=kernel,support=list(sph.SupportScheme)[index%6])
    tolerance=2e-3 if v[0].dtype==torch.float32 else 5e-8
    compare(v,functions,tolerance)
    q,r,d=states(tuple(x.detach() for x in v))
    explicit_compare(q,r,d,p,g,q.supports.new_tensor([.21,.67]),tolerance)


@pytest.mark.parametrize('support',list(sph.SupportScheme))
@pytest.mark.parametrize('direction',list(sph.OperationDirection))
def test_supports_directions_and_masked_supplied_volume(device,support,direction):
    q,r,d,p,g,v,states,functions=case(device,dtype=get_torch_precision(),support=support)
    p=replace(p,operationMode=direction);tolerance=2e-3 if v[0].dtype==torch.float32 else 5e-8
    compare(v,lambda backend:functions(backend,p),tolerance)
    q,r,d=states(tuple(x.detach() for x in v))
    explicit_compare(q,r,d,p,g,q.supports.new_tensor([float('inf'),float('nan')]),tolerance)


@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('separation',['self','ultratiny','clamped','epsilon','near','cutoff','outside','interior'])
def test_cusp_singular_cutoff_and_support_ties(device,dimension,kernel,separation):
    dtype=get_torch_precision();eps=1e-7 if dtype==torch.float32 else 1e-15
    delta={'self':0.,'ultratiny':eps*1e-9,'clamped':eps*.85,'epsilon':eps,'near':eps*10,'cutoff':.8,'outside':.81,'interior':.56}[separation]
    q,r,d,p,g,v,states,functions=case(device,dimension,dtype,kernel=kernel,support=sph.SupportScheme.PartialSymmetric)
    qx=v[0].new_zeros((2,dimension));qx[:,0]=delta;rx=torch.zeros_like(qx);h=qx.new_full((2,),.8);mass=h.new_ones(2)
    q=sph.ParticleState(qx,h,mass,torch.zeros(2,device=device,dtype=torch.int32),mass);r=replace(q,positions=rx,supports=h.clone())
    d=replace(d,periodic=torch.zeros(dimension,device=device,dtype=torch.bool))
    g=sph.AdjacencyList(None,torch.arange(2,device=device,dtype=sph.get_torch_index_dtype()),torch.ones(2,device=device,dtype=torch.int32),torch.arange(2,device=device,dtype=sph.get_torch_offset_dtype()),2,2)
    values=tuple(x.detach().clone().requires_grad_() for x in (qx,rx,h,h.clone(),mass,d.min,d.max))
    def run(backend):
        if backend=='native':from warpSPHCore.crk.native.nt_volume import computeCRKVolume as compute
        else:from warpSPHCore.crk.crk_volume import _computeCRKVolume_stateBackend as compute
        return lambda a,b,c,e,m,lo,hi:compute(replace(q,positions=a,supports=c),p,replace(d,min=lo,max=hi),g,replace(r,positions=b,supports=e,masses=m))
    tolerance=4e-3 if dtype==torch.float32 else 3e-7
    compare(values,run,tolerance)
    explicit_compare(q,r,d,p,g,h.new_tensor([.21,.67]),tolerance)


@pytest.mark.parametrize('calibrated',[False,True])
def test_calibration_grid_unused_fields_and_scope(device,calibrated):
    from warpSPHCore.radiusSearch import buildCompactHashMap
    q,r,d,p,g,v,states,functions=case(device,dtype=get_torch_precision());q,r,d=states(v)
    p=replace(p,positiveDivergence=True,calibrateNormalization=calibrated,n_h=1.2)
    grid=buildCompactHashMap(q.positions,r.positions,q.supports,r.supports,d.periodic,d,p.supportMode)
    compare(v,lambda backend:functions(backend,p,grid),2e-3 if v[0].dtype==torch.float32 else 5e-8)
    # Reference ordinary reverse is first-order; its explicit JVP is the
    # supported forward API. Native additionally supports classic dual inputs.
    y=functions('warp',p)(*v);bars=torch.autograd.grad(y,v,torch.ones_like(y),create_graph=True,allow_unused=True)
    assert not any(x is not None and x.requires_grad for x in bars)
    with pytest.raises(NotImplementedError,match='no JVPSpec'):
        with torch.autograd.forward_ad.dual_level():
            dual=list(x.detach() for x in v);dual[0]=torch.autograd.forward_ad.make_dual(dual[0],torch.ones_like(dual[0]))
            functions('warp',p)(*dual)


@pytest.mark.parametrize('role',[4,7,8,9])
def test_reference_zero_gradient_for_sole_unused_state(device,role):
    q,r,d,p,g,v,states,functions=case(device,dtype=get_torch_precision());v=[x.detach() for x in v];v[role].requires_grad_()
    actual=functions('native')(*v);expected=functions('warp')(*v)
    for y in (actual,expected):torch.testing.assert_close(torch.autograd.grad(y.sum(),v[role])[0],torch.zeros_like(v[role]))


def test_reference_fp64_gradcheck(device):
    if get_torch_precision()!=torch.float64:pytest.skip('FP64 finite-difference qualification')
    for kernel in sph.KernelFunctions:
        q,r,d,p,g,v,states,functions=case(device,kernel=kernel)
        assert torch.autograd.gradcheck(functions('warp'),v,fast_mode=True,atol=5e-5,rtol=8e-4,check_undefined_grad=False)
        from warpSPHCore.crk.crk_volume_jvp import computeCRKVolumeGeometryJVP
        v=list(v);v[5]=v[5].detach();v[6]=v[6].detach();v=tuple(v)
        directions=tuple((x.detach().cos()*.03).requires_grad_() for x in (v[0],v[1],v[2],v[3]))
        V=v[2].detach().clone().requires_grad_()
        def product(*x):
            qi,rj,domain=states(x[:10])
            return computeCRKVolumeGeometryJVP(qi,domain,p.kernel,g,x[14],sph.ParticleTangentState(x[10],x[12],None,None),
                rj,sph.ParticleTangentState(x[11],x[13],None,None),p.operationMode)
        assert torch.autograd.gradcheck(product,(*v,*directions,V),fast_mode=True,atol=5e-5,rtol=8e-4,check_undefined_grad=False)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('separation',['self','tiny','clamped','epsilon','near','interior','cutoff'])
def test_explicit_support_only_product(device,kernel,separation):
    from warpSPHCore.crk.crk_volume_jvp import computeCRKVolumeGeometryJVP as reference
    from warpSPHCore.crk.native.nt_jvp import computeCRKVolumeGeometryJVP as native
    dtype=get_torch_precision();eps=1e-7 if dtype==torch.float32 else 1e-15
    delta={'self':0.,'tiny':eps*1e-9,'clamped':eps*.85,'epsilon':eps,'near':eps*10,'interior':.56,'cutoff':.8}[separation]
    q,r,d,p,g,v,states,functions=case(device,1,dtype,kernel=kernel)
    q=replace(q,positions=q.positions.new_full((2,1),delta),supports=q.supports.new_full((2,),.8))
    r=replace(q,positions=torch.zeros_like(q.positions));d=replace(d,periodic=torch.zeros(1,device=device,dtype=torch.bool))
    g=sph.AdjacencyList(None,torch.arange(2,device=device,dtype=sph.get_torch_index_dtype()),torch.ones(2,device=device,dtype=torch.int32),torch.arange(2,device=device,dtype=sph.get_torch_offset_dtype()),2,2)
    qt=sph.ParticleTangentState(torch.zeros_like(q.positions),q.supports*.03,None,None);V=q.supports.new_tensor([.21,.67])
    args=(q,d,kernel,g,V,qt,r)
    torch.testing.assert_close(native(*args),reference(*args),rtol=4e-3 if dtype==torch.float32 else 3e-7,atol=4e-3 if dtype==torch.float32 else 3e-7,equal_nan=True)
