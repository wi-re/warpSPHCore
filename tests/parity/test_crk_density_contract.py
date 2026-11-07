"""The complete ordinary consistency ratio contract against Warp."""
from dataclasses import replace
import pytest
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.type_config import get_torch_precision
from _support.crk_density_contract import case


@pytest.fixture(scope='module',autouse=True)
def runtime():wp.init()


def compare(values,functions,tolerance,roundoff=False,kernel=None):
    inputs=[x for x in values if x is not None and x.requires_grad]
    for weighted in (False,True):
        actual,expected=functions('native')(*values),functions('warp')(*values)
        torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance,equal_nan=True)
        seed=torch.linspace(-.3,.7,len(actual),device=actual.device,dtype=actual.dtype) if weighted else torch.ones_like(actual)
        bars=torch.autograd.grad(actual,inputs,seed,allow_unused=True);wanted=torch.autograd.grad(expected,inputs,seed,allow_unused=True)
        budgets={}
        if roundoff:
            from _support.crk_density_roundoff import viscosity_ratio_budgets,exact_one_edge_bars
            budgets=viscosity_ratio_budgets(values,kernel,seed);exact=exact_one_edge_bars(values,seed)
            torch.testing.assert_close(actual,values[4][:len(actual)]/values[10][:len(actual)],rtol=tolerance,atol=tolerance)
        for k,(a,b,x) in enumerate(zip(bars,wanted,inputs)):
            slot=next(i for i,value in enumerate(values) if value is x)
            a,b=torch.zeros_like(x) if a is None else a,torch.zeros_like(x) if b is None else b
            if slot not in budgets:
                torch.testing.assert_close(a,b,rtol=tolerance,atol=tolerance,equal_nan=True,msg=lambda message:f'CRK density role {k}: {message}')
            else:
                finite=a.isfinite()&b.isfinite()
                torch.testing.assert_close(a[~finite],b[~finite],rtol=0,atol=0,equal_nan=True)
                assert bool(((a-b).abs()[finite]<=(budgets[slot]+tolerance*(1+b.abs()))[finite]).all()),f'CRK density role {k} exceeds expanded bound'
                for bar in (a,b):
                    allowance=budgets[slot]/2+tolerance*(1+exact[slot].abs())
                    assert bool(((bar-exact[slot]).abs()[finite]<=allowance[finite]).all()),f'CRK density role {k} exceeds exact one-edge oracle bound'


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('dimension',[1,2,3])
def test_all_kernel_values_scalar_and_weighted_reverse(device,kernel,dimension):
    *_,v,states,functions=case(device,dimension,get_torch_precision(),kernel)
    compare(v,functions,9e-3 if v[0].dtype==torch.float32 else 3e-7)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
def test_reference_actual_gradcheck(device,kernel):
    if get_torch_precision()!=torch.float64:pytest.skip('FP64 finite differences')
    dimension=1+list(sph.KernelFunctions).index(kernel)%3
    *_,v,states,functions=case(device,dimension,kernel=kernel)
    assert torch.autograd.gradcheck(functions('warp'),v,fast_mode=True,atol=8e-5,rtol=2e-3,nondet_tol=1e-12 if device=='cuda' else 0.)


@pytest.mark.parametrize('volume_mode',['reference','query','none'])
@pytest.mark.parametrize('coefficient_mode',['full','tuple','none'])
def test_volume_and_coefficient_roles(device,volume_mode,coefficient_mode):
    *_,v,states,functions=case(device,dtype=get_torch_precision(),volume_mode=volume_mode,coefficient_mode=coefficient_mode)
    compare(v,functions,9e-3 if v[0].dtype==torch.float32 else 3e-7)


@pytest.mark.parametrize('support',list(sph.SupportScheme))
@pytest.mark.parametrize('direction',list(sph.OperationDirection))
def test_forced_scatter_direction_and_ignored_flags(device,support,direction):
    q,r,d,p,g,v,states,functions=case(device,dtype=get_torch_precision())
    p=replace(p,supportMode=support,operationMode=direction,calibrateNormalization=True,n_h=1.8,positiveDivergence=True)
    compare(v,lambda backend:functions(backend,p),9e-3 if v[0].dtype==torch.float32 else 3e-7)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('separation',['self','ultratiny','clamped','epsilon','near','cutoff','outside','interior'])
def test_raw_boundaries_and_zero_denominator_reverse(device,kernel,dimension,separation):
    q,r,d,p,g,v,states,functions=case(device,dimension,get_torch_precision(),kernel)
    eps=1e-7 if v[0].dtype==torch.float32 else 1e-15
    distance={'self':0.,'ultratiny':eps*1e-9,'clamped':eps*.85,'epsilon':eps,'near':eps*10,'cutoff':.8,'outside':.81,'interior':.56}[separation]
    d.periodic.zero_();g.paddedJ=torch.arange(2,device=device,dtype=sph.get_torch_index_dtype())
    g.edgeOffsets=torch.arange(2,device=device,dtype=sph.get_torch_offset_dtype());g.numNeighbors=torch.ones(2,device=device,dtype=torch.int32)
    v=list(v);v[0]=torch.zeros_like(v[0]);v[0][:,0]=distance;v[1]=torch.zeros_like(v[1])
    v[2]=torch.full_like(v[2],.8);v[3]=torch.full_like(v[3],.8)
    v=tuple(x.detach().requires_grad_() for x in v)
    bounded=kernel==sph.KernelFunctions.ViscosityKernel and separation in ('ultratiny','clamped','epsilon','near')
    compare(v,functions,2e-2 if v[0].dtype==torch.float32 else 3e-6,bounded,kernel)


def test_grid_and_reference_higher_ad_scope(device):
    from warpSPHCore.radiusSearch import buildCompactHashMap
    q,r,d,p,g,v,states,functions=case(device,dtype=get_torch_precision());q,r,d=states(v[:10])
    grid=buildCompactHashMap(q.positions,r.positions,q.supports,r.supports,d.periodic,d,p.supportMode)
    compare(v,lambda backend:functions(backend,adjacency=grid),9e-3 if v[0].dtype==torch.float32 else 3e-7)
    for backend in ('native','warp'):
        y=functions(backend)(*v);bars=torch.autograd.grad(y,v,torch.ones_like(y),create_graph=True,allow_unused=True)
        assert not any(x is not None and x.requires_grad for x in bars)
    with pytest.raises(NotImplementedError,match='no JVPSpec'):
        with torch.autograd.forward_ad.dual_level():
            dual=[x.detach() for x in v];dual[0]=torch.autograd.forward_ad.make_dual(dual[0],torch.ones_like(dual[0]))
            functions('warp')(*dual)


@pytest.mark.parametrize('role',[7,8,9,14,15])
@pytest.mark.parametrize('value',['zero','nan','absent'])
def test_unused_state_values_and_zero_dependencies(device,role,value):
    *_,v,states,functions=case(device,dtype=get_torch_precision())
    if value=='absent':
        original=v;changed=list(v);changed[role]=None
        actual,baseline=functions('native')(*changed),functions('native')(*original)
        torch.testing.assert_close(actual,baseline,rtol=0,atol=0)
        bars=torch.autograd.grad(actual,original,torch.ones_like(actual),allow_unused=True)
        wanted=torch.autograd.grad(baseline,original,torch.ones_like(baseline),allow_unused=True)
        for a,b,x in zip(bars,wanted,original):
            torch.testing.assert_close(torch.zeros_like(x) if a is None else a,torch.zeros_like(x) if b is None else b,rtol=0,atol=0)
        # The generic reference state bridge can reject absent arrays before
        # reaching this value-only kernel. This is a native zero-dependency
        # extension, with an independent non-absent control as the oracle.
        try:reference=functions('warp')(*changed)
        except AttributeError as exc:assert 'NoneType' in str(exc) and 'view' in str(exc)
        else:torch.testing.assert_close(actual,reference,rtol=9e-3 if actual.dtype==torch.float32 else 3e-7,atol=9e-3 if actual.dtype==torch.float32 else 3e-7)
        return
    v=list(v);v[role]=None if value=='absent' else torch.full_like(v[role],0. if value=='zero' else float('nan')).requires_grad_()
    compare(v,functions,9e-3 if v[0].dtype==torch.float32 else 3e-7)
