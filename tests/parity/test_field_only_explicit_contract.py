"""Independent missing field directions and field-only forward composition."""
import importlib
from dataclasses import replace
import pytest
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.type_config import get_torch_precision

VARIANTS=['interpolate','gradient','divergence','curl',*list(sph.LaplacianScheme)]
CONFIGURATIONS=[(variant,scheme) for variant in VARIANTS for scheme in sph.GradientScheme
                if (variant!='interpolate' or scheme==sph.GradientScheme.Naive)
                and (not isinstance(variant,sph.LaplacianScheme) or scheme in (sph.GradientScheme.Difference,sph.GradientScheme.Symmetric))]


@pytest.fixture(scope='module',autouse=True)
def reference_runtime():wp.init()


def fixture(device,variant,scheme,packed=False,self_cloud=False):
    family='laplacian' if isinstance(variant,sph.LaplacianScheme) else variant
    module=importlib.import_module('_support.'+family+'_contract')
    options=dict(dtype=get_torch_precision(),field_shape=((3,2) if family=='divergence' else (2,3)) if packed else (2,))
    if family=='laplacian':options['mode']=variant
    q,r,d,p,g,v,states,_=module.case(device,**options);p=replace(p,gradientMode=scheme)
    qvalue=v[14] if len(v)>14 else v[10][:len(q.positions)]
    reference=v[10][:len(q.positions)] if self_cloud else v[10]
    tq=(qvalue.detach().cos()*.03).requires_grad_();tr=(reference.detach().sin()*.02).requires_grad_()
    values=(*v,tq,tr)
    if self_cloud:
        n=len(q.positions)
        g=sph.AdjacencyList(None,torch.arange(n,device=device,dtype=sph.get_torch_index_dtype()).repeat(n),
            torch.full((n,),n,device=device,dtype=torch.int32),torch.arange(n,device=device,dtype=sph.get_torch_offset_dtype())*n,n,n)
    def compute(backend,side,corrected=False,ordinary=False,no_primal=False):
        def run(*x):
            qi,rj,domain=states(x[:10]);qv=x[14] if family!='interpolate' else None
            rv=(x[14] if family!='interpolate' else x[10][:len(qi.positions)]) if self_cloud else x[10]
            qt=x[-2] if side=='query' else torch.zeros_like(x[-2])
            rt=x[-1] if side=='reference' else torch.zeros_like(x[-1])
            args=dict(referenceParticles=None if self_cloud else rj,adjacency=g)
            if corrected:
                args.update(referenceVolumes=x[13],crkState=sph.CRKState(x[11],x[12],None if family=='interpolate' else x[15],None if family=='interpolate' else x[16]))
                if family!='interpolate':args.update(gradHState=sph.GradHState(x[19],x[17]),renormalizationState=sph.RenormalizationState(x[18]))
            if ordinary:
                from warpSPHCore.coreOperations.native.wrapper import sphOperation
                return sphOperation(qi,p,domain,qt,rt,backend='warp',**args)
            from warpSPHCore.native.api import warpOperationJVP as native
            return (native if backend=='native' else sph.warpOperationJVP)(qi,p,domain,queryValues=None if no_primal else qv,referenceValues=None if no_primal else rv,
                tangentQueryValues=x[-2] if side=='query' else None,tangentReferenceValues=x[-1] if side=='reference' else None,**args)
        return run
    return values,compute


def compare(values,compute,side,corrected=False):
    tolerance=6e-3 if values[0].dtype==torch.float32 else 2e-7
    inputs=[x for x in values if x.requires_grad]
    for weighted in (False,True):
        expected=compute('warp',side,corrected,ordinary=True)(*values)
        seed=torch.linspace(-.3,.7,expected.numel(),device=expected.device,dtype=expected.dtype).reshape_as(expected) if weighted else torch.ones_like(expected)
        wanted=torch.autograd.grad(expected,inputs,seed,allow_unused=True)
        for backend in ('native','warp'):
            actual=compute(backend,side,corrected)(*values)
            torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance)
            bars=torch.autograd.grad(actual,inputs,seed,allow_unused=True)
            for index,(bar,target,x) in enumerate(zip(bars,wanted,inputs)):
                torch.testing.assert_close(torch.zeros_like(x) if bar is None else bar,torch.zeros_like(x) if target is None else target,
                                           rtol=tolerance,atol=tolerance,msg=lambda msg:f'{backend} field-only role {index}: {msg}')


@pytest.mark.parametrize('variant,scheme',CONFIGURATIONS)
@pytest.mark.parametrize('corrected',[False,True])
@pytest.mark.parametrize('side',['query','reference'])
@pytest.mark.parametrize('packed',[False,True])
def test_one_sided_field_product_and_reverse(device,variant,scheme,corrected,side,packed):
    values,compute=fixture(device,variant,scheme,packed)
    compare(values,compute,side,corrected)


@pytest.mark.parametrize('variant',VARIANTS)
@pytest.mark.parametrize('side',['query','reference'])
def test_self_cloud_field_directions_are_independent(device,variant,side):
    values,compute=fixture(device,variant,sph.GradientScheme.Difference,self_cloud=True)
    compare(values,compute,side)


FORWARD_CASES=[(mode,False,role) for mode in sph.LaplacianScheme for role in (0,2,20)]
FORWARD_CASES += [(mode,True,role) for mode in sph.LaplacianScheme if mode!=sph.LaplacianScheme.Naive for role in (0,2,11,18,20)]
FORWARD_CASES += [(sph.LaplacianScheme.Naive,True,20)]
EXTENSION_CASES=[(mode,corrected,5) for mode in sph.LaplacianScheme for corrected in (False,True)]
EXTENSION_CASES += [(sph.LaplacianScheme.Naive,True,role) for role in (0,2,11,18)]


def forward_fixture(device,mode,corrected):
    from _support.laplacian_contract import case,oracle
    q,r,d,p,g,v,states,_=case(device,dtype=get_torch_precision(),mode=mode,field_shape=(2,))
    values=(*v,v[14].detach().cos()*.03,v[10].detach().cos()*.02)
    def run(backend,*x):
        from warpSPHCore.native.api import warpOperationJVP as native
        qi,rj,domain=states(x[:10]);args=dict(referenceParticles=rj,adjacency=g,queryValues=x[14],referenceValues=x[10],tangentQueryValues=x[20],tangentReferenceValues=x[21])
        if corrected:args.update(referenceVolumes=x[13],crkState=sph.CRKState(x[11],x[12],x[15],x[16]),renormalizationState=sph.RenormalizationState(x[18]))
        return (native if backend=='native' else sph.warpOperationJVP)(qi,p,domain,**args)
    def equation(*x):
        primal=list(x[:20]);primal[14]=x[20];primal[10]=x[21]
        primal[17]=torch.ones_like(primal[17]);primal[19]=torch.ones_like(primal[19])
        return oracle(primal,d,p,g,corrected)
    return values,run,equation


def check_forward(values,run,equation,role,backends):
    values=tuple(x.detach() for x in values);directions=tuple(torch.full_like(x,.02) if k==role else torch.zeros_like(x) for k,x in enumerate(values))
    expected=torch.autograd.functional.jvp(equation,values,directions)[1]
    tolerance=6e-3 if values[0].dtype==torch.float32 else 2e-7
    for backend in backends:
        with torch.autograd.forward_ad.dual_level():
            x=list(values);x[role]=torch.autograd.forward_ad.make_dual(x[role],directions[role])
            primal,tangent=torch.autograd.forward_ad.unpack_dual(run(backend,*x))
            torch.testing.assert_close(torch.zeros_like(primal) if tangent is None else tangent,expected,rtol=tolerance,atol=tolerance)


@pytest.mark.parametrize('mode,corrected,role',FORWARD_CASES)
def test_laplacian_field_only_forward_reference_couplings(device,mode,corrected,role):
    values,run,equation=forward_fixture(device,mode,corrected)
    check_forward(values,run,equation,role,('native','warp'))


@pytest.mark.parametrize('mode,corrected,role',EXTENSION_CASES)
def test_native_field_only_forward_bounds_and_ignored_corrections(device,mode,corrected,role):
    values,run,equation=forward_fixture(device,mode,corrected)
    check_forward(values,run,equation,role,('native',))
    with torch.autograd.forward_ad.dual_level():
        x=[v.detach() for v in values];x[role]=torch.autograd.forward_ad.make_dual(x[role],torch.full_like(x[role],.02))
        with pytest.raises(NotImplementedError):run('warp',*x)


@pytest.mark.parametrize('variant',VARIANTS)
@pytest.mark.parametrize('side',['query','reference'])
def test_one_sided_field_product_without_primal_fields(device,variant,side):
    values,compute=fixture(device,variant,sph.GradientScheme.Difference)
    expected=compute('warp',side,ordinary=True)(*values)
    tolerance=6e-3 if values[0].dtype==torch.float32 else 2e-7
    for backend in ('native','warp'):
        torch.testing.assert_close(compute(backend,side,no_primal=True)(*values),expected,rtol=tolerance,atol=tolerance)


CORRECTION_CASES=[(variant,kind) for variant in VARIANTS for kind in ('volume','crk','renorm')
                  if kind=='volume' or variant not in ('interpolate',sph.LaplacianScheme.Naive)]


def correction_fixture(device,variant,kind):
    family='laplacian' if isinstance(variant,sph.LaplacianScheme) else variant
    field_shape=(2,) if family in ('divergence','curl') or variant==sph.LaplacianScheme.Dot else ()
    options=dict(dtype=get_torch_precision(),field_shape=field_shape)
    if family=='laplacian':options['mode']=variant
    q,r,d,p,g,v,states,_=importlib.import_module('_support.'+family+'_contract').case(device,**options)
    slot=13 if kind=='volume' else 11 if kind=='crk' else 18
    values=(*v,(v[slot].detach().cos()*.03).requires_grad_())
    def compute(backend,force_geometry=False):
        def run(*x):
            from warpSPHCore.native.api import warpOperationJVP as native
            qi,rj,domain=states(x[:10]);args=dict(referenceParticles=rj,adjacency=g,queryValues=x[14] if family!='interpolate' else None,referenceValues=x[10])
            if kind=='volume':args.update(referenceVolumes=x[13],tangentReferenceVolumes=x[-1])
            elif kind=='crk':args.update(crkState=sph.CRKState(x[11],x[12],x[15],x[16]),crkTangentState=sph.CRKTangentState(x[-1],torch.zeros_like(x[12]),torch.zeros_like(x[15]),torch.zeros_like(x[16])))
            else:args.update(renormalizationState=sph.RenormalizationState(x[18]),renormalizationTangentState=sph.RenormalizationTangentState(x[-1]))
            if force_geometry:args['queryTangentState']=sph.ParticleTangentState(torch.zeros_like(x[0]),None,None)
            return (native if backend=='native' else sph.warpOperationJVP)(qi,p,domain,**args)
        return run
    return values,compute


@pytest.mark.parametrize('variant,kind',CORRECTION_CASES)
def test_public_correction_only_dispatch_and_reverse(device,variant,kind):
    values,compute=correction_fixture(device,variant,kind)
    tolerance=6e-3 if values[0].dtype==torch.float32 else 2e-7;inputs=[x for x in values if x.requires_grad]
    for weighted in (False,True):
        expected=compute('warp',True)(*values)
        seed=torch.linspace(-.3,.7,expected.numel(),device=device,dtype=expected.dtype).reshape_as(expected) if weighted else torch.ones_like(expected)
        wanted=torch.autograd.grad(expected,inputs,seed,allow_unused=True)
        for backend in ('native','warp'):
            actual=compute(backend)(*values);torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance)
            bars=torch.autograd.grad(actual,inputs,seed,allow_unused=True)
            for bar,target,x in zip(bars,wanted,inputs):
                torch.testing.assert_close(torch.zeros_like(x) if bar is None else bar,torch.zeros_like(x) if target is None else target,rtol=tolerance,atol=tolerance)
