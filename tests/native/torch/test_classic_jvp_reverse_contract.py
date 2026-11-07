"""Actual gradchecks of ordinary forward-AD tangents on native owners."""
import pytest
import torch
import warpSPHCore as sph
from _support.classic_jvp_contract import VARIANTS,FIELD_VARIANTS,case


@pytest.mark.parametrize('variant',VARIANTS)
@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('corrected',[False,True])
def test_ordinary_geometry_tangent_actual_reverse_gradcheck(device,variant,kernel,corrected):
    if variant=='density' and corrected:pytest.skip('density has no corrections')
    i=list(sph.KernelFunctions).index(kernel)
    *_,values,states,functions=case(device,variant,1+i%3,kernel=kernel,correction='combined' if corrected else 'plain')
    assert torch.autograd.gradcheck(functions('native'),values,fast_mode=True,atol=8e-5,rtol=2e-3)


@pytest.mark.parametrize('variant',FIELD_VARIANTS)
@pytest.mark.parametrize('dimension',[2,3])
def test_packed_corrected_geometry_tangent_actual_reverse_gradcheck(device,variant,dimension):
    *_,values,states,functions=case(device,variant,dimension,correction='combined',packed=True)
    assert torch.autograd.gradcheck(functions('native'),values,fast_mode=True,atol=8e-5,rtol=2e-3)


@pytest.mark.parametrize('variant',FIELD_VARIANTS)
@pytest.mark.parametrize('corrected',[False,True])
def test_packed_field_only_tangent_actual_reverse_gradcheck(device,variant,corrected):
    *_,values,states,functions=case(device,variant,correction='combined' if corrected else 'plain',field_only=True,packed=True)
    assert torch.autograd.gradcheck(functions('native'),values,fast_mode=True,atol=8e-5,rtol=2e-3)


@pytest.mark.gpu
@pytest.mark.parametrize('variant',VARIANTS)
@pytest.mark.parametrize('corrected',[False,True])
def test_capture_ordinary_tangent_and_its_reverse(device,variant,corrected):
    if device!='cuda':pytest.skip('CUDA execution contract')
    if variant=='density' and corrected:pytest.skip('density has no corrections')
    import gc
    gc.collect();enabled=gc.isenabled();gc.disable()
    try:
        *_,values,states,functions=case(device,variant,dtype=torch.float32,correction='combined' if corrected else 'plain')
        run=functions('native');inputs=[x for x in values if x.requires_grad]
        with torch.no_grad():example=run(*values);seed=torch.linspace(-.3,.7,example.numel(),device=device).reshape_as(example)
        del example
        def products():
            y=run(*values);bars=torch.autograd.grad(y,inputs,seed,allow_unused=True)
            return y,*(torch.zeros_like(x) if b is None else b for x,b in zip(inputs,bars))
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):products()
        torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured=products()
        with torch.no_grad():values[0][:,1].add_(.01);values[2].mul_(1.03);values[-1].mul_(1.04)
        graph.replay();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):wanted=products()
        torch.cuda.current_stream().wait_stream(stream)
        for a,b in zip(captured,wanted):torch.testing.assert_close(a,b,rtol=6e-4,atol=6e-4)
    finally:
        if enabled:gc.enable()


@pytest.mark.parametrize('variant',VARIANTS)
@pytest.mark.parametrize('field_only',[False,True])
def test_zero_direction_reverse_remains_linear_actual_gradcheck(device,variant,field_only):
    if field_only and variant not in FIELD_VARIANTS:pytest.skip('operator has no fields')
    *_,values,states,functions=case(device,variant,field_only=field_only)
    # The reference ordinary bridge drops zero-valued directions as an
    # optimization. Native retains the linear map's derivative at zero;
    # qualify that extension by actual finite differences.
    family='laplacian' if isinstance(variant,sph.LaplacianScheme) else variant
    number=10 if family=='density' else 14 if family=='interpolate' else 20
    values=(*values[:number],*(torch.zeros_like(x).requires_grad_() for x in values[number:]))
    assert not functions('native')(*values).count_nonzero()
    assert torch.autograd.gradcheck(functions('native'),values,fast_mode=True,atol=8e-5,rtol=2e-3)


@pytest.mark.parametrize('variant',VARIANTS)
def test_sparse_strided_ordinary_tangent_reverse_is_deterministic(device,variant):
    q,r,d,p,g,values,states,functions=case(device,variant,correction='plain' if variant=='density' else 'combined')
    g.paddedJ=torch.tensor([-1,-1,0,2,2,-1],device=device,dtype=torch.int64);g.edgeOffsets=torch.tensor([2,5],device=device,dtype=torch.int64);g.numNeighbors=torch.tensor([3,0],device=device,dtype=torch.int32)
    def strided(x):
        storage=torch.empty((*x.shape,2),device=x.device,dtype=x.dtype);storage[...,0]=x.detach()
        return storage[...,0].requires_grad_(x.requires_grad)
    values=tuple(strided(x) for x in values);inputs=[x for x in values if x.requires_grad]
    prior=torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        for attempt in range(3):
            y=functions('native')(*values);seed=torch.linspace(-.3,.7,y.numel(),device=device,dtype=y.dtype).reshape_as(y)
            bars=torch.autograd.grad(y,inputs,seed,allow_unused=True)
            assert all(b is None or torch.isfinite(b).all() for b in bars)
            if attempt==0:first=bars
            else:
                for a,b in zip(first,bars):
                    if a is not None:torch.testing.assert_close(a,b,rtol=0,atol=0)
    finally:torch.use_deterministic_algorithms(prior)
    y=functions('native')(*values);g.paddedJ[2]=1
    with pytest.raises(RuntimeError,match='topology changed|modified by an inplace'):torch.autograd.grad(y,inputs,torch.ones_like(y),allow_unused=True)


@pytest.mark.parametrize('mode',list(sph.LaplacianScheme))
@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('corrected',[False,True])
@pytest.mark.parametrize('field_only',[False,True])
def test_positive_laplacian_tangent_actual_reverse_gradcheck(device,mode,dimension,corrected,field_only):
    *_,values,states,functions=case(device,mode,dimension,correction='combined' if corrected else 'plain',field_only=field_only,positive=True)
    assert torch.autograd.gradcheck(functions('native'),values,fast_mode=True,atol=8e-5,rtol=2e-3)


@pytest.mark.parametrize('dot',[False,True])
@pytest.mark.parametrize('consistent',[False,True])
@pytest.mark.parametrize('field_only',[False,True])
def test_divergence_contractions_and_consistency_tangent_actual_gradcheck(device,dot,consistent,field_only):
    *_,values,states,functions=case(device,'divergence',correction='combined',field_only=field_only,packed=True,dot=dot,consistent=consistent)
    assert torch.autograd.gradcheck(functions('native'),values,fast_mode=True,atol=8e-5,rtol=2e-3)


@pytest.mark.parametrize('mode',list(sph.LaplacianScheme))
@pytest.mark.parametrize('corrected',[False,True])
@pytest.mark.parametrize('role',[0,2,5,6,11,13,17,18,19,20,21])
def test_positive_field_explicit_forward_composition_and_reverse_gradcheck(device,mode,corrected,role):
    from _support.positive_field_product_contract import case as positive_case
    values,product,reference,equation=positive_case(device,mode,corrected=corrected)
    outer=(values[role].detach().cos()*.02).requires_grad_();all_values=(*values,outer)
    def run(*x):
        with torch.autograd.forward_ad.dual_level():
            dual=list(x[:-1]);dual[role]=torch.autograd.forward_ad.make_dual(dual[role],x[-1])
            y,t=torch.autograd.forward_ad.unpack_dual(product(*dual))
            return torch.zeros_like(y) if t is None else t
    expected=torch.autograd.functional.jvp(equation,tuple(x.detach() for x in values),tuple(outer.detach() if k==role else torch.zeros_like(x) for k,x in enumerate(values)))[1]
    torch.testing.assert_close(run(*all_values),expected,rtol=2e-7,atol=2e-7)
    assert torch.autograd.gradcheck(run,all_values,fast_mode=True,atol=8e-5,rtol=2e-3)


@pytest.mark.parametrize('variant',VARIANTS)
@pytest.mark.parametrize('field_only',[False,True])
def test_ordinary_tangent_reverse_materializes_efficient_zero_cotangent(device,variant,field_only):
    if field_only and variant not in FIELD_VARIANTS:pytest.skip('operator has no fields')
    *_,values,states,functions=case(device,variant,field_only=field_only)
    y=functions('native')(*values)
    seed=torch._efficientzerotensor(y.shape,device=y.device,dtype=y.dtype)
    bars=torch.autograd.grad(y,[x for x in values if x.requires_grad],seed,allow_unused=True)
    for bar in bars:
        if bar is not None:torch.testing.assert_close(bar,torch.zeros_like(bar),rtol=0,atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize('mode',list(sph.LaplacianScheme))
@pytest.mark.parametrize('corrected',[False,True])
def test_capture_positive_field_product_forward_composition_and_reverse(device,mode,corrected):
    if device!='cuda':pytest.skip('CUDA execution contract')
    import gc
    from _support.positive_field_product_contract import case as positive_case
    gc.collect();enabled=gc.isenabled();gc.disable()
    try:
        values,product,_,_=positive_case(device,mode,dtype=torch.float32,corrected=corrected)
        outer=(values[0].detach().cos()*.02).requires_grad_()
        inputs=[x for x in (*values,outer) if x.requires_grad]
        seed=torch.linspace(-.3,.7,values[20].numel(),device=device).reshape_as(values[20])
        def run():
            with torch.autograd.forward_ad.dual_level():
                dual=list(values);dual[0]=torch.autograd.forward_ad.make_dual(values[0],outer)
                y=torch.autograd.forward_ad.unpack_dual(product(*dual)).tangent
            bars=torch.autograd.grad(y,inputs,seed,allow_unused=True)
            return y,*(torch.zeros_like(x) if b is None else b for x,b in zip(inputs,bars))
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):run()
        torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured=run()
        with torch.no_grad():values[0][:,1].add_(.01);values[2].mul_(1.03);outer.mul_(1.04);values[21].mul_(.97)
        graph.replay();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):wanted=run()
        torch.cuda.current_stream().wait_stream(stream)
        for a,b in zip(captured,wanted):torch.testing.assert_close(a,b,rtol=7e-4,atol=7e-4)
    finally:
        if enabled:gc.enable()
