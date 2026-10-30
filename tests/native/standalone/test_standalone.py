"""Framework-independent native runtime against independent Torch oracles."""
from dataclasses import replace
import pytest
import torch
import warpSPHCore as sph
from _support.operators import fixture, dense, OPS, differentiable_case
from _support.standalone import Request, lib, run, request, output, search, vjp, valid

@pytest.mark.parametrize('op',OPS)
@pytest.mark.parametrize('dim',[1,2,3])
@pytest.mark.parametrize('rank',[0,1,2])
@pytest.mark.parametrize('scheme',list(sph.GradientScheme))
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_standalone_primals(device,op,dim,rank,scheme,dtype):
    valid(op,rank,dim)
    q,r,d,qf,rf=fixture(device,dim,rank)
    def cast(p):return replace(p,positions=p.positions.to(dtype),supports=p.supports.to(dtype),masses=p.masses.to(dtype),densities=p.densities.to(dtype))
    q,r=cast(q),cast(r);d=replace(d,min=d.min.to(dtype),max=d.max.to(dtype));qf=qf.to(dtype);rf=rf.to(dtype)
    prop=sph.OperationProperties(sph.KernelFunctions.Wendland4,operation=op,gradientMode=scheme,
        supportMode=sph.SupportScheme.KernelMeanSymmetric,operationMode=sph.OperationDirection.AllToAll)
    a=search(q,r,d,qf,rf,prop);req=request(q,r,d,qf,rf,prop,a)
    expected=dense(q,r,d,prop,qf,rf)
    actual=output(req,torch.empty_like(expected))
    torch.testing.assert_close(actual,expected,rtol=1e-3 if dtype==torch.float32 else 3e-10,atol=1e-3 if dtype==torch.float32 else 3e-10)


@pytest.mark.parametrize('op',OPS)
@pytest.mark.parametrize('support',list(sph.SupportScheme)[:-1])
@pytest.mark.parametrize('mode',list(sph.LaplacianScheme))
def test_standalone_adjoints(device,op,support,mode):
    q,r,d,qf,rf,values=differentiable_case(device,3,2)
    with torch.no_grad():q.positions[:,0].add_(4);r.positions[:,0].sub_(2)
    prop=sph.OperationProperties(sph.KernelFunctions.CubicSpline,operation=op,gradientMode=sph.GradientScheme.Symmetric,
        supportMode=support,laplacianMode=mode,operationMode=sph.OperationDirection.TrueAllToToAll)
    a=search(q,r,d,qf,rf,prop);req=request(q,r,d,qf,rf,prop,a)
    expected=dense(q,r,d,prop,qf,rf,differentiable=True);cot=torch.arange(expected.numel(),device=device,dtype=expected.dtype).reshape(expected.shape)/max(1,expected.numel())
    slots=[0,1,2,3,4,16,15,13,14,7,8]
    got=vjp(req,values,slots,cot)
    want=torch.autograd.grad(expected,values,cot,allow_unused=True)
    for actual,reference,value in zip(got,want,values):
        torch.testing.assert_close(actual,torch.zeros_like(value) if reference is None else reference,rtol=3e-8,atol=3e-8)
    original=[g.clone() for g in got]
    vjp(req,values,slots,cot,accumulate=True,buffers=got)
    for actual,before in zip(got,original):torch.testing.assert_close(actual,2*before,rtol=3e-8,atol=3e-8)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('op',OPS)
def test_standalone_kernel_families(device,kernel,op):
    q,r,d,qf,rf=fixture(device,2,1)
    prop=sph.OperationProperties(kernel,operation=op,gradientMode=sph.GradientScheme.Difference)
    a=search(q,r,d,qf,rf,prop);req=request(q,r,d,qf,rf,prop,a)
    expected=dense(q,r,d,prop,qf,rf)
    torch.testing.assert_close(output(req,torch.empty_like(expected)),expected,rtol=3e-9,atol=3e-9)


@pytest.mark.parametrize('bits',[(32,32),(32,64),(64,32),(64,64)])
def test_csr_widths_metadata_and_stream(device,bits):
    stream=torch.cuda.Stream() if device=='cuda' else None
    with torch.cuda.stream(stream) if stream else __import__('contextlib').nullcontext():
        q,r,d,qf,rf=fixture(device,3,2)
        prop=sph.OperationProperties(sph.KernelFunctions.Wendland2,operation=sph.WarpOperation.Gradient)
        a=search(q,r,d,qf,rf,prop);a=(a[0].to(torch.int32 if bits[0]==32 else torch.int64),a[1],a[2].to(torch.int32 if bits[1]==32 else torch.int64))
        req=request(q,r,d,qf,rf,prop,a);expected=dense(q,r,d,prop,qf,rf)
        req.validation=1
        torch.testing.assert_close(output(req,torch.empty_like(expected)),expected,rtol=3e-10,atol=3e-10)
    if stream:torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.parametrize('sizes',[(0,5),(7,0),(0,0)])
@pytest.mark.parametrize('op',OPS)
def test_standalone_empty_sets(device,sizes,op):
    q,r,d,qf,rf=fixture(device,2,1);n,m=sizes
    def trim(p,n):return replace(p,positions=p.positions[:n],supports=p.supports[:n],masses=p.masses[:n],densities=p.densities[:n],kinds=p.kinds[:n])
    q,r=trim(q,n),trim(r,m);qf,rf=qf[:n],rf[:m]
    prop=sph.OperationProperties(sph.KernelFunctions.Wendland2,operation=op)
    a=search(q,r,d,qf,rf,prop);req=request(q,r,d,qf,rf,prop,a);req.task=5;size=run(req)
    actual=output(req,torch.full((size,),float('nan'),device=device,dtype=q.positions.dtype))
    assert actual.count_nonzero()==0
    values=[q.positions,r.positions,q.supports,r.supports,r.masses,d.min,d.max,qf,rf,r.densities,q.densities]
    got=vjp(req,values,[0,1,2,3,4,7,8,13,14,15,16],torch.ones_like(actual))
    assert all(g.count_nonzero()==0 for g in got)


def test_standalone_errors_and_collection_bounds(device):
    q,r,d,qf,rf=fixture(device,2,1)
    prop=sph.OperationProperties(sph.KernelFunctions.Wendland2,operation=sph.WarpOperation.Gradient)
    a=search(q,r,d,qf,rf,prop);req=request(q,r,d,qf,rf,prop,a)
    out=torch.empty(28,device=device,dtype=q.positions.dtype)
    req.sizes[0]=0
    with pytest.raises(RuntimeError,match='undersized'):output(req,out)
    req=request(q,r,d,qf,rf,prop,a)
    req.output=q.positions.data_ptr();req.output_size=out.numel();req.task=0
    with pytest.raises(RuntimeError,match='overlap'):run(req)
    bad=q.supports.clone();bad[0]=-1;req.inputs[2]=bad.data_ptr()
    with pytest.raises(RuntimeError,match='contents'):output(req,out)
    req=request(q,r,d,qf,rf,prop,a);badids=a[2].clone();badids[0]=len(r.positions);req.inputs[12]=badids.data_ptr()
    with pytest.raises(RuntimeError,match='reference index'):output(req,out)
    req=request(q,r,d,qf,rf,prop,a);counts=torch.zeros_like(a[1]);offsets=torch.zeros_like(a[0]);req.inputs[10]=offsets.data_ptr();req.inputs[11]=counts.data_ptr()
    canary=torch.tensor([123,456],device=device,dtype=torch.int64);req.task=4;req.output=canary.data_ptr();req.output_size=0
    with pytest.raises(RuntimeError,match='recount'):run(req)
    torch.testing.assert_close(canary,torch.tensor([123,456],device=device,dtype=torch.int64))


@pytest.mark.cpu
@pytest.mark.parametrize('op',OPS)
@pytest.mark.parametrize('threads',[1,4])
def test_parallel_rows_and_scattered_adjoints(device,op,threads):
    q,r,d,qf,rf=fixture(device,3,2)
    q=replace(q,positions=q.positions.repeat(10,1),supports=q.supports.repeat(10),
        masses=q.masses.repeat(10),densities=q.densities.repeat(10),kinds=q.kinds.repeat(10))
    qf=qf.repeat(10,1,1)
    values=[q.positions,r.positions,q.supports,r.supports,r.masses,q.densities,r.densities,qf,rf,d.min,d.max]
    values=[x.detach().clone().requires_grad_() for x in values]
    q=replace(q,positions=values[0],supports=values[2],densities=values[5]);r=replace(r,positions=values[1],supports=values[3],masses=values[4],densities=values[6])
    qf,rf=values[7:9];d=replace(d,min=values[9],max=values[10])
    prop=sph.OperationProperties(sph.KernelFunctions.Wendland4,operation=op,gradientMode=sph.GradientScheme.Symmetric,
        supportMode=sph.SupportScheme.KernelMeanSymmetric)
    a=search(q,r,d,qf,rf,prop);req=request(q,r,d,qf,rf,prop,a);req.threads=threads
    expected=dense(q,r,d,prop,qf,rf,differentiable=True)
    torch.testing.assert_close(output(req,torch.empty_like(expected)),expected,rtol=3e-10,atol=3e-10)
    cot=torch.ones_like(expected);got=vjp(req,values,[0,1,2,3,4,16,15,13,14,7,8],cot)
    want=torch.autograd.grad(expected,values,cot,allow_unused=True)
    for g,w,v in zip(got,want,values):torch.testing.assert_close(g,torch.zeros_like(v) if w is None else w,rtol=3e-8,atol=3e-8)


@pytest.mark.parametrize('op',OPS)
def test_float32_adjoints_and_zero_mass(device,op):
    q,r,d,qf,rf=fixture(device,3,1)
    values=[q.positions,r.positions,q.supports,r.supports,r.masses,q.densities,r.densities,qf,rf,d.min,d.max]
    values=[x.float().detach().requires_grad_() for x in values]
    with torch.no_grad():values[4][0]=0
    q=replace(q,positions=values[0],supports=values[2],masses=q.masses.float(),densities=values[5])
    r=replace(r,positions=values[1],supports=values[3],masses=values[4],densities=values[6])
    qf,rf=values[7:9];d=replace(d,min=values[9],max=values[10])
    prop=sph.OperationProperties(sph.KernelFunctions.Wendland6,operation=op,gradientMode=sph.GradientScheme.Symmetric)
    a=search(q,r,d,qf,rf,prop);req=request(q,r,d,qf,rf,prop,a)
    expected=dense(q,r,d,prop,qf,rf,differentiable=True);cot=torch.ones_like(expected)
    got=vjp(req,values,[0,1,2,3,4,16,15,13,14,7,8],cot)
    want=torch.autograd.grad(expected,values,cot,allow_unused=True)
    for g,w,v in zip(got,want,values):torch.testing.assert_close(g,torch.zeros_like(v) if w is None else w,rtol=5e-4,atol=5e-4)
