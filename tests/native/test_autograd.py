"""Manual native adjoints against independent Torch AD and finite differences."""
from dataclasses import replace
import pytest
import torch
import warpSPHCore as sph
if sph.backend != 'native':pytest.skip('native process only',allow_module_level=True)
from test_kernels import device
from test_primal import fixture, dense

OPS=[sph.WarpOperation.Density,sph.WarpOperation.Interpolate,sph.WarpOperation.Gradient,
     sph.WarpOperation.Divergence,sph.WarpOperation.Curl,sph.WarpOperation.Laplacian,sph.WarpOperation.Covariance]


def differentiable_case(device,dim,rank):
    q,r,d,qf,rf=fixture(device,dim,rank)
    values=[q.positions,r.positions,q.supports,r.supports,r.masses,q.densities,r.densities,
            qf,rf,d.min,d.max]
    values=[x.detach().clone().requires_grad_() for x in values]
    q=replace(q,positions=values[0],supports=values[2],densities=values[5])
    r=replace(r,positions=values[1],supports=values[3],masses=values[4],densities=values[6])
    d=replace(d,min=values[9],max=values[10])
    return q,r,d,values[7],values[8],values


def compare_vjp(q,r,d,qf,rf,values,prop):
    actual=sph.warpOperation(q,prop,d,qf,rf,referenceParticles=r)
    expected=dense(q,r,d,prop,qf,rf,differentiable=True)
    gen=torch.Generator().manual_seed(345)
    cot=torch.randn(actual.shape,generator=gen,dtype=actual.dtype).to(actual.device)
    grads=torch.autograd.grad(actual,values,cot,allow_unused=True)
    oracle=torch.autograd.grad(expected,values,cot,allow_unused=True) if expected.requires_grad else (None,)*len(values)
    for slot,(got,want,original) in enumerate(zip(grads,oracle,values)):
        got=torch.zeros_like(original) if got is None else got
        want=torch.zeros_like(original) if want is None else want
        torch.testing.assert_close(got,want,rtol=3e-4 if got.dtype==torch.float32 else 3e-8,atol=3e-4 if got.dtype==torch.float32 else 3e-8,msg=lambda msg:f'input {slot}: {msg}')


@pytest.mark.parametrize('op',OPS)
@pytest.mark.parametrize('scheme',list(sph.GradientScheme))
@pytest.mark.parametrize('dim',[1,2,3])
@pytest.mark.parametrize('rank',[0,1,2])
def test_native_vjp_matches_independent_ad(device,op,scheme,dim,rank):
    if op==sph.WarpOperation.Divergence and rank==0:pytest.skip('requires vector')
    if op==sph.WarpOperation.Curl and rank==0 and dim>1:pytest.skip('requires vector')
    q,r,d,qf,rf,values=differentiable_case(device,dim,rank)
    prop=sph.OperationProperties(sph.KernelFunctions.Wendland4,operation=op,gradientMode=scheme,
        supportMode=sph.SupportScheme.KernelMeanSymmetric,operationMode=sph.OperationDirection.AllToAll)
    compare_vjp(q,r,d,qf,rf,values,prop)


@pytest.mark.parametrize('op',OPS)
@pytest.mark.parametrize('support',list(sph.SupportScheme)[:-1])
def test_support_policies_and_periodic_domain_adjoints(device,op,support):
    q,r,d,qf,rf,values=differentiable_case(device,2,1)
    # Exercise multiple periodic images, not just positions already in the box.
    with torch.no_grad():q.positions[:,0].add_(4);r.positions[:,0].sub_(2)
    prop=sph.OperationProperties(sph.KernelFunctions.CubicSpline,operation=op,
        gradientMode=sph.GradientScheme.Symmetric,supportMode=support)
    compare_vjp(q,r,d,qf,rf,values,prop)


@pytest.mark.parametrize('mode',list(sph.LaplacianScheme))
@pytest.mark.parametrize('support',list(sph.SupportScheme)[:-1])
def test_laplacian_geometry_adjoints(device,mode,support):
    q,r,d,qf,rf,values=differentiable_case(device,3,1)
    prop=sph.OperationProperties(sph.KernelFunctions.Wendland6,operation=sph.WarpOperation.Laplacian,
        gradientMode=sph.GradientScheme.Symmetric,laplacianMode=mode,supportMode=support)
    compare_vjp(q,r,d,qf,rf,values,prop)


@pytest.mark.parametrize('op',OPS)
def test_finite_difference_gradcheck(device,op):
    q,r,d,qf,rf,values=differentiable_case(device,2,1)
    q=replace(q,positions=q.positions[:2].detach(),supports=q.supports[:2].detach(),densities=q.densities[:2].detach(),masses=q.masses[:2],kinds=q.kinds[:2])
    r=replace(r,positions=r.positions[:2].detach(),supports=r.supports[:2].detach(),densities=r.densities[:2].detach(),masses=r.masses[:2].detach(),kinds=r.kinds[:2])
    qf=qf[:2].detach();rf=rf[:2].detach()
    args=tuple(x.clone().requires_grad_() for x in (q.positions,r.positions,q.supports,r.supports,r.masses,q.densities,r.densities,qf,rf))
    prop=sph.OperationProperties(sph.KernelFunctions.Wendland2,operation=op,gradientMode=sph.GradientScheme.Difference,
        operationMode=sph.OperationDirection.TrueAllToToAll,supportMode=sph.SupportScheme.MeanSymmetric)
    def run(*args):
        qi=replace(q,positions=args[0],supports=args[2],densities=args[5])
        rj=replace(r,positions=args[1],supports=args[3],masses=args[4],densities=args[6])
        return sph.warpOperation(qi,prop,d,args[7],args[8],referenceParticles=rj)
    assert torch.autograd.gradcheck(run,args,eps=1e-6,atol=2e-5,rtol=2e-4,nondet_tol=1e-10,fast_mode=True)


def test_aliasing_chained_density_and_input_versions(device):
    q,_,d,qf,_=fixture(device,2,0)
    q=replace(q,positions=q.positions.detach().requires_grad_(),supports=q.supports.detach().requires_grad_(),
              masses=q.masses.detach().requires_grad_())
    rho_prop=sph.OperationProperties(sph.KernelFunctions.Wendland2,operation=sph.WarpOperation.Density,
        operationMode=sph.OperationDirection.TrueAllToToAll)
    rho=sph.warpOperation(q,rho_prop,d)
    q=replace(q,densities=rho)
    grad_prop=replace(rho_prop,operation=sph.WarpOperation.Gradient,gradientMode=sph.GradientScheme.Difference)
    result=sph.warpOperation(q,grad_prop,d,qf)
    result.square().sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q.positions,q.supports,q.masses))
    with torch.no_grad():q.positions.add_(.001)
    again=sph.warpOperation(q,rho_prop,d)
    with torch.no_grad():q.positions.add_(.001)
    with pytest.raises(RuntimeError,match='modified by an inplace operation'):again.sum().backward()


def test_explicit_ad_capability_limits(device):
    q,r,d,qf,rf,values=differentiable_case(device,2,0)
    prop=sph.OperationProperties(sph.KernelFunctions.Spiky,operation=sph.WarpOperation.Density)
    with pytest.raises(NotImplementedError,match='cusp/singular'):sph.warpOperation(q,prop,d,referenceParticles=r)
    prop=replace(prop,kernel=sph.KernelFunctions.Wendland2)
    previous=torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(NotImplementedError,match='deterministic'):sph.warpOperation(q,prop,d,referenceParticles=r)
    finally:torch.use_deterministic_algorithms(previous)
    output=sph.warpOperation(q,prop,d,referenceParticles=r)
    first=torch.autograd.grad(output.sum(),q.positions,create_graph=True)[0]
    with pytest.raises(RuntimeError):torch.autograd.grad(first.sum(),q.positions)

@pytest.mark.parametrize('kernel',[k for k in sph.KernelFunctions if k.value in {0,1,2,20,30,31,32,33,34,36,41}])
@pytest.mark.parametrize('op',OPS)
def test_reverse_kernel_family(device,kernel,op):
    q,r,d,qf,rf,values=differentiable_case(device,2,1)
    prop=sph.OperationProperties(kernel,operation=op,gradientMode=sph.GradientScheme.Difference,
        laplacianMode=sph.LaplacianScheme.Naive,supportMode=sph.SupportScheme.SuperSymmetric)
    compare_vjp(q,r,d,qf,rf,values,prop)


def test_tensor_dot_axis_adjoint(device):
    q,r,d,qf,rf,values=differentiable_case(device,3,2)
    prop=sph.OperationProperties(sph.KernelFunctions.Wendland2,operation=sph.WarpOperation.Divergence,
        gradientMode=sph.GradientScheme.Symmetric,divergenceDotMode=True)
    compare_vjp(q,r,d,qf,rf,values,prop)


@pytest.mark.parametrize('op',OPS)
def test_float32_zero_mass_adjoints(device,op):
    q,r,d,qf,rf=fixture(device,3,1)
    q=replace(q,positions=q.positions.float(),supports=q.supports.float(),masses=q.masses.float(),densities=q.densities.float())
    r=replace(r,positions=r.positions.float(),supports=r.supports.float(),masses=r.masses.float(),densities=r.densities.float())
    d=replace(d,min=d.min.float(),max=d.max.float())
    r.masses[0]=0
    values=[q.positions,r.positions,q.supports,r.supports,r.masses,q.densities,r.densities,qf.float(),rf.float(),d.min,d.max]
    values=[v.detach().clone().requires_grad_() for v in values]
    q=replace(q,positions=values[0],supports=values[2],densities=values[5])
    r=replace(r,positions=values[1],supports=values[3],masses=values[4],densities=values[6])
    d=replace(d,min=values[9],max=values[10])
    prop=sph.OperationProperties(sph.KernelFunctions.Wendland4,operation=op,gradientMode=sph.GradientScheme.Symmetric)
    compare_vjp(q,r,d,values[7],values[8],values,prop)


def test_unused_density_grad_does_not_enable_reverse(device):
    q,r,d,_,_=fixture(device,2,0)
    q=replace(q,densities=q.densities.detach().requires_grad_())
    prop=sph.OperationProperties(sph.KernelFunctions.Spiky,operation=sph.WarpOperation.Density)
    result=sph.warpOperation(q,prop,d,referenceParticles=r)
    assert not result.requires_grad


@pytest.mark.parametrize('op',[sph.WarpOperation.Density,sph.WarpOperation.Gradient])
def test_smooth_self_pair_gradcheck(device,op):
    q,_,d,field,_=fixture(device,2,0)
    q=replace(q,positions=q.positions[:1].clone(),supports=q.supports[:1].clone(),masses=q.masses[:1].clone(),densities=q.densities[:1].clone(),kinds=q.kinds[:1])
    prop=sph.OperationProperties(sph.KernelFunctions.B7,operation=op,gradientMode=sph.GradientScheme.Naive,
        operationMode=sph.OperationDirection.TrueAllToToAll)
    def run(x,h,m):
        state=replace(q,positions=x,supports=h,masses=m)
        return sph.warpOperation(state,prop,d,field[:1])
    args=tuple(v.detach().clone().requires_grad_() for v in (q.positions,q.supports,q.masses))
    assert torch.autograd.gradcheck(run,args,fast_mode=True,atol=1e-5,rtol=1e-4)


@pytest.mark.parametrize('op',OPS)
@pytest.mark.parametrize('sizes',[(0,5),(7,0),(0,0)])
def test_empty_population_adjoints(device,op,sizes):
    q,r,d,qf,rf,values=differentiable_case(device,2,1)
    nq,nr=sizes
    def trim(state,n):
        return replace(state,positions=state.positions[:n],supports=state.supports[:n],
                       masses=state.masses[:n],densities=state.densities[:n],kinds=state.kinds[:n])
    prop=sph.OperationProperties(sph.KernelFunctions.Wendland4,operation=op,
        gradientMode=sph.GradientScheme.Symmetric)
    out=sph.warpOperation(trim(q,nq),prop,d,qf[:nq],rf[:nr],referenceParticles=trim(r,nr))
    assert out.count_nonzero()==0
    grads=torch.autograd.grad(out.sum(),values,allow_unused=True)
    for value,grad in zip(values,grads):
        if grad is not None:torch.testing.assert_close(grad,torch.zeros_like(value),rtol=0,atol=0)


def test_adjoint_uses_current_cuda_stream(device):
    if device!='cuda':pytest.skip('CUDA stream contract')
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        q,r,d,qf,rf,values=differentiable_case(device,3,2)
        prop=sph.OperationProperties(sph.KernelFunctions.CubicSpline,operation=sph.WarpOperation.Gradient,
            gradientMode=sph.GradientScheme.Symmetric)
        compare_vjp(q,r,d,qf,rf,values,prop)
        assert torch.cuda.current_stream()==stream
    torch.cuda.current_stream().wait_stream(stream)
