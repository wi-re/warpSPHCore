"""Native meshless primals and shared closure controller, bounded by contract."""
import pytest
import torch
from torch.autograd import forward_ad
import warpSPHCore as sph
from warpSPHCore.mfm import MeshlessGeometry,MeshlessNative,mfmRates,signalTimestep


def particles(device,dimension,dtype,n=5):
    axis=torch.linspace(-.4,.4,n,device=device,dtype=dtype)
    positions=axis[:,None] if dimension==1 else torch.cartesian_prod(*([axis]*dimension))
    positions=positions+torch.sin(torch.arange(positions.numel(),device=device,dtype=dtype).reshape_as(positions))*.005
    support=positions.new_full((len(positions),),.65)
    state=sph.ParticleState(positions,support,torch.ones_like(support),
                           torch.zeros(len(support),device=device,dtype=torch.int32),torch.ones_like(support))
    domain=sph.DomainDescription(positions.new_full((dimension,),-1),positions.new_full((dimension,),1),
                                 torch.ones(dimension,device=device,dtype=torch.bool),dimension)
    return state,domain


def test_meshless_geometry_rates_conservation_and_cfl(device):
    for dimension,dtype,kernel in ((1,torch.float32,sph.KernelFunctions.CubicSpline),
                                   (2,torch.float64,sph.KernelFunctions.Wendland4),
                                   (3,torch.float32,sph.KernelFunctions.QuinticSpline)):
        state,domain=particles(device,dimension,dtype)
        reference=MeshlessGeometry.build(state,domain,kernel,closure='none',area_cap=True)
        native=MeshlessNative.build(state.positions,state.supports,domain,kernel,closure='none',area_cap=True,storeI=False)
        assert native.adjacency.paddedI is None
        tolerance=4e-4 if dtype==torch.float32 else 2e-10
        torch.testing.assert_close(native.omega,reference.omega,rtol=tolerance,atol=tolerance)
        torch.testing.assert_close(native.Einv,reference.Einv,rtol=tolerance,atol=tolerance)
        rho=1+.1*torch.arange(len(state.supports),device=device,dtype=dtype).sin()
        velocity=(2*state.positions).sin()*.1
        pressure=1+.1*torch.arange(len(rho),device=device,dtype=dtype).cos()
        for mode,order in (('MFM',1),('MFV',2)):
            rates=native.rates(rho,velocity,pressure,1.4,dt=.001,mode=mode,order=order)
            expected,_=mfmRates(reference,rho,velocity,pressure,1.4,dt=.001,mode=mode,order=order)
            torch.testing.assert_close(rates,expected,rtol=tolerance,atol=tolerance)
            torch.testing.assert_close(rates.sum(0),torch.zeros(dimension+2,device=device,dtype=dtype),rtol=0,
                atol=3e-5 if dtype==torch.float32 else 2e-11)
        torch.testing.assert_close(native.timestep(rho,velocity,pressure,1.4),
                                   signalTimestep(reference,rho,velocity,pressure,1.4),rtol=tolerance,atol=tolerance)


def test_meshless_closure_multigrid_and_captured_rates(device):
    n=10;dtype=torch.float64
    axis=torch.arange(n,device=device,dtype=dtype)/n
    positions=torch.cartesian_prod(axis,axis)
    positions=(positions+.015*torch.arange(positions.numel(),device=device,dtype=dtype).reshape_as(positions).sin())%1
    supports=torch.full((len(positions),),4/n,device=device,dtype=dtype)
    domain=sph.DomainDescription(supports.new_zeros(2),supports.new_ones(2),torch.ones(2,device=device,dtype=torch.bool),2)
    plain=MeshlessNative.build(positions,supports,domain,sph.KernelFunctions.Wendland2,
        closure_solver='cg',cg_iters=100,cg_tol=1e-8,storeI=False)
    multigrid=MeshlessNative.build(positions,supports,domain,sph.KernelFunctions.Wendland2,
        closure_solver='mg',mg_max_coarse=4,cg_iters=100,cg_tol=1e-8,storeI=False)
    # The graph Laplacian leaves a constant potential undetermined. Compare
    # the same gauge; face corrections only depend on potential differences.
    torch.testing.assert_close(multigrid.lam-multigrid.lam.mean(0),
                               plain.lam-plain.lam.mean(0),rtol=2e-5,atol=2e-6)
    assert float(multigrid.closureResidualRatio)<2e-8
    rho=torch.ones_like(supports);velocity=torch.zeros_like(positions);pressure=torch.ones_like(rho)
    expected=multigrid.rates(rho,velocity,pressure,1.4)
    if device!='cpu':
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            multigrid.rates(rho,velocity,pressure,1.4)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):output=multigrid.rates(rho,velocity,pressure,1.4)
            pressure.mul_(2);graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(output,expected*2,rtol=1e-8,atol=1e-9)
    # Bounded closure uses retained per-edge weights, not an unbounded face correction.
    plain.close(cg_iters=60,closure_cap=.2)
    assert torch.isfinite(plain.lam).all()


def test_meshless_forward_only_boundary_and_torch_reference_ad(device):
    state,domain=particles(device,1,torch.float64)
    native=MeshlessNative.build(state.positions,state.supports,domain,sph.KernelFunctions.Wendland2,closure='none')
    rho=torch.ones_like(state.supports,requires_grad=True)
    with pytest.raises(NotImplementedError,match='forward-only'):
        native.rates(rho,torch.zeros_like(state.positions),rho,1.4)
    with forward_ad.dual_level():
        dual=forward_ad.make_dual(state.positions,torch.ones_like(state.positions))
        with pytest.raises(NotImplementedError,match='forward-only'):
            MeshlessNative.build(dual,state.supports,domain,sph.KernelFunctions.Wendland2)
    # The differentiable Torch meshless implementation is also Warp-free.
    positions=state.positions.clone().requires_grad_()
    reference=MeshlessGeometry.build(sph.ParticleState(positions,state.supports,state.masses,state.kinds,state.densities),
                                     domain,sph.KernelFunctions.Wendland2,closure='none')
    gradient,=torch.autograd.grad(reference.omega.sum(),positions)
    assert torch.isfinite(gradient).all()
