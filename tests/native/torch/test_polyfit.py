"""Native polynomial assembly and shared reconstruction, without a new matrix sweep."""
from dataclasses import replace
import torch
from torch.autograd import forward_ad
import warpSPHCore as sph
from warpSPHCore.polyfit.native.nt_polyfit import _sum


def case(device,dimension,dtype=torch.float64):
    axis = torch.linspace(-.3,.3,5,device=device,dtype=dtype)
    positions = axis[:,None] if dimension==1 else torch.cartesian_prod(*([axis]*dimension))
    support = torch.ones(len(positions),device=device,dtype=dtype)
    particles = sph.ParticleState(positions,support,support,
                                  torch.zeros(len(support),device=device,dtype=torch.int32),support)
    domain = sph.DomainDescription(axis.new_full((dimension,),-1),axis.new_full((dimension,),1),
                                   torch.zeros(dimension,device=device,dtype=torch.bool),dimension)
    return particles,domain,sph.radiusSearch(particles,domain)


def test_polynomial_reproduction_counts_and_stencil_policies(device):
    for dimension,dtype in ((1,torch.float64),(2,torch.float32),(3,torch.float64)):
        particles,domain,adjacency = case(device,dimension,dtype)
        x = particles.positions
        field = (x.square()+.3*x).sum(-1)
        for constant in (False,True):
            fit = sph.PolyFit.build(particles,domain,sph.KernelFunctions.Wendland2,2,
                                    constant=constant,adjacency=adjacency)
            tolerance = 3e-4 if dtype==torch.float32 else 3e-10
            torch.testing.assert_close(fit.values(field),field,rtol=tolerance,atol=tolerance)
            torch.testing.assert_close(fit.gradient(field),2*x+.3,rtol=tolerance,atol=tolerance)
            torch.testing.assert_close(fit.laplacian(field),torch.full_like(field,2*dimension),rtol=tolerance,atol=tolerance)
        exponents = torch.tensor(sph.monomialExponents(dimension,2),device=device,dtype=torch.int32)
        properties = sph.OperationProperties(sph.KernelFunctions.Wendland2)
        if dimension < 3:
            _,counts = _sum(particles,properties,domain,exponents,adjacency=adjacency,
                            sector=0,numSectors=4,unitWeights=True)
            displacement = x[None]-x[:,None]
            if dimension==1: selected=displacement[...,0]<0
            else:
                angle=torch.remainder(torch.atan2(displacement[...,1],displacement[...,0]),2*torch.pi)
                selected=(angle>=0)&(angle<torch.pi/2)
            selected |= torch.eye(len(x),device=device,dtype=torch.bool)
            selected &= displacement.norm(dim=-1)<particles.supports[:,None]
            torch.testing.assert_close(counts,selected.sum(-1).int())


def test_polynomial_geometry_field_volume_and_domain_derivatives(device):
    particles,domain,adjacency = case(device,2)
    # A smaller separate source cloud exercises independent CSR/CSC ownership.
    reference = replace(particles,positions=particles.positions[:7].clone(),supports=particles.supports[:7],
                        masses=particles.masses[:7],densities=particles.densities[:7],kinds=particles.kinds[:7])
    adjacency = sph.radiusSearch(particles,domain,reference)
    exponents = torch.tensor(sph.monomialExponents(2,2),device=device,dtype=torch.int32)
    properties = sph.OperationProperties(sph.KernelFunctions.CubicSpline)
    inputs = (particles.positions.clone().requires_grad_(),reference.positions.clone().requires_grad_(),
              particles.supports.clone().requires_grad_(),reference.masses.clone().requires_grad_(),
              (reference.densities*1.3).requires_grad_(),reference.positions.cos().requires_grad_())
    def function(x,y,h,mass,density,field):
        return _sum(replace(particles,positions=x,supports=h),properties,domain,exponents,field,
                    referenceParticles=replace(reference,positions=y,masses=mass,densities=density),
                    adjacency=adjacency)[0]
    assert torch.autograd.gradcheck(function,inputs,fast_mode=True)
    directions = tuple(torch.cos(value.detach())*.02 for value in inputs)
    with forward_ad.dual_level():
        duals = tuple(forward_ad.make_dual(value,tangent) for value,tangent in zip(inputs,directions))
        _,actual = forward_ad.unpack_dual(function(*duals))
    epsilon = 1e-5
    expected = (function(*(v+epsilon*t for v,t in zip(inputs,directions)))-
                function(*(v-epsilon*t for v,t in zip(inputs,directions))))/(2*epsilon)
    torch.testing.assert_close(actual,expected,rtol=1e-7,atol=1e-8)
    # Explicit volume replaces m/rho, and periodic domain derivatives use the
    # same fixed image branch as the primal; both are independent AD inputs.
    periodic = replace(domain,periodic=torch.tensor([True,False],device=device))
    volume = torch.full_like(reference.masses,.7).requires_grad_()
    lower=domain.min.clone().requires_grad_();upper=domain.max.clone().requires_grad_()
    def moments(v,lo,hi):
        return _sum(particles,properties,replace(periodic,min=lo,max=hi),exponents,
                    referenceParticles=replace(reference,positions=reference.positions+reference.positions.new_tensor([2.,0.])),
                    referenceVolumes=v,adjacency=adjacency)[0]
    assert torch.autograd.gradcheck(moments,(volume,lower,upper),fast_mode=True)


def test_shared_reconstruction_and_geometry_tangent(device):
    particles,domain,adjacency = case(device,1)
    field = particles.positions[:,0].square()+.3
    fit = sph.PolyFit.build(particles,domain,sph.KernelFunctions.Wendland2,2,adjacency=adjacency)
    direction = particles.positions.cos()*.01
    with forward_ad.dual_level():
        dual = replace(particles,positions=forward_ad.make_dual(particles.positions,direction))
        dual_fit = sph.PolyFit.build(dual,domain,sph.KernelFunctions.Wendland2,2,adjacency=adjacency)
        primal,tangent = forward_ad.unpack_dual(dual_fit.gradient(field))
    moved = replace(particles,positions=particles.positions.clone().requires_grad_())
    moving_fit = sph.PolyFit.build(moved,domain,sph.KernelFunctions.Wendland2,2,adjacency=adjacency)
    output = moving_fit.gradient(field)
    cotangent = output.sin()
    pullback, = torch.autograd.grad((output*cotangent).sum(),moved.positions)
    torch.testing.assert_close((tangent*cotangent).sum(),(pullback*direction).sum(),rtol=1e-7,atol=1e-8)
    # WENO builds its own search and keeps its stencil and fallback policies.
    reconstructor = sph.Reconstructor.weno(particles,domain,sph.KernelFunctions.Wendland2)
    i,j,displacement = sph.interfacePairs(particles,domain)
    left,right = reconstructor.interfaceStates(field,i,j,displacement)
    assert torch.isfinite(left).all() and torch.isfinite(right).all()
