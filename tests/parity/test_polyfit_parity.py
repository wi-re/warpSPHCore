"""Native polynomial assembly against the module's readable Warp equations."""
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.type_config import get_torch_precision
from warpSPHCore.polyfit import monomialExponents
from warpSPHCore.polyfit.wp_polyfit import _computePolyMoments_stateBackend as warp_moments, _computePolyRHS_stateBackend as warp_rhs
from warpSPHCore.polyfit.native import _computePolyMoments_stateBackend as native_moments, _computePolyRHS_stateBackend as native_rhs


def test_polynomial_moments_rhs_and_pullback_parity(device):
    wp.init()
    for dimension in (1,2,3):
        dtype = get_torch_precision()
        x = torch.arange(13*dimension,device=device,dtype=dtype).reshape(13,dimension).sin()*.3
        h = torch.linspace(.7,1.1,len(x),device=device,dtype=dtype)
        particles = sph.ParticleState(x,h,h*.7,(torch.arange(len(x),device=device)%3).int(),h+1)
        domain = sph.DomainDescription(x.new_full((dimension,),-1),x.new_full((dimension,),1),
                                       torch.zeros(dimension,device=device,dtype=torch.bool),dimension)
        adjacency = sph.radiusSearchCompactHashMap(particles,domain)
        exponents = torch.tensor(monomialExponents(dimension,3),device=device,dtype=torch.int32)
        properties = sph.OperationProperties(sph.KernelFunctions.Wendland4,operationMode=sph.OperationDirection.FluidToFluid)
        field = x.cos()
        for unit in (False,True):
            options = dict(adjacency=adjacency,sector=0 if dimension<3 else -1,
                           numSectors=4,unitWeights=unit)
            actual = native_moments(particles,properties,domain,exponents,**options)
            expected = warp_moments(particles,properties,domain,exponents,**options)
            tolerance = 2e-5 if dtype==torch.float32 else 2e-11
            for left,right in zip(actual,expected):
                torch.testing.assert_close(left,right,rtol=tolerance,atol=tolerance)
            torch.testing.assert_close(native_rhs(particles,properties,domain,exponents,field,True,**options),
                                       warp_rhs(particles,properties,domain,exponents,field,True,**options),rtol=tolerance,atol=tolerance)
        # VJP parity covers both repeated source reads and the query-side
        # subtraction with a field shared by the two directed roles.
        left=field.clone().requires_grad_();right=field.clone().requires_grad_()
        left_bar, = torch.autograd.grad(native_rhs(particles,properties,domain,exponents,left,True,adjacency=adjacency).sum(),left)
        right_bar, = torch.autograd.grad(warp_rhs(particles,properties,domain,exponents,right,True,adjacency=adjacency).sum(),right)
        torch.testing.assert_close(left_bar,right_bar,rtol=tolerance,atol=tolerance)
