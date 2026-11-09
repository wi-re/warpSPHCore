"""Native search preparation preserves one-ulp in-support periodic pairs."""
import pytest
import torch
import warpSPHCore as sph


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_periodic_cutoff_membership_preserves_stored_coordinates(device,dtype):
    a=torch.tensor(.25,dtype=dtype,device=device)
    x=torch.nextafter(a,torch.full_like(a,float('inf'))).reshape(1,1)
    y=x.new_tensor([[.375]])
    h=torch.nextafter(x.new_tensor([.125]),x.new_zeros(1))
    def particles(positions):
        return sph.ParticleState(positions,h,torch.ones_like(h),torch.zeros(1,dtype=torch.int32,device=device),torch.ones_like(h))
    domain=sph.DomainDescription(x.new_tensor([-.5]),x.new_tensor([.5]),torch.ones(1,dtype=torch.bool,device=device),1)
    adjacency=sph.buildVerletList(particles(x),domain,referenceParticles=particles(y),supportMode=sph.SupportScheme.Gather,backend='native')
    i,j=adjacency.livePairs()
    assert i.tolist()==[0] and j.tolist()==[0]
