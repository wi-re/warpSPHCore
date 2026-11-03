"""Lifecycle contracts and independent geometry oracles; no benchmark fixtures."""
from dataclasses import replace
import pytest
import torch
import warpSPHCore as sph


def particles(x, h):
    return sph.ParticleState(x.contiguous(), h.contiguous(), torch.ones_like(h),
                             torch.zeros(len(x), dtype=torch.int32, device=x.device), torch.ones_like(h))


def displacement(a, b, domain):
    delta = a - b
    length = domain.max - domain.min
    return torch.where(domain.periodic, torch.remainder(delta + length/2, length)-length/2, delta)


def oracle_metrics(q, r, old_q, old_r, hq, hr, old_hq, old_hr, domain, scale, case, slack):
    motion = displacement(q, old_q, domain).norm(dim=-1).amax() + displacement(r, old_r, domain).norm(dim=-1).amax()
    drift_q, drift_r = (old_hq/scale-hq).abs().amax(), (old_hr/scale-hr).abs().amax()
    min_q, min_r = torch.minimum(old_hq/scale, hq).amin(), torch.minimum(old_hr/scale, hr).amin()
    drift = drift_q if case == 0 else drift_r if case == 1 else torch.maximum(drift_q, drift_r)
    minimum = min_q if case == 0 else min_r if case == 1 else torch.minimum(min_q, min_r)
    buffer = (scale-1)*minimum
    return (motion+drift > buffer*(1-slack)), motion, drift, minimum, buffer


def test_validity_reductions_and_device_scale(device):
    # Rotate dtype/dimension rather than multiplying a full Cartesian matrix.
    for dim, dtype in ((1, torch.float64), (2, torch.float32), (3, torch.float64)):
        gen = torch.Generator().manual_seed(318+dim)
        def random(*size): return torch.rand(size, generator=gen, dtype=dtype).to(device)
        old_q, old_r = random(513, dim), random(311, dim)
        q, r = old_q + .04*random(513, dim) + 4, old_r - .03*random(311, dim) - 2
        hq, hr = .2+random(513), .4+random(311)
        old_hq, old_hr = (hq+.01)*1.4, (hr-.02)*1.4
        domain = sph.DomainDescription(q.new_zeros(dim), q.new_ones(dim),
                                        torch.ones(dim, dtype=torch.bool, device=device), dim)
        for case in (0, 1, 2):
            args = (q,r,old_q,old_r,hq,hr,old_hq,old_hr,domain.periodic,domain.min,domain.max)
            actual = sph.verletValidityMetrics(*args, hq.new_tensor(1.4), case, .1)
            expected = oracle_metrics(q,r,old_q,old_r,hq,hr,old_hq,old_hr,domain,hq.new_tensor(1.4),case,.1)
            for got, want in zip(actual,expected):
                torch.testing.assert_close(got,want,rtol=2e-5,atol=2e-6)
        same = sph.verletValidityMetrics(q,q,old_q,old_q,hq,hq,old_hq,old_hq,
            domain.periodic,domain.min,domain.max,1.4,2)
        expected = oracle_metrics(q,q,old_q,old_q,hq,hq,old_hq,old_hq,domain,1.4,2,0.)
        for got,want in zip(same,expected): torch.testing.assert_close(got,want,rtol=2e-5,atol=2e-6)


def test_verlet_reuse_filter_capacity_and_snapshots(device):
    x = torch.tensor([[.02],[.23],[.46],[.72],[.96]],device=device,dtype=torch.float64)
    q = particles(x,x.new_full((5,),.3))
    domain = sph.DomainDescription(x.new_zeros(1),x.new_ones(1),torch.ones(1,device=device,dtype=torch.bool),1)
    policy = sph.CapacityPolicy(headroom=1.,shrinkBelow=0.,inPlace=True)
    prior = sph.buildVerletList(q,domain,1.5,capacity=policy,storeI=False)
    moved = replace(q,positions=q.positions+.01)
    assert sph.buildVerletList(moved,domain,1.5,priorNeighborhood=prior,capacity=policy,storeI=False) is prior
    assert not torch.equal(prior.queryPositions,moved.positions)
    assert not bool(sph.verletStaleFlag(moved.positions,moved.supports,prior,domain,1.5,sph.SupportScheme.Gather))
    with sph.deferVerletChecks() as outer:
        with sph.deferVerletChecks() as inner:
            stale = replace(q,positions=q.positions*1.1)
            assert sph.buildVerletList(stale,domain,1.5,priorNeighborhood=prior,storeI=False) is prior
        assert len(inner)==1 and bool(inner[0]) and not outer
    before = (prior.paddedJ.data_ptr(),prior.edgeOffsets.data_ptr(),prior.numNeighbors.data_ptr())
    rebuilt = sph.buildVerletList(moved,domain,1.5,priorNeighborhood=prior,capacity=policy,forceRebuild=True,storeI=False)
    assert (rebuilt.paddedJ.data_ptr(),rebuilt.edgeOffsets.data_ptr(),rebuilt.numNeighbors.data_ptr()) == before
    filtered = sph.filterVerletList(moved,domain,rebuilt,capacity=None,storeI=True)
    distance = displacement(moved.positions[:,None],moved.positions[None,:],domain).norm(dim=-1)
    expected = torch.nonzero(distance<=.3)
    torch.testing.assert_close(filtered.livePairs()[0],expected[:,0])
    torch.testing.assert_close(filtered.livePairs()[1],expected[:,1])
    assert torch.all(rebuilt.paddedJ[rebuilt.numEdges:]==-1)
    other_mode = sph.buildVerletList(moved,domain,1.5,sph.SupportScheme.Scatter,priorNeighborhood=rebuilt,storeI=False)
    assert other_mode is not rebuilt
    changed_domain = replace(domain, max=domain.max.clone())
    assert bool(sph.verletStaleFlag(moved.positions,moved.supports,rebuilt,changed_domain,1.5,sph.SupportScheme.Gather))
    # Mutating a caller-owned state must not mutate the build snapshot.
    snapshot = rebuilt.queryPositions.clone()
    moved.positions.add_(.001)
    torch.testing.assert_close(rebuilt.queryPositions,snapshot)
    changed = replace(q,positions=q.positions[:3],supports=q.supports[:3],masses=q.masses[:3],kinds=q.kinds[:3],densities=q.densities[:3])
    assert sph.buildVerletList(changed,domain,1.5,priorNeighborhood=rebuilt).numRows==3


def test_distinct_population_filter_supports_and_exact_cutoff(device):
    for dim, dtype in ((1,torch.float32),(2,torch.float64),(3,torch.float32)):
        x = torch.zeros((3,dim),device=device,dtype=dtype)
        x[:,0] = torch.tensor([0.,.25,.5],device=device,dtype=dtype)
        y = x.repeat(2,1).contiguous()
        q = particles(x,x.new_tensor([.25,.5,.75]))
        r = particles(y,x.new_tensor([.5,.25,.5,.75,.25,.5]))
        domain = sph.DomainDescription(x.new_full((dim,),-1),x.new_full((dim,),1),
                                        torch.zeros(dim,device=device,dtype=torch.bool),dim)
        candidates = sph.buildVerletList(q,domain,2.,sph.SupportScheme.SuperSymmetric,referenceParticles=r)
        # Filtering accepts CSR32 storage and offsets independently of output width.
        candidates = replace(candidates,paddedJ=candidates.paddedJ.int(),edgeOffsets=candidates.edgeOffsets.int())
        for mode in (sph.SupportScheme.Gather,sph.SupportScheme.Scatter,sph.SupportScheme.MeanSymmetric,
                     sph.SupportScheme.KernelMeanSymmetric,sph.SupportScheme.SuperSymmetric):
            actual = sph.filterVerletList(q,domain,candidates,mode,referenceParticles=r,capacity=None)
            distance = (x[:,None]-y[None,:]).norm(dim=-1)
            hi,hj = q.supports[:,None],r.supports[None,:]
            reach = hi if mode==sph.SupportScheme.Gather else hj if mode==sph.SupportScheme.Scatter else (hi+hj)/2 if mode==sph.SupportScheme.MeanSymmetric else torch.maximum(hi,hj)
            expected = torch.nonzero(distance<=reach)
            torch.testing.assert_close(actual.livePairs()[0],expected[:,0])
            torch.testing.assert_close(actual.livePairs()[1],expected[:,1])


def test_empty_validity_and_deferred_cuda_stream_contract(device):
    x = torch.empty((0,2),device=device,dtype=torch.float32)
    q = particles(x,x.new_empty(0))
    domain = sph.DomainDescription(x.new_zeros(2),x.new_ones(2),torch.ones(2,device=device,dtype=torch.bool),2)
    adjacency = sph.buildVerletList(q,domain,1.4,storeI=False)
    assert adjacency.numEdges==0
    assert not bool(sph.verletStaleFlag(x,q.supports,adjacency,domain,1.4,sph.SupportScheme.Gather))
    assert sph.filterVerletList(q,domain,adjacency,storeI=False).numEdges==0
    if device=='cpu': return
    nonempty = particles(x.new_tensor([[.1,.2],[.3,.4]]),x.new_full((2,),.3))
    adjacency = sph.buildVerletList(nonempty,domain,1.4,storeI=False)
    scale = x.new_tensor(1.4)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda.set_sync_debug_mode('error')
        try:
            flag = sph.verletStaleFlag(nonempty.positions,nonempty.supports,adjacency,domain,scale,sph.SupportScheme.Gather)
        finally:
            torch.cuda.set_sync_debug_mode('default')
    stream.synchronize()
    assert not bool(flag)
