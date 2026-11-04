import pytest
import torch
import sys
import warpSPHCore as sph

pytestmark = pytest.mark.cpu

def test_module_local_native_wrappers_without_warp():
    if sph.backend != 'native': pytest.skip('native process only')
    from warpSPHCore.coreOperations.native import computeDensity, computeGradient
    from warpSPHCore.radiusSearch.native import radiusSearch
    from warpSPHCore._backend import resolve_backend
    x = torch.tensor([[-.2], [.2]])
    p = sph.ParticleState(x, torch.ones(2), torch.ones(2), torch.zeros(2, dtype=torch.int32))
    d = sph.DomainDescription(torch.tensor([-1.]), torch.tensor([1.]), torch.tensor([True]), 1)
    a = radiusSearch(p, d)
    prop = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=sph.WarpOperation.Density)
    p.densities = computeDensity(p, p, d, prop, a)
    torch.testing.assert_close(p.densities, sph.warpOperation(p, prop, d, adjacency=a), rtol=0, atol=0)
    grad_prop = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=sph.WarpOperation.Gradient,
                                         gradientMode=sph.GradientScheme.Difference)
    value = torch.ones(2)
    torch.testing.assert_close(computeGradient(p, p, d, grad_prop, a, value, value), torch.zeros_like(x), rtol=0, atol=0)
    with pytest.raises(ValueError, match='before importing'):
        resolve_backend('warp')
    assert 'warp' not in sys.modules


def test_cpu_only_radius_rejects_device_pointers_before_index_build():
    """CPU-only extensions must fail before borrowing CUDA memory for a CPU index."""
    import os
    import pytest
    if os.environ.get('WARPSPHCORE_NATIVE_BUILD')!='cpu' or not torch.cuda.is_available():
        pytest.skip('Requires CPU-only extension and CUDA-visible host')
    from warpSPHCore.radiusSearch.native import radiusSearch
    from warpSPHCore import ParticleState,DomainDescription
    x=torch.tensor([[.1],[.4]],device='cuda');h=torch.full((2,),.5,device='cuda')
    particles=ParticleState(x,h,torch.ones_like(h),torch.zeros(2,device='cuda',dtype=torch.int32),torch.ones_like(h))
    domain=DomainDescription(h.new_zeros(1),h.new_ones(1),torch.ones(1,device='cuda',dtype=torch.bool),1)
    with pytest.raises(RuntimeError,match='built for CPU only'):
        radiusSearch(particles,domain)
