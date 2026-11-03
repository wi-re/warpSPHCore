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
