"""Spatial query scheduling preserves original CSR rows and particle identity."""
import pytest
import torch
import warp as wp

from warpSPHCore import CapacityPolicy, SupportScheme, buildVerletList
from warpSPHCore.radiusSearch.compactHash import buildCompactHashMap
from warpSPHCore.radiusSearch.compactHash.search import radiusSearchOnCompactHashMap
from warpSPHCore.radiusSearch.options import resolveCollection, resolveQueryOrder
from warpSPHCore.util import generateNeighborTestData


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    wp.init()
    return torch.device(request.param)


def equal(actual, expected):
    assert actual.numEdges == expected.numEdges
    for name in ('paddedI', 'paddedJ', 'numNeighbors', 'edgeOffsets'):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('mode', list(SupportScheme))
def test_spatial_scheduling_preserves_every_array(device, dim, periodic, mode, monkeypatch):
    p, h, n, domain, dx = generateNeighborTestData(5, 15, dim, periodic, device)
    order = torch.randperm(n, generator=torch.Generator().manual_seed(51)).to(device)
    p, h = p[order].contiguous(), h[order].contiguous()
    if periodic:
        p[::3] += 4  # equivalent out-of-domain positions must wrap identically
    query_h = h * torch.linspace(.65, 1., n, device=device)
    grid = buildCompactHashMap(p, p, query_h, h, domain.periodic, domain, mode, n + 1)
    before = p.clone()

    def search(**kwargs):
        return radiusSearchOnCompactHashMap(grid, p, p, query_h, h, domain.periodic, domain, mode, n + 1, **kwargs)

    expected = search(collection='fused', queryOrder='input')
    monkeypatch.setattr(torch, 'argsort', lambda *a, **k: pytest.fail('query scheduling must reuse the existing permutation'))
    for collection in ('fused', 'split', 'auto'):
        for queryOrder in ('spatial', 'auto'):
            got = search(collection=collection, queryOrder=queryOrder)
            equal(got, expected)
            pointers = [getattr(got, f).data_ptr() for f in ('paddedI', 'paddedJ', 'numNeighbors', 'edgeOffsets')]
            reused = search(collection=collection, queryOrder=queryOrder,
                            capacity=CapacityPolicy(inPlace=True), reuse=got)
            equal(reused, expected)
            assert pointers == [getattr(reused, f).data_ptr() for f in ('paddedI', 'paddedJ', 'numNeighbors', 'edgeOffsets')]
    torch.testing.assert_close(p, before, rtol=0, atol=0)
    assert resolveQueryOrder('auto', p.view_as(p), p) == 'spatial'


def test_asymmetric_search_does_not_assume_same_sized_sets_are_identical(device):
    p, h, n, domain, _ = generateNeighborTestData(5, 15, 2, False, device)
    query = p.flip(0).contiguous().clone()
    query[::3] += 10
    grid = buildCompactHashMap(query, p, h, h, domain.periodic, domain, SupportScheme.Gather, n + 1)
    def search(**kwargs):
        return radiusSearchOnCompactHashMap(grid, query, p, h, h, domain.periodic, domain, **kwargs)
    equal(search(), search(collection='fused', queryOrder='input'))
    assert resolveQueryOrder('auto', query, p) == 'input'
    with pytest.raises(ValueError, match='share the same layout'):
        search(queryOrder='spatial')
    with pytest.raises(ValueError, match='queryOrder'):
        search(queryOrder='unknown')


def test_default_and_explicit_kernel_selection(device, monkeypatch):
    import importlib
    module = importlib.import_module('warpSPHCore.radiusSearch.compactHash.search')
    p, h, n, domain, _ = generateNeighborTestData(5, 15, 2, True, device)
    grid = buildCompactHashMap(p, p, h, h, domain.periodic, domain, SupportScheme.Gather, n + 1)
    seen = []
    original = module.wp.launch
    def launch(kernel, *args, **kwargs):
        seen.append(kernel)
        return original(kernel, *args, **kwargs)
    monkeypatch.setattr(module.wp, 'launch', launch)
    radiusSearchOnCompactHashMap(grid, p, p, h, h, domain.periodic, domain)
    assert module.radiusSearchCountNeighborsSpatial in seen
    assert resolveCollection('auto', device) == ('split' if device.type == 'cuda' else 'fused')
    assert (module.fillCSRRows in seen) == (device.type == 'cuda')
    assert (module.radiusSearchCollectJOnlySpatial if device.type == 'cuda' else module.radiusSearchCollectSpatial) in seen
