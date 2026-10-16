"""Production split collection preserves exact CSR order, reuse and stream ownership."""
import pytest
import torch
import warp as wp

from warpSPHCore import CapacityPolicy, SupportScheme
from warpSPHCore.radiusSearch.compactHash import buildCompactHashMap
from warpSPHCore.radiusSearch.compactHash.search import radiusSearchOnCompactHashMap
from warpSPHCore.type_config import get_torch_index_dtype, get_torch_offset_dtype
from warpSPHCore.util import generateNeighborTestData


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    wp.init()
    return torch.device(request.param)


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('mode', list(SupportScheme))
def test_split_exact_order_reuse_and_empty_rows(device, dim, periodic, mode):
    positions, supports, n, domain, dx = generateNeighborTestData(5, 15, dim, periodic, device)
    # Asymmetric populations/supports, with some empty rows outside nonperiodic domains.
    reference_h = supports * torch.linspace(.7, 1., n, device=device)
    queries = positions[::2].clone()
    query_h = supports[::2] * .9
    queries[::7] += 10
    grid = buildCompactHashMap(queries, positions, query_h, reference_h,
                               domain.periodic, domain, mode, n + 1)

    def search(**kwargs):
        return radiusSearchOnCompactHashMap(grid, queries, positions, query_h, reference_h,
                                            domain.periodic, domain, mode, n + 1, **kwargs)

    def equal(actual, expected):
        assert actual.numEdges == expected.numEdges
        for name in ('paddedI', 'paddedJ', 'numNeighbors', 'edgeOffsets'):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)

    for capacity in (None, CapacityPolicy()):
        equal(search(collection='split', capacity=capacity), search(collection='fused',capacity=capacity))
    held = search(collection='split')
    pointers = [getattr(held, name).data_ptr() for name in ('paddedI', 'paddedJ', 'numNeighbors', 'edgeOffsets')]
    # Shrink live rows while retaining capacity, and prove the previous live tail is erased.
    queries[::3] += 20
    expected = search(collection='fused',capacity=CapacityPolicy(shrinkBelow=0), reuse=held)
    got = search(collection='split', capacity=CapacityPolicy(inPlace=True, shrinkBelow=0), reuse=held)
    equal(got, expected)
    assert pointers == [getattr(got, name).data_ptr() for name in ('paddedI', 'paddedJ', 'numNeighbors', 'edgeOffsets')]
    assert bool((got.paddedI[got.numEdges:] == -1).all())
    assert bool((got.paddedJ[got.numEdges:] == -1).all())
    with pytest.raises(ValueError, match='collection'):
        search(collection='unknown')


def test_split_wrapper_forwarding(device):
    from warpSPHCore import ParticleState
    from warpSPHCore.radiusSearch.compactHash import radiusSearchCompactHashMap, radiusSearchCompactHashMap_
    p, h, n, domain, _ = generateNeighborTestData(5, 15, 2, True, device)
    raw = radiusSearchCompactHashMap_(p, p, h, h, domain.periodic, domain, collection='split')
    state = ParticleState(positions=p, supports=h, masses=torch.ones_like(h), kinds=torch.zeros(n,dtype=torch.int32,device=device))
    wrapped = radiusSearchCompactHashMap(state, domain, collection='split')
    torch.testing.assert_close(raw.paddedI, wrapped.paddedI, rtol=0, atol=0)
    torch.testing.assert_close(raw.paddedJ, wrapped.paddedJ, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA stream/replay ownership')
def test_production_row_fill_replay_on_nondefault_stream():
    from warpSPHCore.autograd.stream import torchWarpCapture, torchWarpStream
    from warpSPHCore.radiusSearch.compactHash.wp_collectNeighbors import fillCSRRows
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        counts = torch.tensor([1, 0, 3, 2], dtype=torch.int32, device='cuda')
        offsets = torch.tensor([0, 1, 1, 4], dtype=get_torch_offset_dtype(), device='cuda')
        rows = torch.full((32,), -1, dtype=get_torch_index_dtype(), device='cuda')
        warp_stream = wp.stream_from_torch(stream)

        def fill():
            rows.fill_(-1)
            wp.launch(fillCSRRows, dim=(4, 32), inputs=[wp.from_torch(counts), wp.from_torch(offsets), wp.from_torch(rows)],
                      block_dim=128, device='cuda:0', stream=warp_stream)

        fill()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torchWarpStream(counts.device), torchWarpCapture(graph):
            fill()
        for values in ([0, 17, 0, 1], [2, 1, 3, 0], [0, 0, 0, 0]):
            counts.copy_(torch.tensor(values, dtype=torch.int32, device='cuda'))
            offsets[1:] = counts.cumsum(0)[:-1].to(offsets.dtype)
            graph.replay()
            result = rows.clone()
            stream.synchronize()
            expected = torch.repeat_interleave(torch.arange(4, device='cuda', dtype=rows.dtype), counts.long())
            torch.testing.assert_close(result[:len(expected)], expected, rtol=0, atol=0)
            assert bool((result[len(expected):] == -1).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA stream ownership')
def test_split_search_follows_nondefault_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        p, h, n, domain, _ = generateNeighborTestData(5, 15, 2, False, 'cuda')
        grid = buildCompactHashMap(p, p, h, h, domain.periodic, domain, SupportScheme.Gather, n + 1)
        queries = p.clone()
        torch.cuda._sleep(2_000_000)
        queries[::2] += 10
        expected = radiusSearchOnCompactHashMap(grid, queries, p, h, h, domain.periodic, domain)
        actual = radiusSearchOnCompactHashMap(grid, queries, p, h, h, domain.periodic, domain, collection='split')
        i, j = actual.paddedI.clone(), actual.paddedJ.clone()
    stream.synchronize()
    torch.testing.assert_close(i, expected.paddedI, rtol=0, atol=0)
    torch.testing.assert_close(j, expected.paddedJ, rtol=0, atol=0)
