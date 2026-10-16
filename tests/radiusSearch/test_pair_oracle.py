"""Independent all-pairs oracle, including unwrapped periodic particle IDs."""

import pytest
import torch
import warp as wp

from warpSPHCore import DomainDescription, SupportScheme
from warpSPHCore.radiusSearch.compactHash import radiusSearchCompactHashMap_
from warpSPHCore.type_config import get_torch_precision


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    wp.init()
    return request.param


@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('mode', list(SupportScheme))
def test_search_matches_independent_pairs(device, dim, periodic, mode):
    if dim == 3 and device == 'cpu':
        pytest.skip('3D qualification runs on CUDA')
    dtype = get_torch_precision()
    generator = torch.Generator().manual_seed(982)
    q = (torch.rand((7, dim), generator=generator, dtype=dtype) * 1.8 - .9).to(device)
    r = (torch.rand((13, dim), generator=generator, dtype=dtype) * 1.8 - .9).to(device)
    flags = torch.tensor([periodic and d % 2 == 0 for d in range(dim)], device=device)
    domain = DomainDescription(min=q.new_full((dim,), -1), max=q.new_full((dim,), 1),
                               periodic=flags, dim=dim)
    if not bool(flags.all()):
        q[-1, ~flags] = 9  # A genuine empty row, outside nonperiodic bounds.
    for large_support in (False, True):
        hq = torch.linspace(.22, .73, 7, dtype=dtype, device=device)
        hr = torch.linspace(.15, .81, 13, dtype=dtype, device=device).flip(0)
        if large_support:
            hq = hq * 5
            hr = hr * 5  # One-cell periodic axes must not duplicate neighbors.
        delta = q[:, None] - r[None]
        delta = torch.where(flags, (delta + 1).remainder(2) - 1, delta)
        if mode == SupportScheme.Gather:
            threshold = hq[:, None]
        elif mode == SupportScheme.Scatter:
            threshold = hr[None]
        elif mode == SupportScheme.MeanSymmetric:
            threshold = (hq[:, None] + hr[None]) * .5
        else:
            threshold = torch.maximum(hq[:, None], hr[None])
        expected = delta.norm(dim=-1) <= threshold
        for unwrapped in (False, True):
            # Different integer translations on each public particle, on both
            # roles. Row/column IDs and the physical pair set must be invariant.
            qshift = ((torch.arange(7, device=device) % 5 - 2) * 2)[:, None] * flags
            rshift = ((torch.arange(13, device=device) % 7 - 3) * 2)[:, None] * flags
            query = q + qshift if unwrapped else q
            reference = r + rshift if unwrapped else r
            for collection in ('fused', 'split'):
                adjacency = radiusSearchCompactHashMap_(
                    query, reference, hq, hr, flags, domain, mode,
                    hashMapLength=7, collection=collection)
                i, j = adjacency.livePairs()
                pairs = torch.zeros_like(expected, dtype=torch.int32)
                pairs.index_put_((i.long(), j.long()), torch.ones_like(i, dtype=torch.int32), accumulate=True)
                # Counts detect duplicate cells as well as absent/spurious IDs.
                torch.testing.assert_close(pairs, expected.to(torch.int32), rtol=0, atol=0)
                torch.testing.assert_close(adjacency.numNeighbors, expected.sum(-1).to(torch.int32))
                assert bool((adjacency.paddedI[adjacency.numEdges:] == -1).all())
                assert bool((adjacency.paddedJ[adjacency.numEdges:] == -1).all())
