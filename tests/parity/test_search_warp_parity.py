"""Compare independently built native and Warp compact-hash neighborhoods."""
from dataclasses import replace

import pytest
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.radiusSearch.native.nt_radius import radiusSearch as native_search
from warpSPHCore.radiusSearch.compactHash.wrapper import radiusSearchCompactHashMap
from warpSPHCore.enumTypes import HashMapLengthMode

GEOMETRIES = [(dimension, mask) for dimension in (1, 2, 3) for mask in range(1 << dimension)]


@pytest.mark.parametrize('dimension,mask', GEOMETRIES)
@pytest.mark.parametrize('support', list(sph.SupportScheme)[:-1])
@pytest.mark.parametrize('radius', [.17, .95, 3.5])
def test_independent_compact_hashes_agree(device, dimension, mask, support, radius):
    wp.init()
    dtype = sph.get_torch_precision()
    generator = torch.Generator().manual_seed(197 + dimension + mask)

    def particles(count):
        positions = torch.rand((count, dimension), generator=generator, dtype=dtype).to(device) * 2.6 - 1.3
        supports = (torch.rand(count, generator=generator, dtype=dtype).to(device) * .7 + .3) * radius
        return sph.ParticleState(positions, supports, torch.ones(count, device=device, dtype=dtype),
                                 torch.zeros(count, device=device, dtype=torch.int32))

    query, reference = particles(9), particles(13)
    periodic = torch.tensor([bool(mask & (1 << axis)) for axis in range(dimension)], device=device)
    domain = sph.DomainDescription(query.positions.new_full((dimension,), -1),
                                   query.positions.new_full((dimension,), 1), periodic, dimension)
    # Several box lengths away: both builders must use the nearest image.
    query = replace(query, positions=query.positions + periodic * 8)
    reference = replace(reference, positions=reference.positions - periodic * 6)
    actual = native_search(query, domain, reference, support)
    expected = radiusSearchCompactHashMap(
        query, domain, support, HashMapLengthMode.Fixed, 1,
        referenceParticles=reference, collection='fused', queryOrder='input',
    )

    def canonical(adjacency):
        # Decode CSR directly: this process's Warp COO helper expects its own
        # configured integer widths, while native CSR always emits int64 offsets.
        rows = torch.repeat_interleave(
            torch.arange(adjacency.numRows, device=device, dtype=torch.int64),
            adjacency.numNeighbors.long(), output_size=adjacency.numEdges,
        )
        columns = adjacency.paddedJ[:adjacency.numEdges].long()
        return torch.sort(rows * len(reference.positions) + columns).values

    torch.testing.assert_close(canonical(actual), canonical(expected), rtol=0, atol=0)
    torch.testing.assert_close(actual.numNeighbors, expected.numNeighbors, rtol=0, atol=0)
