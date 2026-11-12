"""Source-typed retained search metadata, independent neighbours and replay."""
from contextlib import nullcontext
from types import SimpleNamespace
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
import warpSPHCore as sph
from warpSPHCore.radiusSearch.native.nt_index import SearchIndex


class NoDouble(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        def check(values):
            leaves, _ = tree_flatten(values)
            assert not any(isinstance(v, torch.Tensor) and v.dtype == torch.float64 for v in leaves), func
        check((args, kwargs or {}))
        result = func(*args, **(kwargs or {}))
        check(result)
        return result


def fixture(device, dtype, dimension, periodic):
    generator = torch.Generator().manual_seed(104 + dimension)
    q = torch.rand((19, dimension), generator=generator, dtype=dtype).to(device)
    r = torch.rand((31, dimension), generator=generator, dtype=dtype).to(device)
    h = q.new_full((len(q),), .23)
    rh = r.new_full((len(r),), .29)
    domain = SimpleNamespace(min=q.new_zeros(dimension), max=q.new_ones(dimension),
                             periodic=torch.full((dimension,), periodic, device=device, dtype=torch.bool))
    return q, r, h, rh, domain


def collect(index, q, h):
    counts = torch.empty(len(q), device=q.device, dtype=torch.int32)
    index.apply(q, h, counts)
    offsets = counts.to(torch.int64).cumsum(0) - counts
    neighbors = torch.empty(int(counts.sum()), device=q.device, dtype=torch.int64)
    index.apply(q, h, counts, offsets, neighbors)
    return counts, offsets, neighbors


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('dimension,periodic', [(1, False), (2, True), (3, False)])
def test_source_precision_and_independent_membership(device, dtype, dimension, periodic):
    with NoDouble() if dtype == torch.float32 else nullcontext():
        q, r, h, rh, domain = fixture(device, dtype, dimension, periodic)
        for mode in [sph.SupportScheme.Gather, sph.SupportScheme.Scatter, sph.SupportScheme.MeanSymmetric,
                     sph.SupportScheme.KernelMeanSymmetric, sph.SupportScheme.SuperSymmetric]:
            index = SearchIndex(q, r, h, rh, domain, mode)
            assert [v.dtype for v in index.metadata] == [torch.int64, dtype]
            assert index.nbytes == sum(v.numel() * v.element_size() for v in [*index.metadata, *index.arrays])
            counts, offsets, neighbors = collect(index, q, h)
            delta = q[:, None] - r[None]
            if periodic:
                delta = torch.remainder(delta + .5, 1.) - .5
            radius = h[:, None] if mode == sph.SupportScheme.Gather else rh[None]
            if mode == sph.SupportScheme.MeanSymmetric:
                radius = (h[:, None] + rh[None]) / 2
            elif mode in (sph.SupportScheme.KernelMeanSymmetric, sph.SupportScheme.SuperSymmetric):
                radius = torch.maximum(h[:, None], rh[None])
            membership = delta.square().sum(-1).sqrt() < radius
            torch.testing.assert_close(counts, membership.sum(1).to(torch.int32), rtol=0, atol=0)
            torch.testing.assert_close(neighbors, torch.where(membership)[1], rtol=0, atol=0)
            # Changed queries reuse the same source-typed descriptor.
            rebuilt = SearchIndex(q + .01, r, h, rh, domain, mode)
            for actual, expected in zip(collect(index, q + .01, h), collect(rebuilt, q + .01, h)):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_descriptor_version_guard(device):
    q, r, h, rh, domain = fixture(device, torch.float32, 2, False)
    index = SearchIndex(q, r, h, rh, domain, sph.SupportScheme.Gather)
    index.metadata[0][0] = 1
    with pytest.raises(RuntimeError, match='Index geometry policy mismatch'):
        collect(index, q, h)


def test_changed_query_replay(device):
    if torch.device(device).type != 'cuda':
        pytest.skip('CUDA replay')
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        q, r, h, rh, domain = fixture('cuda', torch.float32, 2, False)
        index = SearchIndex(q, r, h, rh, domain, sph.SupportScheme.Gather)
        counts = torch.empty(len(q), device=q.device, dtype=torch.int32)
        index.apply(q, h, counts)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        index.apply(q, h, counts)
    q.add_(.017)
    graph.replay()
    eager, _, _ = collect(index, q, h)
    torch.testing.assert_close(counts, eager, rtol=0, atol=0)
