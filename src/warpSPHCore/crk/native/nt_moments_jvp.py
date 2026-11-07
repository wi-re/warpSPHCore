"""Compiled raw moment products and their first reverse on a fixed CSR."""
import math
import torch
from torch.autograd import forward_ad
from torch.autograd.function import once_differentiable
from ...native._inputs import materialize_cotangent
from .nt_preparation import prepare


class _Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx, options, topology, *values):
        from ...native._build import build
        ctx.options, ctx.topology, ctx.token = options, topology, topology.token()
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(*values)
        return build().crk_moments_product(list(values[:17]), values[17], list(values[18:23]), None, [], *options)[0]

    @staticmethod
    @once_differentiable
    def backward(ctx, cotangent):
        from ...native._build import build
        values = ctx.saved_tensors
        if cotangent is None:
            return (None,) * (len(values) + 2)
        if ctx.topology.token() != ctx.token:
            raise RuntimeError('CSR topology changed before native CRK moment JVP reverse')
        csc = ctx.topology.transpose()
        bars = build().crk_moments_product(list(values[:17]), values[17], list(values[18:23]),
            materialize_cotangent(cotangent), [csc.edgeOffsets, csc.numNeighbors, csc.paddedJ], *ctx.options)
        mapping = {0: 0, 1: 1, 2: 2, 3: 3, 17: 4, 18: 5, 19: 6, 20: 7, 21: 8, 22: 9}
        return None, None, *(None if value is None or not ctx.needs_input_grad[k + 2] or k in (7, 8)
                            else bars[mapping[k]] if k in mapping else torch.zeros_like(value)
                            for k, value in enumerate(values))


def compute(query, reference, domain, properties, adjacency, volumes, volume_tangent,
            query_tangent, reference_tangent):
    from ...radiusSearch.chunked import ChunkedAdjacency
    if isinstance(adjacency, ChunkedAdjacency):
        raise NotImplementedError('Row-chunk operators do not support AD')
    if type(adjacency).__name__ in ('CompactHashMap', 'SearchIndex'):
        adjacency = None
    reference = query if reference is None else reference
    unused_volume_tangent = None
    if volumes is None:
        unused_volume_tangent = volume_tangent
        volumes = query.positions.new_zeros((len(reference.positions),))
        volume_tangent = None
    inputs, options, topology = prepare(query, reference, domain, properties, adjacency, volumes)
    inputs[4] = None
    # The public reference launcher freezes domain metadata before its custom
    # autograd function; neither bound directions nor their reverse enter here.
    for k in (7, 8):
        inputs[k] = forward_ad.unpack_dual(inputs[k]).primal.detach()
    directions = [getattr(state, name, None) for state, name in
                  ((query_tangent, 'positions'), (reference_tangent, 'positions'),
                   (query_tangent, 'supports'), (reference_tangent, 'supports'))]
    directions.append(volume_tangent)
    unused_support = []
    for slot in ((2,) if options[0] == 12 else (3,) if options[0] == 11 else ()):
        unused_support.append(directions[slot])
        directions[slot] = None
    ignored = (query.masses, reference.masses, query.densities, reference.densities,
               unused_volume_tangent, *unused_support,
               *(getattr(state, name, None) for state in (query_tangent, reference_tangent)
                 for name in ('masses', 'densities')))
    values = [*inputs, volumes, *directions, *ignored]
    if any(x is not None and forward_ad.unpack_dual(x).tangent is not None for x in values):
        raise NotImplementedError('The explicit reference moment product has no second forward registration')
    values = [x.contiguous() if x is not None else None for x in values]
    product = _Product.apply(options, topology, *values)
    n, d = len(query.positions), domain.dim
    shapes = ((n,), (n, d), (n, d, d), (n, d), (n, d, d), (n, d, d, d))
    result, start = [], 0
    for shape in shapes:
        width = math.prod(shape[1:])
        result.append(product[:, start:start + width].reshape(shape).contiguous())
        start += width
    return tuple(result)
