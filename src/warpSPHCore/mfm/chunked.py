"""Owned face fields for the native meshless closure solver.

Particle vectors remain replicated. Pair weights never acquire a global view;
only per-particle matvec and stencil results are exchanged by row dispatch.
"""
import torch

from ..autograd.operator_spec import launchOperator
from ..autograd.row_dispatch import chunkContexts, validateContext
from ..radiusSearch.pairField import ChunkedPairField
from .._backend import BACKEND
if BACKEND == 'native':
    from .native.nt_mfm import _CLOSURE_W_SPEC,_ctx
else:
    from .wp_mfm import _CLOSURE_W_SPEC,_ctx


def _local(adjacency, tag, function):
    comm = adjacency.communication
    return function() if comm is None else comm.localPhase(tag, function)


def closureWeights(geometry, **extras):
    adjacency = geometry.adjacency
    ctx = _ctx(geometry.particles, geometry.props, geometry.domain, adjacency)
    def validate():
        validateContext(ctx)
        ChunkedPairField._checkLayout(adjacency, (), geometry.volume.dtype)
    _local(adjacency, 'meshless:weights:validate', validate)
    comm = adjacency.communication
    if comm is not None:
        comm.agree('meshless:weights:options', {k: str(v) for k, v in extras.items()
                   if not isinstance(v, torch.Tensor)})
        comm.assertReplicated('meshless:weights:inputs', dict(
            positions=geometry.particles.positions, supports=geometry.particles.supports,
            **{k: v for k, v in extras.items() if isinstance(v, torch.Tensor)}))
    def compute():
        result = ChunkedPairField.empty(adjacency, trailingShape=(), dtype=geometry.volume.dtype)
        for buffer, (interval, local) in zip(result.buffers, chunkContexts(ctx)):
            values = dict(extras, rowStart=interval.start)
            buffer.zero_()
            weights = launchOperator(_CLOSURE_W_SPEC, local, **values)
            live = int(local.adjacency.numEdges)
            buffer[:live].copy_(weights[:live])
        return result
    return _local(adjacency, 'meshless:weights:compute', compute)


def capScale(adjacency, lam, cap):
    def compute():
        result = ChunkedPairField.empty(adjacency, trailingShape=(), dtype=lam.dtype)
        over = lam.new_zeros(())
        total = 0
        for chunk, buffer in zip(adjacency, result.buffers):
            m, csr = chunk.metadata, chunk.adjacency
            rows = torch.repeat_interleave(torch.arange(m.rowStart, m.rowStop, device=lam.device),
                                           csr.numNeighbors.long())
            mag = (lam[csr.paddedJ[:m.liveEdges].long()] - lam[rows]).norm(dim=-1)
            buffer.zero_()
            buffer[:m.liveEdges] = torch.clamp(cap / mag.clamp_min(1e-30), max=1.0)
            over += (mag > cap).to(lam.dtype).sum()
            total += m.liveEdges
        return result, over, total
    result, over, total = _local(adjacency, 'meshless:cap:compute', compute)
    comm = adjacency.communication
    if comm is not None:
        over = comm.reduceScalar(over, operation='sum', tag='meshless:cap:over')
        total = comm.reduceScalar(lam.new_tensor(total), operation='sum', tag='meshless:cap:total')
    return result, over / max(float(total), 1)


def multiply(left, right):
    right.validate(left.adjacency)
    def compute():
        # The freshly computed weights are private to this solve. Scale them
        # in place, without a third resident pair field or cross-chunk alias.
        for a, b in zip(left.buffers, right.buffers):
            a.mul_(b)
        return left
    return _local(left.adjacency, 'meshless:weights:scale', compute)
