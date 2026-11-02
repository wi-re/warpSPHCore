"""Explicit fixed-topology ownership for repeated native linear products."""
import torch
from .._models import AdjacencyList


def tensor_token(tensor):
    if tensor is None:
        return None
    try:
        version = tensor._version
    except RuntimeError as error:
        raise ValueError('Cached native linearizations require versioned tensors; clone outside inference_mode') from error
    return id(tensor), version, tuple(tensor.shape), tensor.dtype, tensor.device


class CSRTopology:
    """Own one CSR lifetime and lazily cache its sorted transpose.

    The input arrays remain borrowed. Tensor replacement/in-place updates clear
    the cache; existing linearizations reject changed topology. Integer atomics
    are used only while constructing CSC; derivative reductions use no floating
    atomics. Unused CSR prefixes, gaps, duplicate edges and tails are preserved
    by their actual offsets/counts rather than reconstructed COO assumptions.
    """
    def __init__(self, adjacency):
        if not isinstance(adjacency, AdjacencyList):
            raise TypeError('CSRTopology requires a Torch AdjacencyList')
        self.adjacency = adjacency
        self._cached_token = None
        self._transpose = None
        self._ready = None
        self._transpose_token = None

    def token(self):
        a = self.adjacency
        return (a.numRows, a.numCols, tensor_token(a.edgeOffsets),
                tensor_token(a.numNeighbors), tensor_token(a.paddedJ))

    def transpose(self):
        token = self.token()
        current = None if self._transpose is None else tuple(tensor_token(t) for t in
            (self._transpose.edgeOffsets, self._transpose.numNeighbors, self._transpose.paddedJ))
        if token != self._cached_token or current != self._transpose_token:
            from ._build import build
            a = self.adjacency
            counts = torch.zeros(a.numCols, device=a.paddedJ.device, dtype=torch.int32)
            with torch.no_grad():
                build().csr_transpose_count(a.edgeOffsets, a.numNeighbors, a.paddedJ, counts)
                offsets = counts.long().cumsum(0) - counts
                ids = torch.full((a.paddedJ.numel(),), -1, device=counts.device, dtype=torch.int64)
                build().csr_transpose_fill(a.edgeOffsets, a.numNeighbors, a.paddedJ,
                                           offsets, counts, torch.zeros_like(counts), ids)
            self._transpose = AdjacencyList(None, ids, counts, offsets, a.numCols, a.numRows)
            self._cached_token = token
            self._transpose_token = tuple(tensor_token(t) for t in (offsets, counts, ids))
            self._ready = None
            if counts.is_cuda:
                self._ready = torch.cuda.Event()
                self._ready.record(torch.cuda.current_stream(counts.device))
        result = self._transpose
        if result.paddedJ.is_cuda:
            stream = torch.cuda.current_stream(result.paddedJ.device)
            stream.wait_event(self._ready)
            for tensor in (result.edgeOffsets, result.numNeighbors, result.paddedJ):
                tensor.record_stream(stream)
        return result
