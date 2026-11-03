"""Owned graph-Laplacian storage for repeated meshless closure applications."""
import torch
from ...native._checks import check_execution


class mfmClosureMatvecCSR:
    def __init__(self,adjacency,kappa,capacity=0):
        self.N = len(adjacency.numNeighbors)
        size = adjacency.paddedJ.numel()
        if capacity > size:
            offsets,counts = adjacency.edgeOffsets.clone(),adjacency.numNeighbors.clone()
            neighbors = torch.empty(capacity,dtype=adjacency.paddedJ.dtype,device=kappa.device)
            weights = torch.empty(capacity,dtype=kappa.dtype,device=kappa.device)
            neighbors[:size].copy_(adjacency.paddedJ);weights[:size].copy_(kappa)
            self._keep = offsets,counts,neighbors,weights
        else:
            self._keep = tuple(value.contiguous() for value in
                (adjacency.edgeOffsets,adjacency.numNeighbors,adjacency.paddedJ,kappa))
        self.capacity = self._keep[2].numel()

    def assign(self,adjacency,kappa):
        size = adjacency.paddedJ.numel()
        if size > self.capacity or len(adjacency.numNeighbors) != self.N:return False
        for target,source in zip(self._keep[:2],(adjacency.edgeOffsets,adjacency.numNeighbors)):
            target.copy_(source)
        self._keep[2][:size].copy_(adjacency.paddedJ);self._keep[3][:size].copy_(kappa)
        return True

    def __call__(self,values):
        from ...native._build import build
        check_execution(values)
        return build().meshless_graph(*self._keep,values.contiguous())
