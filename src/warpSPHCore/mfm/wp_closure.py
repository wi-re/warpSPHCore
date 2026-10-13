"""The closure solve's matvec on the bare CSR arrays, in a warp module of its own.

The closure workspace (``warpBackend._ClosureWorkspace``) captures this kernel in a CUDA graph and
replays it across many builds. A graph holds the kernel's function handle from the loaded warp
module, and warp rebuilds and reloads a whole module whenever one of its kernels is first launched
with a new overload -- in ``wp_mfm.py`` the flux / gradient kernels compile lazily, which would
unload the matvec under a captured graph (a host segfault on replay). Here the module holds only
this kernel, with every overload declared up front (1-, 2-, 3-D), so it is compiled once, on its
first launch, and never reloaded.
"""

import torch
import warp as wp
from typing import Any
from warp.types import vector

from ..type_config import scalar_t
from ..math import zero_like_warp

__all__ = ["mfmClosureMatvecCSR_Kernel", "mfmClosureMatvecCSR"]


@wp.kernel
def mfmClosureMatvecCSR_Kernel(
    neighborOffsets: wp.array(dtype=wp.int32), numNeighbors: wp.array(dtype=wp.int32),  # type: ignore
    neighborList: wp.array(dtype=wp.int64), kappa: wp.array(dtype=scalar_t), p: wp.array(dtype=Any),  # type: ignore
    output_y: wp.array(dtype=Any), output_diag: wp.array(dtype=scalar_t),  # type: ignore
):
    """`wp_mfm.mfmClosureMatvec_Kernel` on the bare CSR arrays: the same loop and arithmetic (bitwise-identical
    results), launched directly (`mfmClosureMatvecCSR`) instead of through the operator ABI. The closure
    solve applies it ~3 times per CG iteration, and the ABI's per-launch host work (~0.5 ms) was most of
    the MFM step's cost."""
    i = wp.tid()
    beginIndex = neighborOffsets[i]
    numIndices = numNeighbors[i]
    acc = zero_like_warp(output_y[i])
    diag = scalar_t(0.0)
    pi = p[i]
    for n in range(numIndices):
        k = kappa[beginIndex + n]
        if k > scalar_t(0.0):
            j = wp.int32(neighborList[beginIndex + n])
            acc += k * (pi - p[j])
            diag += k
    output_y[i] = acc
    output_diag[i] = diag


_VEC = {d: vector(length=d, dtype=scalar_t) for d in (1, 2, 3)}
for _d, _v in _VEC.items():
    wp.overload(mfmClosureMatvecCSR_Kernel, {"p": wp.array(dtype=_v), "output_y": wp.array(dtype=_v)})


class mfmClosureMatvecCSR:
    """The closure matvec repeated on one adjacency: the warp views of the CSR arrays and of ``kappa`` are
    made once; each call wraps ``p`` and two fresh outputs and launches on torch's current stream.
    Returns ``(y, diag)`` as ``wp_mfm.mfmClosureMatvecWarp`` does.

    ``capacity`` (edges) > the adjacency's size allocates the neighbour list and the weights with room to
    spare, and :meth:`assign` then refills them in place for another adjacency over the same particles --
    the closure workspace's CUDA graph reads them at fixed addresses."""

    def __init__(self, adjacency, kappa: torch.Tensor, capacity: int = 0):
        self.N = adjacency.numNeighbors.shape[0]
        self.device = wp.device_from_torch(kappa.device)
        E = adjacency.paddedJ.shape[0]
        if capacity > E:
            offsets, counts = adjacency.edgeOffsets.clone(), adjacency.numNeighbors.clone()
            nbr = torch.empty(capacity, dtype=adjacency.paddedJ.dtype, device=kappa.device)
            kap = torch.empty(capacity, dtype=kappa.dtype, device=kappa.device)
            nbr[:E].copy_(adjacency.paddedJ)
            kap[:E].copy_(kappa)
            self._keep = (offsets, counts, nbr, kap)
        else:
            self._keep = (adjacency.edgeOffsets.contiguous(), adjacency.numNeighbors.contiguous(),
                          adjacency.paddedJ.contiguous(), kappa.contiguous())
        self.capacity = self._keep[2].shape[0]
        self.offsets, self.counts, self.nbr, self.kappa = (wp.from_torch(t) for t in self._keep)

    def assign(self, adjacency, kappa: torch.Tensor) -> bool:
        """Refill in place from another adjacency over the same particles; False if it does not fit."""
        E = adjacency.paddedJ.shape[0]
        if E > self.capacity or adjacency.numNeighbors.shape[0] != self.N:
            return False
        offsets, counts, nbr, kap = self._keep
        offsets.copy_(adjacency.edgeOffsets)
        counts.copy_(adjacency.numNeighbors)
        nbr[:E].copy_(adjacency.paddedJ)
        kap[:E].copy_(kappa)
        return True

    def __call__(self, p: torch.Tensor):
        vec = _VEC[p.shape[1]]
        p = p.contiguous()
        y = torch.empty_like(p)
        diag = torch.empty(self.N, dtype=p.dtype, device=p.device)
        stream = wp.stream_from_torch(torch.cuda.current_stream(p.device)) if p.is_cuda else None
        wp.launch(mfmClosureMatvecCSR_Kernel, dim=self.N,
                  inputs=[self.offsets, self.counts, self.nbr, self.kappa, wp.from_torch(p, dtype=vec)],
                  outputs=[wp.from_torch(y, dtype=vec), wp.from_torch(diag)], device=self.device, stream=stream)
        return y, diag
