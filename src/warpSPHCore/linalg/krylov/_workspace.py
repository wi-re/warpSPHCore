"""Shared owned scratch and bounded guarded-chunk capture lifecycle.

Numerical state and iteration kernels belong to each solver. This base only
owns buffers, validation, graph cache/invalidation and accounting of setup.
"""

from collections import OrderedDict
from time import perf_counter

import torch
import warp as wp

from ...autograd.stream import torchWarpCapture, torchWarpStream
from .common import nonnegativeInt, validateVector


class KrylovWorkspace:
    def __init__(self, example, maxiter, *, history=True, capture=False):
        if (not isinstance(example, torch.Tensor) or example.ndim != 1
                or example.dtype not in (torch.float32, torch.float64)
                or example.device.type not in ('cpu', 'cuda')):
            raise ValueError('workspace example must be a flat CPU/CUDA float32/float64 tensor')
        self.maxiter = nonnegativeInt('maxiter', maxiter)
        if capture and not example.is_cuda:
            raise ValueError('captured chunks require a CUDA workspace')
        wp.init()
        self.history = bool(history)
        self.shape, self.dtype, self.device = example.shape, example.dtype, example.device
        self._t, self._w = {}, {}
        self._busy = False
        self.capture = bool(capture)
        self._graphs = OrderedDict()
        self._captureKey = None
        self._captureRefs = None
        self.captureBuilds = 0
        self.captureMilliseconds = 0.

    def clearCapture(self):
        """Invalidate cached callbacks after rebinding geometry/coefficient storage.

        Same-address content changes are visible on replay. Replacement of
        storage hidden inside a callback requires this explicit invalidation.
        """
        if self._busy:
            raise RuntimeError(f'cannot clear a {type(self).__name__} while it is in use')
        self._graphs.clear()
        self._captureKey = self._captureRefs = None

    def _bindCapture(self, matvec, matvecOut, precond, threshold, dim, checkEvery, precondOut=None):
        if not self.capture:
            return
        layout = (precond.data_ptr(), precond.stride()) if isinstance(precond, torch.Tensor) else None
        key = (id(matvec), id(matvecOut), id(precond), layout, threshold, dim, checkEvery, id(precondOut))
        if key != self._captureKey:
            self.clearCapture()
            self._captureKey = key
            # Keep identities alive for as long as graphs refer to their data.
            self._captureRefs = (matvec, matvecOut, precond, precondOut)

    def _graph(self, steps, iteration, commands, counts):
        if steps in self._graphs:
            self._graphs.move_to_end(steps)
            return self._graphs[steps][0]
        current = torch.cuda.current_stream(self.device)
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(current)
        # Warm one real update, then restore the numerical state. Count that
        # executed work; Python calls made only to record a graph are excluded.
        with torch.cuda.stream(side), torchWarpStream(self.device):
            backup = {key: value.clone() for key, value in self._t.items()}
            try:
                iteration()
                counts['setupIterations'] = counts.get('setupIterations', 0) + 1
            finally:
                for key, value in backup.items():
                    self._t[key].copy_(value)
            side.synchronize()  # explicit setup boundary; never device-wide
            before = perf_counter()
            recordedCounts = counts.copy()
            graph = torch.cuda.CUDAGraph()
            try:
                with torchWarpCapture(graph) as capture:
                    for _ in range(steps):
                        iteration()
            finally:
                counts.clear()
                counts.update(recordedCounts)
            self.captureMilliseconds += (perf_counter() - before) * 1e3
        current.wait_stream(side)
        self.captureBuilds += 1
        self._graphs[steps] = (graph, commands, capture.graph)
        # One regular chunk and one tail; arbitrary budget changes do not grow
        # the graph cache indefinitely. Each graph keeps its bound modules.
        if len(self._graphs) > 2:
            self._graphs.popitem(last=False)
        return graph

    def _allocate(self, name, shape, dtype):
        value = torch.empty(shape, device=self.device, dtype=dtype)
        self._t[name], self._w[name] = value, wp.from_torch(value)

    @property
    def nbytes(self):
        """Owned tensor scratch bytes, excluding graph pools, callbacks and results."""
        return sum(t.numel() * t.element_size() for t in self._t.values())

    def _check(self, b, maxiter):
        validateVector(b, self._t['x'], 'workspace b')
        if maxiter > self.maxiter:
            raise ValueError(f'workspace capacity {self.maxiter} is smaller than maxiter={maxiter}')
        if self._busy:
            raise RuntimeError(f'{type(self).__name__} is already in use')

    @staticmethod
    def _inert(value):
        if value.requires_grad or torch.autograd.forward_ad.unpack_dual(value).tangent is not None:
            raise ValueError('native workspace does not support autograd; use the solver without a workspace')
        return value
