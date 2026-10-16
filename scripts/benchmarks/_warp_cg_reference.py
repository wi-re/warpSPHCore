"""Experimental upstream-CG harness; deliberately not a public core backend.

Keeps upstream recurrence diagnostics separate from an independent true
residual. No claim of core threshold, breakdown, history or budget parity.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch
import warp as wp
from warp.optim import linear

from warpSPHCore.autograd.stream import torchWarpCapture, torchWarpStream


@wp.kernel(enable_backward=False)
def _combine(ax: wp.array(dtype=Any), y: wp.array(dtype=Any), z: wp.array(dtype=Any), alpha: Any, beta: Any):
    i = wp.tid()
    value = alpha * ax[i]
    if beta != type(beta)(0):
        value += beta * y[i]
    z[i] = value


@dataclass
class Observation:
    x: torch.Tensor
    scheduled: int
    recurrence: float
    trueResidual: float
    target: float
    finite: bool

    @property
    def verified(self):
        return self.finite and self.trueResidual <= self.target


class WarpCGReference:
    """Reuse upstream buffers; optionally cache one complete conditional graph.

    ``eager`` disables upstream graph capture. ``internal`` uses its ordinary
    per-call capture. ``cached`` captures the entire call once in a Torch graph
    with a registered external Warp capture for conditional nodes. Shape,
    callback, preconditioner binding and captured scalar options stay fixed;
    RHS, initial guess and referenced coefficient contents may change.
    """

    def __init__(self, matvec, example, *, maxiter=128, rtol=1e-5, atol=0., precond=None,
                 schedule='eager', checkEvery=8, matvecOut=None):
        wp.init()
        if schedule not in ('eager', 'internal', 'cached'):
            raise ValueError('unknown upstream schedule')
        if schedule != 'eager' and not example.is_cuda:
            raise ValueError('captured upstream schedules require CUDA')
        self.schedule, self.rtol, self.atol = schedule, rtol, atol
        self.matvec, self.matvecOut = matvec, matvecOut
        self.b, self.x = torch.zeros_like(example), torch.zeros_like(example)
        self._ax = torch.empty_like(example)
        self._wpb, self._wpx = wp.from_torch(self.b), wp.from_torch(self.x)
        self._wpax = wp.from_torch(self._ax)
        self.precond = precond
        self.graph = None
        self.recorded_matvec_calls = 0
        dtype, device = wp.dtype_from_torch(example.dtype), wp.device_from_torch(example.device)

        ax, wpax = self._ax, self._wpax

        def action(x, y, z, alpha, beta):
            tx = wp.to_torch(x)
            if matvecOut is None:
                ax.copy_(matvec(tx))
            else:
                matvecOut(tx, ax)
            wp.launch(_combine, dim=example.numel(), device=device,
                      inputs=[wpax, y, z, dtype(alpha), dtype(beta)])

        self.operator = linear.LinearOperator((example.numel(), example.numel()), dtype, device, action)
        before = perf_counter()
        with wp.ScopedDevice(device), torchWarpStream(example.device):
            self.state = linear.cg(self.operator, self._wpb, self._wpx, tol=rtol, atol=atol,
                                   maxiter=maxiter, M=None if precond is None else wp.from_torch(precond),
                                   check_every=0 if schedule == 'cached' else checkEvery,
                                   use_cuda_graph=schedule != 'eager', run=False)
        self.setup_ms = (perf_counter() - before) * 1e3
        self.capture_ms = 0.

    def prepare(self, b, x0=None):
        """Warm kernels, then capture on a side stream; exclude from steady timing."""
        if self.schedule != 'cached':
            return
        side = torch.cuda.Stream(device=b.device)
        side.wait_stream(torch.cuda.current_stream(b.device))
        with torch.cuda.stream(side), torchWarpStream(b.device):
            self.b.copy_(b)
            self.x.zero_() if x0 is None else self.x.copy_(x0)
            self.state()  # load tiled reduction and conditional kernels before capture
            side.synchronize()
            self.x.zero_() if x0 is None else self.x.copy_(x0)
            before = perf_counter()
            graph = torch.cuda.CUDAGraph()
            with torchWarpCapture(graph) as capture:
                self._device_result = self.state()
            self._warp_capture = capture.graph  # retain Warp modules and child graphs
            self.graph = graph
            self.capture_ms = (perf_counter() - before) * 1e3
        torch.cuda.current_stream(b.device).wait_stream(side)

    def raw(self, b, x0=None):
        self.b.copy_(b)
        self.x.zero_() if x0 is None else self.x.copy_(x0)
        with wp.ScopedDevice(wp.device_from_torch(b.device)), torchWarpStream(b.device):
            if self.schedule == 'cached':
                if self.graph is None:
                    raise RuntimeError('call prepare before replay')
                self.graph.replay()
                return self._device_result
            return self.state()

    def observe(self, b, x0=None):
        raw = self.raw(b, x0)
        x = self.x.clone()
        if self.schedule == 'cached':
            iterations, rr, _ = raw
            scheduled = wp.to_torch(iterations)[0].double()
            recurrence = wp.to_torch(rr)[0].sqrt().double()
        else:
            scheduled, recurrence, _ = raw
            scheduled = b.new_tensor(scheduled, dtype=torch.float64)
            recurrence = b.new_tensor(recurrence, dtype=torch.float64)
        from warpSPHCore.linalg.krylov.common import absoluteTolerance
        true = (b - self.matvec(x)).norm()
        target = absoluteTolerance(b.norm(), 0., self.rtol, self.atol)
        # A single diagnostic transfer, outside the upstream iteration loop.
        summary = torch.stack([scheduled, recurrence, true.double(), target.double(),
                               x.isfinite().all().double()]).cpu().tolist()
        return Observation(x, int(summary[0]), *summary[1:4], bool(summary[4]))
