"""Forward-only fast launches of the MFM operator kernels on one geometry.

``launchOperator`` rebuilds the canonical ABI prefix (particle / domain / adjacency / grid /
correction / kernel-property structs) and runs the autograd bridge on every call -- ~0.5 ms of
host work per launch, which made the meshless step host-bound (a ``MeshlessWarp`` build and its
rates launch ~10 operator kernels on the same particles and adjacency). :class:`PreparedLauncher`
does the prefix once, with the same ``extractStateInfo`` / ``build_fn`` code ``launchOperator``
uses (so the kernels receive the same arguments and give bitwise the same results), and then
launches a spec's kernel directly: outputs allocated on torch's allocator, extras converted like
the operator's no-grad path.

No gradients: the MFM warp backend is forward-only (the torch reference is the differentiable
path). A call with any extra that requires grad -- or under forward-mode AD -- goes through
``launchOperator`` instead.
"""

from __future__ import annotations

import torch
import torch.autograd.forward_ad as fwAD
import warp as wp

from ..autograd.arg_extract import extractStateInfo
from ..autograd.cache import getCachedWarpArray
from ..autograd.launcher import _allocate_output, _launchMaybeTiled
from ..autograd.lanes import neighborLanes
from ..autograd.operator_spec import launchOperator

__all__ = ["PreparedLauncher"]


def _inert(t: torch.Tensor) -> bool:
    return not t.requires_grad and fwAD.unpack_dual(t)[1] is None


class PreparedLauncher:
    """The ABI prefix of one ``SPHContext``, built once; ``launch(spec, **extras)`` as ``launchOperator``."""

    def __init__(self, ctx):
        self.ctx = ctx
        qVol, rVol = ctx.corrections.volumes
        state = (ctx.query, ctx.properties, ctx.domain, qVol, rVol, ctx.adjacency, ctx.reference,
                 ctx.corrections.crk, ctx.corrections.gradH, ctx.corrections.renorm)
        flat, build_fn, device, dim = extractStateInfo(*state)
        self._flat = flat                                   # keep the viewed tensors alive
        self._ok = all(_inert(t) for t in flat if isinstance(t, torch.Tensor))
        if self._ok:
            wa = []
            for t in flat:
                if isinstance(t, torch.Tensor):
                    a = getCachedWarpArray(t, use_cache=False)
                    a.requires_grad = False
                else:
                    a = t.view()
                    a.requires_grad = False
                wa.append(a)
            self._wa = wa
            self.prefix = build_fn(wa, use_bundle=False)
        self.cuda = ctx.query.positions.is_cuda

    def launch(self, spec, **extras):
        tensors = [v for v in extras.values() if isinstance(v, torch.Tensor)]
        if not self._ok or (torch.is_grad_enabled() and not all(_inert(t) for t in tensors)):
            return launchOperator(spec, self.ctx, **extras)
        ctx = self.ctx
        args = []
        for e in spec.extras:
            v = extras[e.name]
            if isinstance(v, torch.Tensor):
                a = getCachedWarpArray(v, use_cache=False)
                a.requires_grad = False
                args.append(a)
            else:
                args.append(v)
        outs_t, outs_w = [], []
        device = None
        for out in spec.outputs:
            shape, dtype = out.resolve(ctx, extras)
            if device is None:
                device = self._wa[0].device
            t, w = _allocate_output(shape, dtype, device, False)
            del t._warp_array
            outs_t.append(t)
            outs_w.append(w)
        first = spec.outputs[0].resolve(ctx, extras)[0]
        n = first if isinstance(first, int) else first[0]
        if spec.numThreads is not None:
            n = spec.numThreads(ctx, extras)
        kernel, lanes = spec.kernel, None
        if spec.tiledKernel is not None and self.cuda:
            nLanes = neighborLanes(default=spec.defaultNeighborLanes)
            if nLanes > 1:
                kernel, lanes = spec.tiledKernel, nLanes
        _launchMaybeTiled(kernel, n, list(self.prefix) + args + outs_w, device, lanes)
        return outs_t[0] if len(outs_t) == 1 else tuple(outs_t)
