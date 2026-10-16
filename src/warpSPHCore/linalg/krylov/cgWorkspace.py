"""Reusable, dtype-generic native CG updates with host checkpoints.

The workspace is serial-use scratch; returned SolveResults own their storage.
Optional cached CUDA chunks retain eager checkpoints and verification. Neither Warp's
process precision nor autograd is used for these inference-only updates.
"""

from typing import Any

import torch
import warp as wp

from ...autograd.stream import torchWarpStream
from .common import PassResult, thresholdExceeded
from ._workspace import KrylovWorkspace

__all__ = ['CGWorkspace']


@wp.kernel(enable_backward=False)
def _initialize(rr: wp.array(dtype=Any), target: wp.array(dtype=Any),
                state: wp.array(dtype=wp.int32), norms: wp.array(dtype=Any),
                valid: wp.array(dtype=wp.bool), initialThreshold: wp.array(dtype=wp.bool),
                zeroRhs: wp.array(dtype=wp.bool), history: bool):
    norm = wp.sqrt(rr[0])
    conv = wp.isfinite(norm) and norm <= target[0]
    live = not conv and not initialThreshold[0]
    state[0] = wp.int32(live)
    state[1] = -14
    if conv:
        state[1] = 0
    if initialThreshold[0]:
        state[1] = -12
    state[2] = 0  # accepted updates
    state[3] = 0  # scheduled updates
    state[4] = 0  # accept current candidate
    if history:
        norms[0] = norm
        valid[0] = not (zeroRhs[0] and conv)


@wp.kernel(enable_backward=False)
def _candidate(x: wp.array(dtype=Any), r: wp.array(dtype=Any), p: wp.array(dtype=Any),
               ap: wp.array(dtype=Any), rz: wp.array(dtype=Any), pap: wp.array(dtype=Any),
               xn: wp.array(dtype=Any), rn: wp.array(dtype=Any), flags: wp.array(dtype=wp.int32),
               threshold: Any, hasThreshold: bool, dim: int):
    i = wp.tid()
    alpha = x.dtype(0)
    if wp.isfinite(pap[0]) and pap[0] != x.dtype(0):
        alpha = rz[0] / pap[0]
    xn[i] = x[i] + alpha * p[i]
    rn[i] = r[i] - alpha * ap[i]
    if not wp.isfinite(xn[i]) or not wp.isfinite(rn[i]):
        wp.atomic_max(flags, 0, 1)
    if hasThreshold and i % dim == 0:
        length2 = x.dtype(0)
        for j in range(dim):
            value = x[i + j] + alpha * p[i + j]
            length2 += value * value
        if wp.sqrt(length2) > x.dtype(threshold):
            wp.atomic_max(flags, 1, 1)


@wp.kernel(enable_backward=False)
def _decide(rz: wp.array(dtype=Any), pap: wp.array(dtype=Any), rr: wp.array(dtype=Any),
            target: wp.array(dtype=Any), flags: wp.array(dtype=wp.int32),
            state: wp.array(dtype=wp.int32), norms: wp.array(dtype=Any),
            valid: wp.array(dtype=wp.bool), history: bool):
    norm = wp.sqrt(rr[0])
    live = state[0] != 0
    ok = (wp.isfinite(rz[0]) and rz[0] > rz.dtype(0) and wp.isfinite(pap[0])
          and pap[0] > pap.dtype(0) and wp.isfinite(norm) and flags[0] == 0)
    go = live and ok
    conv = go and norm <= target[0] and flags[1] == 0
    if live and not ok:
        state[1] = -16
    elif go and flags[1] != 0:
        state[1] = -12
    elif conv:
        state[1] = state[3]
    state[0] = wp.int32(go and not conv and flags[1] == 0)
    state[4] = wp.int32(go)
    state[2] += wp.int32(go)
    state[3] += 1
    if history:
        norms[state[3]] = norm
        valid[state[3]] = go


@wp.kernel(enable_backward=False)
def _commit(x: wp.array(dtype=Any), r: wp.array(dtype=Any), xn: wp.array(dtype=Any),
            rn: wp.array(dtype=Any), state: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if state[4] != 0:
        x[i] = xn[i]
        r[i] = rn[i]


@wp.kernel(enable_backward=False)
def _diagonal(r: wp.array(dtype=Any), inverse: wp.array(dtype=Any), z: wp.array(dtype=Any)):
    i = wp.tid()
    z[i] = r[i] * inverse[i]


@wp.kernel(enable_backward=False)
def _direction(z: wp.array(dtype=Any), p: wp.array(dtype=Any), rz: wp.array(dtype=Any),
               rzn: wp.array(dtype=Any), state: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if state[0] != 0:
        beta = p.dtype(0)
        if wp.isfinite(rz[0]) and rz[0] != p.dtype(0):
            beta = rzn[0] / rz[0]
        p[i] = z[i] + beta * p[i]


class CGWorkspace(KrylovWorkspace):
    """Fixed native scratch for ``cg(..., workspace=workspace)``.

    Construct with a flat float32/float64 example vector and a maximum scheduled
    iteration capacity. ``history=False`` omits recurrence history; status,
    completed updates, work counts and the verified residual remain available.
    Reuse serially on the same Torch stream, or establish a stream dependency
    before reuse. Concurrent solves require separate workspaces. Inputs and
    callback outputs must not require gradients or carry forward AD tangents.

    ``capture=True`` caches CUDA chunks of ``checkEvery`` updates and a tail.
    Bind stable, graph-safe callbacks; call ``clearCapture`` after replacing
    captured storage hidden inside a callback. Content updates at the same
    address are visible on replay. Perform first-use setup serially before
    launching concurrent GPU work. The outer solve still reads the host.
    """

    def __init__(self, example: torch.Tensor, maxiter: int, *, history: bool = True, capture: bool = False):
        super().__init__(example, maxiter, history=history, capture=capture)
        for name in ('x', 'r', 'p', 'z', 'ap', 'xn', 'rn'):
            self._allocate(name, self.shape, self.dtype)
        for name in ('rz', 'pap', 'rzn', 'rr', 'target'):
            self._allocate(name, (1,), self.dtype)
        self._allocate('state', (5,), torch.int32)
        self._allocate('flags', (2,), torch.int32)
        self._allocate('norms', (self.maxiter + 1 if history else 1,), self.dtype)
        self._allocate('valid', self._t['norms'].shape, torch.bool)

    def _run(self, matvec, psolve, precond, x, residual, target, budget, zeroRhs,
             threshold, dim, checkEvery, counts, matvecOut=None):
        t, w = self._t, self._w
        n = x.numel()
        with wp.ScopedDevice(wp.device_from_torch(self.device)), torchWarpStream(self.device):
            t['x'].copy_(x)
            t['r'].copy_(residual)
            t['target'].copy_(target)
            initialThreshold = thresholdExceeded(x, threshold, dim)
            if initialThreshold is None:
                initialThreshold = torch.zeros((), dtype=torch.bool, device=self.device)
            self._precondition(psolve, precond, 'r', counts)
            t['p'].copy_(t['z'])
            wp.utils.array_inner(w['r'], w['z'], out=w['rz'])
            wp.utils.array_inner(w['r'], w['r'], out=w['rr'])
            wp.launch(_initialize, dim=1, inputs=[w['rr'], w['target'], w['state'], w['norms'], w['valid'],
                      wp.from_torch(initialThreshold.reshape(1)), wp.from_torch(zeroRhs.reshape(1)), self.history])
            used = 0
            if budget and bool(t['state'][0]):
                # Bind invariant arguments once per pass. Launch commands
                # avoid Python's full kernel argument marshalling each update.
                candidate = wp.launch(_candidate, dim=n, record_cmd=True,
                    inputs=[w[key] for key in ('x', 'r', 'p', 'ap', 'rz', 'pap', 'xn', 'rn', 'flags')]
                    + [wp.dtype_from_torch(self.dtype)(threshold or 0), threshold is not None, dim])
                decide = wp.launch(_decide, dim=1, record_cmd=True,
                    inputs=[w[key] for key in ('rz', 'pap', 'rr', 'target', 'flags', 'state', 'norms', 'valid')]
                    + [self.history])
                commit = wp.launch(_commit, dim=n, record_cmd=True,
                    inputs=[w[key] for key in ('x', 'r', 'xn', 'rn', 'state')])
                direction = wp.launch(_direction, dim=n, record_cmd=True,
                    inputs=[w[key] for key in ('z', 'p', 'rz', 'rzn', 'state')])
                diagonal = None if not isinstance(precond, torch.Tensor) else wp.launch(
                    _diagonal, dim=n, record_cmd=True, inputs=[w['rn'], wp.from_torch(precond), w['z']])
                def iteration():
                    if matvecOut is None:
                        t['ap'].copy_(self._inert(matvec(t['p'])))
                    else:
                        matvecOut(t['p'], t['ap'])
                    wp.utils.array_inner(w['p'], w['ap'], out=w['pap'])
                    w['flags'].zero_()
                    candidate.launch()
                    wp.utils.array_inner(w['rn'], w['rn'], out=w['rr'])
                    decide.launch()
                    commit.launch()
                    self._precondition(psolve, precond, 'rn', counts, diagonal)
                    wp.utils.array_inner(w['rn'], w['z'], out=w['rzn'])
                    direction.launch()
                    t['rz'].copy_(t['rzn'])
                if self.capture:
                    while used < budget:
                        steps = min(checkEvery, budget - used)
                        graph = self._graph(steps, iteration, (candidate, decide, commit, direction, diagonal), counts)
                        graph.replay()
                        counts['matvecs'] += steps
                        if precond is not None:
                            counts['preconditionerApplications'] += steps
                        used += steps
                        if not bool(t['state'][0]):
                            break
                else:
                    for k in range(budget):
                        iteration()
                        used = k + 1
                        if used % checkEvery == 0 and not bool(t['state'][0]):
                            break
            # Copies occur once per pass, never once per iteration. They protect
            # histories from the next residual-replacement pass overwriting them.
            norms = list(t['norms'][:used + 1].clone().unbind()) if self.history else []
            valid = list(t['valid'][:used + 1].clone().unbind()) if self.history else []
            return PassResult(t['x'], t['state'][1], norms, valid, t['state'][2], used)

    def _precondition(self, psolve, precond, source, counts, command=None):
        if isinstance(precond, torch.Tensor):
            # The public boundary validated this inverse diagonal. Count its
            # application here while avoiding psolve's elementwise allocation.
            counts['preconditionerApplications'] += 1
            if command is None:
                wp.launch(_diagonal, dim=self.shape[0], inputs=[self._w[source], wp.from_torch(precond), self._w['z']])
            else:
                command.launch()
        elif precond is None:
            self._t['z'].copy_(self._t[source])
        else:
            self._t['z'].copy_(self._inert(psolve(self._t[source])))
