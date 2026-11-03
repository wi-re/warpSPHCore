"""Shared cg workspace orchestration; numerical updates are backend-owned."""
import torch
from ._workspace import KrylovWorkspace
from ._execution import engine
from .common import PassResult,thresholdExceeded

kernels = engine.kernels('cg')
__all__ = ['CGWorkspace']

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
        with engine.scope(self.device):
            t['x'].copy_(x)
            t['r'].copy_(residual)
            t['target'].copy_(target)
            initialThreshold = thresholdExceeded(x, threshold, dim)
            if initialThreshold is None:
                initialThreshold = torch.zeros((), dtype=torch.bool, device=self.device)
            self._precondition(psolve, precond, 'r', counts)
            t['p'].copy_(t['z'])
            engine.inner(w['r'], w['z'], out=w['rz'])
            engine.inner(w['r'], w['r'], out=w['rr'])
            engine.launch(kernels._initialize, dim=1, inputs=[w['rr'], w['target'], w['state'], w['norms'], w['valid'],
                      engine.array(initialThreshold.reshape(1)), engine.array(zeroRhs.reshape(1)), self.history])
            used = 0
            if budget and bool(t['state'][0]):
                # Bind invariant arguments once per pass. Launch commands
                # avoid Python's full kernel argument marshalling each update.
                candidate = engine.launch(kernels._candidate, dim=n, record_cmd=True,
                    inputs=[w[key] for key in ('x', 'r', 'p', 'ap', 'rz', 'pap', 'xn', 'rn', 'flags')]
                    + [engine.scalar(threshold or 0,self.dtype), threshold is not None, dim])
                decide = engine.launch(kernels._decide, dim=1, record_cmd=True,
                    inputs=[w[key] for key in ('rz', 'pap', 'rr', 'target', 'flags', 'state', 'norms', 'valid')]
                    + [self.history])
                commit = engine.launch(kernels._commit, dim=n, record_cmd=True,
                    inputs=[w[key] for key in ('x', 'r', 'xn', 'rn', 'state')])
                direction = engine.launch(kernels._direction, dim=n, record_cmd=True,
                    inputs=[w[key] for key in ('z', 'p', 'rz', 'rzn', 'state')])
                diagonal = None if not isinstance(precond, torch.Tensor) else engine.launch(
                    kernels._diagonal, dim=n, record_cmd=True, inputs=[w['rn'], engine.array(precond), w['z']])
                def iteration():
                    if matvecOut is None:
                        t['ap'].copy_(self._inert(matvec(t['p'])))
                    else:
                        matvecOut(t['p'], t['ap'])
                    engine.inner(w['p'], w['ap'], out=w['pap'])
                    w['flags'].zero_()
                    candidate.launch()
                    engine.inner(w['rn'], w['rn'], out=w['rr'])
                    decide.launch()
                    commit.launch()
                    self._precondition(psolve, precond, 'rn', counts, diagonal)
                    engine.inner(w['rn'], w['z'], out=w['rzn'])
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
                engine.launch(kernels._diagonal, dim=self.shape[0], inputs=[self._w[source], engine.array(precond), self._w['z']])
            else:
                command.launch()
        elif precond is None:
            self._t['z'].copy_(self._t[source])
        else:
            self._t['z'].copy_(self._inert(psolve(self._t[source])))
