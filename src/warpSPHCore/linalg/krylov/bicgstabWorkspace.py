"""Shared bicgstab workspace orchestration; numerical updates are backend-owned."""
import torch
from ._workspace import KrylovWorkspace
from ._execution import engine
from .common import PassResult,thresholdExceeded

kernels = engine.kernels('bicgstab')
__all__ = ['BiCGStabWorkspace']

class BiCGStabWorkspace(KrylovWorkspace):
    """Owned native scratch for right-preconditioned BiCGStab.

    Flat CPU/CUDA float32/float64 vectors; serial inference-only reuse.
    Optional history and cached guarded CUDA chunks follow CGWorkspace's
    ownership, setup, invalidation and host-checkpoint contracts.
    """

    def __init__(self, example, maxiter, *, history=True, capture=False):
        super().__init__(example, maxiter, history=history, capture=capture)
        for name in ('x', 'r', 'r0', 'p', 'z', 'ap', 's', 'candidate', 'zs', 'ast'):
            self._allocate(name, self.shape, self.dtype)
        for name in ('rho', 'rv', 'aa', 'ss', 'tt', 'ts', 'rr', 'rhon', 'alpha', 'omega', 'beta', 'r0norm', 'rnorm', 'target'):
            self._allocate(name, (1,), self.dtype)
        self._allocate('state', (9,), torch.int32)
        self._allocate('flags', (3,), torch.int32)
        self._allocate('norms', (max(1, self.maxiter) if history else 1,), self.dtype)
        self._allocate('valid', self._t['norms'].shape, torch.bool)

    def _run(self, matvec, psolve, precond, x, residual, target, budget, zeroRhs,
             threshold, dim, checkEvery, counts, matvecOut=None, precondOut=None):
        t, w = self._t, self._w
        n = x.numel()
        with engine.scope(self.device):
            for key, value in (('x', x), ('r', residual), ('r0', residual), ('p', residual), ('target', target)):
                t[key].copy_(value)
            thr = thresholdExceeded(x, threshold, dim)
            if thr is None:
                thr = torch.zeros((), dtype=torch.bool, device=self.device)
            engine.inner(w['r'], w['r'], out=w['rr'])
            engine.launch(kernels._initialize, dim=1, inputs=[w[k] for k in ('rr', 'target', 'r0norm', 'rnorm', 'state')]
                      + [engine.array(thr.reshape(1))])
            used = 0
            if budget and bool(t['state'][0]):
                def command(kernel, keys, size=1, extra=()):
                    return engine.launch(kernel, dim=size, record_cmd=True, inputs=[w[k] for k in keys.split()] + list(extra))

                bounds = (engine.scalar(threshold or 0,self.dtype), threshold is not None, dim)
                alpha = command(kernels._alpha, 'rho rv aa r0norm rnorm alpha state')
                half = command(kernels._halfCandidate, 'x r z ap alpha candidate s flags', n, bounds)
                halfDecide = command(kernels._halfDecision, 'ss target flags state')
                omega = command(kernels._omega, 'ts tt omega')
                full = command(kernels._fullCandidate, 'x z zs ast s candidate alpha omega flags state', n, bounds)
                fullDecide = command(kernels._fullDecision, 'rr rhon rho alpha omega tt beta target flags state')
                direction = command(kernels._direction, 'x candidate s p ap z beta omega flags state', n)
                finish = command(kernels._finish, 'rr rnorm flags state norms valid', extra=(self.history,))
                commit = command(kernels._commit, 'r p s z state', n)
                diagonal = []
                if isinstance(precond, torch.Tensor):
                    for source, output in (('p', 'z'), ('s', 'zs')):
                        diagonal.append(engine.launch(kernels._diagonal, dim=n, record_cmd=True,
                            inputs=[w[source], engine.array(precond), w[output]]))

                def condition(source, output, slot):
                    if precondOut is not None:
                        precondOut(t[source], t[output])
                    elif diagonal:
                        counts['preconditionerApplications'] += 1
                        diagonal[slot].launch()
                    elif precond is None:
                        t[output].copy_(t[source])
                    else:
                        t[output].copy_(self._inert(psolve(t[source])))

                def action(source, output):
                    if matvecOut is not None:
                        matvecOut(t[source], t[output])
                    else:
                        t[output].copy_(self._inert(matvec(t[source])))

                def inner(a, b, out):
                    engine.inner(w[a], w[b], out=w[out])

                def iteration():
                    condition('p', 'z', 0)
                    action('z', 'ap')
                    inner('r', 'r0', 'rho')
                    inner('ap', 'r0', 'rv')
                    inner('ap', 'ap', 'aa')
                    alpha.launch()
                    w['flags'].zero_()
                    half.launch()
                    inner('s', 's', 'ss')
                    halfDecide.launch()
                    condition('s', 'zs', 1)
                    action('zs', 'ast')
                    inner('ast', 's', 'ts')
                    inner('ast', 'ast', 'tt')
                    omega.launch()
                    w['flags'].zero_()
                    full.launch()
                    inner('s', 's', 'rr')
                    inner('s', 'r0', 'rhon')
                    fullDecide.launch()
                    direction.launch()
                    finish.launch()
                    commit.launch()

                commands = (alpha, half, halfDecide, omega, full, fullDecide, direction, finish, commit, *diagonal)
                if self.capture:
                    while used < budget:
                        steps = min(checkEvery, budget-used)
                        graph = self._graph(steps, iteration, commands, counts)
                        graph.replay()
                        counts['matvecs'] += 2 * steps
                        if precond is not None:
                            counts['preconditionerApplications'] += 2 * steps
                        used += steps
                        if not bool(t['state'][0]):
                            break
                else:
                    for k in range(budget):
                        iteration()
                        used = k + 1
                        if used % checkEvery == 0 and not bool(t['state'][0]):
                            break
            norms = list(t['norms'][:used].clone().unbind()) if self.history else []
            valid = list(t['valid'][:used].clone().unbind()) if self.history else []
            return PassResult(t['x'], t['state'][1], norms, valid, t['state'][2], used)
