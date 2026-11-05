"""Per-call neighborhood dispatch, independent of the runtime type backend.

A Warp-runtime frontend can select native search without switching its public
particle models or importing a second runtime. Adjacency records its producer;
validity/filter calls inherit that producer unless explicitly overridden.
"""
from contextlib import contextmanager
from .._backend import BACKEND, resolve_backend
from .._models import DEFAULT_CAPACITY
from ..enumTypes import SupportScheme

__all__ = ['buildVerletList', 'filterVerletList', 'verletStaleFlag', 'verletUsage', 'deferVerletChecks']


def _selected(backend, adjacency=None):
    if backend is None:
        backend = getattr(adjacency, 'searchBackend', BACKEND)
    return resolve_backend(backend)


def buildVerletList(queryParticles, domain, verletScale=1., supportMode=SupportScheme.Gather,
                    priorNeighborhood=None, verbose=False, referenceParticles=None,
                    capacity=DEFAULT_CAPACITY, rebuildSlack=0., gridFactory=None, forceRebuild=False,
                    *, collection='auto', queryOrder='auto', storeI=True, backend=None):
    selected = _selected(backend, priorNeighborhood)
    from .chunked import ChunkedAdjacency, buildChunkedVerletList
    if isinstance(priorNeighborhood, ChunkedAdjacency):
        if gridFactory is not None or capacity is not DEFAULT_CAPACITY:
            raise NotImplementedError('Chunked Verlet lists do not support flat capacities or grid factories.')
        if collection not in ('auto', 'split') or queryOrder not in ('auto', 'input'):
            raise NotImplementedError('Chunked Verlet lists require direct split collection in input row order.')
        return buildChunkedVerletList(queryParticles, domain, referenceParticles=referenceParticles,
            policy=priorNeighborhood.plan.policy, verletScale=verletScale, supportMode=supportMode,
            priorNeighborhood=priorNeighborhood, storeI=storeI, rebuildSlack=rebuildSlack,
            forceRebuild=forceRebuild, orderingKey=priorNeighborhood.orderingKey, backend=selected)
    if selected == 'native':
        from .native.nt_verlet import buildVerletList as build
    else:
        from .verlet.build import buildVerletList as build
    prior = priorNeighborhood
    if prior is not None and prior.searchBackend != selected:
        prior = None
    result = build(queryParticles, domain, verletScale, supportMode, prior, verbose,
        referenceParticles, capacity, rebuildSlack, gridFactory, forceRebuild,
        collection=collection, queryOrder=queryOrder, storeI=storeI)
    result.searchBackend = selected
    return result


def filterVerletList(queryParticles, domain, adjacency, supportMode=SupportScheme.Gather,
                     referenceParticles=None, capacity=DEFAULT_CAPACITY, *, storeI=None, backend=None):
    selected = _selected(backend, adjacency)
    from .chunked import ChunkedAdjacency, filterChunkedVerletList
    if isinstance(adjacency, ChunkedAdjacency):
        if capacity is not DEFAULT_CAPACITY or supportMode != adjacency.mode:
            raise ValueError('Chunked filtering retains its support mode and bounded row policy.')
        return filterChunkedVerletList(queryParticles, domain, adjacency,
            referenceParticles=referenceParticles, storeI=storeI, backend=selected)
    if selected == 'native':
        from .native.nt_verlet import filterVerletList as filter_list
    else:
        from .verlet.filter import filterVerletList as filter_list
    result = filter_list(queryParticles, domain, adjacency, supportMode, referenceParticles,
                         capacity, storeI=storeI)
    result.searchBackend = selected
    return result


def verletStaleFlag(positions, supports, adjacency, domain, verletScale,
                    supportMode=SupportScheme.SuperSymmetric, rebuildSlack=0., *, backend=None):
    selected = _selected(backend, adjacency)
    from .chunked import ChunkedAdjacency, chunkedVerletStaleFlag_
    if isinstance(adjacency, ChunkedAdjacency):
        return chunkedVerletStaleFlag_(positions, positions, supports, supports, domain,
            adjacency, mode=supportMode, verletScale=verletScale, rebuildSlack=rebuildSlack, backend=selected)
    if selected == 'native':
        from .native.nt_verlet import verletStaleFlag as stale
    else:
        from .verlet.build import verletStaleFlag as stale
    return stale(positions, supports, adjacency, domain, verletScale, supportMode, rebuildSlack)


def verletUsage(positions, supports, adjacency, domain, verletScale,
                supportMode=SupportScheme.SuperSymmetric, *, backend=None):
    if _selected(backend, adjacency) == 'native':
        from .native.nt_verlet import verletUsage as usage
    else:
        from .verlet.build import verletUsage as usage
    return usage(positions, supports, adjacency, domain, verletScale, supportMode)


@contextmanager
def deferVerletChecks():
    from .native.nt_verlet import _DEFERRED
    if BACKEND == 'native':
        from .native.nt_verlet import deferVerletChecks as deferred
        with deferred() as flags:
            yield flags
    else:
        from .verlet.build import deferVerletChecks as deferred
        with deferred() as flags:
            token = _DEFERRED.set(flags)
            try:
                yield flags
            finally:
                _DEFERRED.reset(token)
