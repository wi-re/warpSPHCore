"""Warp-free Verlet lifecycle over Torch-owned CSR storage.

Membership and rebuild decisions are discrete. Only the eager rebuild decision
and variable-size output allocation read a scalar from the device. Validity
metrics and deferred checks otherwise stay on the current Torch stream.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace

import torch

from ..._models import AdjacencyList, DEFAULT_CAPACITY
from ...enumTypes import SupportScheme
from .nt_radius import radiusSearch

_DEFERRED = ContextVar('native_verlet_deferred_checks', default=None)


@contextmanager
def deferVerletChecks():
    flags = []
    token = _DEFERRED.set(flags)
    try:
        yield flags
    finally:
        _DEFERRED.reset(token)


def _context(domain, mode):
    bounds = (domain.min, domain.max, domain.periodic)
    if any(value.is_inference() for value in bounds):
        # Inference tensors cannot report mutation versions. Conservatively
        # rebuild rather than accepting an undetectable changed domain.
        return None
    signature = (mode.value, *((id(value), value._version) for value in bounds))
    return signature, bounds  # retain owners so Python identities cannot recycle


def _same_context(prior, current):
    return prior is not None and current is not None and prior[0] == current[0]


def _case(mode):
    if not isinstance(mode, SupportScheme):
        raise TypeError('supportMode must be a SupportScheme')
    return 0 if mode == SupportScheme.Gather else 1 if mode == SupportScheme.Scatter else 2


@torch.no_grad()
def verletValidityMetrics(queryPositions, referencePositions, priorQueryPositions, priorReferencePositions,
                          querySupports, referenceSupports, supports_a, supports_b,
                          periodicity, domainMin, domainMax, verletScale, support_case, rebuildSlack=0.):
    if queryPositions.shape != priorQueryPositions.shape or referencePositions.shape != priorReferencePositions.shape:
        zero = querySupports.new_zeros(())
        return torch.ones((), dtype=torch.bool, device=queryPositions.device), zero, zero, zero, zero
    if not len(queryPositions) or not len(referencePositions):
        zero = querySupports.new_zeros(())
        return torch.zeros((), dtype=torch.bool, device=queryPositions.device), zero, zero, zero, zero
    if support_case not in (0, 1, 2):
        raise ValueError('support_case must be 0, 1 or 2')
    same = (queryPositions is referencePositions and priorQueryPositions is priorReferencePositions
            and querySupports is referenceSupports and supports_a is supports_b)
    scale = (verletScale.detach().to(dtype=querySupports.dtype, device=querySupports.device)
             if isinstance(verletScale, torch.Tensor) else querySupports.new_tensor(verletScale))
    scale = scale.reshape(1).contiguous()
    from ...native._build import build
    flag, values = build().verlet_metrics(
        [queryPositions, referencePositions, querySupports, referenceSupports, domainMin, domainMax, periodicity],
        priorQueryPositions, priorReferencePositions, supports_a, supports_b, scale,
        support_case, rebuildSlack, same)
    return flag, *values.unbind()


def _metrics(positions, supports, adjacency, domain, scale, mode, slack):
    return verletValidityMetrics(positions, positions, adjacency.queryPositions, adjacency.referencePositions,
                                 supports, supports, adjacency.querySupports, adjacency.referenceSupports,
                                 domain.periodic, domain.min, domain.max, scale, _case(mode), slack)


def verletStaleFlag(positions, supports, adjacency, domain, verletScale,
                    supportMode=SupportScheme.SuperSymmetric, rebuildSlack=0.):
    _case(supportMode)
    if not _same_context(adjacency.searchContext, _context(domain, supportMode)):
        return torch.ones((), dtype=torch.bool, device=positions.device)
    return _metrics(positions, supports, adjacency, domain, verletScale, supportMode, rebuildSlack)[0]


def verletUsage(positions, supports, adjacency, domain, verletScale,
                supportMode=SupportScheme.SuperSymmetric):
    _, motion, drift, _, buffer = _metrics(positions, supports, adjacency, domain, verletScale, supportMode, 0.)
    return (motion + drift) / buffer


def _storage(neighbors, counts, offsets, query, reference, capacity, reuse, storeI):
    """Apply capacity policy using scalar sizes already known by the builder."""
    edges = neighbors.numel()
    compatible = (reuse is not None and reuse.numRows == len(query.positions)
                  and reuse.numCols == len(reference.positions) and reuse.paddedJ.device == neighbors.device)
    previous = reuse.paddedJ.numel() if compatible else None
    size = edges if capacity is None else capacity.capacityFor(edges, previous)
    if size < edges:
        raise ValueError('Capacity must cover all live edges')
    inplace = (compatible and capacity is not None and capacity.inPlace and previous == size
               and (reuse.paddedI is not None) == storeI)
    if inplace:
        target = reuse.paddedJ
        target[:edges].copy_(neighbors)
        target[edges:].fill_(-1)
        reuse.numNeighbors.copy_(counts)
        reuse.edgeOffsets.copy_(offsets)
        counts, offsets = reuse.numNeighbors, reuse.edgeOffsets
    else:
        target = torch.full((size,), -1, dtype=neighbors.dtype, device=neighbors.device)
        target[:edges].copy_(neighbors)
    rows = None
    if storeI:
        rows = reuse.paddedI if inplace else torch.full_like(target, -1)
        live_rows = torch.repeat_interleave(torch.arange(len(counts), dtype=rows.dtype, device=rows.device),
                                           counts.long(), output_size=edges)
        rows[:edges].copy_(live_rows)
        rows[edges:].fill_(-1)
    return AdjacencyList(rows, target, counts, offsets, len(query.positions), len(reference.positions), numEdges=edges)


def _capture_check(positions):
    if positions.is_cuda and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError('Native Verlet build/filter allocates variable-size storage; use fixed adjacency during capture')


@torch.no_grad()
def buildVerletList(queryParticles, domain, verletScale=1., supportMode=SupportScheme.Gather,
                    priorNeighborhood=None, verbose=False, referenceParticles=None,
                    capacity=DEFAULT_CAPACITY, rebuildSlack=0., gridFactory=None, forceRebuild=False,
                    *, collection='auto', queryOrder='auto', storeI=True):
    _case(supportMode)
    if type(storeI) is not bool:
        raise ValueError('storeI must be a bool')
    if gridFactory is not None:
        raise NotImplementedError('Native Verlet owns its compact-hash index; Warp grid factories are not accepted')
    if collection not in ('auto', 'split') or queryOrder not in ('auto', 'input'):
        raise ValueError('Native Verlet uses direct CSR collection in input row order')
    reference = referenceParticles if referenceParticles is not None else queryParticles
    prior = priorNeighborhood
    snapshots = prior is not None and all(getattr(prior, field) is not None for field in (
        'queryPositions', 'referencePositions', 'querySupports', 'referenceSupports'))
    compatible = (snapshots and prior.queryPositions.shape == queryParticles.positions.shape
                  and prior.referencePositions.shape == reference.positions.shape
                  and prior.queryPositions.dtype == queryParticles.positions.dtype
                  and prior.referencePositions.dtype == reference.positions.dtype
                  and prior.queryPositions.device == queryParticles.positions.device
                  and prior.referencePositions.device == reference.positions.device)
    if compatible and _same_context(prior.searchContext, _context(domain, supportMode)) and not forceRebuild and (prior.paddedI is not None) == storeI:
        flag = verletValidityMetrics(queryParticles.positions, reference.positions,
            prior.queryPositions, prior.referencePositions, queryParticles.supports, reference.supports,
            prior.querySupports, prior.referenceSupports, domain.periodic, domain.min, domain.max,
            verletScale, _case(supportMode), rebuildSlack)[0]
        deferred = _DEFERRED.get()
        if deferred is not None:
            deferred.append(flag)
            return prior
        if not bool(flag):
            return prior
    _capture_check(queryParticles.positions)
    query = replace(queryParticles, supports=queryParticles.supports * verletScale)
    references = query if reference is queryParticles else replace(reference, supports=reference.supports * verletScale)
    adjacency = radiusSearch(query, domain, references, supportMode)
    result = _storage(adjacency.paddedJ, adjacency.numNeighbors, adjacency.edgeOffsets,
                      query, references, capacity, prior if compatible else None, storeI)
    result.searchContext = _context(domain, supportMode)
    result.queryPositions = query.positions.clone()
    result.referencePositions = result.queryPositions if reference is queryParticles else references.positions.clone()
    result.querySupports = query.supports.clone()
    result.referenceSupports = result.querySupports if reference is queryParticles else references.supports.clone()
    if verbose:
        print(f'Built native Verlet CSR: {result.numRows} rows, {result.numEdges} edges, {result.paddedJ.numel()} capacity')
    return result


@torch.no_grad()
def filterVerletList(queryParticles, domain, adjacency, supportMode=SupportScheme.Gather,
                     referenceParticles=None, capacity=DEFAULT_CAPACITY, *, storeI=None):
    _case(supportMode)
    storeI = adjacency.paddedI is not None if storeI is None else storeI
    if type(storeI) is not bool:
        raise ValueError('storeI must be a bool')
    reference = referenceParticles if referenceParticles is not None else queryParticles
    _capture_check(queryParticles.positions)
    from ...native._build import build
    from ...native._inputs import prepare_adjacency, prepare_geometry
    # Reuse metadata checks; this does not scan CSR contents or particle data.
    if adjacency is None:
        raise ValueError('Verlet filtering requires an existing candidate list')
    source = prepare_adjacency(queryParticles, reference, domain, None, adjacency)
    neighbors, counts, offsets = build().verlet_filter(
        list(prepare_geometry(queryParticles, reference, domain, allow_reverse=True)),
        source.paddedJ, source.numNeighbors, source.edgeOffsets, supportMode.value)
    result = _storage(neighbors, counts, offsets, queryParticles, reference, capacity, None, storeI)
    result.queryPositions, result.referencePositions = queryParticles.positions, reference.positions
    result.querySupports, result.referenceSupports = queryParticles.supports, reference.supports
    return result
