"""Opt-in eager whole-row neighbour storage, using the existing search kernels.

Only query positions/supports and CSR row metadata are sliced. References and J
particle IDs stay global. This API is deliberately distinct from AdjacencyList:
there is no implicit global J/I view or production operator fallback. The host
reference planner synchronizes counts at rebuild time; this is not a device-only
planner or a graph/AD backend. An explicitly bound c10d group enables
distributed counts, local collection and replicated particle-output exchange.
"""
from dataclasses import asdict, dataclass, field, fields
from hashlib import sha256
import json
import math
import sys

import torch
import warp as wp

from ..dataTypes import AdjacencyList, DomainDescription, ParticleState
from ..enumTypes import SupportScheme
from ..type_config import (get_torch_index_dtype, get_torch_offset_dtype,
                           get_torch_precision, index_t, offset_t, scalar_t)
from ..util import castTorchToWarp, castTorchToWarpAsBuiltins
from .compactHash.buildHashmap import buildCompactHashMap
from .compactHash.wp_countNeighbors import radiusSearchCountNeighborsCompactHashMap
from .compactHash.wp_collectNeighbors import radiusSearchCollectJOnly
from .csr import validateParticleCounts
from .rowChunks import RowChunk, RowChunkGeneration, RowChunkPlan, RowChunkPolicy, buildRowChunkPlan
from .verlet.build import _DEFERRED_CHECKS
from .verlet.wp_countVerlet import countNeighborsVerlet
from .verlet.wp_updateVerlet import updateNeighborsVerlet
from .verlet.wp_validity import verletValidityMetrics

__all__ = ['OwnedRowChunk', 'ChunkedAdjacency', 'radiusSearchRowChunks_',
           'radiusSearchRowChunks', 'buildChunkedVerletList_', 'buildChunkedVerletList',
           'chunkedVerletStaleFlag_', 'filterChunkedVerletList_', 'filterChunkedVerletList']


@wp.kernel
def _fillGlobalRows(counts: wp.array(dtype=wp.int32), offsets: wp.array(dtype=offset_t),
                    rowStart: wp.int32, rows: wp.array(dtype=index_t)):
    row, lane = wp.tid()
    p = wp.int64(lane)
    while p < wp.int64(counts[row]):
        rows[offsets[row] + offset_t(p)] = index_t(rowStart + row)
        p += wp.int64(32)


@dataclass(frozen=True, slots=True)
class OwnedRowChunk:
    metadata: RowChunk
    adjacency: AdjacencyList


@dataclass(slots=True)
class _StorageLease:
    valid: bool = True


@dataclass(frozen=True, slots=True)
class ChunkedAdjacency:
    """Iterate owned chunks explicitly; adjacency row numbers are chunk-local.

    Each chunk's optional stored I and all J values are GLOBAL particle IDs.
    Operator consumers must map local query rows with metadata.rowStart; these
    small AdjacencyLists must not be passed to unmodified global-row operators.
    Snapshots and the replicated C/P are read-only by contract. inPlace rebuilds
    explicitly obsolete old pair/CSR views, just like flat CapacityPolicy.inPlace.
    """
    plan: RowChunkPlan
    chunks: tuple[OwnedRowChunk, ...]
    owner: int
    numNeighbors: torch.Tensor
    globalPrefix: torch.Tensor
    queryPositions: torch.Tensor
    referencePositions: torch.Tensor
    querySupports: torch.Tensor
    referenceSupports: torch.Tensor
    domain: DomainDescription
    mode: SupportScheme
    storeI: bool
    verletScale: float
    orderingKey: str
    kind: str
    rebuildReason: str
    searchScratchBytes: int = 0
    sourceSignature: str | None = None
    communication: object | None = field(default=None, compare=False, repr=False)
    _storageLease: _StorageLease = field(default_factory=_StorageLease, compare=False, repr=False)

    def assertStorageValid(self):
        """An in-place rebuild may have overwritten this generation's CSR/J."""
        if not self._storageLease.valid:
            raise ValueError('Chunked adjacency storage was overwritten by an in-place rebuild; use the new generation.')

    def __iter__(self):
        return iter(self.chunks)

    @property
    def numRows(self):
        return self.plan.numRows

    @property
    def numCols(self):
        return self.plan.generation.numReferences

    @property
    def numEdges(self):
        """Global live edge count; this does not imply a global edge buffer."""
        return self.plan.totalEdges

    def memoryUsage(self) -> dict[str, int]:
        """Logical bytes, not allocator/process peaks; no hidden pair export."""
        pairs = [t for c in self for t in (c.adjacency.paddedI, c.adjacency.paddedJ) if t is not None]
        snapshots = [self.queryPositions, self.referencePositions, self.querySupports,
                     self.referenceSupports, self.domain.min, self.domain.max, self.domain.periodic]
        rows = [self.numNeighbors, self.globalPrefix, *(c.adjacency.edgeOffsets for c in self)]
        return dict(pairCapacityBytes=_storageBytes(pairs),
                    pairLiveBytes=sum(c.metadata.liveEdges for c in self)
                        * torch.empty((), dtype=get_torch_index_dtype()).element_size() * (1 + self.storeI),
                    pairBudgetBytes=sum(c.metadata.capacity for c in self) * self.plan.policy.bytesPerEdge,
                    rowMetadataBytes=_storageBytes(rows), snapshotBytes=_storageBytes(snapshots),
                    hostPrefixBytes=sys.getsizeof(self.plan.prefix) + sum(sys.getsizeof(p) for p in self.plan.prefix),
                    searchScratchBytes=self.searchScratchBytes)


def _storageBytes(tensors):
    storages = {(str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
                for t in tensors}
    return sum(storages.values())


def _stream(device):
    return wp.stream_from_torch(torch.cuda.current_stream(device)) if device.type == 'cuda' else None


def _checkInputs(q, r, qh, rh, domain, mode, policy, storeI):
    if not isinstance(policy, RowChunkPolicy):
        raise TypeError('policy must be a RowChunkPolicy.')
    if not isinstance(mode, SupportScheme):
        raise TypeError('mode must be a SupportScheme.')
    if type(storeI) is not bool:
        raise TypeError('storeI must be a bool.')
    if q.device.type not in ('cpu', 'cuda'):
        raise ValueError('Chunked search supports CPU and CUDA tensors.')
    if _DEFERRED_CHECKS[0] is not None or (q.is_cuda and torch.cuda.is_current_stream_capturing()):
        raise NotImplementedError('Chunked search does not support deferred/graph capture.')
    if any(t.requires_grad for t in (q, r, qh, rh)):
        raise NotImplementedError('Chunked search AD is not supported.')
    if q.ndim != 2 or r.ndim != 2 or q.shape[1] != r.shape[1] or q.shape[1] not in (2, 3):
        raise ValueError('Query/reference positions must have matching 2D or 3D coordinates.')
    if qh.shape != (len(q),) or rh.shape != (len(r),):
        raise ValueError('Each population requires one support per particle.')
    if any(t.device != q.device or t.dtype != get_torch_precision() for t in (q, r, qh, rh)):
        raise ValueError('Particle inputs must share the configured precision and device.')
    if domain.dim != q.shape[1] or any(not isinstance(t, torch.Tensor) or t.shape != (domain.dim,)
                                      or t.device != q.device for t in (domain.min, domain.max, domain.periodic)):
        raise ValueError('Chunked search requires explicit domain bounds/periodicity on the particle device.')
    if domain.min.dtype != q.dtype or domain.max.dtype != q.dtype or domain.periodic.dtype != torch.bool:
        raise ValueError('Domain bounds must use particle precision and periodicity must be bool.')
    if domain.min.requires_grad or domain.max.requires_grad:
        raise NotImplementedError('Chunked search AD is not supported.')
    validateParticleCounts(len(q), len(r))
    pairBytes = torch.empty((), dtype=get_torch_index_dtype()).element_size() * (1 + storeI)
    if policy.bytesPerEdge < pairBytes:
        raise ValueError('bytesPerEdge must include J and optional stored I.')


def _config(q, r, domain, mode, policy, storeI, scale, orderingKey, numOwners, owner, kind):
    if not isinstance(orderingKey, str):
        raise TypeError('orderingKey must be a string identifying particle ordering changes.')
    data = dict(numRows=len(q), numCols=len(r), dim=domain.dim, dtype=str(q.dtype),
                deviceType=q.device.type, mode=mode.name, policy=asdict(policy), storeI=storeI,
                scale=float(scale), orderingKey=orderingKey, numOwners=numOwners, kind=kind,
                minimum=domain.min.tolist(), maximum=domain.max.tolist(), periodic=domain.periodic.tolist())
    if any(not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi
           for lo, hi in zip(data['minimum'], data['maximum'])):
        raise ValueError('Domain bounds must be finite and strictly increasing.')
    # Owner is not part of replica agreement; its local selection is recorded on
    # the container, while every owner must derive the SAME global plan digest.
    return sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _snapshots(q, r, qh, rh, domain):
    memo = {}
    def clone(t):
        key = (t.data_ptr(), t.shape, t.stride(), t.dtype, t.device)
        if key not in memo:
            memo[key] = t.detach().clone().contiguous()
        return memo[key]
    return (clone(q), clone(r), clone(qh), clone(rh),
            DomainDescription(clone(domain.min), clone(domain.max), clone(domain.periodic), domain.dim))


def _plan(counts, generationNumber, config, policy, numReferences, numOwners, expectedSignature):
    plan = buildRowChunkPlan(counts.cpu(), generation=RowChunkGeneration(generationNumber, numReferences, config),
                             policy=policy, numOwners=numOwners)
    if expectedSignature is not None:
        plan.validateSignature(expectedSignature)
    # The host planner is the Phase1 reference. Only counts cross to the host;
    # no pair list or particle fields are copied for planning.
    prefix = torch.empty(len(counts) + 1, dtype=torch.int64, device=counts.device)
    prefix[0] = 0
    torch.cumsum(counts, 0, dtype=torch.int64, out=prefix[1:])
    return plan, prefix


def _allocateChunks(plan, counts, prefix, owner, storeI, reuse, inPlace):
    old = {(c.metadata.rowStart, c.metadata.rowStop): c.adjacency for c in reuse} if reuse is not None else {}
    chunks = []
    for metadata in plan.chunks:
        if metadata.owner != owner:
            continue
        a, b = metadata.rowStart, metadata.rowStop
        # Always subtract the GLOBAL base in int64 before narrowing local offsets.
        offsets = (prefix[a:b] - prefix[a]).to(get_torch_offset_dtype())
        prior = old.get((a, b))
        compatible = (inPlace and prior is not None and prior.paddedJ.numel() == metadata.capacity
                      and prior.paddedJ.device == counts.device and prior.paddedJ.dtype == get_torch_index_dtype()
                      and (prior.paddedI is not None) == storeI
                      and (not storeI or (prior.paddedI.shape == prior.paddedJ.shape
                                          and prior.paddedI.device == counts.device
                                          and prior.paddedI.dtype == get_torch_index_dtype()))
                      and prior.edgeOffsets.shape == offsets.shape and prior.edgeOffsets.dtype == offsets.dtype
                      and prior.edgeOffsets.device == counts.device)
        if compatible:
            reuse._storageLease.valid = False
        j = prior.paddedJ if compatible else torch.empty(metadata.capacity, dtype=get_torch_index_dtype(), device=counts.device)
        i = (prior.paddedI if compatible else torch.empty_like(j)) if storeI else None
        j.fill_(-1)
        if i is not None:
            i.fill_(-1)
        if compatible:
            prior.edgeOffsets.copy_(offsets)
            offsets = prior.edgeOffsets
        chunks.append(OwnedRowChunk(metadata, AdjacencyList(i, j, counts[a:b], offsets,
                                                          b - a, plan.generation.numReferences, numEdges=metadata.liveEdges)))
    return tuple(chunks)


def _fillRows(chunks, device):
    for chunk in chunks:
        m, adj = chunk.metadata, chunk.adjacency
        if adj.paddedI is not None and m.liveEdges:
            wp.launch(_fillGlobalRows, dim=(m.rowCount, 32), block_dim=128,
                      inputs=[castTorchToWarp(adj.numNeighbors), castTorchToWarp(adj.edgeOffsets),
                              m.rowStart, castTorchToWarp(adj.paddedI)], device=str(device), stream=_stream(device))


def _search(q, r, qh, rh, domain, *, policy, mode, storeI, numOwners, owner,
            generationNumber, reuse, inPlace, expectedSignature, orderingKey, kind, scale, reason, communication=None):
    local = (lambda tag, fn: fn()) if communication is None else communication.localPhase
    def inputs():
        _checkInputs(q, r, qh, rh, domain, mode, policy, storeI)
        if communication is not None and q.device != communication.device:
            raise ValueError('Search tensors must use the process group device.')
    local('search:inputs', inputs)
    if type(numOwners) is not int or numOwners < 1 or type(owner) is not int or not 0 <= owner < numOwners:
        raise ValueError('owner must identify one of numOwners positive logical owners.')
    if reuse is not None:
        reuse.assertStorageValid()
    config = local('search:config', lambda: _config(q, r, domain, mode, policy, storeI, scale, orderingKey, numOwners, owner, kind))
    if communication is not None:
        communication.agree('search:configuration', (config, generationNumber, expectedSignature))
    q, r, qh, rh, domain = local('search:snapshots', lambda: _snapshots(q, r, qh, rh, domain))
    if communication is not None:
        communication.assertReplicated('search:replicas', dict(q=q, r=r, qh=qh, rh=rh))
    counts = local('search:count-allocation', lambda: torch.zeros(len(q), dtype=torch.int32, device=q.device))
    def count():
        grid, scratch = None, 0
        if not len(q) or not len(r):
            return grid, scratch, None, None

        grid = buildCompactHashMap(q, r, qh, rh, domain.periodic, domain, mode, len(q) + 1)
        y = torch.vstack([part if not periodic else torch.remainder(part - domain.min[d], domain.max[d] - domain.min[d]) + domain.min[d]
                          for d, (part, periodic) in enumerate(zip(q.mT, domain.periodic.tolist()))]).mT.contiguous()
        shared = [castTorchToWarp(t) for t in (grid.sortedPositions, grid.sortedSupports,
                  grid.hashTable, grid.sortedCellTable, grid.qMin)]
        shared += [scalar_t(grid.hCell), grid.D, castTorchToWarp(grid.cellOffsets), castTorchToWarp(grid.numCells),
                   castTorchToWarp(grid.qMax), castTorchToWarp(grid.qMin), castTorchToWarp(domain.periodic), wp.uint32(mode.value)]
        # Counting partitions and compute partitions are independent. The slice
        # base is explicit here; each kernel addresses all query fields locally.
        width, extra = divmod(len(q), numOwners)
        ranks = range(numOwners) if communication is None else (owner,)
        for rank in ranks:
            a, b = rank * width + min(rank, extra), (rank + 1) * width + min(rank + 1, extra)
            if a < b:
                wp.launch(radiusSearchCountNeighborsCompactHashMap, dim=b-a,
                          inputs=[castTorchToWarp(y[a:b]), castTorchToWarp(qh[a:b]), *shared, castTorchToWarp(counts[a:b])],
                          device=str(q.device), stream=_stream(q.device))
        scratch = _storageBytes([y, *(getattr(grid, f.name) for f in fields(grid) if isinstance(getattr(grid, f.name), torch.Tensor))])
        return grid, scratch, y, shared
    grid, scratch, y, shared = local('search:count', count)
    if communication is not None:
        from .rowChunks import _balancedRanges
        communication.exchangeRows(counts, _balancedRanges(len(q), numOwners),
                                   tag='search:counts', signature=config)
        if grid is not None:
            communication.assertReplicated('search:grid-order', dict(sortIndex=grid.sortIndex))
    plan, prefix = local('search:plan', lambda: _plan(counts, generationNumber, config, policy, len(r), numOwners, expectedSignature))
    if communication is not None:
        communication.agree('search:partition', plan.signature)
    chunks = local('search:allocate-owned', lambda: _allocateChunks(plan, counts, prefix, owner, storeI, reuse, inPlace))
    def collect():
        if grid is None:
            return
        for chunk in chunks:
            m, adj = chunk.metadata, chunk.adjacency
            if m.liveEdges:
                wp.launch(radiusSearchCollectJOnly, dim=m.rowCount,
                          inputs=[castTorchToWarp(y[m.rowStart:m.rowStop]), castTorchToWarp(qh[m.rowStart:m.rowStop]), *shared,
                                  castTorchToWarp(adj.numNeighbors), castTorchToWarp(adj.edgeOffsets), castTorchToWarp(grid.sortIndex),
                                  wp.array(dtype=index_t, device=str(q.device)), castTorchToWarp(adj.paddedJ)],
                          device=str(q.device), stream=_stream(q.device))
    local('search:collect-owned', collect)
    local('search:fill-I', lambda: _fillRows(chunks, q.device))
    return ChunkedAdjacency(plan, chunks, owner, counts, prefix, q, r, qh, rh, domain, mode,
                            storeI, scale, orderingKey, kind, reason, scratch, communication=communication)


def _bindCommunication(communication, prior):
    previous = getattr(prior, 'communication', None)
    if communication is None:
        communication = previous
    if communication is not None:
        from ..distributed import RowChunkProcessGroup
        if not isinstance(communication, RowChunkProcessGroup):
            raise TypeError('communication must be an explicit RowChunkProcessGroup.')
        if prior is not None:
            communication.localPhase('bind:prior', lambda: communication.validateAdjacency(prior))
    return communication


def radiusSearchRowChunks_(queryPositions, referencePositions, querySupports, referenceSupports, domain,
                           *, policy, mode=SupportScheme.Gather, storeI=False, numOwners=1, owner=0,
                           reuse=None, inPlace=False, generationNumber=None, expectedSignature=None, orderingKey='input', communication=None):
    """Count disjoint intervals, plan whole rows, allocate/collect ONLY this owner.

    Without communication, logical owners are a single-device sequencing fixture.
    With an explicit RowChunkProcessGroup, only this rank counts its interval;
    counts/plan agreement precede owned pair allocation and collection.
    inPlace reuses only exact row intervals with matching planned capacities;
    views of the replaced list are then obsolete. Default rebuilds preserve them.
    """
    communication = _bindCommunication(communication, reuse)
    local = (lambda tag, fn: fn()) if communication is None else communication.localPhase
    def validate():
        if reuse is not None and not isinstance(reuse, ChunkedAdjacency):
            raise TypeError('reuse must be a ChunkedAdjacency.')
        if type(inPlace) is not bool:
            raise TypeError('inPlace must be a bool.')
        number = (0 if reuse is None else reuse.plan.generation.number + 1) if generationNumber is None else generationNumber
        if reuse is not None and number <= reuse.plan.generation.number:
            raise ValueError('A rebuild must advance the generation number.')
        return number
    generationNumber = local('search:arguments', validate)
    if communication is not None:
        numOwners, owner = communication.size, communication.rank
    return _search(queryPositions, referencePositions, querySupports, referenceSupports, domain,
                   policy=policy, mode=mode, storeI=storeI, numOwners=numOwners, owner=owner, reuse=reuse,
                   inPlace=inPlace, generationNumber=generationNumber, expectedSignature=expectedSignature,
                   orderingKey=orderingKey, kind='search', scale=1.0, reason='explicit search', communication=communication)


def chunkedVerletStaleFlag_(queryPositions, referencePositions, querySupports, referenceSupports,
                           domain, adjacency, *, mode=None, verletScale=None, orderingKey=None, rebuildSlack=0.0):
    """Use the existing validity reduction; configuration changes always rebuild."""
    if not isinstance(adjacency, ChunkedAdjacency):
        raise TypeError('adjacency must be a ChunkedAdjacency.')
    adjacency.assertStorageValid()
    mode = adjacency.mode if mode is None else mode
    _checkInputs(queryPositions, referencePositions, querySupports, referenceSupports, domain,
                 mode, adjacency.plan.policy, adjacency.storeI)
    scale = adjacency.verletScale if verletScale is None else verletScale
    orderingKey = adjacency.orderingKey if orderingKey is None else orderingKey
    if queryPositions.device != adjacency.queryPositions.device or referencePositions.device != adjacency.referencePositions.device:
        return torch.ones((), dtype=torch.bool, device=queryPositions.device)
    config = _config(queryPositions, referencePositions, domain, mode, adjacency.plan.policy, adjacency.storeI,
                     scale, orderingKey, adjacency.plan.numOwners, adjacency.owner, adjacency.kind)
    if config != adjacency.plan.generation.configSignature:
        return torch.ones((), dtype=torch.bool, device=queryPositions.device)
    case = 0 if mode == SupportScheme.Gather else 1 if mode == SupportScheme.Scatter else 2
    return verletValidityMetrics(queryPositions, referencePositions, adjacency.queryPositions, adjacency.referencePositions,
                                  querySupports, referenceSupports, adjacency.querySupports, adjacency.referenceSupports,
                                  domain.periodic, domain.min, domain.max, scale, case, rebuildSlack)[0]


def buildChunkedVerletList_(queryPositions, referencePositions, querySupports, referenceSupports, domain,
                            *, policy, verletScale=1.0, mode=SupportScheme.SuperSymmetric, priorNeighborhood=None,
                            storeI=False, inPlace=False, forceRebuild=False, rebuildSlack=0.0, orderingKey='input', communication=None):
    """Retain resident chunks until existing motion/support validity requests rebuild.

    Eager scalar-scale API. Deferred checks, graph capture
    and AD require later integration. An explicit process group coordinates
    distributed validity, counts, planning and owned collection.
    """
    communication = _bindCommunication(communication, priorNeighborhood)
    local = (lambda tag, fn: fn()) if communication is None else communication.localPhase
    numOwners, owner = (1, 0) if communication is None else (communication.size, communication.rank)
    def validate():
        _checkInputs(queryPositions, referencePositions, querySupports, referenceSupports, domain, mode, policy, storeI)
        if isinstance(verletScale, torch.Tensor) or isinstance(verletScale, bool) or not isinstance(verletScale, (int, float)):
            raise TypeError('Chunked eager Verlet currently requires a scalar Python verletScale.')
        if not math.isfinite(verletScale) or verletScale < 1:
            raise ValueError('verletScale must be finite and at least one.')
        if type(inPlace) is not bool or type(forceRebuild) is not bool:
            raise TypeError('inPlace and forceRebuild must be bool.')
        if not 0 <= rebuildSlack < 1:
            raise ValueError('rebuildSlack must be in [0, 1).')
        prior = priorNeighborhood
        if prior is not None:
            if not isinstance(prior, ChunkedAdjacency):
                raise TypeError('priorNeighborhood must be a ChunkedAdjacency.')
            prior.assertStorageValid()
        return _config(queryPositions, referencePositions, domain, mode, policy, storeI, float(verletScale),
                       orderingKey, numOwners, owner, 'verlet')
    config = local('verlet:inputs', validate)
    prior = priorNeighborhood
    if communication is not None:
        communication.agree('verlet:configuration', (config, rebuildSlack, inPlace,
                             None if prior is None else prior.plan.signature))
        communication.assertReplicated('verlet:replicas', dict(q=queryPositions, r=referencePositions,
                                      qh=querySupports, rh=referenceSupports))
    compatible = (prior is not None and config == prior.plan.generation.configSignature and prior.kind == 'verlet'
                  and queryPositions.device == prior.queryPositions.device
                  and referencePositions.device == prior.referencePositions.device)
    reason = 'initial build' if prior is None else 'configuration changed'
    def stale():
        return forceRebuild or not compatible or bool(chunkedVerletStaleFlag_(
            queryPositions, referencePositions, querySupports, referenceSupports,
            domain, prior, rebuildSlack=rebuildSlack).item())
    rebuild = local('verlet:validity', stale)
    if communication is not None:
        rebuild = communication.any(rebuild, tag='verlet:global-rebuild')
    if not rebuild:
        return prior
    if compatible:
        reason = 'coordinated rebuild' if communication is not None else ('forced rebuild' if forceRebuild else 'Verlet validity exceeded')
    def scaleSupports():
        qh = querySupports * float(verletScale)
        rh = qh if querySupports is referenceSupports else referenceSupports * float(verletScale)
        return qh, rh
    qh, rh = local('verlet:scale', scaleSupports)
    return _search(queryPositions, referencePositions, qh, rh, domain, policy=policy, mode=mode, storeI=storeI,
                   numOwners=numOwners, owner=owner, reuse=prior, inPlace=inPlace,
                   generationNumber=0 if prior is None else prior.plan.generation.number + 1,
                   expectedSignature=None, orderingKey=orderingKey, kind='verlet', scale=float(verletScale),
                   reason=reason, communication=communication)


def filterChunkedVerletList_(queryPositions, referencePositions, querySupports, referenceSupports, domain,
                             adjacency, *, policy=None, storeI=None, expectedSignature=None):
    """Filter resident J directly into newly planned chunks, preserving row order.

    Source/destination boundaries may differ. Each overlap uses local CSR slices
    and global reference fields; no concatenated source or destination exists.
    A stale source fails before collection. Filtering requires complete ownership;
    distributed filtering/pair migration remains unsupported.
    """
    if not isinstance(adjacency, ChunkedAdjacency) or adjacency.kind != 'verlet' or adjacency.plan.numOwners != 1:
        raise ValueError('Filtering requires a complete single-owner chunked Verlet list.')
    policy = adjacency.plan.policy if policy is None else policy
    storeI = adjacency.storeI if storeI is None else storeI
    _checkInputs(queryPositions, referencePositions, querySupports, referenceSupports, domain, adjacency.mode, policy, storeI)
    if bool(chunkedVerletStaleFlag_(queryPositions, referencePositions, querySupports, referenceSupports, domain, adjacency).item()):
        raise ValueError('Rebuild the stale chunked Verlet list before filtering.')
    q, r, qh, rh, domain = _snapshots(queryPositions, referencePositions, querySupports, referenceSupports, domain)
    counts = torch.zeros(len(q), dtype=torch.int32, device=q.device)
    def arguments(a, b, source):
        adj, m = source.adjacency, source.metadata
        return [castTorchToWarpAsBuiltins(q[a:b]), castTorchToWarpAsBuiltins(r),
                castTorchToWarp(qh[a:b]), castTorchToWarp(rh), castTorchToWarp(domain.min),
                castTorchToWarp(domain.max), castTorchToWarp(domain.periodic), wp.uint32(adjacency.mode.value),
                castTorchToWarp(adj.paddedJ), castTorchToWarp(adj.edgeOffsets[a-m.rowStart:b-m.rowStart]),
                castTorchToWarp(adj.numNeighbors[a-m.rowStart:b-m.rowStart])]
    for source in adjacency:
        m = source.metadata
        wp.launch(countNeighborsVerlet, dim=m.rowCount,
                  inputs=[*arguments(m.rowStart, m.rowStop, source), castTorchToWarp(counts[m.rowStart:m.rowStop])],
                  device=str(q.device), stream=_stream(q.device))
    config = _config(q, r, domain, adjacency.mode, policy, storeI, 1.0, adjacency.orderingKey, 1, 0, 'filtered')
    config = sha256((config + adjacency.plan.signature).encode()).hexdigest()
    plan, prefix = _plan(counts, adjacency.plan.generation.number, config, policy, len(r), 1, expectedSignature)
    chunks = _allocateChunks(plan, counts, prefix, 0, storeI, None, False)
    first = 0
    for target in chunks:
        tm, ta = target.metadata, target.adjacency
        while first < len(adjacency.chunks) and adjacency.chunks[first].metadata.rowStop <= tm.rowStart:
            first += 1
        cursor = first
        while cursor < len(adjacency.chunks) and adjacency.chunks[cursor].metadata.rowStart < tm.rowStop:
            source = adjacency.chunks[cursor]
            a, b = max(tm.rowStart, source.metadata.rowStart), min(tm.rowStop, source.metadata.rowStop)
            if a < b:
                wp.launch(updateNeighborsVerlet, dim=b-a,
                          inputs=[*arguments(a, b, source), castTorchToWarp(ta.edgeOffsets[a-tm.rowStart:b-tm.rowStart]),
                                  castTorchToWarp(ta.numNeighbors[a-tm.rowStart:b-tm.rowStart]),
                                  wp.array(dtype=index_t, device=str(q.device)), castTorchToWarp(ta.paddedJ), False],
                          device=str(q.device), stream=_stream(q.device))
            cursor += 1
    _fillRows(chunks, q.device)
    return ChunkedAdjacency(plan, chunks, 0, counts, prefix, q, r, qh, rh, domain, adjacency.mode,
                            storeI, 1.0, adjacency.orderingKey, 'filtered', 'filtered resident chunks',
                            sourceSignature=adjacency.plan.signature)


def _particleArgs(queryParticles, referenceParticles):
    r = queryParticles if referenceParticles is None else referenceParticles
    return queryParticles.positions, r.positions, queryParticles.supports, r.supports


def radiusSearchRowChunks(queryParticles: ParticleState, domain, *, referenceParticles=None, **kwargs):
    return radiusSearchRowChunks_(*_particleArgs(queryParticles, referenceParticles), domain, **kwargs)


def buildChunkedVerletList(queryParticles: ParticleState, domain, *, referenceParticles=None,
                           supportMode=SupportScheme.Gather, **kwargs):
    """ParticleState wrapper, matching buildVerletList's Gather default."""
    return buildChunkedVerletList_(*_particleArgs(queryParticles, referenceParticles), domain,
                                   mode=supportMode, **kwargs)


def filterChunkedVerletList(queryParticles: ParticleState, domain, adjacency, *, referenceParticles=None, **kwargs):
    return filterChunkedVerletList_(*_particleArgs(queryParticles, referenceParticles), domain, adjacency, **kwargs)
