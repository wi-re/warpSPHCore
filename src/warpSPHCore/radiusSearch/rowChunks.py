"""CPU reference planner for whole-row neighbour storage (no pair allocation).

This is deliberately separate from CUDA reduction lanes and the flat search API.
Counts must describe the *stored* list, including Verlet padding in support.
The caller supplies a signature covering particle ordering, reference fields,
supports, domain, grid configuration and acceptance rules for one rebuild.
Device scans/collection and production configuration are subsequent work.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json

import torch

from .csr import WARP_ARRAY_MAX, validateParticleCounts


def _integer(name: str, value: int, minimum: int = 0) -> None:
    if type(value) is not int:
        raise TypeError(f'{name} must be a Python integer.')
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}.')


@dataclass(frozen=True, slots=True)
class RowChunkGeneration:
    number: int
    numReferences: int
    configSignature: str

    def __post_init__(self):
        _integer('generation number', self.number)
        _integer('numReferences', self.numReferences)
        validateParticleCounts(0, self.numReferences)
        if not isinstance(self.configSignature, str) or not self.configSignature:
            raise ValueError('A nonempty rebuild configuration signature is required.')


@dataclass(frozen=True, slots=True)
class RowChunkPolicy:
    """Bounds apply to padded capacity, never just live edges.

    bytesPerEdge covers all colocated pair buffers (e.g. J plus retained W).
    maxElementsPerEdge bounds the largest scalar array extent per edge; use
    dimension for a [capacity, dimension] scalar work tensor. This conservative
    limit keeps both extents and flattened scalar addressing within int32.
    Headroom is an exact rational fraction; capacity is rounded UP, then aligned.
    Zero-edge chunks have zero capacity. Oversized rows fail without truncation.
    """
    edgeBudget: int | None = None
    byteBudget: int | None = None
    bytesPerEdge: int = 4
    maxElementsPerEdge: int = 1
    headroomNumerator: int = 1
    headroomDenominator: int = 10
    alignment: int = 1
    maxRows: int | None = None

    def __post_init__(self):
        if self.edgeBudget is None and self.byteBudget is None:
            raise ValueError('Specify an edgeBudget or byteBudget for row chunks.')
        for name in ('edgeBudget', 'byteBudget', 'maxRows'):
            value = getattr(self, name)
            if value is not None:
                _integer(name, value, 1)
        for name in ('bytesPerEdge', 'maxElementsPerEdge', 'headroomDenominator', 'alignment'):
            _integer(name, getattr(self, name), 1)
        _integer('headroomNumerator', self.headroomNumerator)

    @property
    def capacityLimit(self) -> int:
        limit = WARP_ARRAY_MAX // self.maxElementsPerEdge
        if self.edgeBudget is not None:
            limit = min(limit, self.edgeBudget)
        if self.byteBudget is not None:
            limit = min(limit, self.byteBudget // self.bytesPerEdge)
        return limit

    def capacityFor(self, liveEdges: int) -> int:
        _integer('liveEdges', liveEdges)
        d = self.headroomDenominator
        padded = (liveEdges * (d + self.headroomNumerator) + d - 1) // d
        return ((padded + self.alignment - 1) // self.alignment) * self.alignment


@dataclass(frozen=True, slots=True)
class RowChunk:
    rowStart: int
    rowStop: int
    edgeStart: int
    liveEdges: int
    capacity: int
    owner: int

    @property
    def rowCount(self) -> int:
        return self.rowStop - self.rowStart


@dataclass(frozen=True, slots=True)
class RowChunkPlan:
    generation: RowChunkGeneration
    policy: RowChunkPolicy
    numOwners: int
    prefix: tuple[int, ...]
    countingRanges: tuple[tuple[int, int], ...]
    chunks: tuple[RowChunk, ...]
    signature: str
    ownershipPolicy: str = 'contiguous-balanced-chunks-v1'

    @property
    def numRows(self) -> int:
        return len(self.prefix) - 1

    @property
    def totalEdges(self) -> int:
        return self.prefix[-1]

    @property
    def ownerRanges(self) -> tuple[tuple[int, int], ...]:
        """Contiguous particle ownership derived from compute chunks, not counts."""
        cursor, ranges = 0, []
        for owner in range(self.numOwners):
            owned = tuple(c for c in self.chunks if c.owner == owner)
            stop = owned[-1].rowStop if owned else cursor
            ranges.append((cursor, stop))
            cursor = stop
        return tuple(ranges)

    def globalOffsets(self) -> torch.Tensor:
        """Fresh CPU int64 P[N+1], independent of the flat offset default."""
        return torch.tensor(self.prefix, dtype=torch.int64)

    def localOffsets(self, chunkIndex: int) -> torch.Tensor:
        chunk = self.chunks[chunkIndex]
        # Subtract in wide host arithmetic BEFORE constructing any local view.
        return torch.tensor(tuple(p - chunk.edgeStart for p in
                                  self.prefix[chunk.rowStart:chunk.rowStop]), dtype=torch.int64)

    def validateSignature(self, expected: str) -> None:
        """Check peer/count/collect agreement before local pair allocation."""
        if self.signature != expected:
            raise ValueError('Row-chunk generation/configuration/partition signature mismatch.')


def _balancedRanges(size: int, owners: int) -> tuple[tuple[int, int], ...]:
    width, extra = divmod(size, owners)
    return tuple((i * width + min(i, extra),
                  (i + 1) * width + min(i + 1, extra)) for i in range(owners))


def buildRowChunkPlan(counts: torch.Tensor, *, generation: RowChunkGeneration,
                      policy: RowChunkPolicy, numOwners: int = 1) -> RowChunkPlan:
    """Greedily pack maximal whole-row intervals under explicit capacity bounds.

    CPU reference only: no implicit CUDA synchronization/copy. Counts are the
    already assembled int32 C[N], from disjoint countingRanges, not a sum of
    duplicated full counts. Compute owners receive balanced contiguous numbers
    of chunks, independently of the equal-row counting assignment. Empty owners
    own no chunks; zero-neighbour particles still belong to exactly one chunk.
    Python integers hold the immutable reference prefix; all tensor offsets are
    explicitly int64. No arrays proportional to the number of edges are made.
    """
    if not isinstance(generation, RowChunkGeneration) or not isinstance(policy, RowChunkPolicy):
        raise TypeError('Expected RowChunkGeneration and RowChunkPolicy metadata.')
    _integer('numOwners', numOwners, 1)
    if counts.ndim != 1 or counts.dtype != torch.int32:
        raise TypeError('Neighbor counts must be a one-dimensional torch.int32 tensor.')
    if counts.device.type != 'cpu':
        raise ValueError('The CPU reference planner requires CPU counts explicitly.')
    validateParticleCounts(len(counts), generation.numReferences)
    prefix = [0]
    for count in counts.tolist():
        if count < 0 or count > generation.numReferences:
            raise ValueError('Each stored row count must be between zero and numReferences.')
        prefix.append(prefix[-1] + count)
    if prefix[-1] > torch.iinfo(torch.int64).max:
        raise OverflowError('Global row-chunk prefix does not fit int64.')

    intervals = []
    start = 0
    for row in range(len(counts)):
        rowEdges = prefix[row + 1] - prefix[row]
        if policy.capacityFor(rowEdges) > policy.capacityLimit:
            raise OverflowError(f'Row {row} with {rowEdges} stored edges exceeds the padded chunk capacity limit {policy.capacityLimit}; increase the budget or reduce headroom. Rows cannot be split.')
        candidate = prefix[row + 1] - prefix[start]
        rowLimit = policy.maxRows is not None and row - start >= policy.maxRows
        if row > start and (rowLimit or policy.capacityFor(candidate) > policy.capacityLimit):
            intervals.append((start, row))
            start = row
    if start < len(counts):
        intervals.append((start, len(counts)))

    chunks = []
    for owner, (first, stop) in enumerate(_balancedRanges(len(intervals), numOwners)):
        for start, end in intervals[first:stop]:
            live = prefix[end] - prefix[start]
            chunks.append(RowChunk(start, end, prefix[start], live, policy.capacityFor(live), owner))
    countingRanges = _balancedRanges(len(counts), numOwners)
    ownershipPolicy = 'contiguous-balanced-chunks-v1'
    digest = sha256()
    metadata = dict(generation=asdict(generation), policy=asdict(policy),
                    numOwners=numOwners, ownershipPolicy=ownershipPolicy,
                    countingRanges=countingRanges, chunks=[asdict(c) for c in chunks])
    digest.update(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode('utf-8'))
    # Include every row boundary, not just per-chunk totals. Stream the prefix
    # into the digest to avoid a second full JSON encoding of a large count list.
    for offset in prefix:
        digest.update(offset.to_bytes(8, byteorder='little', signed=True))
    return RowChunkPlan(generation, policy, numOwners, tuple(prefix), countingRanges,
                        tuple(chunks), digest.hexdigest(), ownershipPolicy)
