"""Portable resident-search history: particle snapshots and counts, never pairs.

Restore replays the same deterministic compact-hash traversal on the frozen
search inputs. A new chunk budget/rank count changes storage ownership only.
This is an eager versioned recipe, not an AD/graph or arbitrary search importer.
"""
from dataclasses import asdict
from hashlib import sha256
import json
import math

import torch

from ..dataTypes import DomainDescription
from ..enumTypes import SupportScheme
from ..type_config import get_torch_precision
from .chunked import ChunkedAdjacency, _config, _search
from .rowChunks import RowChunkGeneration, RowChunkPolicy, buildRowChunkPlan

__all__ = ['rowChunkCheckpointRecipe', 'rowChunkCheckpointState', 'restoreRowChunkCheckpointState']

_ALGORITHM = 'compact-hash-input-split-v1'


def _digest(metadata, tensors):
    digest = sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
    for name, tensor in sorted(tensors.items()):
        digest.update(json.dumps((name, list(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor.detach().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def rowChunkCheckpointRecipe(*, queryPositions, referencePositions, querySupports,
        referenceSupports, counts, domain, policy, mode, storeI, verletScale,
        orderingKey='input', generation=0, sourceOwners=1):
    """Encode explicit frozen search inputs, e.g. from a legacy CSR checkpoint.

    No pair arrays are accepted. Restore validates this recipe and recounts the
    geometry before pair allocation; encoding alone is not validation.
    """
    metadata = dict(version=1, algorithm=_ALGORITHM, numRows=len(queryPositions),
        numCols=len(referencePositions), dim=domain.dim, generation=generation,
        totalEdges=int(counts.sum(dtype=torch.int64)), policy=asdict(policy), mode=mode.name,
        storeI=storeI, verletScale=verletScale, orderingKey=orderingKey,
        sourceOwners=sourceOwners, samePositions=queryPositions is referencePositions,
        sameSupports=querySupports is referenceSupports)
    tensors = dict(queryPositions=queryPositions, querySupports=querySupports,
        counts=counts, domainMin=domain.min, domainMax=domain.max, periodic=domain.periodic)
    if not metadata['samePositions']:
        tensors['referencePositions'] = referencePositions
    if not metadata['sameSupports']:
        tensors['referenceSupports'] = referenceSupports
    tensors = {name:t.detach().cpu().contiguous().clone() for name,t in tensors.items()}
    return dict(metadata=metadata, tensors=tensors, checksum=_digest(metadata, tensors))


def rowChunkCheckpointState(adjacency):
    """Independent CPU recipe; no per-edge allocation, transfer or serialization."""
    if not isinstance(adjacency, ChunkedAdjacency) or adjacency.kind != 'verlet':
        raise ValueError('Checkpoint history requires a resident chunked Verlet list.')
    adjacency.assertStorageValid()
    a = adjacency
    counts = a.numNeighbors.cpu().tolist()
    if tuple(counts) != tuple(stop-start for start,stop in zip(a.plan.prefix, a.plan.prefix[1:])):
        raise ValueError('Resident counts no longer match the immutable chunk plan.')
    return rowChunkCheckpointRecipe(queryPositions=a.queryPositions, referencePositions=a.referencePositions,
        querySupports=a.querySupports, referenceSupports=a.referenceSupports, counts=a.numNeighbors,
        domain=a.domain, policy=a.plan.policy, mode=a.mode, storeI=a.storeI, verletScale=a.verletScale,
        orderingKey=a.orderingKey, generation=a.plan.generation.number, sourceOwners=a.plan.numOwners)


def restoreRowChunkCheckpointState(state, *, device, policy=None, communication=None, storeI=None):
    """Reconstruct only owned J, preserving the saved global row traversal.

    Local offsets use the current configured offset width, independent of the
    source process/offset configuration. Particle precision must match. The new
    plan/count signature is verified before pair allocation by the search path.
    """
    local = (lambda tag, fn: fn()) if communication is None else communication.localPhase
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    def prepare():
        if not isinstance(state, dict) or state.keys() != {'metadata','tensors','checksum'}:
            raise ValueError('Invalid row-chunk checkpoint recipe envelope.')
        m, tensors = state['metadata'], state['tensors']
        keys = {'version','algorithm','numRows','numCols','dim','generation','totalEdges',
                'policy','mode','storeI','verletScale','orderingKey','sourceOwners','samePositions','sameSupports'}
        if not isinstance(m, dict) or m.keys() != keys or m['version'] != 1 or m['algorithm'] != _ALGORITHM:
            raise ValueError('Unsupported row-chunk checkpoint recipe version/algorithm.')
        for name in ('storeI','samePositions','sameSupports'):
            if type(m[name]) is not bool:
                raise TypeError(f'Checkpoint {name} must be a bool.')
        for name in ('numRows','numCols','generation','totalEdges','sourceOwners','dim'):
            if type(m[name]) is not int or m[name] < (1 if name in ('sourceOwners','dim') else 0):
                raise ValueError(f'Invalid checkpoint {name}.')
        expected = {'queryPositions','querySupports','counts','domainMin','domainMax','periodic'}
        if not m['samePositions']:
            expected.add('referencePositions')
        if not m['sameSupports']:
            expected.add('referenceSupports')
        if not isinstance(tensors, dict) or tensors.keys() != expected:
            raise ValueError('Incomplete row-chunk checkpoint snapshots/counts.')
        if any(not isinstance(t,torch.Tensor) or t.device.type != 'cpu' or not t.is_contiguous()
               or t.requires_grad or torch.autograd.forward_ad.unpack_dual(t).tangent is not None for t in tensors.values()):
            raise ValueError('Checkpoint recipe tensors must be inert contiguous CPU tensors.')
        if state['checksum'] != _digest(m,tensors):
            raise ValueError('Row-chunk checkpoint recipe checksum mismatch.')
        q, qh, counts = tensors['queryPositions'], tensors['querySupports'], tensors['counts']
        r = q if m['samePositions'] else tensors['referencePositions']
        rh = qh if m['sameSupports'] else tensors['referenceSupports']
        if (q.shape != (m['numRows'],m['dim']) or r.shape != (m['numCols'],m['dim'])
            or qh.shape != (len(q),) or rh.shape != (len(r),)
            or counts.shape != (len(q),) or counts.dtype != torch.int32):
            raise ValueError('Checkpoint snapshot/count dimensions are inconsistent.')
        if any(t.dtype != get_torch_precision() or not bool(torch.isfinite(t).all()) for t in (q,r,qh,rh)):
            raise ValueError('Checkpoint particle snapshots must be finite and match configured precision.')
        if type(m['verletScale']) not in (int,float) or not math.isfinite(m['verletScale']) or m['verletScale'] < 1:
            raise ValueError('Invalid checkpoint Verlet scale.')
        if m['mode'] not in SupportScheme.__members__:
            raise ValueError('Invalid checkpoint support mode.')
        selected = RowChunkPolicy(**m['policy']) if policy is None else policy
        if not isinstance(selected, RowChunkPolicy):
            raise TypeError('Restore policy must be a RowChunkPolicy.')
        selectedI = m['storeI'] if storeI is None else storeI
        if type(selectedI) is not bool:
            raise TypeError('Restore storeI must be a bool.')
        memo = {}
        def transfer(t):
            key = id(t)
            if key not in memo:
                memo[key] = t.to(device).clone().contiguous()
            return memo[key]
        q,r,qh,rh = map(transfer,(q,r,qh,rh))
        domain = DomainDescription(transfer(tensors['domainMin']), transfer(tensors['domainMax']),
                                   transfer(tensors['periodic']), m['dim'])
        owners, owner = (1,0) if communication is None else (communication.size,communication.rank)
        mode = SupportScheme[m['mode']]
        config = _config(q,r,domain,mode,selected,selectedI,m['verletScale'],m['orderingKey'],owners,owner,'verlet')
        expectedPlan = buildRowChunkPlan(counts,generation=RowChunkGeneration(m['generation'],len(r),config),
                                         policy=selected,numOwners=owners)
        if expectedPlan.totalEdges != m['totalEdges']:
            raise ValueError('Checkpoint edge total disagrees with row counts.')
        return q,r,qh,rh,domain,mode,selected,selectedI,expectedPlan
    q,r,qh,rh,domain,mode,selected,selectedI,expectedPlan = local('checkpoint:search-prepare',prepare)
    if communication is not None:
        communication.agree('checkpoint:search-contract',(state['checksum'],expectedPlan.signature))
    return _search(q,r,qh,rh,domain,policy=selected,mode=mode,storeI=selectedI,
        numOwners=expectedPlan.numOwners,owner=0 if communication is None else communication.rank,
        generationNumber=expectedPlan.generation.number,reuse=None,inPlace=False,
        expectedSignature=expectedPlan.signature,orderingKey=state['metadata']['orderingKey'],
        kind='verlet',scale=state['metadata']['verletScale'],reason='checkpoint reconstructed',communication=communication)
