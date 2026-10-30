"""Explicit eager c10d coordination for replicated particles and owned CSR rows.

The caller owns process-group initialization, timeout and destruction. No object
collectives, pair transfers, global communicator or package/runtime changes.
Control handshakes detect differing call order/layout/generation before payload
collectives. A launcher (e.g. torchrun) must terminate peers if a process dies.
"""
from collections import deque
from hashlib import sha256
import json
import time

import torch
import torch.distributed as dist
import torch.autograd.forward_ad as fwAD

__all__ = ['RowChunkProcessGroup', 'DistributedContractError']


class DistributedContractError(RuntimeError):
    """A local phase failed, or ranks disagreed on the collective contract."""


class RowChunkProcessGroup:
    """Bind an existing NCCL/CUDA or Gloo/CPU group to one explicit device.

    All ranks must call this interface in the same order. Operations are eager
    on Torch's current stream, including Warp producers/consumers using that
    stream. Tensor exchanges broadcast disjoint particle rows in owner order;
    only O(N) particle storage is replicated. Timings include control/host waits
    and are diagnostics, not transport benchmarks.
    """
    def __init__(self, group, device):
        if not dist.is_initialized():
            raise RuntimeError('Initialize torch.distributed with a finite timeout first.')
        self.group = group
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.backend = dist.get_backend(group)
        if (self.backend, self.device.type) not in (('gloo', 'cpu'), ('nccl', 'cuda')):
            raise ValueError('Row chunks require Gloo CPU or NCCL CUDA tensors.')
        if self.device.type == 'cuda' and self.device.index != torch.cuda.current_device():
            raise ValueError('Set the current CUDA device explicitly before binding the process group.')
        self.rank = dist.get_rank(group)
        self.size = dist.get_world_size(group)
        if self.rank < 0 or self.size < 1:
            raise ValueError('This process must belong to the row-chunk process group.')
        self.sequence = 0
        self.events = deque(maxlen=512)
        self.payloadBytes = 0
        self.controlBytes = 0

    def _event(self, tag, start, payload=0):
        self.payloadBytes += payload
        self.events.append(dict(sequence=self.sequence, tag=tag, payloadBytes=payload,
                                wallSeconds=time.perf_counter() - start))

    def _control(self, tag, metadata=None, error=None):
        start = time.perf_counter()
        try:
            encoded = json.dumps((self.sequence, tag, metadata), sort_keys=True,
                                 separators=(',', ':'), allow_nan=False).encode()
        except Exception as exc:
            encoded = b'invalid control metadata'
            error = error or exc
        digest = sha256(encoded).digest()
        # Fixed-size diagnostics let peers report a local validation/kernel
        # failure without entering a differently shaped payload collective.
        message = b'' if error is None else (type(error).__name__ + ': ' + str(error)).encode()[:479]
        local = torch.tensor(list(digest + message + bytes(480-len(message))),
                             dtype=torch.uint8, device=self.device)
        peers = [torch.empty_like(local) for _ in range(self.size)]
        dist.all_gather(peers, local, group=self.group)
        self.sequence += 1
        self.controlBytes += local.numel() * self.size
        records = [bytes(p.cpu().tolist()) for p in peers]
        failures = [(rank, p[32:].split(b'\0', 1)[0].decode(errors='replace'))
                    for rank, p in enumerate(records) if p[32] != 0]
        self._event(tag, start)
        if failures:
            detail = '; '.join(f'rank {rank}: {message}' for rank, message in failures)
            raise DistributedContractError(f'{tag}: {detail}') from error
        if len({p[:32] for p in records}) != 1:
            raise DistributedContractError(
                f'{tag}: ranks disagree on collective sequence, operation, generation or layout (sequence {self.sequence-1}).')

    def agree(self, tag, metadata):
        """Agree on JSON metadata before a branch, allocation or payload transfer."""
        self._control(tag, metadata)

    def localPhase(self, tag, function):
        """Run a LOCAL callback, then propagate exceptions to every live peer.

        The callback must not contain collectives. CUDA asynchronous device
        faults/process death still rely on c10d's timeout and the job launcher.
        """
        error, result = None, None
        started = time.perf_counter()
        try:
            result = function()
        except Exception as exc:
            error = exc
        hostSeconds = time.perf_counter() - started
        try:
            self._control(tag, error=error)
        finally:
            if self.events:
                self.events[-1]['hostLocalSeconds'] = hostSeconds
        return result

    def _tensor(self, tensor):
        if not isinstance(tensor, torch.Tensor) or tensor.device != self.device:
            raise ValueError(f'Collective tensor must be on {self.device}.')
        if tensor.requires_grad or fwAD.unpack_dual(tensor).tangent is not None:
            raise NotImplementedError('Distributed row chunks do not support AD.')
        if tensor.is_cuda and torch.cuda.is_current_stream_capturing():
            raise NotImplementedError('Distributed row chunks do not support capture.')
        if not tensor.is_contiguous():
            raise ValueError('Collective tensor must be contiguous.')
        if tensor.dtype not in (torch.float32, torch.float64, torch.int32, torch.int64, torch.uint8):
            raise TypeError('Unsupported row-chunk collective dtype.')

    def assertReplicated(self, tag, tensors):
        """Exact rebuild/debug agreement on ordered reference fields (O(N))."""
        def fingerprints():
            result = []
            for name, tensor in tensors.items():
                # bool domain fields are valid replica metadata, not payloads.
                if not isinstance(tensor, torch.Tensor) or tensor.device != self.device:
                    raise ValueError(f'Replicated field {name} must be on {self.device}.')
                if tensor.requires_grad or fwAD.unpack_dual(tensor).tangent is not None:
                    raise NotImplementedError('Distributed replicas do not support AD.')
                raw = tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
                result.append((name, list(tensor.shape), str(tensor.dtype), sha256(raw).hexdigest()))
            return result
        metadata = self.localPhase(tag + ':validate', fingerprints)
        self.agree(tag, metadata)

    def exchangeRows(self, tensor, ranges, *, tag, signature):
        """Replicate uniquely owned particle rows in place; never sum replicas."""
        def validate():
            self._tensor(tensor)
            if tensor.ndim < 1 or len(ranges) != self.size:
                raise ValueError('One particle-row interval is required per rank.')
            cursor = 0
            for a, b in ranges:
                if type(a) is not int or type(b) is not int or a != cursor or b < a:
                    raise ValueError('Owner intervals must form an ordered disjoint partition.')
                cursor = b
            if cursor != len(tensor):
                raise ValueError('Owner intervals must cover all particle rows.')
        self.localPhase(tag + ':validate', validate)
        self.agree(tag, (signature, list(tensor.shape), str(tensor.dtype), ranges))
        start = time.perf_counter()
        for owner, (a, b) in enumerate(ranges):
            if b > a and tensor[a:b].numel():
                source = dist.get_global_rank(self.group, owner) if self.group is not None else owner
                dist.broadcast(tensor[a:b], src=source, group=self.group)
        self._event(tag + ':rows', start, tensor.numel() * tensor.element_size())
        return tensor

    def reduceScalar(self, value, *, operation, tag):
        """MIN timestep, MAX flag or SUM uniquely owned local contributions."""
        def validate():
            self._tensor(value)
            if value.ndim != 0 or operation not in ('min', 'max', 'sum'):
                raise ValueError('Scalar reduction requires min/max/sum and a scalar tensor.')
        self.localPhase(tag + ':validate', validate)
        self.agree(tag, (operation, str(value.dtype)))
        out = self.localPhase(tag + ':allocate', value.clone)
        start = time.perf_counter()
        dist.all_reduce(out, op={'min': dist.ReduceOp.MIN, 'max': dist.ReduceOp.MAX,
                                'sum': dist.ReduceOp.SUM}[operation], group=self.group)
        self._event(tag + ':reduce', start, out.element_size())
        return out

    def any(self, flag, *, tag):
        flag = self.localPhase(tag + ':flag', lambda: torch.as_tensor(flag, device=self.device, dtype=torch.int32).reshape(()))
        return bool(self.reduceScalar(flag, operation='max', tag=tag).item())

    def validateAdjacency(self, adjacency):
        adjacency.assertStorageValid()
        if (adjacency.communication is not self or adjacency.owner != self.rank
            or adjacency.plan.numOwners != self.size or adjacency.queryPositions.device != self.device):
            raise ValueError('Chunked adjacency is bound to a different rank, group or device.')
        expected = tuple(m for m in adjacency.plan.chunks if m.owner == self.rank)
        if tuple(c.metadata for c in adjacency) != expected:
            raise ValueError('Rank must hold exactly its planned chunks, including zero-edge rows.')

    def exchangeParticleOutputs(self, adjacency, tensors, *, tag):
        self.localPhase(tag + ':adjacency', lambda: self.validateAdjacency(adjacency))
        self.agree(tag + ':outputs', (adjacency.plan.signature, len(tensors)))
        for index, tensor in enumerate(tensors):
            self.exchangeRows(tensor, adjacency.plan.ownerRanges, tag=f'{tag}:{index}',
                              signature=adjacency.plan.signature)

    def sumOwned(self, tensor, adjacency, *, tag):
        def local():
            self.validateAdjacency(adjacency)
            self._tensor(tensor)
            if tensor.ndim < 1 or len(tensor) != adjacency.numRows:
                raise ValueError('Owned sum requires one contribution per particle row.')
            a, b = adjacency.plan.ownerRanges[self.rank]
            return tensor[a:b].to(torch.float64).sum()
        value = self.localPhase(tag + ':owned', local)
        return self.reduceScalar(value, operation='sum', tag=tag)

    def summary(self):
        return dict(backend=self.backend, rank=self.rank, ranks=self.size, device=str(self.device),
                    controls=self.sequence, logicalPayloadBytes=self.payloadBytes,
                    controlReceiveBytes=self.controlBytes, recentEvents=list(self.events))
