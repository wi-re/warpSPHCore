"""Explicit eager c10d coordination for replicated particles and owned CSR rows.

The caller owns process-group initialization, timeout and destruction. No object
collectives, global communicator or package/runtime changes. Pair redistribution
uses explicit bounded owner-to-owner transfers.
Control handshakes detect differing call order/layout/generation before payload
collectives. A launcher (e.g. torchrun) must terminate peers if a process dies.
"""
from collections import deque
from hashlib import sha256
import json
import time

import numpy as np

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
    def __init__(self, group, device, *, replicaChecks='exact', particleWorkspaceBytes=4*1024*1024):
        if not dist.is_initialized():
            raise RuntimeError('Initialize torch.distributed with a finite timeout first.')
        if replicaChecks not in ('exact','contract'):
            raise ValueError('replicaChecks must be exact or contract.')
        if type(particleWorkspaceBytes) is not int or particleWorkspaceBytes<0:
            raise ValueError('particleWorkspaceBytes must be a nonnegative integer.')
        self.replicaChecks = replicaChecks
        # One lazy, fixed-capacity allocation, independent of tags/shapes/dtypes.
        # Zero disables packing. Rows too wide for the budget use direct fields.
        self._particleWorkspaceBytes = particleWorkspaceBytes
        self._particleWorkspace = None
        self._particleReady = None
        self._particleRecorded = False
        self.particleBroadcasts = 0
        self.particlePackedBroadcasts = 0
        self.particlePackedBatches = 0
        self.particleDirectFields = 0
        self.particleWorkspaceAllocations = 0
        self.particleWorkspaceWaits = 0
        self.ownedTransfers = 0
        self.ownedSendBytes = 0
        self.ownedReceiveBytes = 0
        self.ownedLocalCopyBytes = 0
        self.group = group
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.backend = dist.get_backend(group)
        if (self.backend, self.device.type) not in (('gloo', 'cpu'), ('nccl', 'cuda')):
            raise ValueError('Row chunks require Gloo CPU or NCCL CUDA tensors.')
        if self.device.type == 'cuda' and self.device.index != torch.cuda.current_device():
            raise ValueError('Set the current CUDA device explicitly before binding the process group.')
        self._eager()
        self.rank = dist.get_rank(group)
        self.size = dist.get_world_size(group)
        if self.rank < 0 or self.size < 1:
            raise ValueError('This process must belong to the row-chunk process group.')
        self.sequence = 0
        self.events = deque(maxlen=512)
        self.payloadBytes = 0
        self.controlBytes = 0
        self.replicaHashBytes = 0
        self.replicaHashDeviceBytes = 0
        self.replicaFields = dict(exact=0,contract=0)
        # A control finishes its host read before returning, so these buffers
        # can be reused even when the next call selects another CUDA stream.
        self._controlSend = torch.empty(512,dtype=torch.uint8,device=self.device)
        self._controlReceive = torch.empty((self.size,512),dtype=torch.uint8,device=self.device)
        self._controlPeers = list(self._controlReceive.unbind(0))
        self._hostSend = (torch.empty(512,dtype=torch.uint8,pin_memory=True)
                          if self.device.type=='cuda' else self._controlSend)
        self._hostReceive = (torch.empty((self.size,512),dtype=torch.uint8,pin_memory=True)
                             if self.device.type=='cuda' else self._controlReceive)
        self._hostSendArray = self._hostSend.numpy()
        self._hostReceiveArray = self._hostReceive.numpy()

    @property
    def particleWorkspaceBytes(self):
        """Fixed packing capacity; choose it when constructing the group."""
        return self._particleWorkspaceBytes

    def _eager(self):
        if self.device.type=='cuda' and torch.cuda.is_current_stream_capturing():
            raise NotImplementedError('Distributed row chunks cannot run during CUDA graph capture; select eager execution before capture.')

    def _event(self, tag, start, payload=0):
        self.payloadBytes += payload
        self.events.append(dict(sequence=self.sequence, tag=tag, payloadBytes=payload,
                                wallSeconds=time.perf_counter() - start))

    def _control(self, tag, metadata=None, error=None):
        self._eager()
        start = time.perf_counter()
        try:
            encoded = json.dumps((self.sequence, tag, self.replicaChecks, metadata), sort_keys=True,
                                 separators=(',', ':'), allow_nan=False).encode()
        except Exception as exc:
            encoded = b'invalid control metadata'
            error = error or exc
        digest = sha256(encoded).digest()
        # Fixed-size diagnostics let peers report a local validation/kernel
        # failure without entering a differently shaped payload collective.
        message = b'' if error is None else (type(error).__name__ + ': ' + str(error)).encode()[:479]
        record = digest + message + bytes(480-len(message))
        self._hostSendArray[:] = np.frombuffer(record,dtype=np.uint8)
        if self.device.type=='cuda':
            self._controlSend.copy_(self._hostSend,non_blocking=True)
        dist.all_gather(self._controlPeers,self._controlSend,group=self.group)
        if self.device.type=='cuda':
            # One blocking host transfer for the complete gathered packet,
            # instead of one new device->host allocation/copy per peer.
            self._hostReceive.copy_(self._controlReceive)
        self.sequence += 1
        self.controlBytes += 512 * self.size
        records = [row.tobytes() for row in self._hostReceiveArray]
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
        self._eager()
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

    def _checkedAgree(self,tag,function):
        """One handshake for local validation/allocation AND its JSON contract."""
        self._eager()
        error,metadata=None,None
        started=time.perf_counter()
        try:
            metadata=function()
        except Exception as exc:
            error=exc
        hostSeconds=time.perf_counter()-started
        try:
            self._control(tag,metadata,error)
        finally:
            if self.events:self.events[-1]['hostLocalSeconds']=hostSeconds
        return metadata

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

    def assertReplicated(self, tag, tensors, *, exact=False):
        """Check ordered replica layouts; hash contents in exact/forced checks.

        Contract mode assumes the dependency exchanges maintain replica values.
        Search, accepted-state and checkpoint boundaries explicitly force exact
        checks. Both modes retain layout/device/AD and collective-order checks.
        """
        def fingerprints():
            if type(exact) is not bool:
                raise TypeError('exact must be a bool.')
            full = exact or self.replicaChecks=='exact'
            result = [('exact',full)]
            for name, tensor in tensors.items():
                # bool domain fields are valid replica metadata, not payloads.
                if not isinstance(tensor, torch.Tensor) or tensor.device != self.device:
                    raise ValueError(f'Replicated field {name} must be on {self.device}.')
                if tensor.requires_grad or fwAD.unpack_dual(tensor).tangent is not None:
                    raise NotImplementedError('Distributed replicas do not support AD.')
                digest=None
                if full:
                    raw=tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
                    digest=sha256(raw).hexdigest()
                    self.replicaHashBytes+=len(raw)
                    if tensor.is_cuda:self.replicaHashDeviceBytes+=len(raw)
                self.replicaFields['exact' if full else 'contract']+=1
                result.append((name, list(tensor.shape), str(tensor.dtype), digest))
            return result
        self._checkedAgree(tag,fingerprints)

    def _rowLayout(self,tensor,ranges):
        self._tensor(tensor)
        if tensor.ndim<1 or len(ranges)!=self.size:
            raise ValueError('One particle-row interval is required per rank.')
        cursor=0
        for a,b in ranges:
            if type(a) is not int or type(b) is not int or a!=cursor or b<a:
                raise ValueError('Owner intervals must form an ordered disjoint partition.')
            cursor=b
        if cursor!=len(tensor):
            raise ValueError('Owner intervals must cover all particle rows.')
        return (list(tensor.shape),str(tensor.dtype))

    def _broadcastRows(self,tensor,ranges,tag):
        start=time.perf_counter()
        for owner,(a,b) in enumerate(ranges):
            if b>a and tensor[a:b].numel():
                source=dist.get_global_rank(self.group,owner) if self.group is not None else owner
                dist.broadcast(tensor[a:b],src=source,group=self.group)
                self.particleBroadcasts+=1
        self.particleDirectFields+=1
        self._event(tag+':rows',start,tensor.numel()*tensor.element_size())
        return tensor

    def exchangeRows(self,tensor,ranges,*,tag,signature):
        """Replicate uniquely owned particle rows in place; never sum replicas."""
        self._checkedAgree(tag,lambda:(signature,self._rowLayout(tensor,ranges),ranges))
        return self._broadcastRows(tensor,ranges,tag)

    def reduceScalar(self,value,*,operation,tag):
        """MIN timestep, MAX flag or SUM uniquely owned local contributions."""
        out=None
        def prepare():
            nonlocal out
            self._tensor(value)
            if value.ndim!=0 or operation not in ('min','max','sum'):
                raise ValueError('Scalar reduction requires min/max/sum and a scalar tensor.')
            out=value.clone()
            return operation,str(value.dtype)
        self._checkedAgree(tag,prepare)
        start=time.perf_counter()
        dist.all_reduce(out,op={'min':dist.ReduceOp.MIN,'max':dist.ReduceOp.MAX,
                               'sum':dist.ReduceOp.SUM}[operation],group=self.group)
        self._event(tag+':reduce',start,out.element_size())
        return out

    def transferOwned(self,send,receive,*,source,target,count,dtype,tag,signature):
        """Copy a bounded 1D owner interval directly to its new owner.

        Every rank enters the contract; only source/target enter the payload.
        Caller-owned storage and a deterministic schedule avoid pair gathers.
        Source and destination may coincide. No allocation/object collective.
        """
        sizes={torch.float32:4,torch.float64:8,torch.int32:4,torch.int64:8,torch.uint8:1}
        def validate():
            if (type(source) is not int or type(target) is not int or not 0<=source<self.size
                or not 0<=target<self.size or type(count) is not int or count<0 or dtype not in sizes):
                raise ValueError('Owned transfer requires valid ranks, count and payload dtype.')
            for tensor,owner in ((send,source),(receive,target)):
                if self.rank==owner:
                    self._tensor(tensor)
                    if tensor.shape!=(count,) or tensor.dtype!=dtype:
                        raise ValueError('Owned transfer tensor must match the agreed count/dtype.')
                elif tensor is not None:
                    raise ValueError('Only the participating owner may supply transfer storage.')
            return signature,source,target,count,str(dtype)
        self._checkedAgree(tag,validate)
        start=time.perf_counter();size=count*sizes[dtype]
        if count:
            if source==target:
                if self.rank==source:
                    receive.copy_(send)
                    self.ownedLocalCopyBytes+=size
            elif self.rank==source:
                peer=dist.get_global_rank(self.group,target) if self.group is not None else target
                dist.send(send,dst=peer,group=self.group)
                self.ownedSendBytes+=size
            elif self.rank==target:
                peer=dist.get_global_rank(self.group,source) if self.group is not None else source
                dist.recv(receive,src=peer,group=self.group)
                self.ownedReceiveBytes+=size
        self.ownedTransfers+=1
        self._event(tag+':owned-transfer',start,size)

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
        """Replicate outputs, packing compatible fields without arithmetic/casts.

        All local validation/allocation precedes one control handshake. Scratch
        is bounded by particleWorkspaceBytes, with whole-row tiles and direct
        fallback for oversized rows. Outputs never borrow scratch storage.
        Like controls, this API is serial eager, not concurrent-thread safe.
        """
        batches=[]
        def prepare():
            self.validateAdjacency(adjacency)
            ranges=adjacency.plan.ownerRanges
            layouts=[self._rowLayout(t,ranges) for t in tensors]
            # Contiguous address intervals permit disjoint views of one storage
            # and exact aliases, but reject partial/cross-row or retyped overlap.
            spans=[]
            aliases=[]
            groups={}
            for index,t in enumerate(tensors):
                aliases.append(index)
                if not t.numel():continue
                lo=t.data_ptr();hi=lo+t.numel()*t.element_size()
                for begin,end,dtype,alias in spans:
                    if lo<end and begin<hi:
                        if lo!=begin or hi!=end or t.dtype!=dtype:
                            raise ValueError('Particle outputs must not partially overlap or reinterpret storage.')
                        aliases[-1]=alias
                spans.append((lo,hi,t.dtype,aliases[-1]))
                groups.setdefault(t.dtype,[]).append(index)
            for indices in groups.values():
                width=sum(tensors[i][0].numel() for i in indices)
                rowBytes=width*tensors[indices[0]].element_size()
                tileRows=self.particleWorkspaceBytes//rowBytes
                # Very small budgets/large populations can need more tiled
                # broadcasts than direct fields; pack only when calls decrease.
                packed=(len(indices)>1 and tileRows>0 and
                        sum((b-a+tileRows-1)//tileRows for a,b in ranges)<
                        len(indices)*sum(b>a for a,b in ranges))
                batches.append((indices,width,packed))
            if any(packed for _,_,packed in batches) and self._particleWorkspace is None:
                workspace=torch.empty(self.particleWorkspaceBytes,dtype=torch.uint8,device=self.device)
                ready=torch.cuda.Event() if self.device.type=='cuda' else None
                self._particleWorkspace=workspace
                self._particleReady=ready
                self.particleWorkspaceAllocations+=1
            return (adjacency.plan.signature,ranges,layouts,self.particleWorkspaceBytes,aliases)
        self._checkedAgree(tag+':outputs',prepare)
        ranges=adjacency.plan.ownerRanges
        for indices,width,packed in batches:
            if not packed:
                for index in indices:
                    self._broadcastRows(tensors[index],ranges,f'{tag}:{index}')
                continue
            first=tensors[indices[0]]
            rows=self.particleWorkspaceBytes//(width*first.element_size())
            storage=self._particleWorkspace[:rows*width*first.element_size()].view(first.dtype).view(rows,width)
            if self._particleReady is not None and self._particleRecorded:
                # The prior unpack may still read scratch on another stream.
                # A control's host read does not fence that stream's payload.
                torch.cuda.current_stream(self.device).wait_event(self._particleReady)
                self.particleWorkspaceWaits+=1
            if self._particleReady is not None:
                # Also protect allocator lifetime if the group is released
                # before a non-creation stream finishes reading the workspace.
                self._particleWorkspace.record_stream(torch.cuda.current_stream(self.device))
            started=time.perf_counter()
            try:
                for owner,(a,b) in enumerate(ranges):
                    source=dist.get_global_rank(self.group,owner) if self.group is not None else owner
                    for start in range(a,b,rows):
                        stop=min(start+rows,b);packet=storage[:stop-start]
                        cursor=0
                        if owner==self.rank:
                            for index in indices:
                                field=tensors[index][start:stop].reshape(stop-start,-1)
                                packet[:,cursor:cursor+field.shape[1]].copy_(field)
                                cursor+=field.shape[1]
                        dist.broadcast(packet,src=source,group=self.group)
                        self.particleBroadcasts+=1
                        self.particlePackedBroadcasts+=1
                        if owner!=self.rank:
                            for index in indices:
                                field=tensors[index][start:stop].reshape(stop-start,-1)
                                field.copy_(packet[:,cursor:cursor+field.shape[1]])
                                cursor+=field.shape[1]
            finally:
                if self._particleReady is not None:
                    self._particleReady.record(torch.cuda.current_stream(self.device))
                    self._particleRecorded=True
            self.particlePackedBatches+=1
            self._event(tag+':packed-rows',started,sum(tensors[i].numel()*tensors[i].element_size() for i in indices))

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
                    replicaChecks=self.replicaChecks,controls=self.sequence,logicalPayloadBytes=self.payloadBytes,
                    controlReceiveBytes=self.controlBytes,controlBufferAllocations=4 if self.device.type=='cuda' else 2,
                    controlBufferBytes=512*(self.size+1)*(2 if self.device.type=='cuda' else 1),
                    controlHostReadbacks=self.sequence if self.device.type=='cuda' else 0,
                    particleBroadcasts=self.particleBroadcasts,particlePackedBroadcasts=self.particlePackedBroadcasts,
                    particlePackedBatches=self.particlePackedBatches,particleDirectFields=self.particleDirectFields,
                    particleWorkspaceLimitBytes=self.particleWorkspaceBytes,
                    particleWorkspaceBytes=0 if self._particleWorkspace is None else self._particleWorkspace.numel(),
                    particleWorkspaceAllocations=self.particleWorkspaceAllocations,
                    particleWorkspaceWaits=self.particleWorkspaceWaits,
                    ownedTransfers=self.ownedTransfers,ownedSendBytes=self.ownedSendBytes,
                    ownedReceiveBytes=self.ownedReceiveBytes,ownedLocalCopyBytes=self.ownedLocalCopyBytes,
                    replicaHashBytes=self.replicaHashBytes,replicaHashDeviceBytes=self.replicaHashDeviceBytes,
                    replicaFields=dict(self.replicaFields),recentEvents=list(self.events))
