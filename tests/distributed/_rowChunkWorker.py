"""Multiprocess row ownership, dependency exchange and failure regressions.

Run through torchrun; pytest's driver uses Gloo CPU. CUDA runs explicitly use
the selected GPU (including two ranks on cuda:0 with a suitable NCCL runtime).
Tiny flat lists below are independent test oracles, never production fallbacks.
"""
from dataclasses import replace
from datetime import timedelta
import argparse
import importlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import warp as wp

from warpSPHCore import (DomainDescription, ParticleState, OperationProperties,
    KernelFunctions, SupportScheme, WarpOperation, GradientScheme, warpOperation,
    radiusSearchCompactHashMap, setNeighborLanes, get_torch_precision)
from warpSPHCore.autograd import SPHContext, PreparedLauncher
from warpSPHCore.coreOperations.wp_density import _DENSITY_SPEC
from warpSPHCore.distributed import RowChunkProcessGroup, DistributedContractError, DistributedAllocationError
from warpSPHCore.radiusSearch import (radiusSearchRowChunks, RowChunkPolicy,
    buildChunkedVerletList, ChunkedPairField, chunkedVerletStaleFlag_)
from warpSPHCore.crk import computeCRKFactors


def sample(device, nq=19, nr=None, dim=2):
    dtype = get_torch_precision()
    def population(n, shift):
        t = torch.arange(n, dtype=dtype, device=device)
        x = torch.stack([torch.sin(t*(k+.7)+shift)*.4 for k in range(dim)], 1)
        return ParticleState(x, 1.1+.2*torch.cos(t), .8+.03*t,
                             torch.zeros(n, device=device, dtype=torch.int32), 1.4+.02*t)
    q = population(nq, .12)
    r = q if nr is None else population(nr, .77)
    domain = DomainDescription(torch.full((dim,), -2., dtype=dtype, device=device),
        torch.full((dim,), 2., dtype=dtype, device=device), torch.zeros(dim, dtype=torch.bool, device=device), dim)
    return q, r, domain


def equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    else:
        for x, y in zip(a, b):
            equal(x, y)


def check_rows(adj, flat):
    equal(adj.numNeighbors, flat.numNeighbors)
    assert adj.globalPrefix.dtype == torch.int64
    equal(adj.globalPrefix[:-1], flat.edgeOffsets.to(torch.int64))
    for chunk in adj:
        a, m = chunk.adjacency, chunk.metadata
        assert m.owner == adj.owner and a.paddedJ.numel() == m.capacity
        assert a.paddedJ.numel() <= adj.plan.policy.capacityLimit
        for row in range(m.rowCount):
            n = int(a.numNeighbors[row])
            start, old = int(a.edgeOffsets[row]), int(flat.edgeOffsets[row+m.rowStart])
            equal(a.paddedJ[start:start+n], flat.paddedJ[old:old+n])
            if a.paddedI is not None:
                assert bool((a.paddedI[start:start+n] == row+m.rowStart).all())
        assert bool((a.paddedJ[m.liveEdges:] == -1).all())


def correctness(comm):
    checks = []
    policy = RowChunkPolicy(edgeBudget=80, maxRows=3, bytesPerEdge=32, maxElementsPerEdge=3)
    for dim, nq, nr in ((2,19,None),(3,11,19),(2,0,0),(2,0,5),(2,3,0),(2,1,4)):
        q, r, domain = sample(comm.device, nq, nr, dim)
        calls = []
        module = importlib.import_module('warpSPHCore.radiusSearch.chunked')
        launch = module.wp.launch
        def audit(kernel, *a, **kw):
            if kernel == module.radiusSearchCountNeighborsCompactHashMap:
                calls.append(kw['dim'])
            return launch(kernel, *a, **kw)
        module.wp.launch = audit
        try:
            adj = radiusSearchRowChunks(q, domain, referenceParticles=r, policy=policy,
                mode=SupportScheme.SuperSymmetric, communication=comm, storeI=True)
        finally:
            module.wp.launch = launch
        a,b = adj.plan.countingRanges[comm.rank]
        assert sum(calls) == (b-a if len(r.positions) else 0)
        flat = radiusSearchCompactHashMap(q, domain, referenceParticles=r,
            mode=SupportScheme.SuperSymmetric, storeI=True)
        check_rows(adj, flat)
        # Particle-output collectives replicate fields between every dependency.
        for operation in (WarpOperation.Density, WarpOperation.Covariance, WarpOperation.Gradient):
            props = OperationProperties(kernel=KernelFunctions.Wendland2, operation=operation,
                supportMode=SupportScheme.SuperSymmetric, gradientMode=GradientScheme.Symmetric)
            kw = dict(referenceParticles=r, queryValues=q.positions.square()+.1,
                      referenceValues=r.positions.sin()-.2, covarianceReturnNumNeighbors=True)
            equal(warpOperation(q, props, domain, adjacency=adj, **kw),
                  warpOperation(q, props, domain, adjacency=flat, **kw))
        ctx = SPHContext(q, OperationProperties(kernel=KernelFunctions.Wendland2), domain, adj, r)
        prepared = PreparedLauncher(ctx)
        equal(prepared.launch(_DENSITY_SPEC), PreparedLauncher(replace(ctx, adjacency=flat)).launch(_DENSITY_SPEC))
        if nq and len(r.positions):
            actual = computeCRKFactors(q, domain, KernelFunctions.Wendland2, referenceState=r, adjacency=adj)
            expected = computeCRKFactors(q, domain, KernelFunctions.Wendland2, referenceState=r, adjacency=flat)
            equal(actual[:2], expected[:2])
            equal(tuple(vars(actual[2]).values()), tuple(vars(expected[2]).values()))
        values = torch.arange(nq, dtype=q.positions.dtype, device=comm.device)+1
        expected = values.double().sum()
        # Poison replicas outside ownership to detect duplicate summation.
        a,b = adj.plan.ownerRanges[comm.rank]
        values[:a] = -1e6
        values[b:] = -1e6
        equal(comm.sumOwned(values, adj, tag='test:owned-sum'), expected)
        field = ChunkedPairField.empty(adj, trailingShape=(dim,), dtype=q.positions.dtype)
        assert len(field.buffers) == len(adj.chunks)
        checks.append(dict(dim=dim, queries=nq, references=len(r.positions), chunks=len(adj.chunks),
                           countRows=sum(calls), pairCapacityBytes=adj.memoryUsage()['pairCapacityBytes'],
                           workCapacityBytes=field.memoryUsage()['capacityBytes']))
        print(f'[distributed core rank {comm.rank}] rows {nq}/{len(r.positions)} dim {dim}: passed', flush=True)
    q,r,domain = sample(comm.device)
    prior = buildChunkedVerletList(q, domain, policy=policy, verletScale=1.1,
        supportMode=SupportScheme.SuperSymmetric, communication=comm)
    assert buildChunkedVerletList(q, domain, policy=policy, verletScale=1.1,
        supportMode=SupportScheme.SuperSymmetric, priorNeighborhood=prior) is prior
    rebuilt = buildChunkedVerletList(q, domain, policy=replace(policy,maxRows=2), verletScale=1.1,
        supportMode=SupportScheme.SuperSymmetric, priorNeighborhood=prior,
        forceRebuild=comm.rank == comm.size-1)
    assert rebuilt.plan.generation.number == prior.plan.generation.number+1
    forced = buildChunkedVerletList(q, domain, policy=rebuilt.plan.policy, verletScale=1.1,
        supportMode=SupportScheme.SuperSymmetric, priorNeighborhood=rebuilt,
        forceRebuild=comm.rank == comm.size-1)
    assert forced.plan.generation.number == rebuilt.plan.generation.number+1
    # Independent old buffers remain valid; an in-place reuse invalidates leases.
    inplace = buildChunkedVerletList(q, domain, policy=forced.plan.policy, verletScale=1.1,
        supportMode=SupportScheme.SuperSymmetric, priorNeighborhood=forced,
        forceRebuild=True, inPlace=True)
    if forced.chunks:
        try:
            forced.assertStorageValid()
        except ValueError:
            pass
        else:
            raise AssertionError('in-place reuse retained the old lease')
    assert comm.any(comm.rank == comm.size-1, tag='test:any')
    scalar = torch.tensor(.25+comm.rank, device=comm.device, dtype=q.positions.dtype)
    equal(comm.reduceScalar(scalar, operation='min', tag='test:dt'), scalar.new_tensor(.25))
    checks.append(dict(lifecycle='resident/repartition/global-rebuild/in-place', status='passed'))
    return checks


def failures(comm):
    def rejected(function, match):
        try:
            function()
        except DistributedContractError as exc:
            assert match in str(exc), str(exc)
        else:
            raise AssertionError('all ranks must reject a mismatched contract')
    rejected(lambda: comm.agree('test:generation', {'generation':comm.rank}), 'disagree')
    rejected(lambda: comm.agree('test:call-'+str(comm.rank), {}), 'disagree')
    def broken():
        if comm.rank == comm.size-1:
            raise ValueError('injected local failure')
    rejected(lambda: comm.localPhase('test:injected', broken), 'injected local failure')
    for allocationType in (MemoryError, torch.OutOfMemoryError):
        def allocation():
            if comm.rank == comm.size-1:
                raise allocationType('injected bulk allocation')
        try:
            comm.localPhase('test:allocation-failure', allocation)
        except DistributedAllocationError as exc:
            assert isinstance(exc, MemoryError) and 'injected bulk allocation' in str(exc)
        else:
            raise AssertionError('allocation classification must reach healthy peers')
    bad = torch.zeros(2 if comm.rank == 0 else 3, device=comm.device, dtype=torch.int32)
    ranges = ((0,1),(1,2)) if comm.size == 2 else ((0,1),(1,2),(2,2))
    rejected(lambda: comm.exchangeRows(bad, ranges, tag='test:bad-shape', signature='same'), 'cover all')
    q,r,domain = sample(comm.device)
    rejected(lambda: radiusSearchRowChunks(q, domain, referenceParticles=r,
        policy=RowChunkPolicy(edgeBudget=2), communication=comm), 'exceeds')
    rejected(lambda: radiusSearchRowChunks(q, domain, referenceParticles=r,
        policy=RowChunkPolicy(edgeBudget=80,maxRows=comm.rank+1), communication=comm), 'disagree')
    q.positions[0,0] += comm.rank*.01
    rejected(lambda: radiusSearchRowChunks(q, domain, referenceParticles=r,
        policy=RowChunkPolicy(edgeBudget=80), communication=comm), 'disagree')
    q,r,domain = sample(comm.device)
    adj = radiusSearchRowChunks(q, domain, referenceParticles=r,
        policy=RowChunkPolicy(edgeBudget=80,maxRows=3), communication=comm)
    props = OperationProperties(kernel=KernelFunctions.Wendland2, operation=WarpOperation.Density)
    if comm.device.type=='cuda':
        from warpSPHCore.autograd import launchOperator
        rejected(lambda:launchOperator(replace(_DENSITY_SPEC,tiledKernel=None,blockDim=64 if comm.rank==0 else 128),
            SPHContext(q,props,domain,adj,r)), 'disagree')
    # Divergent reference data must fail BEFORE outputs can hide the discrepancy
    # by broadcasting a mixture of owner results.
    q.masses[0] += comm.rank*.01
    if comm.replicaChecks=='exact':
        rejected(lambda: warpOperation(q, props, domain, adjacency=adj, referenceParticles=r), 'disagree')
    else:
        rejected(lambda:comm.assertReplicated('test:accepted-boundary',dict(masses=q.masses),exact=True),'disagree')
    print(f'[distributed core rank {comm.rank}] coordinated failure checks: passed', flush=True)
    return 8



def protocolAudit(comm):
    """Check whole-batch validation, forced exact checks and cached-control reuse."""
    cached=hasattr(comm,'_controlSend')
    addresses=None if not cached else tuple(t.data_ptr() for t in
        (comm._controlSend,comm._controlReceive,comm._hostSend,comm._hostReceive))
    factories={name:getattr(torch,name) for name in ('empty','empty_like','tensor')}
    calls=[]
    def tracked(name,function):
        def create(*a,**kw):
            calls.append(name)
            return function(*a,**kw)
        return create
    for name,function in factories.items():setattr(torch,name,tracked(name,function))
    try:
        for index in range(12):comm.agree('test:cached-control',index)
    finally:
        for name,function in factories.items():setattr(torch,name,function)
    if cached:assert not calls,calls
    def rejected(function,text):
        try:function()
        except DistributedContractError as exc:assert text in str(exc),str(exc)
        else:raise AssertionError('expected coordinated rejection: '+text)
    value=torch.arange(5,dtype=torch.float32,device=comm.device)+comm.rank
    if comm.replicaChecks=='contract':
        before=comm.replicaHashBytes
        comm.assertReplicated('test:layout-only',dict(value=value))
        assert comm.replicaHashBytes==before
    rejected(lambda:comm.assertReplicated('test:forced-exact',dict(value=value),exact=True),'disagree')
    mode=comm.replicaChecks
    comm.replicaChecks='exact' if comm.rank==0 else 'contract'
    try:rejected(lambda:comm.agree('test:mismatched-policy',None),'disagree')
    finally:comm.replicaChecks=mode
    # An error message must not survive in a reused packet on the next call.
    comm.agree('test:after-policy-failure',True)
    rejected(lambda:comm.reduceScalar(value,operation='min',tag='test:bad-scalar'),'Scalar reduction')
    q,r,domain=sample(comm.device,nq=5)
    adj=radiusSearchRowChunks(q,domain,policy=RowChunkPolicy(edgeBudget=80,maxRows=2),communication=comm)
    a,b=adj.plan.ownerRanges[comm.rank]
    expected=torch.arange(10,dtype=torch.float32,device=comm.device).reshape(5,2)
    output=expected.clone()
    other=expected.long().clone()
    output[:a]=-999;output[b:]=-999;other[:a]=-999;other[b:]=-999
    before=comm.sequence
    comm.exchangeParticleOutputs(adj,(output,other),tag='test:batch-outputs')
    if cached:assert comm.sequence-before==1
    equal(output,expected);equal(other,expected.long())
    broadcast=dist.broadcast
    def forbidden(*a,**kw):raise AssertionError('invalid batch entered payload broadcast')
    dist.broadcast=forbidden
    try:
        bad=other if comm.rank==0 else other[:2]
        rejected(lambda:comm.exchangeParticleOutputs(adj,(output,bad),tag='test:bad-last-output'),'cover all')
    finally:dist.broadcast=broadcast
    comm.agree('test:after-batch-failure',True)
    if cached and comm.device.type=='cuda':
        streams=[torch.cuda.Stream(device=comm.device) for _ in range(2)]
        for index in range(12):
            with torch.cuda.stream(streams[index%2]):
                comm.agree('test:control-stream-switch',index)
    if cached:
        assert addresses==tuple(t.data_ptr() for t in
            (comm._controlSend,comm._controlReceive,comm._hostSend,comm._hostReceive))
    return dict(cachedControlBuffers=cached,explicitTensorFactoriesFor12Controls=len(calls),
                coordinatedFailures=4,wholeBatchValidation=True)


def packingAudit(parent):
    """Bounded tiles, bitwise packing, aliases, allocation errors and scratch life."""
    from unittest.mock import patch
    comm=RowChunkProcessGroup(parent.group,parent.device,replicaChecks=parent.replicaChecks,
                              particleWorkspaceBytes=127)
    def rejected(function,text):
        try:function()
        except DistributedContractError as exc:assert text in str(exc),str(exc)
        else:raise AssertionError('expected coordinated rejection: '+text)
    def forbidden(*a,**kw):raise AssertionError('invalid packing batch entered payload')
    address=None
    retained=[]
    totalBroadcasts=0
    for n in (0,11,1):
        q,r,domain=sample(comm.device,nq=n)
        adj=radiusSearchRowChunks(q,domain,policy=RowChunkPolicy(edgeBudget=80,maxRows=2),communication=comm)
        a,b=adj.plan.ownerRanges[comm.rank]
        expected=[]
        for dtype,shape in ((torch.float32,(2,)),(torch.float32,()),(torch.float32,(2,2)),
                            (torch.float64,()),(torch.float64,(2,)),
                            (torch.int32,()),(torch.int32,(2,)),
                            (torch.int64,(2,)),(torch.int64,()),
                            (torch.uint8,(3,)),(torch.uint8,())):
            count=n
            for width in shape:count*=width
            expected.append(torch.arange(count,device=comm.device).to(dtype).reshape(n,*shape))
        if n:
            # Preserve signed zero and NaN payload bits, not only finite values.
            expected[0].view(torch.int32).reshape(-1)[0]=-2147483648
            expected[1].view(torch.int32)[0]=0x7fc00031
        outputs=[t.clone() for t in expected]
        # Exact aliases with different trailing shapes and disjoint views of
        # one allocation are valid; partial/cross-row aliases are not.
        expected.append(expected[0].reshape(n,1,2));outputs.append(outputs[0].reshape(n,1,2))
        backing=torch.arange(n*4,device=comm.device,dtype=torch.float32)
        expected.extend((backing[:n*2].reshape(n,2).clone(),backing[n*2:].reshape(n,2).clone()))
        outputs.extend((backing[:n*2].reshape(n,2),backing[n*2:].reshape(n,2)))
        expected.append(torch.empty((n,0,3),device=comm.device));outputs.append(expected[-1].clone())
        for t in outputs:t[:a].fill_(99);t[b:].fill_(99)
        if n==11:
            empty=torch.empty
            def fail(*args,**kw):
                if args==(127,) and kw.get('dtype')==torch.uint8 and comm.rank==0:
                    raise MemoryError('injected particle workspace allocation')
                return empty(*args,**kw)
            with patch.object(torch,'empty',fail),patch.object(dist,'broadcast',forbidden):
                rejected(lambda:comm.exchangeParticleOutputs(adj,outputs,tag='test:packing-oom'),'injected particle')
            comm.agree('test:after-packing-oom',True)
        before=comm.particleBroadcasts
        sequence=comm.sequence
        comm.exchangeParticleOutputs(adj,outputs,tag='test:mixed-packed')
        assert comm.sequence-sequence==1
        for t,want in zip(outputs,expected):equal(t.reshape(-1).view(torch.uint8),want.reshape(-1).view(torch.uint8))
        widths={}
        if n:
            for t in outputs:
                if t.numel():widths[t.dtype]=widths.get(t.dtype,0)+t.numel()//n
        count=sum((end-start+127//(width*torch.empty((),dtype=dtype).element_size())-1)//
                  (127//(width*torch.empty((),dtype=dtype).element_size()))
                  for dtype,width in widths.items() for start,end in adj.plan.ownerRanges)
        assert comm.particleBroadcasts-before==count
        totalBroadcasts+=count
        if n:
            assert comm._particleWorkspace.numel()==127
            if address is None:address=comm._particleWorkspace.data_ptr()
            assert address==comm._particleWorkspace.data_ptr()
        else:assert comm._particleWorkspace is None
        retained.extend((t,want.clone()) for t,want in zip(outputs,expected))
        if n==11:
            overlap=torch.empty(12,device=comm.device)
            partial=overlap[1:] if comm.rank else overlap[:11]
            with patch.object(dist,'broadcast',forbidden):
                rejected(lambda:comm.exchangeParticleOutputs(adj,(overlap[:11],partial),tag='test:packing-overlap'),'overlap')
                alias=overlap[:11] if comm.rank==0 else overlap[:11].clone()
                rejected(lambda:comm.exchangeParticleOutputs(adj,(overlap[:11],alias),tag='test:packing-alias-contract'),'disagree')
                rejected(lambda:comm.exchangeParticleOutputs(adj,(overlap[:11],overlap[:11].view(torch.int32)),
                                                             tag='test:packing-retyped-alias'),'reinterpret')
            # Oversized combined rows fall back directly without growing scratch.
            wide=[torch.full((n,40),float(i),device=comm.device) for i in range(2)]
            for t in wide:t[:a]=-99;t[b:]=-99
            before=comm.particleBroadcasts
            comm.exchangeParticleOutputs(adj,wide,tag='test:packing-wide-fallback')
            assert comm.particleBroadcasts-before==2*sum(end>start for start,end in adj.plan.ownerRanges)
            for i,t in enumerate(wide):equal(t,torch.full_like(t,float(i)))
            # Warm the workspace, then reuse it with varied shapes repeatedly.
            factories={name:getattr(torch,name) for name in ('empty','empty_like','tensor')}
            def noFactory(*a,**kw):raise AssertionError('warm exchange allocated Tensor storage')
            for name in factories:setattr(torch,name,noFactory)
            try:
                for _ in range(4):comm.exchangeParticleOutputs(adj,outputs,tag='test:packing-warm')
            finally:
                for name,function in factories.items():setattr(torch,name,function)
    # A disabled budget has no retained particle workspace.
    disabled=RowChunkProcessGroup(parent.group,parent.device,replicaChecks=parent.replicaChecks,particleWorkspaceBytes=0)
    q,r,domain=sample(comm.device,nq=1)
    adj=radiusSearchRowChunks(q,domain,policy=RowChunkPolicy(edgeBudget=80,maxRows=2),communication=disabled)
    a,b=adj.plan.ownerRanges[disabled.rank]
    fields=[torch.full((1,),float(i),device=comm.device) for i in range(2)]
    for t in fields:t[:a]=-9;t[b:]=-9
    disabled.exchangeParticleOutputs(adj,fields,tag='test:packing-disabled')
    assert disabled._particleWorkspace is None
    for i,t in enumerate(fields):equal(t,torch.full_like(t,float(i)))
    # Budgets affect the payload schedule, so differing live-rank budgets
    # must reject before any payload even when both locally fit the fields.
    mismatch=RowChunkProcessGroup(parent.group,parent.device,replicaChecks=parent.replicaChecks,
                                 particleWorkspaceBytes=127+int(parent.rank!=0))
    adj=radiusSearchRowChunks(q,domain,policy=RowChunkPolicy(edgeBudget=80,maxRows=2),communication=mismatch)
    with patch.object(dist,'broadcast',forbidden):
        rejected(lambda:mismatch.exchangeParticleOutputs(adj,fields,tag='test:packing-budget-mismatch'),'disagree')
    mismatch.agree('test:after-budget-mismatch',True)
    guarded=RowChunkProcessGroup(parent.group,parent.device,replicaChecks=parent.replicaChecks,particleWorkspaceBytes=8)
    q,r,domain=sample(comm.device,nq=11)
    adj=radiusSearchRowChunks(q,domain,policy=RowChunkPolicy(edgeBudget=80,maxRows=2),communication=guarded)
    a,b=adj.plan.ownerRanges[guarded.rank]
    fields=[torch.full((11,),float(i),device=comm.device) for i in range(2)]
    for t in fields:t[:a]=-9;t[b:]=-9
    before=guarded.particleBroadcasts
    guarded.exchangeParticleOutputs(adj,fields,tag='test:packing-call-count-guard')
    assert guarded.particleBroadcasts-before==2*sum(end>start for start,end in adj.plan.ownerRanges)
    assert guarded._particleWorkspace is None
    for i,t in enumerate(fields):equal(t,torch.full_like(t,float(i)))
    if comm.device.type=='cuda':
        # Inputs are produced on each call's stream. A delayed unpack keeps old
        # scratch reads outstanding while the next stream enters a new control.
        streams=[torch.cuda.Stream(device=comm.device) for _ in range(2)]
        for stream in streams:stream.wait_stream(torch.cuda.current_stream(comm.device))
        q,r,domain=sample(comm.device,nq=11)
        adj=radiusSearchRowChunks(q,domain,policy=RowChunkPolicy(edgeBudget=80,maxRows=2),communication=comm)
        a,b=adj.plan.ownerRanges[comm.rank]
        copy=torch.Tensor.copy_
        def delayed(t,source,*args,**kw):
            if source.untyped_storage().data_ptr()==address:torch.cuda._sleep(200000)
            return copy(t,source,*args,**kw)
        with patch.object(torch.Tensor,'copy_',delayed):
            for index in range(12):
                with torch.cuda.stream(streams[index%2]):
                    torch.cuda._sleep(200000)
                    fields=[torch.full((11,width),float(index+width),device=comm.device) for width in (1,2)]
                    for t in fields:t[:a]=-1;t[b:]=-1
                    comm.exchangeParticleOutputs(adj,fields,tag='test:packing-stream-switch')
                    retained.extend((t,torch.full_like(t,float(index+t.shape[1]))) for t in fields)
        for stream in streams:torch.cuda.current_stream(comm.device).wait_stream(stream)
        assert comm.particleWorkspaceWaits>=12
    for t,want in retained:equal(t.reshape(-1).view(torch.uint8),want.reshape(-1).view(torch.uint8))
    assert comm.particleWorkspaceAllocations==1
    assert comm._particleWorkspace.data_ptr()==address
    comm.agree('test:after-packing-audit',True)
    return dict(status='passed',mixedBatchBroadcasts=totalBroadcasts,retainedOutputs=len(retained),coordinatedFailures=5,
                guardedWorkspaceBytes=guarded.summary()['particleWorkspaceBytes'],
                disabledWorkspaceBytes=disabled.summary()['particleWorkspaceBytes'],**comm.summary())


def filteringAudit(comm):
    """Real owner migration against independent tiny flat filtering oracles."""
    from unittest.mock import patch
    from warpSPHCore.radiusSearch import filterChunkedVerletList
    from warpSPHCore.radiusSearch.verlet.build import buildVerletList_
    from warpSPHCore.radiusSearch.verlet.filter import filterVerletList_
    from warpSPHCore.type_config import get_torch_offset_dtype
    import time
    started=time.perf_counter();reports=[]
    for number,(nq,nr,dim,mode,oldRows,newRows) in enumerate([
        (19,None,dim,mode,oldRows,newRows) for dim in (2,3) for mode in SupportScheme
        for oldRows,newRows in ((1,7),(7,1))]+[(0,5,2,SupportScheme.Gather,1,7),
            (3,0,3,SupportScheme.Scatter,7,1),(1,None,2,SupportScheme.SuperSymmetric,1,80)]):
        q,r,domain=sample(comm.device,nq,nr,dim)
        if number%3==0:domain.periodic[0]=True
        oldI=number%2==0;newI=not oldI
        policy=RowChunkPolicy(edgeBudget=80,maxRows=oldRows,bytesPerEdge=8+dim*q.positions.element_size(),maxElementsPerEdge=dim)
        args=(q.positions,r.positions,q.supports,r.supports,domain)
        old=buildChunkedVerletList(q,domain,referenceParticles=r,policy=policy,verletScale=1.5,
                                  supportMode=mode,storeI=oldI,communication=comm)
        flat=buildVerletList_(*args,verletScale=1.5,mode=mode,storeI=oldI)
        saved=[c.adjacency.paddedJ.clone() for c in old]
        work=ChunkedPairField.empty(old,trailingShape=(dim,),dtype=q.positions.dtype)
        for w in work.buffers:w.fill_(comm.rank+.125)
        workSaved=[w.clone() for w in work.buffers]
        q.positions.add_(.0001)
        targetPolicy=replace(policy,maxRows=newRows)
        expected=filterVerletList_(*args,flat,mode,capacity=None,storeI=newI)
        def forbidden(*a,**kw):raise AssertionError('distributed filter used a global search/pair fallback')
        module=importlib.import_module('warpSPHCore.radiusSearch.chunked')
        before=comm.ownedTransfers
        with patch.object(module,'_search',forbidden):
            actual=filterChunkedVerletList(q,domain,old,referenceParticles=r,policy=targetPolicy,storeI=newI)
        assert actual.communication is comm and actual.sourceSignature==old.plan.signature
        assert actual.kind=='filtered' and actual.plan.generation.number==old.plan.generation.number
        check_rows(actual,expected)
        for chunk,j,w,want in zip(old,saved,work.buffers,workSaved):
            equal(chunk.adjacency.paddedJ,j);equal(w,want)
        work.validate(old)
        try:work.validate(actual)
        except ValueError as exc:assert 'replay or migrate' in str(exc)
        else:raise AssertionError('old W accepted a changed filtered list')
        # Live temporary J and local offsets fit one original source chunk.
        offsetBytes=torch.empty((),dtype=get_torch_offset_dtype()).element_size()
        bound=max((c.metadata.liveEdges*4+c.metadata.rowCount*offsetBytes for c in old),default=0)
        assert actual.searchScratchBytes<=bound
        reports.append(dict(rows=nq,references=len(r.positions),dim=dim,mode=mode.name,
            sourceRanges=old.plan.ownerRanges,targetRanges=actual.plan.ownerRanges,
            transfers=comm.ownedTransfers-before,scratchBytes=actual.searchScratchBytes,scratchBound=bound))
        print(f'[distributed core rank {comm.rank}] filter {number}: passed',flush=True)
    def rejected(function,text):
        try:function()
        except (DistributedContractError,ValueError) as exc:assert text in str(exc),str(exc)
        else:raise AssertionError('expected filter rejection: '+text)
    q,r,domain=sample(comm.device)
    policy=RowChunkPolicy(edgeBudget=80,maxRows=1)
    old=buildChunkedVerletList(q,domain,policy=policy,verletScale=1.5,
                              supportMode=SupportScheme.SuperSymmetric,communication=comm)
    rejected(lambda:filterChunkedVerletList(q,domain,old,policy=replace(policy,maxRows=1+comm.rank)),'disagree')
    rejected(lambda:filterChunkedVerletList(q,domain,old,policy=replace(policy,edgeBudget=1)),'exceeds')
    module=importlib.import_module('warpSPHCore.radiusSearch.chunked');allocate=module._allocateChunks
    def fail(*a,**kw):
        if comm.rank==comm.size-1:raise MemoryError('injected filter target allocation')
        return allocate(*a,**kw)
    def noPayload(*a,**kw):raise AssertionError('invalid filter entered owner transfer')
    with patch.object(module,'_allocateChunks',fail),patch.object(dist,'send',noPayload),patch.object(dist,'recv',noPayload):
        rejected(lambda:filterChunkedVerletList(q,domain,old),'injected filter target')
    # One source rank fails its temporary allocation after targets exist.
    empty=torch.empty
    def failTemporary(*a,**kw):
        import sys
        caller=sys._getframe(1).f_code
        if comm.rank==0 and caller.co_name=='prepare' and caller.co_filename.endswith('/radiusSearch/chunked.py'):
            raise MemoryError('injected filter source allocation')
        return empty(*a,**kw)
    with patch.object(torch,'empty',failTemporary),patch.object(dist,'send',noPayload),patch.object(dist,'recv',noPayload):
        rejected(lambda:filterChunkedVerletList(q,domain,old),'injected filter source')
    rejected(lambda:filterChunkedVerletList(replace(q,positions=q.positions+2),domain,old),'stale')
    send=torch.zeros(2 if comm.rank==0 else 1,device=comm.device,dtype=torch.int32) if comm.rank==0 else None
    receive=torch.empty(1,device=comm.device,dtype=torch.int32) if comm.rank==comm.size-1 else None
    with patch.object(dist,'send',noPayload),patch.object(dist,'recv',noPayload):
        rejected(lambda:comm.transferOwned(send,receive,source=0,target=comm.size-1,count=1,dtype=torch.int32,
                    tag='test:bad-owned-transfer',signature='same'),'agreed count/dtype')
    actual=filterChunkedVerletList(q,domain,old,policy=replace(policy,maxRows=7))
    old.assertStorageValid();comm.validateAdjacency(actual)
    assert comm.ownedSendBytes+comm.ownedReceiveBytes>0
    comm.agree('test:filter-recovered',True)
    return dict(status='passed',fixtures=reports,coordinatedFailures=6,wallSeconds=time.perf_counter()-started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replica-checks',choices=('exact','contract'),default='exact')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output')
    parser.add_argument('--die-rank', type=int)
    args = parser.parse_args()
    if args.device.startswith('cuda'):
        torch.cuda.set_device(torch.device(args.device))
    dist.init_process_group('nccl' if args.device.startswith('cuda') else 'gloo', timeout=timedelta(seconds=30))
    comm = RowChunkProcessGroup(dist.group.WORLD,args.device,replicaChecks=args.replica_checks)
    wp.init()
    setNeighborLanes(1)
    try:
        if args.die_rank is not None:
            if comm.rank == args.die_rank:
                raise RuntimeError('injected fatal rank failure')
            comm.agree('test:peer-died', {})
            raise AssertionError('peer death must fail the job')
        # Non-default streams cover Warp producers -> c10d -> Warp consumers.
        stream = torch.cuda.Stream(device=comm.device) if comm.device.type == 'cuda' else None
        import contextlib
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            checks = correctness(comm)
            errors = failures(comm)
            protocol=protocolAudit(comm)
            packing=packingAudit(comm)
            filtering=filteringAudit(comm)
        if stream is not None:
            stream.synchronize()
        summary = dict(status='passed', pid=os.getpid(), checks=checks, coordinatedFailures=errors,protocol=protocol,packing=packing,filtering=filtering,
                       **comm.summary())
        if comm.device.type == 'cuda':
            import ctypes
            version = ctypes.c_int()
            ctypes.CDLL('libnccl.so.2').ncclGetVersion(ctypes.byref(version))
            summary['runtimeNccl'] = version.value
            summary['gpuUuid'] = str(torch.cuda.get_device_properties(comm.device).uuid)
            summary['loadedNccl'] = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                                           if 'libnccl.so' in line})
        if args.output:
            Path(args.output).mkdir(parents=True, exist_ok=True)
            (Path(args.output)/f'core-rank{comm.rank}.json').write_text(json.dumps(summary,indent=2))
        print(f'[distributed core rank {comm.rank}] all checks passed', flush=True)
    finally:
        setNeighborLanes(None)
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
