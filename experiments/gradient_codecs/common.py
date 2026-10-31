"""Fixture preparation and qualification, never called inside timed operators."""
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import warpSPHCore as sph
from warpSPHCore.native.api import radiusSearch
from warpSPHCore.crk.crk_volume import _computeCRKVolume_stateBackend
from warpSPHCore.crk.crk_moments import _computeCRKMoments_stateBackend
from warpSPHCore.crk.crk_terms import computeCRKTermsWarp


def activity():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory',
                                  '--format=csv,noheader'],text=True).strip()
    import os
    return dict(processes=raw,other_processes=[line for line in raw.splitlines()
                if line.split(',')[0].strip()!=str(os.getpid())],
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,clocks.sm,power.draw',
                                     '--format=csv,noheader'],text=True).strip())


def fixture(n,dimension,dtype,nh):
    generator=torch.Generator().manual_seed(9067+n*3+dimension)
    x=torch.rand((n,dimension),generator=generator,dtype=torch.float64).to(device='cuda',dtype=dtype)
    h=x.new_full((n,),nh*n**(-1/dimension))
    state=sph.ParticleState(x,h,h.new_full((n,),1/n),torch.zeros(n,device='cuda',dtype=torch.int32),h.new_ones(n))
    # Smooth periodic fields avoid an artificial discontinuity at the domain edge.
    vector=torch.sin(x*(2*math.pi))+.2*torch.cos(x*(4*math.pi))
    scalar=(vector*torch.arange(1,dimension+1,device=x.device,dtype=x.dtype)).sum(-1)
    domain=sph.DomainDescription(x.new_zeros(dimension),x.new_ones(dimension),
                                 torch.ones(dimension,device='cuda',dtype=torch.bool),dimension)
    return state,scalar,vector,domain


def morton_order(state,planner_library,dimension):
    if dimension==1:
        return torch.argsort(state.positions[:,0],stable=True)
    spec=importlib.util.spec_from_file_location('gradient_cell_planner',ROOT/'experiments/cluster_pairs/runtime.py')
    planner=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=planner
    spec.loader.exec_module(planner)
    library=planner.Library(planner_library)
    grid=math.floor(1/float(state.supports[0]))
    # Same refined Morton order as the interpolation studies, without retaining
    # cell traversal buffers that this exact-CSR experiment does not consume.
    keys=library.keys(state.positions,grid*8)
    return torch.argsort(keys,stable=True)


def reorder(state,order):
    return replace(state,**{name:getattr(state,name)[order] for name in
                            ('positions','supports','masses','densities','kinds')})


def properties(work):
    operation=(sph.WarpOperation.Gradient,sph.WarpOperation.Gradient,
               sph.WarpOperation.Divergence,sph.WarpOperation.Covariance)[work]
    return sph.OperationProperties(sph.KernelFunctions.Wendland2,operation=operation,
        gradientMode=sph.GradientScheme.Difference,supportMode=sph.SupportScheme.Gather,
        operationMode=sph.OperationDirection.TrueAllToToAll)


def crk_factors(state,domain,adjacency,batch=262144,batch_plan=None):
    """Use actual apparent-volume moments, with dimensionless coefficient solves.

    The production solver has an absolute det(m2)<1e-14 limiter. In a unit box,
    shrinking h alone eventually activates it for essentially every 3D particle.
    Normalize moment coordinates by each query's h before that unchanged solve,
    then transform B, gradA and gradB back to physical coordinates. This keeps
    the benchmark a nontrivial CRK workload at large N. It changes no runtime op.
    """
    begin=time.perf_counter()
    props=properties(0)
    if batch_plan is not None:
        volumes=batch_plan.volumes(props)
        raw=batch_plan.moments(props,volumes)
    else:
        volumes=_computeCRKVolume_stateBackend(state,props,domain,adjacency=adjacency)
        raw=_computeCRKMoments_stateBackend(state,props,domain,volumes,volumes,adjacency=adjacency)
    m0,m1,m2,dm0,dm1,dm2,counts=raw
    n,dimension=state.positions.shape
    A=torch.empty_like(m0);B=torch.empty_like(m1);gradA=torch.empty_like(dm0);gradB=torch.empty_like(dm1)
    for start in range(0,n,batch):
        part=slice(start,min(start+batch,n))
        h=state.supports[part]
        aa,bb,ga,gb=computeCRKTermsWarp(m0[part],m1[part]/h[:,None],m2[part]/h[:,None,None].square(),
            dm0[part]*h[:,None],dm1[part],dm2[part]/h[:,None,None,None],counts[part],torch.ones_like(h))
        A[part]=aa;B[part]=bb/h[:,None];gradA[part]=ga/h[:,None];gradB[part]=gb/h[:,None,None].square()
    torch.cuda.synchronize()
    correction=sph.CRKState(A.contiguous(),B.contiguous(),gradA.contiguous(),gradB.contiguous())
    statistics=dict(wall_ms=(time.perf_counter()-begin)*1000,
        physical_limiter_fraction=float(((counts<2)|(torch.det(m2).abs()<1e-14)).float().mean()),
        identity_fraction=float(((A==1)&(B==0).all(-1)&(gradA==0).all(-1)&(gradB==0).all((-1,-2))).float().mean()),
        finite=all(bool(torch.isfinite(value).all()) for value in (volumes,A,B,gradA,gradB)),
        note='Apparent-volume Warp moments; batched dimensionless solve with unchanged CRK term solver; coefficients transformed back to physical coordinates.')
    if not statistics['finite']:raise AssertionError('Nonfinite CRK factors')
    limited=(A==1)&(B==0).all(-1)&(gradA==0).all(-1)&(gradB==0).all((-1,-2))
    if bool((~limited).any()):
        constant=A*(m0+(B*m1).sum(-1))-1
        linear=A[:,None]*(m1+torch.einsum('nij,nj->ni',m2,B))/state.supports[:,None]
        statistics['constant_constraint_max_abs']=float(constant[~limited].abs().max())
        statistics['linear_constraint_scaled_max_abs']=float(linear[~limited].abs().max())
    if n<=4096 and statistics['physical_limiter_fraction']==0:
        original=computeCRKTermsWarp(m0,m1,m2,dm0,dm1,dm2,counts,state.supports)
        statistics['normalized_vs_physical']={name:error(value,expected) for name,value,expected in
            zip(('A','B','gradA','gradB'),(A,B,gradA,gradB),original)}
        tolerance=2e-3 if A.dtype==torch.float32 else 3e-10
        assert all(item['relative_l2']<tolerance for item in statistics['normalized_vs_physical'].values())
    return volumes,correction,statistics


def error(actual,expected):
    difference=actual.double()-expected.double()
    denom=float(expected.double().norm())
    return dict(relative_l2=float(difference.norm())/max(denom,1e-30),max_abs=float(difference.abs().max()),
                reference_rms=float(expected.double().square().mean().sqrt()),finite=bool(torch.isfinite(actual).all()))


def sample(function,iterations):
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize();start=time.perf_counter();begin.record()
    for _ in range(iterations):result=function()
    end.record();end.synchronize()
    return dict(event_ms=begin.elapsed_time(end)/iterations,wall_ms=(time.perf_counter()-start)*1000/iterations)


class WarpBatches:
    """Planning adapter for Warp's signed-int32 array-shape limit.

    Neighbor IDs remain global reference IDs. Only query rows, their offsets and
    query-side fields are sliced. The operation includes concatenating outputs;
    no topology inspection or offset readback occurs in a timed call.
    """
    def __init__(self,state,adjacency,domain,max_rows=4000000,edge_limit=2**31-1):
        self.reference=state;self.domain=domain;self.batches=[]
        n=len(state.positions);start=0
        while start<n:
            stop=min(start+max_rows,n)
            begin=int(adjacency.edgeOffsets[start])
            while True:
                end=int(adjacency.edgeOffsets[stop]) if stop<n else adjacency.numEdges
                if end-begin<=edge_limit:break
                if stop-start==1:raise ValueError('A single row exceeds the Warp array-shape limit')
                stop=start+(stop-start)//2
            query=reorder(state,slice(start,stop))
            block=sph.AdjacencyList(None,adjacency.paddedJ[begin:end],adjacency.numNeighbors[start:stop],
                adjacency.edgeOffsets[start:stop]-begin,stop-start,n,numEdges=end-begin)
            self.batches.append((query,block,start,stop));start=stop

    def operation(self,props,values,volumes=None,correction=None):
        calls=[]
        for query,adjacency,start,stop in self.batches:
            kwargs={}
            if correction is not None:
                kwargs=dict(queryVolumes=volumes[start:stop],referenceVolumes=volumes,
                    crkState=sph.CRKState(**{name:getattr(correction,name)[start:stop] for name in ('A','B','gradA','gradB')}))
            calls.append((query,adjacency,values[start:stop],kwargs))
        def run():
            results=[sph.warpOperation(query,props,self.domain,query_values,referenceValues=values,
                adjacency=adjacency,referenceParticles=self.reference,**kwargs)
                for query,adjacency,query_values,kwargs in calls]
            return torch.cat(results) if len(results)>1 else results[0]
        return run

    def volumes(self,props):
        return torch.cat([_computeCRKVolume_stateBackend(query,props,self.domain,adjacency=adjacency,
            referenceParticles=self.reference) for query,adjacency,_,_ in self.batches])

    def moments(self,props,volumes):
        parts=[_computeCRKMoments_stateBackend(query,props,self.domain,volumes[start:stop],volumes,
            adjacency=adjacency,referenceParticles=self.reference) for query,adjacency,start,stop in self.batches]
        return tuple(torch.cat(values) for values in zip(*parts))
