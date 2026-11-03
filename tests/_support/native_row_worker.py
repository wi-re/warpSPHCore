"""Focused native owned-row integration; run with torchrun, never as a benchmark."""
import importlib.abc,sys
class BlockWarp(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name=='warp' or name.startswith('warp.'):raise RuntimeError('Unexpected Warp import '+name)
sys.meta_path.insert(0,BlockWarp())
from datetime import timedelta
from dataclasses import replace
import torch
import torch.distributed as dist
import warpSPHCore as sph
from warpSPHCore.distributed import RowChunkProcessGroup,DistributedContractError
from warpSPHCore.mfm import MeshlessNative

def check(left,right):
 if isinstance(left,torch.Tensor):torch.testing.assert_close(left,right,rtol=1e-9,atol=1e-9)
 else:
  for a,b in zip(left,right):check(a,b)

def main():
 torch.set_num_threads(2)
 dist.init_process_group('gloo',timeout=timedelta(seconds=60))
 try:
  for checks in ('exact','contract'):
   comm=RowChunkProcessGroup(dist.group.WORLD,'cpu',replicaChecks=checks)
   D=2;n=13;t=torch.arange(n,dtype=torch.float64)
   x=torch.stack((t.sin(),(t*.7).cos()),-1)*.5
   h=torch.ones_like(t)*1.1
   p=sph.ParticleState(x,h,h*.02,torch.zeros(n,dtype=torch.int32),torch.ones_like(h))
   domain=sph.DomainDescription(x.new_full((D,),-1),x.new_full((D,),1),torch.ones(D,dtype=torch.bool),D)
   indexBytes=4 if sph.get_torch_index_dtype()==torch.int32 else 8
   policy=sph.RowChunkPolicy(edgeBudget=70,maxRows=3,bytesPerEdge=indexBytes+8,maxElementsPerEdge=1)
   adjacency=sph.buildChunkedVerletList(p,domain,policy=policy,verletScale=1.2,communication=comm)
   assert sph.buildChunkedVerletList(p,domain,policy=policy,verletScale=1.2,communication=comm,priorNeighborhood=adjacency) is adjacency
   filtered=sph.filterChunkedVerletList(p,domain,adjacency,policy=replace(policy,maxRows=2),communication=comm)
   flat=sph.radiusSearch(p,domain,mode=sph.SupportScheme.SuperSymmetric)
   for op in (sph.WarpOperation.Density,sph.WarpOperation.Gradient,sph.WarpOperation.Covariance):
    props=sph.OperationProperties(sph.KernelFunctions.Wendland2,operation=op,gradientMode=sph.GradientScheme.Difference)
    check(sph.warpOperation(p,props,domain,x,adjacency=filtered,covarianceReturnNumNeighbors=True),
          sph.warpOperation(p,props,domain,x,adjacency=flat,covarianceReturnNumNeighbors=True))
   a=sph.prepareCRK(p,domain,sph.KernelFunctions.Wendland2,adjacency=filtered)
   b=sph.prepareCRK(p,domain,sph.KernelFunctions.Wendland2,adjacency=flat)
   check((a.queryVolumes,a.crkState.A,a.consistencyDensity),(b.queryVolumes,b.crkState.A,b.consistencyDensity))
   fit=sph.PolyFit.build(p,domain,sph.KernelFunctions.Wendland2,2,constant=False,adjacency=filtered)
   ref=sph.PolyFit.build(p,domain,sph.KernelFunctions.Wendland2,2,constant=False,adjacency=flat)
   check(fit.gradient(x[:,0].square()),ref.gradient(x[:,0].square()))
   for solver in ('cg','mg'):
    a=MeshlessNative.build(x,h,domain,sph.KernelFunctions.Wendland2,adjacency=filtered,
        closure_solver=solver,mg_max_coarse=4,cg_iters=50)
    b=MeshlessNative.build(x,h,domain,sph.KernelFunctions.Wendland2,adjacency=flat,
        closure_solver=solver,mg_max_coarse=4,cg_iters=50)
    check(a.rates(p.densities,x.sin()*.1,p.densities,1.4),b.rates(p.densities,x.sin()*.1,p.densities,1.4))
   state=sph.rowChunkCheckpointState(adjacency)
   restored=sph.restoreRowChunkCheckpointState(state,device='cpu',policy=replace(policy,maxRows=2),communication=comm)
   check(restored.numNeighbors,adjacency.numNeighbors)
   # A rank-local invalid field must become a coordinated contract failure.
   bad=x[:1] if comm.rank==0 else x
   try:sph.warpOperation(p,sph.OperationProperties(sph.KernelFunctions.Wendland2),domain,bad,adjacency=filtered)
   except DistributedContractError:pass
   else:raise AssertionError('Expected coordinated field-layout failure')
   comm.agree('native:recovered',checks)
   print(f'rank {comm.rank}: native {checks} owned-row integration passed',flush=True)
 finally:dist.destroy_process_group()

if __name__=='__main__':main()
