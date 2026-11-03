"""Native CG/MG closure, reconstruction and flux across owned rows."""
from datetime import timedelta
from pathlib import Path
import argparse
import contextlib
import json
from unittest.mock import patch

import torch
import warp as wp
import torch.distributed as dist
from _rowChunkWorker import sample
from warpSPHCore import KernelFunctions, SupportScheme
from warpSPHCore.distributed import RowChunkProcessGroup, DistributedAllocationError
from warpSPHCore.radiusSearch import RowChunkPolicy, buildChunkedVerletList, ChunkedPairField
from warpSPHCore.mfm import MeshlessWarp


def qualify(comm):
    reports=[]
    for dim in (2,3):
        q,_,domain=sample(comm.device,nq=19,dim=dim)
        flat=MeshlessWarp.build(q.positions,q.supports,domain,KernelFunctions.Wendland2,closure='none')
        adjacency=buildChunkedVerletList(q,domain,policy=RowChunkPolicy(edgeBudget=1024,maxRows=2,
            bytesPerEdge=4+q.positions.element_size()),supportMode=SupportScheme.SuperSymmetric,communication=comm)
        for solver,cap in (('cg',0.),('mg',0.),('cg',.02)):
            control=MeshlessWarp.build(q.positions,q.supports,domain,KernelFunctions.Wendland2,
                adjacency=flat.adjacency,closure='project',closure_solver=solver,closure_cap=cap,mg_max_coarse=16)
            owned=MeshlessWarp.build(q.positions,q.supports,domain,KernelFunctions.Wendland2,
                adjacency=adjacency,closure='project',closure_solver=solver,closure_cap=cap,mg_max_coarse=16)
            tolerance=5e-5 if q.positions.dtype==torch.float32 else 2e-11
            torch.testing.assert_close(owned.lam,control.lam,rtol=tolerance,atol=tolerance)
            rho,vel,pressure=q.densities,.2*q.positions.cos(),1+.1*q.positions[:,0].cos()
            for mode in ('MFM','MFV'):
                expected=control.rates(rho,vel,pressure,1.4,mode=mode,dt=.001)
                actual=owned.rates(rho,vel,pressure,1.4,mode=mode,dt=.001)
                torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance)
                comm.assertReplicated('meshless:test:rates',dict(rate=actual),exact=True)
            reports.append(dict(dim=dim,solver=solver,closureCap=cap,passed=True))
        empty=ChunkedPairField.empty
        @classmethod
        def fail(cls,*args,**kwargs):
            if comm.rank==comm.size-1:
                raise torch.OutOfMemoryError('injected meshless owned weights')
            return empty(*args,**kwargs)
        with patch.object(ChunkedPairField,'empty',fail):
            try:
                owned.close(closure_solver='cg')
            except DistributedAllocationError as error:
                assert 'injected meshless' in str(error)
            else:
                raise AssertionError('expected coordinated allocation failure')
        comm.agree('meshless:test:recovered',dim)
    return reports


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',required=True)
    parser.add_argument('--replica-checks',default='exact')
    args=parser.parse_args()
    if args.device.startswith('cuda'):torch.cuda.set_device(torch.device(args.device))
    dist.init_process_group('nccl' if args.device.startswith('cuda') else 'gloo',timeout=timedelta(seconds=60))
    wp.init()
    comm=RowChunkProcessGroup(dist.group.WORLD,args.device,replicaChecks=args.replica_checks)
    try:
        stream=torch.cuda.Stream() if comm.device.type=='cuda' else None
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            report=dict(status='passed',cases=qualify(comm),communication=comm.summary())
        if stream is not None:stream.synchronize()
        output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
        (output/f'rank-{comm.rank}.json').write_text(json.dumps(report,indent=2))
        print(f'meshless core rank {comm.rank}: passed',flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
