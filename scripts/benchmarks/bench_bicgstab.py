#!/usr/bin/env python3
"""BiCGStab Torch/native/cached comparison on a fixed nonsymmetric SPH action.

Separate eight scheduled steps from verified solve-to-tolerance measurements.
Report whole-call time, actual pair work, first-use setup and owned scratch.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
import warp as wp

from _work_metrics import WorkLedger
from warpSPHCore import (
    GradientScheme, KernelFunctions, LaplacianScheme, OperationProperties,
    ParticleState, SupportScheme, WarpOperation, radiusSearchCompactHashMap, warpOperation,
)
from warpSPHCore.linalg import BiCGStabWorkspace, bicgstab
from warpSPHCore.type_config import get_torch_precision
from warpSPHCore.util import generateNeighborTestData


def timed(solve):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    before = time.perf_counter()
    result = solve()
    end.record()
    end.synchronize()
    return result, (time.perf_counter()-before)*1e3, start.elapsed_time(end)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sides', type=int, nargs='+', default=[32,256,1024])
    parser.add_argument('--neighbors', type=int, nargs='+', default=[25,55])
    parser.add_argument('--engines', nargs='+', choices=['torch','native','native-cached','native-out','native-cached-out'],
                        default=['torch','native','native-cached','native-out','native-cached-out'])
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--fixed-steps', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('requires an accessible CUDA device')
    if min(*args.sides,*args.neighbors,args.repeat,args.fixed_steps) < 1:
        parser.error('sizes, sample counts and budgets must be positive')
    wp.init()
    torch.set_num_threads(2)
    dtype = get_torch_precision()
    rows=[]
    for side in args.sides:
        for requested in args.neighbors:
            positions,supports,_,domain,dx = generateNeighborTestData(side,requested,2,True,'cuda')
            n = positions.shape[0]
            particles = ParticleState(positions=positions,supports=supports,
                masses=torch.full((n,),dx**2,dtype=dtype,device='cuda'),
                densities=torch.ones(n,dtype=dtype,device='cuda'),kinds=torch.zeros(n,dtype=torch.int32,device='cuda'))
            adjacency = radiusSearchCompactHashMap(particles,domain,mode=SupportScheme.Gather)
            edges = int(adjacency.numNeighbors.sum())
            props = OperationProperties(operation=WarpOperation.Laplacian,kernel=KernelFunctions.Wendland2,
                gradientMode=GradientScheme.Difference,laplacianMode=LaplacianScheme.Brookshaw,supportMode=SupportScheme.Gather)
            scale = dx**2*(.5+.25*torch.sin(9*positions[:,0]))
            inverse = 1/(1+scale/dx**2)

            def laplacian(v):
                return warpOperation(particles,props,domain,adjacency=adjacency,queryValues=v,referenceValues=v)

            def action(v):
                return v-scale*laplacian(v)

            def into(v,out):
                torch.mul(laplacian(v),scale,out=out)
                torch.sub(v,out,out=out)

            generator = torch.Generator(device='cuda').manual_seed(17)
            known = torch.randn(n,dtype=dtype,device='cuda',generator=generator)
            b = action(known)
            for mode in ('fixed','verified'):
                rtol = 0 if mode=='fixed' else (2e-5 if dtype==torch.float32 else 1e-10)
                budget = args.fixed_steps if mode=='fixed' else 128
                checkEvery = args.fixed_steps if mode=='fixed' else 8
                for engine in args.engines:
                    torch.cuda.current_stream().synchronize()
                    heldInputs = torch.cuda.memory_allocated()
                    workspace = (None if engine=='torch' else BiCGStabWorkspace(b,budget,history=False,capture='cached' in engine))
                    kwargs = dict(rtol=rtol,maxiter=budget,checkEvery=checkEvery,precond=inverse,
                                  maxRestarts=0 if mode=='fixed' else 2,workspace=workspace,
                                  matvecOut=into if engine.endswith('-out') else None)
                    solve = lambda: bicgstab(action,b,**kwargs)
                    warm,firstWall,firstDevice = timed(solve)
                    firstSetup = warm.setupIterations
                    del warm
                    torch.cuda.current_stream().synchronize()
                    baseline = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    walls, intervals, diagnostics, callPeaks = [],[],[],[]
                    for _ in range(args.repeat):
                        torch.cuda.reset_peak_memory_stats()
                        result,wall,interval = timed(solve)
                        callPeaks.append(torch.cuda.max_memory_allocated())
                        walls.append(wall)
                        intervals.append(interval)
                        actual = (b-action(result.x)).norm()
                        torch.testing.assert_close(result.trueResidual,actual,rtol=0,atol=0)
                        if not bool(result.x.isfinite().all()):
                            raise RuntimeError('nonfinite returned iterate')
                        if mode=='verified' and (result.statusInt()<0 or not bool(actual<=rtol*b.norm())):
                            raise RuntimeError('solve failed independent original-residual verification')
                        if mode=='fixed' and result.loopIterations!=args.fixed_steps:
                            raise RuntimeError('fixed scheduled work was not matched')
                        diagnostics.append(dict(status=result.statusInt(),iterations=int(result.iterations),
                            loops=result.loopIterations,matvecs=result.matvecs,verifications=result.verifications,
                            preconditionerApplications=result.preconditionerApplications,restarts=result.restarts,
                            setupIterations=result.setupIterations,relativeResidual=float(actual/b.norm()),
                            relativeSolutionError=float((result.x-known).norm()/known.norm())))
                        del result
                    if any(row!=diagnostics[0] for row in diagnostics):
                        raise RuntimeError('work changed between repeated identical solves')
                    count = diagnostics[0]
                    median = statistics.median(walls)
                    ledger = WorkLedger()
                    for role,passes in (('scheduled_recurrence',count['matvecs']-count['verifications']),
                                        ('original_residual_verification',count['verifications'])):
                        ledger.pair('Brookshaw Laplacian',evaluated_nonself_pairs=edges-n,candidate_entries=edges,
                            passes=passes,role=role,basis='actual periodic equal-support CSR; all callback evaluations')
                    ledger.particle('identity_minus_row_scaled_laplacian',particles=n,passes=count['matvecs'])
                    ledger.particle('inverse_diagonal',particles=n,passes=count['preconditionerApplications'])
                    for name in ('BiCGStab half candidate','BiCGStab full candidate','BiCGStab direction and commit'):
                        ledger.particle(name,particles=n,passes=count['loops'],basis='named algorithmic stage; not a kernel or FLOP count')
                    row = dict(n=n,edges=edges,meanDegree=edges/n,requestedNeighbors=requested,mode=mode,engine=engine,
                        rtol=rtol,wallMs=walls,medianWallMs=median,deviceIntervalsMs=intervals,
                        firstCallWallMs=firstWall,firstCallDeviceIntervalMs=firstDevice,firstCallSetupIterations=firstSetup,
                        workspaceBytes=0 if workspace is None else workspace.nbytes,
                        captureBuilds=0 if workspace is None else workspace.captureBuilds,
                        captureMilliseconds=0 if workspace is None else workspace.captureMilliseconds,
                        peakTorchBytes=max(callPeaks),incrementalPeakTorchBytes=max(callPeaks)-baseline,
                        residentTorchBytesAfterWarmup=baseline-heldInputs,
                        peakAdditionalTorchBytes=max(callPeaks)-heldInputs,
                        nativeRecurrenceReductions=None if workspace is None else count['restarts']+1+8*(count['loops']+count['setupIterations']),
                        **count,work=ledger.report(wall_ms=median,timing_scope='whole warmed BiCGStab call, including residual certification'))
                    rows.append(row)
                    print(json.dumps({k:row[k] for k in ('n','meanDegree','mode','engine','medianWallMs','iterations','loops','relativeResidual')}),flush=True)
                    del solve,kwargs,workspace
                    gc.collect()
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__).resolve(),root/'scripts/benchmarks/_work_metrics.py',
        root/'src/warpSPHCore/linalg/krylov/bicgstab.py',root/'src/warpSPHCore/linalg/krylov/bicgstabWorkspace.py',
        root/'src/warpSPHCore/linalg/krylov/_workspace.py',root/'src/warpSPHCore/linalg/krylov/common.py',
        root/'src/warpSPHCore/coreOperations/wp_laplacian.py',root/'src/warpSPHCore/autograd/stream.py']
    payload = dict(schemaVersion=1,scope=__doc__,torch=torch.__version__,warp=wp.__version__,
        gpu=torch.cuda.get_device_name(),dtype=str(dtype),records=rows,
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        sourceSha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+'\n')


if __name__=='__main__':
    main()
