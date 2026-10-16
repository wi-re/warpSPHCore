#!/usr/bin/env python3
"""Preconditioner setup/application costs and verified vector-SPH GMRES solves.

Local inverse actions use entries/blocks per second; solver ledgers count real
SPH interactions separately from local matrices. No whole-scheme claim.
"""

import argparse
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
    ParticleState, SupportScheme, WarpOperation, radiusSearchCompactHashMap,
    warpOperation,
)
from warpSPHCore.linalg import BlockJacobiPreconditioner, DiagonalPreconditioner, gmres
from warpSPHCore.type_config import get_torch_precision
from warpSPHCore.util import generateNeighborTestData


def measure(action, repeat, *, batch=1, validate=None):
    walls, intervals = [], []
    for _ in range(repeat):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        before = time.perf_counter()
        output = action()
        end.record()
        end.synchronize()
        walls.append((time.perf_counter() - before) * 1e3 / batch)
        intervals.append(start.elapsed_time(end) / batch)
        if validate is not None:
            validate(output)
    return output, dict(wallMs=walls, deviceIntervalMs=intervals,
                        medianWallMs=statistics.median(walls), medianDeviceIntervalMs=statistics.median(intervals))


def captured(action, repeat, batch):
    # Warm on an explicitly ordered side stream before capture; retain the
    # final output so the captured ordinary-call allocation stays alive.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    outputs = []
    with torch.cuda.stream(stream):
        for _ in range(3):
            output = action()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(batch):
            output = action()
        outputs.append(output)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.current_stream().synchronize()
    _, timing = measure(graph.replay, repeat, batch=batch)
    return outputs[0], timing


def localBlocks(n, d, dtype):
    matrix = torch.eye(d, device='cuda', dtype=dtype)
    matrix[0, 0], matrix[1, 1] = 12., 19.
    matrix[0, 1], matrix[1, 0] = 14., 14.
    # Eigenvalues of the 2x2 part are positive (~1.07 and 29.93).
    return matrix[None].expand(n, d, d).contiguous()


def applications(args, dtype):
    rows = []
    for n in args.ns:
        for d in (1, 2, 3):
            coefficients = (torch.linspace(1, 3, n, device='cuda', dtype=dtype) if d == 1
                            else localBlocks(n, d, dtype))
            v = torch.linspace(-1, 1, n*d, device='cuda', dtype=dtype)
            kinds = ['diagonal'] if d == 1 else ['block', 'block_spd', 'block_native']
            for kind in kinds:
                def build():
                    return (DiagonalPreconditioner(coefficients, requireSPD=True) if kind == 'diagonal'
                            else BlockJacobiPreconditioner(coefficients, requireSPD=kind != 'block',
                                                          application='native' if kind == 'block_native' else 'torch'))

                helper = build()
                helper, setup = measure(build, args.repeat)
                inverse = helper.inverse
                expected = inverse * v if d == 1 else torch.einsum('nab,nb->na', inverse, v.reshape(n, d)).flatten()
                out = torch.empty_like(v)
                helper.apply(v, out)
                _, eager = measure(lambda: helper(v), args.repeat)
                observed, graph = captured(lambda: helper(v), args.repeat, args.batch)
                torch.testing.assert_close(observed, expected)
                supplied, graphOut = captured(lambda: helper.apply(v, out), args.repeat, args.batch)
                torch.testing.assert_close(supplied, expected)
                baseline = (lambda: inverse * v) if d == 1 else (lambda: torch.einsum('nab,nb->na', inverse, v.reshape(n, d)).flatten())
                _, reference = captured(baseline, args.repeat, args.batch)
                # Count only named local actions, never particle pairs.
                row = dict(n=n, d=d, kind=kind, setup=setup, eagerApplication=eager,
                           capturedApplication=graph, capturedSuppliedOutput=graphOut,
                           cachedTorchReference=reference, nbytes=helper.nbytes,
                           localActionsPerSecond=n / (graph['medianDeviceIntervalMs'] * 1e-3),
                           workUnit='scalar entries' if d == 1 else 'local block-matrix applications')
                rows.append(row)
                print(json.dumps(dict(scope='application', n=n, d=d, kind=kind,
                                      setupMs=setup['medianWallMs'], applicationUs=graph['medianDeviceIntervalMs']*1e3)), flush=True)
    return rows


def solves(args, dtype):
    rows = []
    rtol = 2e-5 if dtype == torch.float32 else 1e-10
    for side in args.sph_sides:
        for requested in args.neighbors:
            positions, supports, _, domain, dx = generateNeighborTestData(side, requested, 2, True, 'cuda')
            n = positions.shape[0]
            particles = ParticleState(positions=positions, supports=supports,
                masses=torch.full((n,), dx**2, dtype=dtype, device='cuda'),
                densities=torch.ones(n, dtype=dtype, device='cuda'), kinds=torch.zeros(n, dtype=torch.int32, device='cuda'))
            adjacency = radiusSearchCompactHashMap(particles, domain, mode=SupportScheme.Gather)
            edges = int(adjacency.numNeighbors.sum())
            props = OperationProperties(operation=WarpOperation.Laplacian, kernel=KernelFunctions.Wendland2,
                gradientMode=GradientScheme.Difference, laplacianMode=LaplacianScheme.Brookshaw,
                supportMode=SupportScheme.Gather)
            B = localBlocks(n, 2, dtype)
            # Positive local scale variation preserves block conditioning.
            B = B * (1 + .1*positions[:, 0].sin())[:, None, None]

            def action(v):
                values = v.reshape(n, 2)
                lap = warpOperation(particles, props, domain, adjacency=adjacency,
                                    queryValues=values, referenceValues=values)
                local = torch.bmm(B, values[:, :, None])[:, :, 0]
                return (local - .05*dx**2*lap).flatten()

            generator = torch.Generator(device='cuda').manual_seed(29)
            known = torch.randn(n*2, dtype=dtype, device='cuda', generator=generator)
            b = action(known)
            for kind in ('none', 'diagonal', 'block', 'block_spd', 'block_native'):
                torch.cuda.current_stream().synchronize()
                before = time.perf_counter()
                if kind == 'none':
                    precond = None
                elif kind == 'diagonal':
                    precond = DiagonalPreconditioner(B.diagonal(dim1=-2, dim2=-1).flatten(), requireSPD=True)
                else:
                    precond = BlockJacobiPreconditioner(B, requireSPD=kind != 'block',
                        application='native' if kind == 'block_native' else 'torch')
                torch.cuda.current_stream().synchronize()
                setupMs = 0.0 if kind == 'none' else (time.perf_counter() - before) * 1e3
                kw = dict(precond=precond, rtol=rtol, maxiter=160, restart=20, history=False)
                warm = gmres(action, b, **kw)
                if warm.statusInt() < 0:
                    raise RuntimeError(f'GMRES warmup failed: {warm.statusInt()}')
                del warm
                diagnostics = []

                def verify(result):
                    actual = (b-action(result.x)).norm()
                    if result.statusInt() < 0 or not bool(actual <= rtol*b.norm()):
                        raise RuntimeError('independent original-residual verification failed')
                    torch.testing.assert_close(result.trueResidual, actual, rtol=0, atol=0)
                    diagnostics.append(dict(status=result.statusInt(), iterations=int(result.iterations), loops=result.loopIterations,
                        matvecs=result.matvecs, verifications=result.verifications, restarts=result.restarts,
                        preconditionerApplications=result.preconditionerApplications))

                result, timing = measure(lambda: gmres(action, b, **kw), args.repeat, validate=verify)
                if any(row != diagnostics[0] for row in diagnostics):
                    raise RuntimeError('solver work changed between matched samples')
                actual = (b-action(result.x)).norm()
                if result.statusInt() < 0 or not bool(actual <= rtol*b.norm()):
                    raise RuntimeError('independent original-residual verification failed')
                torch.testing.assert_close(result.trueResidual, actual, rtol=0, atol=0)
                count = diagnostics[0]
                ledger = WorkLedger().pair('vector Brookshaw Laplacian', evaluated_nonself_pairs=edges-n,
                    candidate_entries=edges, passes=count['matvecs'], basis='actual fixed CSR, one vector pair visit per matvec')
                ledger.local_matrix('local operator block', matrices=n, rows=2, cols=2, method='multiply', passes=count['matvecs'])
                if kind != 'none':
                    if kind == 'diagonal':
                        ledger.particle('diagonal inverse action (2 components)', particles=n, passes=count['preconditionerApplications'])
                    else:
                        ledger.local_matrix('block inverse action', matrices=n, rows=2, cols=2, method='multiply', passes=count['preconditionerApplications'])
                row = dict(n=n, d=2, edges=edges, meanDegree=edges/n, requestedNeighbors=requested, kind=kind,
                    setupMs=setupMs, timing=timing, **count, rtol=rtol,
                    relativeResidual=float(actual/b.norm()), relativeSolutionError=float((result.x-known).norm()/known.norm()),
                    preconditionerBytes=0 if precond is None else precond.nbytes,
                    work=ledger.report(wall_ms=timing['medianWallMs'], timing_scope='whole warmed verified GMRES, setup excluded'))
                rows.append(row)
                print(json.dumps(dict(scope='solve', n=n, meanDegree=edges/n, kind=kind,
                                      wallMs=timing['medianWallMs'], iterations=count['iterations'])), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ns', nargs='+', type=int, default=[1024, 65536, 1048576])
    parser.add_argument('--sph-sides', nargs='*', type=int, default=[32, 256, 1024])
    parser.add_argument('--neighbors', nargs='+', type=int, default=[25, 55])
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('an accessible CUDA device is required')
    if min(*args.ns, *args.neighbors, *args.sph_sides, args.repeat, args.batch) < 1:
        parser.error('sizes and counts must be positive')
    torch.set_num_threads(2)
    wp.init()
    dtype = get_torch_precision()
    applicationRows = applications(args, dtype)
    solverRows = solves(args, dtype)
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__).resolve(), root/'scripts/benchmarks/_work_metrics.py',
        root/'src/warpSPHCore/linalg/preconditioner.py', root/'src/warpSPHCore/linalg/operator.py',
        root/'src/warpSPHCore/linalg/krylov/gmres.py', root/'src/warpSPHCore/coreOperations/wp_laplacian.py']
    payload = dict(schemaVersion=1, scope=__doc__, torch=torch.__version__, warp=wp.__version__,
        gpu=torch.cuda.get_device_name(), dtype=str(dtype), applications=applicationRows, solves=solverRows,
        arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        sourceSha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2)+'\n')


if __name__ == '__main__':
    main()
