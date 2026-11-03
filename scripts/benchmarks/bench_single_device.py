#!/usr/bin/env python3
"""CPU/CUDA operator, search and memory worker; see warpSPH benchmark README.

The JSON request protocol permits a frontend sweep to isolate every case in a
fresh process. This entry point can also be called directly with a request file.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import traceback

from _single_device import configure, metadata, measure, profile_call, write_result


def workload(request, device):
    import torch
    if request['kind'] == 'memory':
        size = request['memory_mib'] * 1024**2 // torch.empty((), dtype=getattr(torch, request['precision'])).element_size()
        a = torch.ones(size, device=device, dtype=getattr(torch, request['precision']))
        b, out = torch.full_like(a, 2), torch.empty_like(a)
        operation = request['operation']
        if operation == 'copy':
            fn = lambda: out.copy_(a)
            traffic = 2 * a.numel() * a.element_size()
            expected = 1.
        elif operation == 'triad':
            fn = lambda: torch.add(a, b, alpha=.5, out=out)
            traffic = 3 * a.numel() * a.element_size()
            expected = 2.
        elif operation == 'gather':
            a.uniform_(-1., 1.)
            index = torch.randperm(size, device=device)
            fn = lambda: torch.index_select(a, 0, index, out=out)
            traffic = size * (2 * a.element_size() + index.element_size())
            expected = 1.
        else:
            raise ValueError(operation)
        def validate(value):
            if operation == 'gather':
                torch.testing.assert_close(value, a[index], rtol=0, atol=0)
                return
            if not bool((value == expected).all()):
                raise AssertionError('memory control output mismatch')
        return fn, validate, dict(elements=size, logical_bytes_per_call=traffic,
                                 bandwidth_note='Logical traffic, not measured DRAM traffic; cache effects possible.')

    from _operator_cases import Case, Workload, check_output, work_counts
    from warpSPHCore import setNeighborLanes, radiusSearchCompactHashMap
    w = Workload(request['nx'] ** request['dim'], request['dim'], request['neighbors'], str(device))
    order = request.get('particle_order', 'input')
    if order == 'morton':
        from dataclasses import replace
        # Stable 21-bit Z order, same coordinate quantization as the simulator.
        # This is setup work. Full-step cases time production layout adoption.
        p = w.particles
        cell = torch.floor((p.positions-w.domain.min)/(w.domain.max-w.domain.min)*(1 << 21)).long().clamp_(0, (1 << 21)-1)
        keys = torch.zeros(w.n, dtype=torch.int64, device=device)
        for bit in range(21):
            for axis in range(w.dim):
                keys.bitwise_or_(((cell[:, axis] >> bit) & 1) << (3*bit+axis))
        permutation = torch.argsort(keys, stable=True)
        fields = {name: getattr(p, name)[permutation].contiguous() for name in
                  ('positions', 'supports', 'masses', 'densities', 'kinds')}
        w.particles = replace(p, **fields)
        w.scalar, w.vector, w.matrix = (value[permutation].contiguous() for value in (w.scalar, w.vector, w.matrix))
    w.adjacency, w.grid = None, None
    search_options = dict(collection=request.get('collection', 'auto'), queryOrder=request.get('query_order', 'auto'),
                          storeI=not bool(request.get('neighbor_omit_i')))
    w.adjacency, w.grid = radiusSearchCompactHashMap(w.particles, w.domain, returnCompactHashMap=True, **search_options)
    w.edges = int(w.adjacency.numNeighbors.sum())
    setNeighborLanes(request['lanes'])
    common = dict(particles=w.n, particle_order=order, search_options=search_options, directed_edges_including_self=w.edges,
                  mean_neighbors_including_self=w.edges / w.n,
                  search_options_scope='explicit fixture topology; implicit operator rebuild uses public operator defaults')
    if request['kind'] == 'operator':
        # Case's legacy lane field is integer-only; runtime dispatch uses the
        # process selection above. Keep None in the reported request/metadata.
        c = Case(w.n, dim=w.dim, neighbors=request['neighbors'], operation=request['operation'],
                 traversal=request['traversal'], lanes=request['lanes'] or 1, kernel=request['kernel'])
        fn = w.call(c)
        # Independent lane-1/CSR reference outside timing, including grid/rebuild paths.
        from dataclasses import replace
        setNeighborLanes(1)
        reference = w.call(replace(c, traversal='adjacency', lanes=1))()
        setNeighborLanes(request['lanes'])
        return fn, lambda out: check_output(out, reference), dict(common, case=asdict(c) | dict(lanes=request['lanes']), **work_counts(c, w.edges))
    if request['kind'] != 'search':
        raise ValueError(request['kind'])
    from warpSPHCore import radiusSearchCompactHashMap
    from warpSPHCore.radiusSearch.compactHash.buildHashmap import buildCompactHashMap
    from warpSPHCore.radiusSearch.compactHash.search import radiusSearchOnCompactHashMap
    from warpSPHCore.enumTypes import SupportScheme
    from warpSPHCore.radiusSearch.compactHash.wrapper import getNextPrime
    p = w.particles
    periodic = torch.as_tensor(w.domain.periodic, device=device)
    length = getNextPrime(w.n)
    args = (p.positions, p.positions, p.supports, p.supports, periodic, w.domain, SupportScheme.Gather, length)
    if request['operation'] == 'build':
        fn = lambda: buildCompactHashMap(*args)
        def validate(grid):
            check_output(grid)
    elif request['operation'] in ('query', 'rebuild'):
        fn = (lambda: radiusSearchOnCompactHashMap(w.grid, *args, **search_options)) if request['operation'] == 'query' else (
              lambda: radiusSearchCompactHashMap(p, w.domain, mode=SupportScheme.Gather, **search_options))
        def validate(adj):
            # Frozen geometry must reproduce the exact directed CSR topology.
            torch.testing.assert_close(adj.numNeighbors, w.adjacency.numNeighbors, rtol=0, atol=0)
            torch.testing.assert_close(adj.paddedJ[:adj.numEdges], w.adjacency.paddedJ[:w.adjacency.numEdges], rtol=0, atol=0)
    else:
        raise ValueError(request['operation'])
    return fn, validate, common


def execute(request, output):
    result = dict(schema_version=1, request=request, status='failed')
    started = time.perf_counter()
    try:
        device = configure(request)
        result['machine'] = metadata(device)
        if request['backend'] != 'warp' and request['kind'] != 'memory':
            raise NotImplementedError('operator/search fixtures currently qualify Warp only; native harness integration remains separate')
        fn, validate, details = workload(request, device)
        result['workload'] = details
        result['timing'] = measure(fn, device, warmup=request['warmup'], repeats=request['repeats'],
                                   calls=request['calls'], validate=validate)
        if 'logical_bytes_per_call' in details:
            result['logical_GBps_wall'] = details['logical_bytes_per_call'] / result['timing']['wall_ms']['median'] / 1e6
            stream = result['timing']['stream_ms']
            result['logical_GBps_stream'] = details['logical_bytes_per_call'] / stream['median'] / 1e6 if stream else None
        if request.get('profile'):
            result['profile'] = profile_call(fn, device, Path(output).with_suffix('.trace.json'))
        result['status'] = 'ok'
    except NotImplementedError as exc:
        result.update(status='unsupported', error=str(exc))
    except Exception as exc:
        result.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    result['process_wall_s'] = time.perf_counter() - started
    write_result(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = execute(json.loads(args.request.read_text()), args.output)
    print(f"{result['status']}: {args.output}")
    raise SystemExit(1 if result['status'] == 'failed' else 0)
