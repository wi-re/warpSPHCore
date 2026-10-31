#!/usr/bin/env python3
"""Screen cluster tiles, then measure fixed selections against CSR and cells.

Completed calls include allocations, field preparation and Python dispatch.
Resident events batch calls in C++; planning and fresh execution are separate.
No correctness checks or host reads occur inside an interpolation call.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--native-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell-selection-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=["float32", "float64"], default="float32")
    parser.add_argument("--dims", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--ns", type=int, nargs="+", default=[32768, 262144, 1000000])
    parser.add_argument("--n-h", type=float, nargs="+", default=[4., 2.57])
    parser.add_argument("--refinement", type=int, default=8)
    parser.add_argument("--layout", choices=["random", "lattice"], default="random")
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    os.environ.update(WARPSPHCORE_BACKEND="warp", warpSPHCore_PRECISION=args.precision,
                      warpSPHCore_INDEX_DTYPE="int64", warpSPHCore_OFFSET_DTYPE="int64")
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/benchmarks")]
    import torch
    import warp as wp
    import warpSPHCore as sph
    import warpSPHCore.native._build as build
    from warpSPHCore.native.api import radiusSearch, sphOperation
    import bench_backends as bench
    from runtime import Library, build_cells, build_clusters, call, cell
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    wp.init()
    wp.set_module_options({"enable_backward": False}, module="warpSPHCore.coreOperations.native.wp_interpolate")
    torch.ops.load_library(str(args.native_artifact.resolve()))
    build._extension = torch.ops.warpsph_native
    library = Library(args.library)
    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2,
        operation=sph.WarpOperation.Interpolate, operationMode=sph.OperationDirection.TrueAllToToAll)
    dtype = getattr(torch, args.precision)
    selection = json.loads(args.selection.read_text())["selection"] if args.selection else {}
    cell_choices = json.loads((args.cell_selection_dir/f"screen-squared-{args.precision}.json").read_text())["selection"]
    report = dict(complete=False, pid=os.getpid(), configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source_revision=bench.command_output(["git", "rev-parse", "HEAD"]),
        source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob("*") if f.is_file()},
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        native_sha256=hashlib.sha256(args.native_artifact.read_bytes()).hexdigest(),
        environment=dict(torch=torch.__version__, warp=wp.__version__, gpu=torch.cuda.get_device_name()),
        scope=__doc__, cases=[], activity=[], selection={})
    bench.save(args.output, report)

    def sample(function, iterations=1):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin = time.perf_counter()
        start.record()
        for _ in range(iterations):
            output = function()
        end.record()
        torch.cuda.synchronize()
        return dict(wall_ms=(time.perf_counter()-begin)*1000/iterations,
                    event_ms=start.elapsed_time(end)/iterations)

    def summarize(samples):
        wall = [s["wall_ms"] for s in samples]
        return dict(samples=samples, median_ms=statistics.median(wall), min_ms=min(wall), max_ms=max(wall),
                    event_ms=statistics.median(s["event_ms"] for s in samples))

    def list_stats(plan):
        b = plan.buffers
        entries = b["entries"].long()
        src = entries >> 5
        lengths = (b["source_end"]-b["source_begin"])[src].long()
        group = torch.repeat_interleave(torch.arange(plan.descriptor.groups, device="cuda"),
                                        b["counts"], output_size=len(entries))
        visited = lengths.new_zeros(())
        for bit in range(plan.group_size):
            query = b["group_first"][group].long()+bit
            valid = query < b["group_end"][group]
            query = query.clamp_max(plan.descriptor.queries-1)
            qlen = (b["query_end"]-b["query_begin"])[query].long()
            visited += (lengths*qlen*valid*((b["masks"].long() & (1 << bit)) != 0)).sum()
        return dict(entries=len(entries), groups=plan.descriptor.groups, queries=plan.descriptor.queries,
                    sources=plan.descriptor.sources, visited_pairs=int(visited),
                    raw_bytes=plan.memory_bytes(False), compressed_bytes=plan.memory_bytes(True),
                    encoded_index_bytes=b["compressed"].numel())

    for dimension, nh, n in itertools.product(args.dims, args.n_h, args.ns):
        report["activity"].append(bench.activity())
        generator = torch.Generator().manual_seed(9067+n*3+dimension)
        x = torch.rand((n, dimension), generator=generator, dtype=torch.float64)
        side = round(n**(1/dimension))
        spacing = 1/side if side**dimension == n else n**(-1/dimension)
        if args.layout == "lattice":
            if side**dimension != n:
                raise ValueError("Lattice needs an exact integer power")
            linear = torch.arange(n)
            x = torch.stack([((linear//side**d) % side+.5)/side for d in range(dimension)], dim=1).double()
            x = x[torch.randperm(n, generator=generator)]
        x = x.to(device="cuda", dtype=dtype)
        h = x.new_full((n,), nh*spacing)
        support = float(h[0])
        state = sph.ParticleState(x, h, h.new_full((n,), 1/n), torch.zeros(n, device="cuda", dtype=torch.int32), h.new_ones(n))
        values = torch.sin(x*(2*math.pi))+.1*x
        domain = sph.DomainDescription(x.new_zeros(dimension), x.new_ones(dimension),
                                        torch.ones(dimension, device="cuda", dtype=torch.bool), dimension)
        cells = build_cells(library, state, values, support, args.refinement)
        adjacency = radiusSearch(cells.state, domain)
        native = lambda: sphOperation(cells.state, properties, domain, cells.values, adjacency=adjacency)
        warp = lambda: sph.warpOperation(cells.state, properties, domain, cells.values, adjacency=adjacency)
        expected = native()
        tolerance = 3e-4 if dtype == torch.float32 else 3e-9
        warp_result = warp()
        warp_qualified = bool(torch.allclose(expected, warp_result, rtol=tolerance, atol=tolerance))
        if not warp_qualified and not (dimension == 1 and dtype == torch.float32):
            raise AssertionError("Warp parity failed outside the known FP32 1D fixture")
        key = f"{args.precision}:{dimension}:{nh:g}"
        record = dict(dimension=dimension, n_h=nh, n=n, precision=args.precision,
            cells=len(cells.counts), refinement=cells.refinement, support=support, neighbors=adjacency.numEdges/n,
            edges=adjacency.numEdges, cell_candidate_pairs=cells.candidate_pairs(),
            warp_qualified=warp_qualified, warp_max_abs=float((expected-warp_result).abs().max()),
            screening=[], records=[], resident=[])
        plans, plan_costs = {}, {}

        def get_plan(configuration):
            tile = tuple(configuration[:4])
            if tile not in plans:
                holder = {}
                def make():
                    holder["plan"] = build_clusters(library, cells, *tile)
                    return holder["plan"]
                plan_costs[str(tile)] = sample(make)
                plans[tile] = holder["plan"]
            return plans[tile]

        if args.screen:
            # Screen schedule choices at a separate size. Each candidate gets
            # three independent samples, in rotated order, after warm-up.
            candidates = list(itertools.product([4, 8], [4, 8], [1, 8], [False, True], range(3), [False, True], [4, 8]))
            candidates += [(i, j, 8, exact, 3, compressed, 8*((i*j+31)//32))
                           for i, j, exact, compressed in itertools.product([4, 8], [4, 8], [False, True], [False, True])]
            functions = {}
            for configuration in candidates:
                plan = get_plan(configuration)
                mode, compressed, warps = configuration[4:]
                function = lambda p=plan, m=mode, c=compressed, w=warps: call(library, p, m, c, w)
                torch.testing.assert_close(function(), expected, rtol=tolerance, atol=tolerance)
                functions[configuration] = function
            samples = {c: [] for c in candidates}
            for repeat in range(3):
                random.Random(1103+repeat).shuffle(candidates)
                for configuration in candidates:
                    samples[configuration].append(sample(functions[configuration], 3))
            for configuration, data in samples.items():
                record["screening"].append(dict(configuration=configuration, **summarize(data)))
            winners = {}
            for compressed in [False, True]:
                for mode in range(4):
                    best = min((r for r in record["screening"] if r["configuration"][4:6] == (mode, compressed)), key=lambda r: r["median_ms"])
                    name = f"cluster_{'compressed' if compressed else 'raw'}_{['register', 'shared', 'serial', 'super_shared'][mode]}"
                    winners[name] = list(best["configuration"])
            report["selection"][key] = winners
            record["plans"] = {str(k): list_stats(p) for k, p in plans.items()}
            record["planning_screen_ms"] = plan_costs
            report["cases"].append(record)
            report["activity"].append(bench.activity())
            bench.save(args.output, report)
            print(f"SCREEN {key} N={n}: "+", ".join(f"{name}={c}" for name, c in winners.items()), flush=True)
            continue

        winners = selection[key]
        report["selection"][key] = winners
        functions = {"native_refined": native, "warp_refined": warp}
        owners = []
        for mode, name in [(8, "direct"), (6, "shared"), (7, "flat")]:
            block, width = cell_choices[key][name]
            tasks = cells.tasks(block//width)
            owners.append(tasks)
            functions[f"cell_{name}"] = lambda t=tasks, m=mode, b=block, w=width: cell.call(library, cells, t, m, b, w)
        coarse = build_cells(library, state, values, support, 1)
        coarse_adjacency = radiusSearch(coarse.state, domain)
        coarse_native = lambda: sphOperation(coarse.state, properties, domain, coarse.values, adjacency=coarse_adjacency)
        torch.testing.assert_close(coarse_native()[torch.argsort(coarse.order)][cells.order], expected, rtol=tolerance, atol=tolerance)
        functions["native_coarse"] = coarse_native
        for name, configuration in winners.items():
            plan = get_plan(configuration)
            mode, compressed, warps = configuration[4:]
            functions[name] = lambda p=plan, m=mode, c=compressed, w=warps: call(library, p, m, c, w)
        for name, function in functions.items():
            result = function()
            if name != "native_coarse" and (name != "warp_refined" or warp_qualified):
                torch.testing.assert_close(result, expected, rtol=tolerance, atol=tolerance)
            sample(function, 2)
        samples = {name: [] for name in functions}
        for repeat in range(args.repeat):
            order = list(functions)
            random.Random(3301+repeat).shuffle(order)
            for name in order:
                samples[name].append(sample(functions[name], args.iterations))
        for name, data in samples.items():
            summary = summarize(data)
            record["records"].append(dict(backend=name, **summary,
                useful_gpairs_s=record["edges"]/(summary["median_ms"]*1e6)))
        record["plans"] = {str(k): list_stats(p) for k, p in plans.items()}
        # Resident-buffer measurements use the same selected schedules. These
        # include preparation but exclude allocation and per-call Python setup.
        packed = values.new_empty((2*dimension, n))
        output = torch.empty_like(values)
        resident_functions = {}
        for name, configuration in winners.items():
            plan = get_plan(configuration)
            descriptor = plan.request(packed, output)
            mode, compressed, warps = configuration[4:]
            library.run(descriptor, mode, compressed, warps)
            torch.testing.assert_close(output, expected, rtol=tolerance, atol=tolerance)
            resident_functions[name] = lambda a=descriptor, m=mode, c=compressed, w=warps: library.run(a, m, c, w, True, args.iterations)
        for mode, name in [(8, "direct"), (6, "shared"), (7, "flat")]:
            block, width = cell_choices[key][name]
            tasks = cells.tasks(block//width)
            owners.append(tasks)
            descriptor = cell.request(library, cells, tasks, packed, output)
            resident_functions[f"cell_{name}"] = lambda a=descriptor, m=mode, b=block, w=width: library.interpolate(a, m, b, w, True, args.iterations)
        for function in resident_functions.values():
            function()
        resident_samples = {name: [] for name in resident_functions}
        for repeat in range(args.repeat):
            order = list(resident_functions)
            random.Random(4401+repeat).shuffle(order)
            for name in order:
                result = sample(resident_functions[name])
                resident_samples[name].append({k: v/args.iterations for k, v in result.items()})
        record["resident"] = [dict(backend=name, **summarize(data)) for name, data in resident_samples.items()]
        record["cell_planning"] = summarize([sample(lambda: build_cells(library, state, values, support, args.refinement)) for _ in range(3)])
        record["cluster_planning"] = {}
        record["fresh"] = {}
        for name, configuration in winners.items():
            i, j, group, exact, mode, compressed, warps = configuration
            def build_only():
                return build_clusters(library, cells, i, j, group, exact, compression=compressed)
            record["cluster_planning"][name] = summarize([sample(build_only) for _ in range(3)])
            def fresh():
                fresh_cells = build_cells(library, state, values, support, args.refinement)
                fresh_plan = build_clusters(library, fresh_cells, i, j, group, exact, compression=compressed)
                return call(library, fresh_plan, mode, compressed, warps)
            record["fresh"][name] = summarize([sample(fresh) for _ in range(3)])
        def fresh_native():
            fresh_cells = build_cells(library, state, values, support, args.refinement)
            return sphOperation(fresh_cells.state, properties, domain, fresh_cells.values, adjacency=radiusSearch(fresh_cells.state, domain))
        record["fresh"]["native_refined"] = summarize([sample(fresh_native) for _ in range(3)])
        for mode, name in [(8, "direct"), (6, "shared"), (7, "flat")]:
            block, width = cell_choices[key][name]
            def fresh_cell():
                p = build_cells(library, state, values, support, args.refinement)
                return cell.call(library, p, p.tasks(block//width), mode, block, width)
            record["fresh"][f"cell_{name}"] = summarize([sample(fresh_cell) for _ in range(3)])
        report["cases"].append(record)
        report["activity"].append(bench.activity())
        bench.save(args.output, report)
        print(f"{key} N={n}: "+", ".join(f"{r['backend']}={r['median_ms']:.4f}ms" for r in record["records"]), flush=True)
        del functions, resident_functions, plans, adjacency, coarse_adjacency, owners
    report["complete"] = True
    bench.save(args.output, report)


if __name__ == "__main__":
    main()
