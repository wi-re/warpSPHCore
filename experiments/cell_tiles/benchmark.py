#!/usr/bin/env python3
"""Compare cell tiles, Morton CSR and input-order CSR on identical inputs."""
import argparse
from dataclasses import replace
import hashlib
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
    parser.add_argument("--precision", choices=["float32", "float64"], default="float32")
    parser.add_argument("--dims", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--ns", type=int, nargs="+", default=[32768, 262144, 1000000])
    parser.add_argument("--n-h", type=float, nargs="+", default=[4.0, 2.57])
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--screen", action="store_true", help="Screen every scheduling candidate")
    parser.add_argument("--squared", action="store_true", help="Conservative squared-distance rejection before sqrt")
    parser.add_argument("--grid-reference", action="store_true", help="Also measure Warp without particle CSR")
    parser.add_argument("--flat", action="store_true", help="Stage full tiles across cell boundaries")
    parser.add_argument("--cell-images", action="store_true", help="Precomputed cell-image variant")
    parser.add_argument("--selection", type=Path, help="Use winners from an independent screening run")
    parser.add_argument("--layout", choices=["random", "lattice"], default="random")
    args = parser.parse_args()
    if any(n < 16 for n in args.ns) or any(h <= 0 for h in args.n_h):
        parser.error("Positive neighborhood ratio and N >= 16 required")
    os.environ.update(WARPSPHCORE_BACKEND="warp", warpSPHCore_PRECISION=args.precision,
                      warpSPHCore_INDEX_DTYPE="int64", warpSPHCore_OFFSET_DTYPE="int64")
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/benchmarks")]
    import torch
    import warp as wp
    import warpSPHCore as sph
    import warpSPHCore.native._build as build
    from warpSPHCore.native.api import radiusSearch, sphOperation
    import bench_backends as bench
    from runtime import Library, build_plan, sort_state, call, request
    bench.torch, bench.sph = torch, sph
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    wp.init()
    wp.set_module_options({"enable_backward": False},
                         module="warpSPHCore.coreOperations.native.wp_interpolate")
    torch.ops.load_library(str(args.native_artifact.resolve()))
    build._extension = torch.ops.warpsph_native
    library = Library(args.library)
    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2,
        operation=sph.WarpOperation.Interpolate,
        operationMode=sph.OperationDirection.TrueAllToToAll)
    dtype = getattr(torch, args.precision)
    modes = [(4, "direct"), (3, "shared"), (5, "flat")] if args.cell_images else [(0, "direct"), (1, "shared")]
    if args.flat:
        modes = [(4, "direct"), (5, "shared")]
    if args.squared:
        modes = [(8,"direct"),(6,"shared"),(7,"flat")]
    choices = json.loads(args.selection.read_text())["selection"] if args.selection else {}
    report = dict(complete=False, pid=os.getpid(), configuration={
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source_revision=bench.command_output(["git", "rev-parse", "HEAD"]),
        source_sha256={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                       for f in [Path(__file__), Path(__file__).with_name("runtime.py"),
                                 Path(__file__).with_name("tiles.cu")]},
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        native_sha256=hashlib.sha256(args.native_artifact.read_bytes()).hexdigest(),
        environment=dict(torch=torch.__version__, warp=wp.__version__, gpu=torch.cuda.get_device_name()),
        scope="Completed calls include output/scratch allocation and coefficient preparation; plans/CSR reused. GPU event time is recorded separately. Planning is measured separately.",
        activity=[], cases=[], selection={})
    bench.save(args.output, report)

    def sample(function, iterations):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin = time.perf_counter()
        start.record()
        for _ in range(iterations):
            output = function()
        end.record()
        torch.cuda.synchronize()
        return dict(wall_ms=(time.perf_counter() - begin) * 1000 / iterations,
                    event_ms=start.elapsed_time(end) / iterations)

    def summarize(samples):
        wall = [s["wall_ms"] for s in samples]
        events = [s["event_ms"] for s in samples]
        return dict(samples=samples, median_ms=statistics.median(wall),
                    min_ms=min(wall), max_ms=max(wall), event_ms=statistics.median(events))

    for dimension in args.dims:
        for nh in args.n_h:
            for n in args.ns:
                report["activity"].append(bench.activity())
                generator = torch.Generator().manual_seed(9067 + n * 3 + dimension)
                positions = torch.rand((n, dimension), generator=generator, dtype=torch.float64)
                if args.layout == "lattice":
                    side = round(n ** (1 / dimension))
                    if side ** dimension != n:
                        raise ValueError("Lattice requires an exact integer power of N")
                    linear = torch.arange(n)
                    positions = torch.stack([((linear // side ** d) % side + .5) / side
                                             for d in range(dimension)], dim=1).double()
                    # Input order is deliberately randomized; Morton sorting is a real transformation.
                    positions = positions[torch.randperm(n, generator=generator)]
                # Exact integer powers avoid an ulp bump in h that would
                # otherwise change floor(1/h), e.g. 32^3 at nh=4: grid 8 -> 7.
                side = round(n ** (1 / dimension))
                spacing = 1 / side if side ** dimension == n else n ** (-1 / dimension)
                h = nh * spacing
                positions = positions.to(device="cuda", dtype=dtype)
                supports = positions.new_full((n,), h)
                state = sph.ParticleState(positions, supports, supports.new_full((n,), 1 / n),
                    torch.zeros(n, device="cuda", dtype=torch.int32), supports.new_ones(n))
                values = torch.sin(positions * (2 * math.pi)) + .1 * positions
                domain = sph.DomainDescription(positions.new_zeros(dimension), positions.new_ones(dimension),
                    torch.ones(dimension, device="cuda", dtype=torch.bool), dimension)
                h = float(supports[0])
                if h >= 1 / 3:
                    raise ValueError("Support must fit at least three periodic cells per dimension")
                # Include allocations, sort, permutation, unique and neighbor-cell lookup in planning.
                for _ in range(2):
                    plan = build_plan(library, state, values, h)
                plan_samples = []
                for _ in range(3):
                    result = {}
                    def make_plan():
                        result["plan"] = build_plan(library, state, values, h)
                        return result["plan"]
                    plan_samples.append(sample(make_plan, 1))
                    plan = result["plan"]
                planning = summarize(plan_samples)
                sorting = summarize([sample(lambda: sort_state(library, state, values, h), 1) for _ in range(3)])
                # Each comparator reuses its own CSR. All interpolation paths see
                # identical particle attributes and the same mathematical support.
                original_adjacency = radiusSearch(state, domain)
                sorted_adjacency = radiusSearch(plan.state, domain)
                torch.cuda.synchronize()
                native_original = lambda: sphOperation(state, properties, domain, values,
                                                        adjacency=original_adjacency)
                warp_original = lambda: sph.warpOperation(state, properties, domain, values,
                                                          adjacency=original_adjacency)
                native_sorted = lambda: sphOperation(plan.state, properties, domain, plan.values,
                                                      adjacency=sorted_adjacency)
                warp_sorted = lambda: sph.warpOperation(plan.state, properties, domain, plan.values,
                                                        adjacency=sorted_adjacency)
                expected = native_sorted()
                tolerance = 3e-4 if dtype == torch.float32 else 3e-9
                torch.testing.assert_close(expected, native_original()[plan.order],
                                           rtol=tolerance, atol=tolerance)
                warp_result = warp_sorted()
                difference = (expected - warp_result).double()
                warp_error = dict(max_abs=difference.abs().max().item(),
                    relative_l2=(difference.norm() / expected.double().norm().clamp_min(1e-30)).item())
                # High-resolution 1D FP32 Warp wraps (dx+.5)-.5, losing bits.
                # Preserve/report that known reference discrepancy instead of
                # changing either kernel or silently relaxing a correctness gate.
                warp_qualified = bool(torch.allclose(expected, warp_result, rtol=tolerance, atol=tolerance))
                if not warp_qualified and not (dimension == 1 and dtype == torch.float32):
                    raise AssertionError(f"Warp discrepancy: {warp_error}")
                case = dict(dimension=dimension, n=n, n_h=nh, precision=args.precision,
                    support=h, grid=plan.grid, cells=len(plan.counts),
                    mean_cell=float(plan.counts.double().mean()), max_cell=int(plan.counts.max()),
                    edges=sorted_adjacency.numEdges, neighbors=sorted_adjacency.numEdges / n,
                    candidate_pairs=plan.candidate_pairs(), plan_bytes=plan.storage_bytes(),
                    planning=planning, sorting=sorting, warp_qualified=warp_qualified, warp_error=warp_error,
                    screening=[], records=[])
                task_cache = {}
                def tasks_for(block, width):
                    capacity = block // width
                    if capacity not in task_cache:
                        task_cache[capacity] = plan.tasks(capacity)
                    return task_cache[capacity]
                key = f"{args.precision}:{dimension}:{nh:g}"
                selected = choices.get(key, {"shared": [128, 4], "direct": [128, 4], "flat": [128, 4]})
                if args.screen:
                    candidates = [(mode, block, width) for mode, _ in modes
                                  for block in [32, 128, 256] for width in [1, 4, 8]]
                    functions = {}
                    for mode, block, width in candidates:
                        tasks = tasks_for(block, width)
                        function = lambda m=mode, b=block, w=width, t=tasks: call(library, plan, t, m, b, w)
                        actual = function()
                        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                        functions[mode, block, width] = function
                    samples = {c: [] for c in candidates}
                    for repeat in range(3):
                        random.Random(1103 + repeat).shuffle(candidates)
                        for candidate in candidates:
                            samples[candidate].append(sample(functions[candidate], 3))
                    for candidate, data in samples.items():
                        mode, block, width = candidate
                        case["screening"].append(dict(mode=mode, block=block, width=width,
                                                      **summarize(data)))
                    selected = {}
                    for mode, name in modes:
                        best = min((r for r in case["screening"] if r["mode"] == mode),
                                   key=lambda r: r["median_ms"])
                        selected[name] = [best["block"], best["width"]]
                report["selection"][key] = selected
                functions = {"native_input": native_original, "warp_input": warp_original,
                             "native_morton": native_sorted, "warp_morton": warp_sorted}
                for mode, name in modes:
                    block, width = selected[name]
                    tasks = tasks_for(block, width)
                    function = lambda m=mode, b=block, w=width, t=tasks: call(library, plan, t, m, b, w)
                    torch.testing.assert_close(function(), expected, rtol=tolerance, atol=tolerance)
                    functions[f"tile_{name}"] = function
                    case[f"{name}_configuration"] = dict(block=block, width=width,
                        query_capacity=block // width, tasks=len(tasks.cells),
                        task_bytes=sum(x.numel() * x.element_size() for x in (tasks.cells,tasks.begin,tasks.end)))
                if args.grid_reference:
                    from warpSPHCore.radiusSearch import buildCompactHashMap
                    def make_grid(particles):
                        return buildCompactHashMap(particles.positions, particles.positions,
                            particles.supports, particles.supports, domain.periodic, domain,
                            sph.SupportScheme.Gather, n)
                    grid = make_grid(plan.state)
                    grid_call = lambda: sph.warpOperation(plan.state, properties, domain, plan.values, adjacency=grid)
                    torch.testing.assert_close(grid_call(), warp_result, rtol=tolerance, atol=tolerance)
                    functions["warp_grid_morton"] = grid_call
                # Independent repetitions after screening; rotate all comparator order.
                for function in functions.values():
                    sample(function, 2)
                samples = {name: [] for name in functions}
                for repeat in range(args.repeat):
                    order = list(functions)
                    random.Random(3301 + repeat).shuffle(order)
                    for name in order:
                        samples[name].append(sample(functions[name], args.iterations))
                for name, data in samples.items():
                    result = summarize(data)
                    case["records"].append(dict(backend=name, **result,
                        useful_gpairs_s=case["edges"] / (result["median_ms"] * 1e6),
                        visited_gpairs_s=case["candidate_pairs"] / (result["median_ms"] * 1e6)
                        if name.startswith("tile_") else None))
                # Build/search cost is separate: sorted callers can reuse order,
                # whereas input-order callers must also pay permutation cost.
                case["search_input"] = summarize([sample(lambda: radiusSearch(state, domain), 1) for _ in range(3)])
                case["search_morton"] = summarize([sample(lambda: radiusSearch(plan.state, domain), 1) for _ in range(3)])
                case["tasks_build"] = summarize([sample(lambda: plan.tasks(selected["shared"][0] // selected["shared"][1]), 1) for _ in range(3)])
                block, width = selected["shared"]
                shared_mode = dict((name, mode) for mode, name in modes)["shared"]
                def fresh_tiled():
                    fresh = build_plan(library, state, values, h)
                    return call(library, fresh, fresh.tasks(block // width), shared_mode, block, width)
                def fresh_native():
                    return sphOperation(state, properties, domain, values,
                                        adjacency=radiusSearch(state, domain))
                def fresh_morton_native():
                    ordered, field, _, _, _ = sort_state(library, state, values, h)
                    return sphOperation(ordered, properties, domain, field,
                                        adjacency=radiusSearch(ordered, domain))
                if args.grid_reference:
                    def fresh_grid():
                        return sph.warpOperation(state, properties, domain, values, adjacency=make_grid(state))
                    case["fresh_warp_grid"] = summarize([sample(fresh_grid, 1) for _ in range(3)])
                if any(name == "flat" for _, name in modes):
                    flat_block, flat_width = selected["flat"]
                    def fresh_flat():
                        fresh = build_plan(library, state, values, h)
                        return call(library, fresh, fresh.tasks(flat_block // flat_width), dict((name,mode) for mode,name in modes)["flat"], flat_block, flat_width)
                    case["fresh_flat"] = summarize([sample(fresh_flat, 1) for _ in range(3)])
                case["fresh_tile"] = summarize([sample(fresh_tiled, 1) for _ in range(3)])
                case["fresh_native_input"] = summarize([sample(fresh_native, 1) for _ in range(3)])
                case["fresh_native_morton"] = summarize([sample(fresh_morton_native, 1) for _ in range(3)])
                def restore_output():
                    output = torch.empty_like(expected)
                    output[plan.order] = expected
                    return output
                case["restore_output"] = summarize([sample(restore_output, 1) for _ in range(3)])
                case["candidate_overhead"] = case["candidate_pairs"] / case["edges"]
                case["scratch_bytes"] = 2 * dimension * n * torch.tensor([],dtype=dtype).element_size()
                report["cases"].append(case)
                report["activity"].append(bench.activity())
                bench.save(args.output, report)
                print(f"D={dimension} nh={nh:g} N={n}: {case['neighbors']:.2f} neighbors, "
                      f"{case['candidate_overhead']:.2f}x candidate pairs; "
                      + ", ".join(f"{r['backend']}={r['median_ms']:.4f}ms" for r in case["records"]), flush=True)
    report["complete"] = True
    bench.save(args.output, report)


if __name__ == "__main__":
    main()
