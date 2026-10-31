#!/usr/bin/env python3
"""GPU throughput with resident output/scratch and one C++ batch per sample.

Unlike benchmark.py's completed calls, allocation and per-call Python descriptor
construction are excluded. Report this as a separate experiment, not a public API
speedup. The batch includes preparation unless explicitly labeled interaction-only.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

import torch
from runtime import Library, build_plan, request, call
from verify import State


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--squared", action="store_true")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=7)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/benchmarks"))
    from bench_backends import activity, save
    torch.set_num_threads(16)
    library = Library(args.library)
    report = dict(complete=False, pid=os.getpid(), scope=__doc__, library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  iterations=args.iterations, repeat=args.repeat, cases=[], activity=[])
    for precision in ["float32", "float64"]:
        selection_path = args.archive / (f"screen-squared-{precision}.json" if args.squared else ("selection-float32.json" if precision == "float32" else "screen-float64.json"))
        selection = json.loads(selection_path.read_text())["selection"]
        dtype = getattr(torch, precision)
        for dimension in [1, 2, 3]:
            for nh in [4.0, 2.57]:
                for n in [32768, 1000000]:
                    report["activity"].append(activity())
                    generator = torch.Generator().manual_seed(9067 + n * 3 + dimension)
                    x = torch.rand((n, dimension), generator=generator, dtype=torch.float64).to(device="cuda",dtype=dtype)
                    side = round(n ** (1 / dimension))
                    spacing = 1/side if side**dimension == n else n**(-1/dimension)
                    h = x.new_full((n,), nh*spacing)
                    state = State(x,h,h.new_full((n,),1/n),torch.zeros(n,device="cuda",dtype=torch.int32),h.new_ones(n))
                    values = torch.sin(x * (2*math.pi)) + .1*x
                    plan = build_plan(library,state,values,float(h[0]))
                    selected = selection[f"{precision}:{dimension}:{nh:g}"]
                    packed = x.new_empty((2*dimension,n))
                    output = torch.empty_like(values)
                    functions = {}
                    owners = []
                    for mode,name in ([(8,"direct"),(6,"shared"),(7,"flat")] if args.squared else [(4,"direct"),(3,"shared"),(5,"flat")]):
                        block,width = selected[name]
                        tasks = plan.tasks(block//width)
                        owners.append(tasks)
                        descriptor = request(library,plan,tasks,packed,output)
                        library.interpolate(descriptor,mode,block,width)
                        reference = call(library,plan,tasks,mode,block,width)
                        tolerance = 3e-4 if dtype == torch.float32 else 3e-9
                        torch.testing.assert_close(output,reference,rtol=tolerance,atol=tolerance)
                        for prepare in [True,False]:
                            key = f"{name}_{'prepare' if prepare else 'interactions'}"
                            functions[key] = lambda a=descriptor,m=mode,b=block,w=width,p=prepare: library.interpolate(a,m,b,w,p,args.iterations)
                    for function in functions.values():
                        for _ in range(2): function()
                    samples = {name:[] for name in functions}
                    for repeat in range(args.repeat):
                        order = list(functions)
                        random.Random(4811+repeat).shuffle(order)
                        for name in order:
                            torch.cuda.synchronize()
                            start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            begin = time.perf_counter()
                            start.record()
                            functions[name]()
                            end.record()
                            torch.cuda.synchronize()
                            samples[name].append(dict(event_ms=start.elapsed_time(end)/args.iterations,
                                wall_ms=(time.perf_counter()-begin)*1000/args.iterations))
                    records=[]
                    for name,data in samples.items():
                        events=[s["event_ms"] for s in data]
                        records.append(dict(backend=name,samples=data,event_ms=statistics.median(events),
                                            min_ms=min(events),max_ms=max(events)))
                    report["cases"].append(dict(precision=precision,dimension=dimension,n_h=nh,n=n,records=records))
                    report["activity"].append(activity())
                    save(args.archive/"resident.json",report)
                    print(f"{precision} D={dimension} nh={nh:g} N={n}: "+", ".join(f"{r['backend']}={r['event_ms']:.4f}ms" for r in records),flush=True)
    report["complete"]=True
    save(args.archive/"resident.json",report)


if __name__ == "__main__":
    main()
