#!/usr/bin/env python3
"""Plot raw completed-call samples and export useful/visited pair throughput."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args = parser.parse_args()
    runs = {}
    for precision in ["float32","float64"]:
        run = json.loads((args.directory/f"final-{precision}.json").read_text())
        if not run["complete"]: parser.error(f"Incomplete {precision} run")
        runs[precision] = run
    rows = []
    for precision,run in runs.items():
        for case in run["cases"]:
            for record in case["records"]:
                rows.append(dict(precision=precision,dimension=case["dimension"],n_h=case["n_h"],n=case["n"],
                    backend=record["backend"],median_ms=record["median_ms"],min_ms=record["min_ms"],max_ms=record["max_ms"],
                    event_ms=record["event_ms"],mean_neighbors=case["neighbors"],candidate_ratio=case["candidate_overhead"],
                    useful_gpairs_s=record["useful_gpairs_s"],visited_gpairs_s=record["visited_gpairs_s"],
                    warp_qualified=case["warp_qualified"]))
    with (args.directory/"throughput.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    colors={"native_input":"#9aabb9","native_morton":"#3977a8","tile_direct":"#a59780",
            "tile_shared":"#e07820","tile_flat":"#399667","warp_morton":"#b35b80","warp_grid_morton":"#8466aa"}
    labels={"native_input":"Native CSR / input order","native_morton":"Native CSR / Morton",
            "tile_direct":"Cell tiles / cached loads","tile_shared":"Cell tiles / shared memory",
            "tile_flat":"Cell tiles / flattened stencil","warp_morton":"Warp CSR / Morton","warp_grid_morton":"Warp direct grid / Morton"}
    displayed=["native_input","native_morton","tile_shared","tile_flat","warp_morton","warp_grid_morton"]
    for precision,run in runs.items():
        fig,axes=plt.subplots(2,3,figsize=(16,8.6),constrained_layout=True)
        for row,nh in enumerate([4.,2.57]):
            for col,dimension in enumerate([1,2,3]):
                axis=axes[row,col]
                case=next(c for c in run["cases"] if c["n"]==1000000 and c["n_h"]==nh and c["dimension"]==dimension)
                records={r["backend"]:r for r in case["records"]}
                data=[records[name] for name in displayed]
                axis.barh(np.arange(len(data)),[r["median_ms"] for r in data],
                    color=[colors[name] for name in displayed],height=.65,
                    xerr=[[r["median_ms"]-r["min_ms"] for r in data],[r["max_ms"]-r["median_ms"] for r in data]],
                    error_kw=dict(elinewidth=.7,capsize=1.5))
                axis.set_yticks(np.arange(len(data)),[labels[name] for name in displayed],fontsize=8)
                axis.invert_yaxis();axis.set_xscale("log");axis.grid(axis="x",alpha=.2);axis.set_axisbelow(True)
                axis.xaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
                axis.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f"{x:g}"))
                axis.xaxis.set_minor_formatter(NullFormatter())
                axis.set_xlim(min(r["min_ms"] for r in data)*.75,max(r["max_ms"] for r in data)*2)
                for y,r in enumerate(data):
                    axis.text(r["max_ms"]*1.08,y,f"{r['median_ms']:.3f}",va="center",fontsize=8)
                axis.set_xlabel("Completed call (ms; log scale)")
                axis.set_title(f"{dimension}D · nh={nh:g} · {case['neighbors']:.1f} neighbors")
        note="; 1D random FP32 Warp values fail parity—see binary-lattice control" if precision=="float32" else ""
        fig.suptitle(f"Cell-to-cell interpolation · {precision.upper().replace("FLOAT","FP")} · 1,000,000 particles\n"
                     f"Allocation + coefficient preparation included; geometric plans reused{note}",fontsize=12)
        fig.savefig(args.directory/f"interpolation-{precision}.png",dpi=150);plt.close(fig)
    # Scaling uses the full public call, not private resident-buffer timings.
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for row,nh in enumerate([4.,2.57]):
        for col,dimension in enumerate([1,2,3]):
            axis=axes[row,col]
            cases=sorted([c for c in runs["float32"]["cases"] if c["dimension"]==dimension and c["n_h"]==nh],key=lambda c:c["n"])
            for name in ["native_input","native_morton","tile_shared","tile_flat","warp_morton"]:
                records=[next(r for r in c["records"] if r["backend"]==name) for c in cases]
                ns=[c["n"] for c in cases]
                axis.plot(ns,[r["median_ms"] for r in records],marker="o",color=colors[name],label=labels[name])
                axis.fill_between(ns,[r["min_ms"] for r in records],[r["max_ms"] for r in records],color=colors[name],alpha=.12)
            axis.set(xscale="log",yscale="log",xlabel="Particles",ylabel="Completed call (ms)",title=f"{dimension}D · nh={nh:g}")
            axis.grid(alpha=.2)
    fig.suptitle("FP32 scaling · 32K / 262K / 1M · shaded bands are sample min–max")
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="outside lower center",ncol=3,frameon=False)
    fig.savefig(args.directory/"scaling-float32.png",dpi=150);plt.close(fig)
    # Actual fresh calls, not sums of independently measured stage medians.
    fresh=[("fresh_native_input","native_input","Native CSR / input"),
           ("fresh_native_morton","native_morton","Sort + native CSR"),
           ("fresh_tile","tile_shared","Sort + shared cells"),
           ("fresh_flat","tile_flat","Sort + flat cells"),
           ("fresh_warp_grid","warp_grid_morton","Warp fresh grid")]
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for row,nh in enumerate([4.,2.57]):
        for col,dimension in enumerate([1,2,3]):
            axis=axes[row,col]
            case=next(c for c in runs["float32"]["cases"] if c["n"]==1000000 and c["dimension"]==dimension and c["n_h"]==nh)
            data=[case[key] for key,_,_ in fresh]
            axis.barh(range(len(data)),[r["median_ms"] for r in data],color=[colors[name] for _,name,_ in fresh],
                xerr=[[r["median_ms"]-r["min_ms"] for r in data],[r["max_ms"]-r["median_ms"] for r in data]],error_kw=dict(elinewidth=.7,capsize=1.5))
            axis.set_yticks(range(len(data)),[label for _,_,label in fresh],fontsize=9)
            axis.invert_yaxis();axis.set_xscale("log");axis.grid(axis="x",alpha=.2);axis.set_axisbelow(True)
            axis.xaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
            axis.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f"{x:g}"))
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.set_xlim(min(r["min_ms"] for r in data)*.75,max(r["max_ms"] for r in data)*1.6)
            for y,r in enumerate(data):
                axis.text(r["max_ms"]*1.06,y,f"{r['median_ms']:.2f}",va="center",fontsize=8)
            axis.set(xlabel="Fresh build + interpolation (ms; log scale)",title=f"{dimension}D · nh={nh:g}")
    fig.suptitle("FP32 · 1,000,000 particles · actual fresh calls\nSorting, permutation, metadata/CSR allocation and interpolation included")
    fig.savefig(args.directory/"fresh-float32.png",dpi=150);plt.close(fig)
    lines=["# Cell-to-cell experiment graphs","",
        "![FP32 interpolation](interpolation-float32.png)","",
        "![FP64 interpolation](interpolation-float64.png)","",
        "![FP32 scaling](scaling-float32.png)","",
        "![Fresh FP32 calls](fresh-float32.png)","",
        "All bars/lines use complete final sweeps; whiskers/bands show observed min–max, not confidence intervals.",
        "throughput.csv exports all completed-call useful and visited pair rates. resident.json records separate C++ batches.","",
        "## Sampled activity",""]
    for precision,run in runs.items():
        pid=str(run["pid"])
        other=sum(any(line.split(",")[0].strip()!=pid for line in s["processes"].get("stdout","").splitlines()[1:]) for s in run["activity"])
        lines.append(f"- {precision}: {other}/{len(run['activity'])} snapshots contain another compute process.")
    (args.directory/"graphs.md").write_text("\n".join(lines)+"\n")
    print(args.directory/"graphs.md")


if __name__ == "__main__":
    main()
