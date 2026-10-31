#!/usr/bin/env python3
"""Plot complete paired sweeps; select schedules only from the training screen."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODES = ["register", "shared", "serial", "super_shared"]
COLORS = {"native_coarse": "#777777", "native_refined": "#1764ab", "cell": "#e88c20",
          "raw": "#22883c", "compressed": "#8953ae", "warp_refined": "#bd4d4d"}
LABELS = {"native_coarse": "Native CSR, coarse Morton", "native_refined": "Native CSR, refined Morton",
          "cell": "Best cell schedule", "raw": "Cluster, uncompressed", "compressed": "Cluster, compressed",
          "warp_refined": "Warp CSR, refined Morton"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    rows, summary = [], []
    for precision in ["float32", "float64"]:
        report = json.loads((args.archive/f"final-{precision}.json").read_text())
        screen = json.loads((args.archive/f"screen-{precision}.json").read_text())
        # Cell schedules are retained from the independent earlier screen. For
        # the cell-family aggregate, use its 262K completed training measurement.
        training = { (c["dimension"], c["n_h"]): c for c in report["cases"] if c["n"] == 262144 }
        if len(training) != 6:
            raise ValueError("Aggregate plots require a 262K training/control case for every dimension/load")
        selected = {}
        for c in screen["cases"]:
            key = c["dimension"], c["n_h"]
            selected[key] = {}
            for compressed in [False, True]:
                best = min((r for r in c["screening"] if r["configuration"][5] == compressed), key=lambda r: r["median_ms"])
                selected[key]["compressed" if compressed else "raw"] = f"cluster_{'compressed' if compressed else 'raw'}_{MODES[best['configuration'][4]]}"
            best_cell = min((r for r in training[key]["records"] if r["backend"].startswith("cell_")), key=lambda r: r["median_ms"])
            selected[key]["cell"] = best_cell["backend"]

        for c in report["cases"]:
            for r in c["records"]:
                rows.append(dict(precision=precision, dimension=c["dimension"], n_h=c["n_h"], n=c["n"],
                    backend=r["backend"], milliseconds=r["median_ms"], min_ms=r["min_ms"], max_ms=r["max_ms"],
                    useful_gpairs_s=r["useful_gpairs_s"], warp_qualified=c["warp_qualified"]))
            if c["n"] == 1000000:
                records = {r["backend"]: r for r in c["records"]}
                choice = selected[c["dimension"], c["n_h"]]
                summary.append(dict(precision=precision, dimension=c["dimension"], n_h=c["n_h"],
                    neighbors=c["neighbors"], selected=choice,
                    milliseconds={k: records[choice.get(k, k)]["median_ms"] for k in COLORS},
                    fresh_ms={k: c["fresh"][choice.get(k, k)]["median_ms"] for k in ["native_refined", "cell", "raw", "compressed"]},
                    plans=c["plans"]))

        fig, axes = plt.subplots(2, 3, figsize=(13.5, 7), sharex=True)
        for dimension in [1, 2, 3]:
            for row, nh in enumerate([4., 2.57]):
                ax = axes[row, dimension-1]
                cases = sorted((c for c in report["cases"] if c["dimension"] == dimension and c["n_h"] == nh), key=lambda c: c["n"])
                for family in COLORS:
                    name = selected[dimension, nh].get(family, family)
                    times, lows, highs, sizes = [], [], [], []
                    for c in cases:
                        # Preserve the known high-resolution FP32 Warp failure;
                        # do not draw an unqualified speedup comparison.
                        if family == "warp_refined" and not c["warp_qualified"]:
                            continue
                        r = next(r for r in c["records"] if r["backend"] == name)
                        sizes.append(c["n"])
                        times.append(r["useful_gpairs_s"])
                        lows.append(c["edges"]/(r["max_ms"]*1e6))
                        highs.append(c["edges"]/(r["min_ms"]*1e6))
                    ax.plot(sizes, times, "o-", color=COLORS[family], label=LABELS[family], markersize=4)
                    ax.fill_between(sizes, lows, highs, color=COLORS[family], alpha=.12)
                ax.set_xscale("log")
                ax.set_title(f"{dimension}D, n_h={nh:g}")
                ax.grid(True, alpha=.2)
                if row == 1: ax.set_xlabel("Particles")
        handles, labels = axes[0, 1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .005), ncol=3, frameon=False)
        fig.supylabel("Useful billion interactions / second", x=.005)
        fig.suptitle(f"{precision.upper()} interpolation, completed calls; plans reused\nSchedules chosen at 262K; bands show observed min–max")
        fig.subplots_adjust(left=.07, right=.99, bottom=.18, top=.85, wspace=.22, hspace=.35)
        fig.savefig(args.archive/f"throughput-{precision}.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(12.5, 7))
        for dimension in [1, 2, 3]:
            for row, nh in enumerate([4., 2.57]):
                ax = axes[row, dimension-1]
                c = next(c for c in report["cases"] if c["dimension"] == dimension and c["n_h"] == nh and c["n"] == 1000000)
                families = ["native_refined", "cell", "raw", "compressed"]
                names = [selected[dimension, nh].get(f, f) for f in families]
                times = [c["fresh"][name]["median_ms"] for name in names]
                ax.bar(range(4), times, color=[COLORS[f] for f in families])
                ax.set_xticks(range(4), ["CSR", "Cells", "Clusters", "Compressed"], rotation=15)
                ax.set_yscale("log")
                ax.set_ylim(min(times)*.8, max(times)*1.5)
                ax.set_title(f"{dimension}D, n_h={nh:g}")
                ax.grid(True, axis="y", alpha=.2)
                for i, t in enumerate(times): ax.text(i, t*1.05, f"{t:.2f}", ha="center", fontsize=8)
        fig.supylabel("Fresh build + interpolation (ms)", x=.005)
        fig.suptitle(f"{precision.upper()}, one million particles\nIncludes sorting, permutation, list construction and one interpolation")
        fig.subplots_adjust(left=.075, right=.99, bottom=.1, top=.85, wspace=.24, hspace=.5)
        fig.savefig(args.archive/f"fresh-{precision}.png", dpi=160)
        plt.close(fig)
    with (args.archive/"throughput.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.archive/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (args.archive/"graphs.md").write_text("# Cluster-pair experiment figures\n\n"+"\n".join(
        f"![{kind} {precision}]({kind}-{precision}.png)\n" for kind in ["throughput", "fresh"] for precision in ["float32", "float64"]))


if __name__ == "__main__":
    main()
