#!/usr/bin/env python3
"""Compare fixed CSR operators with identical owned-row chunks.

See CHUNKING.md. Setup/search/ordering are outside operator timings. Native and
Warp execute in separate processes. Profiler passes never supply base timings.
"""

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import random
import sys
import traceback
import hashlib

p = argparse.ArgumentParser()
p.add_argument("--backend", choices=["warp", "native"], default="warp")
p.add_argument("--device", default="cuda:0")
p.add_argument("--nx", type=int, default=48)
p.add_argument("--n-h", type=float, default=4.0)
p.add_argument("--list-scale", type=float, default=1.24)
p.add_argument("--precision", choices=["float32", "float64"], default="float32")
p.add_argument("--index-dtype", choices=["int32", "int64"], default="int32")
p.add_argument("--particle-order", choices=["input", "morton"], default="input")
p.add_argument("--budget-gib", type=float, default=8.0)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--chunks", nargs="+", type=int, default=[0, 1, 2, 4, 8])
p.add_argument(
    "--lanes", nargs="+", type=int, choices=[1, 2, 4, 8, 16, 32], default=[1]
)
p.add_argument(
    "--operations",
    nargs="+",
    choices=[
        "density",
        "gradient",
        "divergence",
        "laplacian",
        "covariance",
        "crk_factors",
    ],
    default=["density", "gradient", "laplacian", "crk_factors"],
)
p.add_argument("--repeats", type=int, default=7)
p.add_argument("--calls", type=int, default=10)
p.add_argument("--warmup", type=int, default=5)
p.add_argument("--profile", action="store_true")
p.add_argument("--output", required=True, type=Path)
a = p.parse_args()
if a.output.exists():
    p.error("--output must be a fresh path")
if a.nx < 2 or a.threads < 1 or min(a.repeats, a.calls, a.warmup) < 1:
    p.error("sizes, threads and timing counts must be positive")
if (
    not math.isfinite(a.n_h)
    or a.n_h <= 0
    or not math.isfinite(a.list_scale)
    or a.list_scale < 1
):
    p.error("support must be positive and list scale at least one")
if a.n_h * a.list_scale / a.nx >= 0.5:
    p.error("periodic search support must be less than half the unit box; increase nx")
if not math.isfinite(a.budget_gib) or a.budget_gib <= 0:
    p.error("capacity budget must be positive and finite")
if min(a.chunks) < 0 or max(a.chunks) > a.nx**3:
    p.error("chunk counts must be between zero (CSR) and particle count")
if a.backend == "native" and a.lanes != [1]:
    p.error("--lanes controls Warp only; native kernels own their launch configuration")
a.chunks = list(dict.fromkeys(a.chunks))
a.operations = list(dict.fromkeys(a.operations))
a.lanes = list(dict.fromkeys(a.lanes))
# Screen all resident topologies, candidates, snapshots and reference/output fields.
# This is an estimate, not an allocator limit; actual edge counts are recorded.
max_degree = min(a.nx**3, (2 * math.ceil(a.n_h * a.list_scale) + 1) ** 3)
index_bytes = 4 if a.index_dtype == "int32" else 8
estimated = a.nx**3 * (
    max_degree * index_bytes * (1 + sum(c > 0 for c in a.chunks)) * 1.25 + 2048
)
if estimated > a.budget_gib * 1024**3:
    p.error(
        f"estimated resident fixture {estimated/1024**3:.2f} GiB exceeds --budget-gib; raise budget explicitly"
    )
a.output.parent.mkdir(parents=True, exist_ok=True)
source = Path(__file__).read_bytes()
a.output.with_suffix(".source.py").write_bytes(source)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from _single_device import configure, metadata, measure, profile_call, write_result

q = dict(
    precision=a.precision,
    backend=a.backend,
    device=a.device,
    threads=a.threads,
    index_dtype=a.index_dtype,
    offset_dtype="int64",
)
os.environ["WARPSPHCORE_OPERATORS"] = a.backend
device = configure(q)
import torch
import warpSPHCore as sph
from warpSPHCore.autograd.lanes import setNeighborLanes
from warpSPHCore.radiusSearch import radiusSearchRowChunks, RowChunkPolicy
from warpSPHCore.radiusSearch.native import radiusSearch
from _operator_cases import check_output


def fixture():
    dtype = getattr(torch, a.precision)
    n = a.nx**3
    axis = (torch.arange(a.nx, device=device, dtype=dtype) + 0.5) / a.nx
    pos = (
        torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
        .reshape(-1, 3)
        .contiguous()
    )
    if a.particle_order == "morton":
        cell = torch.floor(pos * (1 << 21)).long().clamp_(0, (1 << 21) - 1)
        keys = torch.zeros(n, dtype=torch.int64, device=device)
        for bit in range(21):
            for axis_id in range(3):
                keys.bitwise_or_(((cell[:, axis_id] >> bit) & 1) << (3 * bit + axis_id))
        permutation = torch.argsort(keys, stable=True)
        pos = pos[permutation].contiguous()
    h = torch.full((n,), a.n_h / a.nx, device=device, dtype=dtype)
    particles = sph.ParticleState(
        pos,
        h,
        torch.full_like(h, 1 / n),
        torch.zeros(n, device=device, dtype=torch.int32),
        torch.ones_like(h),
    )
    domain = sph.DomainDescription(
        torch.zeros(3, device=device, dtype=dtype),
        torch.ones(3, device=device, dtype=dtype),
        torch.ones(3, device=device, dtype=torch.bool),
        3,
    )
    g = torch.Generator(device=device).manual_seed(173)
    scalar = torch.randn(n, device=device, dtype=dtype, generator=g)
    vector = torch.randn(n, 3, device=device, dtype=dtype, generator=g)
    if a.particle_order == "morton":
        scalar, vector = (value[permutation].contiguous() for value in (scalar, vector))
    search = replace(particles, supports=h * a.list_scale)
    flat = radiusSearch(
        search, domain, mode=sph.SupportScheme.Gather, backend=a.backend
    )
    # Native search emits int64 IDs. Normalize the frozen operator fixture,
    # outside timing, so backend comparisons use the requested storage width.
    index_dtype = getattr(torch, a.index_dtype)
    flat = replace(
        flat,
        paddedJ=flat.paddedJ.to(index_dtype),
        paddedI=None if flat.paddedI is None else flat.paddedI.to(index_dtype),
    )
    edges = int(flat.numNeighbors.sum())
    topologies = {0: flat}
    for count in a.chunks:
        if count == 0:
            continue
        policy = RowChunkPolicy(
            edgeBudget=edges * 2 + 1,
            maxRows=math.ceil(n / count),
            headroomNumerator=0,
            bytesPerEdge=flat.paddedJ.element_size(),
        )
        adj = radiusSearchRowChunks(
            search, domain, policy=policy, mode=sph.SupportScheme.Gather, storeI=False
        )
        torch.testing.assert_close(adj.numNeighbors, flat.numNeighbors, rtol=0, atol=0)
        # Exact row/pair preservation against the same backend's canonical list.
        for chunk in adj:
            m = chunk.metadata
            start = int(flat.edgeOffsets[m.rowStart])
            live = chunk.adjacency.numEdges
            torch.testing.assert_close(
                chunk.adjacency.paddedJ[:live],
                flat.paddedJ[start : start + live],
                rtol=0,
                atol=0,
            )
        topologies[count] = adj
    return particles, domain, scalar, vector, edges, topologies


particles, domain, scalar, vector, edges, topologies = fixture()


def call(op, adj):
    props = sph.OperationProperties(
        kernel=sph.KernelFunctions.Wendland2,
        supportMode=sph.SupportScheme.Gather,
        gradientMode=sph.GradientScheme.Difference,
        laplacianMode=sph.LaplacianScheme.Brookshaw,
    )
    if op == "crk_factors":
        return lambda: sph.computeCRKFactors(
            particles, domain, sph.KernelFunctions.Wendland2, adjacency=adj
        )
    props = replace(props, operation=getattr(sph.WarpOperation, op.capitalize()))
    values = vector if op in ("divergence", "curl") else scalar
    return lambda: sph.warpOperation(
        particles,
        props,
        domain,
        adjacency=adj,
        queryValues=values,
        referenceValues=values,
    )


def validate(op, out, reference):
    # CRK B/gradA/gradB have inverse-length dimensions. Compare their
    # dimensionless forms so an absolute FP32 tolerance does not depend on dx.
    if op != "crk_factors":
        return check_output(out, reference)
    check_output(out)
    tol = 2e-4 if a.precision == "float32" else 2e-10
    h = particles.supports
    for actual, expected, scale in [
        (out[0], reference[0], h**-3),
        (out[1], reference[1], 1.0),
    ]:
        torch.testing.assert_close(actual * scale, expected * scale, rtol=tol, atol=tol)
    for name, power in [("A", 0), ("B", 1), ("gradA", 1), ("gradB", 2)]:
        x, y = getattr(out[2], name), getattr(reference[2], name)
        scale = h.reshape(-1, *([1] * (x.ndim - 1))) ** power
        torch.testing.assert_close(x * scale, y * scale, rtol=tol, atol=tol)


report = dict(
    schema_version=1,
    arguments=vars(a) | dict(output=str(a.output)),
    machine=metadata(device),
    probe_sha256=hashlib.sha256(source).hexdigest(),
    estimated_resident_bytes=estimated,
    warp_lanes_note="Warp override only; native kernels own their cooperative launch geometry",
    particles=a.nx**3,
    candidate_edges=edges,
    mean_candidates=edges / (a.nx**3),
    index_dtype=str(topologies[0].paddedJ.dtype),
    offset_dtype=str(topologies[0].edgeOffsets.dtype),
    fixture_note="Native search IDs normalized to requested index width outside timing; same-backend exact topology and CSR output references checked",
    chunk_rows={
        str(c): [x.metadata.rowCount for x in adj.chunks]
        for c, adj in topologies.items()
        if c
    },
    records=[],
)
requests = [
    (op, chunk, lane) for op in a.operations for chunk in a.chunks for lane in a.lanes
]
references = {}
setNeighborLanes(1)
for op in a.operations:
    references[op] = call(op, topologies[0])()
import numpy as np

saved = {}
for op, value in references.items():
    if op == "crk_factors":
        for name, tensor in [
            ("volume", value[0]),
            ("density", value[1]),
            *[(name, getattr(value[2], name)) for name in ("A", "B", "gradA", "gradB")],
        ]:
            saved[op + "_" + name] = tensor.detach().cpu().numpy()
    else:
        saved[op] = value.detach().cpu().numpy()
np.savez_compressed(a.output.with_suffix(".reference.npz"), **saved)
random.Random(173).shuffle(requests)
# Group lane variants to avoid repeated Warp module compilation as block sizes change.
requests.sort(key=lambda item: item[2])
for op, chunk, lane in requests:
    setNeighborLanes(lane)
    fn = call(op, topologies[chunk])
    record = dict(
        operation=op,
        requested_chunks=chunk,
        actual_chunks=0 if chunk == 0 else len(topologies[chunk].chunks),
        lanes=lane,
        status="failed",
    )
    try:
        record["timing"] = measure(
            fn,
            device,
            warmup=a.warmup,
            repeats=a.repeats,
            calls=a.calls,
            validate=lambda out: validate(op, out, references[op]),
        )
        if a.profile:
            trace = a.output.with_name(
                f"{a.output.stem}-{op}-chunks{chunk}-lanes{lane}.trace.json"
            )
            record["profile"] = profile_call(fn, device, trace)
            # Keep base timing uninstrumented for every subsequent case.
            import warpSPHCore.profiling as hooks

            hooks._torchRecordFunction = None
            import cProfile, pstats

            prof = cProfile.Profile()
            prof.enable()
            for _ in range(a.calls):
                out = fn()
                del out
            torch.cuda.synchronize(device) if device.type == "cuda" else None
            prof.disable()
            stats = pstats.Stats(prof)
            entries = []
            for (filename, line, name), (
                primitive,
                total,
                self_s,
                cumulative_s,
                callers,
            ) in stats.stats.items():
                entries.append(
                    dict(
                        file=filename,
                        line=line,
                        function=name,
                        calls=total,
                        self_s=self_s,
                        cumulative_s=cumulative_s,
                    )
                )
            entries.sort(key=lambda r: r["cumulative_s"], reverse=True)
            record["python_profile"] = dict(
                calls=a.calls, total_elapsed_s=stats.total_tt, functions=entries[:65]
            )
        record["status"] = "ok"
    except Exception as error:
        record.update(
            error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()
        )
    report["records"].append(record)
    write_result(a.output, report)
    print(
        op,
        chunk,
        lane,
        record["status"],
        record.get("timing", {}).get("wall_ms", {}).get("median"),
        record.get("error", ""),
        flush=True,
    )
raise SystemExit(any(r["status"] != "ok" for r in report["records"]))
