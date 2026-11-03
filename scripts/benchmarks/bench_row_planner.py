#!/usr/bin/env python3
"""CPU-only shared row planner timing; see CHUNKING.md. No device build/init."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import traceback

from _single_device import metadata, measure, write_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', nargs='+', type=int, default=[48**3, 128**3])
    parser.add_argument('--neighbors', type=int, default=515)
    parser.add_argument('--distribution', choices=('constant', 'uniform'), default='constant')
    parser.add_argument('--owners', nargs='+', type=int, default=[1])
    parser.add_argument('--policy', type=Path, help='optional JSON object of RowChunkPolicy fields')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--calls', type=int, default=1)
    parser.add_argument('--budget-gib', type=float, default=8.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output must be a fresh path')
    if min(args.rows) < 0 or max(args.rows) > 2**31-1 or not 0 <= args.neighbors <= 2**31-1:
        parser.error('rows and neighbors must fit nonnegative int32')
    if min(args.owners) < 1 or min(args.warmup, args.repeats, args.calls) < 1:
        parser.error('owners and timing counts must be positive')
    if not math.isfinite(args.budget_gib) or args.budget_gib <= 0:
        parser.error('--budget-gib must be positive and finite')
    if max(args.rows)*80 > args.budget_gib*1024**3:
        parser.error('estimated host fixture exceeds --budget-gib; raise it explicitly')

    # The planner is shared Python/NumPy code. Select the light native import by
    # default; neither native compilation nor CUDA initialization is needed.
    os.environ.setdefault('WARPSPHCORE_BACKEND', 'native')
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from warpSPHCore.radiusSearch.rowChunks import RowChunkGeneration, RowChunkPolicy, buildRowChunkPlan
    policy_fields = json.loads(args.policy.read_text()) if args.policy else dict(
        edgeBudget=8000000, maxRows=65536, bytesPerEdge=16, maxElementsPerEdge=3)
    policy = RowChunkPolicy(**policy_fields)
    device = torch.device('cpu')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).read_bytes()
    args.output.with_suffix('.source.py').write_bytes(source)
    report = dict(schema_version=1, kind='row_planner', machine=metadata(device),
                  arguments=vars(args) | dict(output=str(args.output), policy=asdict(policy)),
                  benchmark_sha256=hashlib.sha256(source).hexdigest(),
                  timing_scope='CPU counts to immutable plan, including output allocation/release; count generation and validation excluded',
                  records=[])
    for rows in dict.fromkeys(args.rows):
        generator = torch.Generator().manual_seed(173)
        counts = (torch.full((rows,), args.neighbors, dtype=torch.int32) if args.distribution == 'constant'
                  else torch.randint(args.neighbors+1, (rows,), generator=generator, dtype=torch.int32))
        generation = RowChunkGeneration(0, max(rows, args.neighbors), 'row-planner-benchmark-v1')
        counts_hash = hashlib.sha256(counts.numpy()).hexdigest()
        for owners in dict.fromkeys(args.owners):
            record = dict(rows=rows, num_references=generation.numReferences, owners=owners,
                          counts_sha256=counts_hash, status='failed')
            try:
                def build():
                    return buildRowChunkPlan(counts, generation=generation, policy=policy, numOwners=owners)
                reference = build()
                signature = reference.signature
                record.update(chunks=len(reference.chunks), total_edges=reference.totalEdges, signature=signature)
                del reference
                def validate(output):
                    if output.signature != signature:
                        raise AssertionError('planner partition/signature changed for fixed counts')
                record['timing'] = measure(build, device, warmup=args.warmup,
                    repeats=args.repeats, calls=args.calls, validate=validate)
                record['status'] = 'ok'
            except Exception as error:
                record.update(error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
            report['records'].append(record)
            write_result(args.output, report)
            print(rows, owners, record['status'], record.get('timing', {}).get('wall_ms', {}).get('median'), flush=True)
    return int(any(row['status'] != 'ok' for row in report['records']))


if __name__ == '__main__':
    raise SystemExit(main())
