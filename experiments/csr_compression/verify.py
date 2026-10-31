#!/usr/bin/env python3
"""Offline exact codec checks and independent dense interpolation references."""
import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import torch
from runtime import CELL, DELTA, RAW32, RAW64, Library, build_plan, descriptor, prefix


@dataclass
class State:
    positions: torch.Tensor
    supports: torch.Tensor
    masses: torch.Tensor
    densities: torch.Tensor


def decode_cpu(plan):
    """Scalar reader independent of the device's ballot/popcount/prefix scan."""
    buffers = {key: value.cpu().tolist() for key, value in plan.buffers.items()}
    rows = []
    for row, count in enumerate(buffers['counts']):
        result = []
        for block in range(buffers['row_blocks'][row], buffers['row_blocks'][row+1]):
            cursor = buffers['meta_offsets'][block] & 0xffffffff
            previous = (buffers.get('bases', [0])[block] & 0xffffffff) if plan.encoding == DELTA else 0
            base = 0
            starts = (buffers['masks'][block] & 0xffffffff) if plan.encoding == CELL else 0
            for lane in range(min(plan.width, count-len(result))):
                code = buffers['codes'][block*plan.width+lane]
                if plan.encoding == DELTA:
                    if code == 255:
                        code = buffers['extras'][cursor] & 0xffffffff
                        cursor += 1
                    previous += code
                    result.append(previous)
                else:
                    if starts & (1 << lane):
                        base = buffers['extras'][cursor] & 0xffffffff
                        cursor += 1
                    result.append(base+code)
        rows.extend(result)
    return torch.tensor(rows, dtype=torch.int64)


def check_codec(library, plan, offsets, indices):
    actual = torch.empty_like(indices)
    request = descriptor(dict(plan.buffers, row_offsets=offsets, decoded=actual), len(plan.buffers['counts']))
    for action in (2,3):
        actual.fill_(-1)
        library.codec(request, plan.encoding, plan.width, action)
        torch.testing.assert_close(actual, indices, rtol=0, atol=0)
    torch.testing.assert_close(decode_cpu(plan), indices.cpu(), rtol=0, atol=0)


def synthetic(library):
    delta_rows = [[], [0], list(range(31)), list(range(32)), list(range(33)), list(range(65)),
                  [0, 1, 255, 510, 766, 65535, 65536, 16777217, 2147483647]]
    cell_rows = [[], [0], list(range(31)), list(range(32)), list(range(33)), list(range(1537)),
                 [0, 1, 254, 255, 256, 257, 511, 512, 768, 1199, 1200, 1536]]
    cell_offsets = torch.tensor([0, 0, 1, 257, 1200, 1537], device='cuda', dtype=torch.int64)
    records = []
    for encoding, rows in ((DELTA, delta_rows), (CELL, cell_rows), (DELTA, []), (CELL, [])):
        counts = torch.tensor([len(row) for row in rows], device='cuda', dtype=torch.int32)
        offsets = prefix(counts)
        indices = torch.tensor([j for row in rows for j in row], device='cuda', dtype=torch.int64)
        for width in (4, 8, 16, 32):
            plan = build_plan(library, indices, counts, offsets, width, encoding, cell_offsets, 1537)
            check_codec(library, plan, offsets, indices)
            records.append(dict(kind='synthetic', encoding=encoding, width=width, rows=len(rows)))
    return records


def dense(state, values, periodic):
    x = state.positions.double()
    dx = x[:, None, :] - x[None, :, :]
    dx -= torch.floor(dx+.5)*periodic
    r = dx.norm(dim=-1)
    dimension = x.shape[1]
    h = state.supports.double()
    q = r/h[:, None].clamp_min(1e-100)
    t = (1-q).clamp_min(0)
    coefficient = [1.25, 2.228169203286535, 3.3422538049298023][dimension-1]
    weight = coefficient*t.pow(3 if dimension == 1 else 4)*(1+(3 if dimension == 1 else 4)*q)
    weight /= h[:, None].clamp_min(1e-100).pow(dimension)
    weight[h <= 0] = 0
    field = values.double()*(state.masses.double()/state.densities.double())[:, None]
    return weight @ field, r <= h[:, None]


def fixture(n, dimension, dtype, periodic=True):
    generator = torch.Generator().manual_seed(51031+n+dimension)
    x = torch.rand((n, dimension), generator=generator, dtype=torch.float64)
    x[:n//2] = .47+.02*x[:n//2]
    edge_points = min(7, n-n//2)
    x[n//2:n//2+edge_points, 0] = torch.linspace(0, .999999, edge_points)
    # Sort by coarse Morton-like integer keys, independently of the GPU planner.
    cell = torch.floor(x*7).long()
    key = torch.zeros(n, dtype=torch.int64)
    for bit in range(3):
        for d in range(dimension):
            key |= ((cell[:, d] >> bit) & 1) << (bit*dimension+d)
    order = torch.argsort(key, stable=True)
    x = x[order].to(device='cuda', dtype=dtype)
    h = .14*(.65+.35*torch.rand(n, generator=generator, dtype=torch.float64))
    h[::71] = 0
    masses = (.5+torch.rand(n, generator=generator, dtype=torch.float64))/n
    densities = .7+torch.rand(n, generator=generator, dtype=torch.float64)
    state = State(x, h.to(x), masses.to(x), densities.to(x))
    values = torch.randn((n, dimension), generator=generator, dtype=torch.float64).to(x)
    # Occupied cell boundaries, including dense cells >256 for the 769 case.
    _, cell_counts = torch.unique_consecutive(key[order], return_counts=True)
    cell_offsets = prefix(cell_counts.to('cuda'))
    flags = torch.full((dimension,), periodic, device='cuda', dtype=torch.bool)
    if dimension > 1:
        flags[-1] = not periodic
    return state, values, cell_offsets, flags


def check_numerical(library, n, dimension, dtype, periodic, quick):
    state, values, cell_offsets, flags = fixture(n, dimension, dtype, periodic)
    expected, mask = dense(state, values, flags)
    counts = mask.sum(1).int()
    offsets = prefix(counts)
    indices = mask.nonzero()[:, 1].contiguous()
    packed = values.new_empty((n, ((2*dimension+3)//4)*4))
    output = torch.empty_like(values)
    tensors = {name: getattr(state, name) for name in ('positions', 'supports', 'masses', 'densities')}
    tensors.update(values=values, lower=values.new_zeros(dimension), upper=values.new_ones(dimension),
                   periodic=flags, packed=packed, output=output)
    records = []
    configs = [(8, 2, 128, v) for v in (False,True)] if quick else (
        list(itertools.product((4,8,16,32), (1,2,4), (128,256), (False,))) +
        list(itertools.product((4,8,16,32), (1,2), (128,256), (True,))))
    tolerance = 4e-5 if dtype == torch.float32 else 4e-10
    for width in sorted({config[0] for config in configs}):
        for encoding in (DELTA, CELL):
            plan = build_plan(library, indices, counts, offsets, width, encoding, cell_offsets, n)
            check_codec(library, plan, offsets, indices)
            request = plan.request(tensors, n, dimension, int(dtype == torch.float64))
            # The retained operator really has no CSR pointer to fall back to.
            assert not request.indices and not request.row_offsets and not request.particle_bases
            for w, batch, threads, vector in configs:
                if w != width:
                    continue
                library.run(request, encoding, width, batch, threads, vector=vector)
                torch.testing.assert_close(output.double(), expected, rtol=tolerance, atol=tolerance)
                # Identical raw-code schedule must produce bit-identical output.
                for raw in (RAW64, RAW32):
                    raw_ids = indices if raw == RAW64 else indices.int()
                    raw_output = torch.empty_like(values)
                    raw_request = descriptor(dict(tensors, counts=counts, row_offsets=offsets,
                                                  indices=raw_ids, output=raw_output), n,
                                             dimension=dimension, precision=int(dtype == torch.float64))
                    library.run(raw_request, raw, width, batch, threads, vector=vector)
                    torch.testing.assert_close(output, raw_output, rtol=0, atol=0)
                records.append(dict(kind='numerical', n=n, dimension=dimension, precision=str(dtype),
                                    periodic=periodic, encoding=encoding, width=width,
                                    batch=batch, threads=threads, vector=vector))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(8)
    library = Library(args.library)
    results = synthetic(library)
    dimensions = (3,) if args.quick else (1,2,3)
    sizes = (33,769) if args.quick else (1,33,257,769)
    for n, dimension, dtype, periodic in itertools.product(sizes, dimensions,
                                                         (torch.float32,torch.float64), (True,False)):
        results.extend(check_numerical(library, n, dimension, dtype, periodic, args.quick))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        results.extend(check_numerical(library, 33, 3, torch.float32, True, True))
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    args.output.write_text(json.dumps(dict(complete=True, cases=len(results), nondefault_stream=True,
                                          results=results), indent=2)+'\n')
    print(f'Passed {len(results)} codec/numerical configurations and a nondefault stream', flush=True)

if __name__ == '__main__':
    main()
