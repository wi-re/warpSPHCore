#!/usr/bin/env python3
"""Independent numerical, list and codec checks; never imported by timed calls."""
import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path

import torch
from runtime import ClusterCase, Library, build_cells, build_clusters, call, prefix


@dataclass
class State:
    positions: torch.Tensor
    supports: torch.Tensor
    masses: torch.Tensor
    kinds: torch.Tensor
    densities: torch.Tensor


def dense(state, values):
    x = state.positions.double()
    dx = x[:, None, :] - x[None, :, :]
    dx -= torch.floor(dx + .5)
    distance = dx.norm(dim=-1)
    dimension = x.shape[1]
    q = distance / state.supports.double()[:, None]
    t = (1-q).clamp_min(0)
    coefficient = [1.25, 2.228169203286535, 3.3422538049298023][dimension-1]
    weight = coefficient*t.pow(3 if dimension == 1 else 4)*(1+(3 if dimension == 1 else 4)*q)
    weight /= state.supports.double()[:, None].pow(dimension)
    field = values.double()*(state.masses.double()/state.densities.double())[:, None]
    return weight @ field


def decode_bytes(data, count):
    """Independent CPU reader for the block format, including partial blocks."""
    result, cursor, previous = [], 0, 0
    for base in range(0, count, 32):
        size = min(32, count-base)
        unity = int.from_bytes(bytes(data[cursor:cursor+4]), "little")
        non_unity = [i for i in range(size) if not (unity >> i) & 1]
        infos = data[cursor+4:cursor+4+(len(non_unity)+1)//2]
        tags = [(infos[i//2] >> (4*(i % 2))) & 15 for i in range(len(non_unity))]
        payload_begin = cursor+4+(len(non_unity)+1)//2
        nibble_cursor = 0
        delta = [1]*size
        for i, tag in zip(non_unity, tags):
            if tag >= 8:
                delta[i] = tag-6
            else:
                value = 0
                for digit in range(tag+1):
                    nibble = (data[payload_begin+nibble_cursor//2] >> (4*(nibble_cursor % 2))) & 15
                    value |= nibble << (4*digit)
                    nibble_cursor += 1
                delta[i] = value
        for value in delta:
            previous += value
            result.append(previous-1)
        cursor = payload_begin+(nibble_cursor+1)//2
    assert cursor == len(data)
    return result


def check_lists(plan):
    buffers = plan.buffers
    offsets = buffers["offsets"].cpu().tolist()
    entries = buffers["entries"].cpu().tolist()
    masks = buffers["masks"].cpu().tolist()
    byte_offsets = buffers["byte_offsets"].cpu().tolist()
    encoded = buffers["compressed"].cpu().tolist()
    images = buffers["image_codes"].cpu().tolist()
    for g, (lo, hi) in enumerate(zip(offsets, offsets[1:])):
        ids = [int(e) >> 5 for e in entries[lo:hi]]
        assert ids == sorted(set(ids))
        assert all(0 < m < 2**plan.group_size for m in masks[lo:hi])
        assert decode_bytes(encoded[byte_offsets[g]:byte_offsets[g+1]], hi-lo) == ids
        assert images[lo:hi] == [e & 31 for e in entries[lo:hi]]


def check_synthetic_codec(library):
    # Independently cover unity runs, all direct tags, payload digit boundaries,
    # empty rows, both sides of 32 entries, and high valid 27-bit source IDs.
    rows = [[], [0], list(range(32)), list(range(65)),
            [0, 2, 5, 9, 14, 20, 27, 35, 44, 54, 69, 85, 340,
             596, 4691, 8787, 74322, 139858, 1188433, 2237009, 19014224, 35791440, 102900304]]
    counts = torch.tensor([len(row) for row in rows], device="cuda", dtype=torch.int64)
    offsets = prefix(counts)
    ids = torch.tensor([i for row in rows for i in row], device="cuda", dtype=torch.int32)
    image = torch.arange(len(ids), device="cuda", dtype=torch.int32) % 27
    entries = (ids << 5) | image
    byte_counts = torch.empty_like(counts)
    descriptor = ClusterCase()
    descriptor.groups = len(rows)
    descriptor.offsets, descriptor.entries = offsets.data_ptr(), entries.data_ptr()
    descriptor.byte_counts = byte_counts.data_ptr()
    library.invoke("cluster_compress", descriptor, 0)
    byte_offsets = prefix(byte_counts)
    encoded = torch.empty(int(byte_offsets[-1]), device="cuda", dtype=torch.uint8)
    image_codes = torch.empty(len(ids), device="cuda", dtype=torch.uint8)
    descriptor.byte_offsets, descriptor.compressed = byte_offsets.data_ptr(), encoded.data_ptr()
    descriptor.image_codes = image_codes.data_ptr()
    library.invoke("cluster_compress", descriptor, 1)
    actual = torch.empty_like(ids)
    descriptor.output = actual.data_ptr()
    library.invoke("cluster_decode", descriptor)
    torch.testing.assert_close(actual, ids, rtol=0, atol=0)
    torch.testing.assert_close(image_codes.int(), image, rtol=0, atol=0)
    encoded, byte_offsets = encoded.cpu().tolist(), byte_offsets.cpu().tolist()
    for g, row in enumerate(rows):
        assert decode_bytes(encoded[byte_offsets[g]:byte_offsets[g+1]], len(row)) == row
    return len(rows)


def fixture(n, dimension, dtype, max_support):
    generator = torch.Generator().manual_seed(5521+dimension+n)
    x = torch.rand((n, dimension), generator=generator, dtype=torch.float64)
    # Empty cells, a heavily populated cell, periodic pairs, and partial tiles.
    x[:n//2] = .47+.02*x[:n//2]
    x[n//2:n//2+7, 0] = torch.linspace(0, .999999, 7)
    h = max_support*(.65+.35*torch.rand(n, generator=generator, dtype=torch.float64))
    masses = (.5+torch.rand(n, generator=generator, dtype=torch.float64))/n
    densities = .7+torch.rand(n, generator=generator, dtype=torch.float64)
    state = State(x.to(device="cuda", dtype=dtype), h.to(device="cuda", dtype=dtype),
                  masses.to(device="cuda", dtype=dtype), torch.zeros(n, device="cuda", dtype=torch.int32),
                  densities.to(device="cuda", dtype=dtype))
    values = torch.randn((n, dimension), generator=generator, dtype=torch.float64).to(device="cuda", dtype=dtype)
    return state, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    library = Library(args.library)
    synthetic_rows = check_synthetic_codec(library)
    results = []
    for dtype, dimension, support, refinement in itertools.product(
            [torch.float32, torch.float64], [1, 2, 3], [.140001, .310001], [1, 8]):
        if args.quick and (dtype != torch.float32 or dimension != 3 or support > .2 or refinement != 8):
            continue
        state, values = fixture(257, dimension, dtype, support)
        cells = build_cells(library, state, values, support, refinement)
        expected = dense(state, values)[cells.order]
        for i, j, group, exact in itertools.product([4, 8], [4, 8], [1, 8], [False, True]):
            plan = build_clusters(library, cells, i, j, group, exact)
            check_lists(plan)
            for mode, compressed, warps in itertools.product(range(4), [False, True], [4, 8]):
                actual = call(library, plan, mode, compressed, warps).double()
                tolerance = 3e-5 if dtype == torch.float32 else 3e-10
                torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                results.append(dict(precision=str(dtype), dimension=dimension, support=support,
                                    refinement=refinement, tile=[i, j], group=group, exact=exact,
                                    mode=mode, compressed=compressed, warps=warps,
                                    max_abs=float((actual-expected).abs().max())))
    for dtype in [torch.float32, torch.float64]:
        state, values = fixture(67, 1, dtype, 2.57e-6)
        state.supports.fill_(2.57e-6)
        cells = build_cells(library, state, values, float(state.supports[0]), 8)
        assert cells.refinement == 4
        expected = dense(state, values)[cells.order]
        for compressed in [False, True]:
            plan = build_clusters(library, cells)
            actual = call(library, plan, compressed=compressed)
            tolerance = 3e-5 if dtype == torch.float32 else 3e-10
            torch.testing.assert_close(actual.double(), expected, rtol=tolerance, atol=tolerance)
    periodic_cutoff_cases = 0
    for dtype in [torch.float32, torch.float64]:
        # Around 1, adding the image first rounds at twice the spacing of
        # subtracting below 1. The box test must keep the subtraction order used
        # by the pair kernel. The query's self field is zero so a missing small
        # off-diagonal contribution cannot hide behind a huge self term.
        unit = 2.0**-24
        x = torch.tensor([[1-3*unit], [51*unit]], device="cuda", dtype=dtype)
        h = x.new_full((2,), 54.5*unit)
        state = State(x, h, h.new_ones(2), torch.zeros(2, device="cuda", dtype=torch.int32), h.new_ones(2))
        values = x.new_tensor([[0.], [1.]])
        cells = build_cells(library, state, values, float(h[0]))
        expected = dense(state, values)[cells.order]
        assert float(dense(state, values)[0, 0]) > .1
        for exact, compressed, mode in itertools.product([False, True], [False, True], [0, 3]):
            plan = build_clusters(library, cells, exact=exact)
            actual = call(library, plan, mode=mode, compressed=compressed)
            tolerance = 3e-5 if dtype == torch.float32 else 3e-10
            torch.testing.assert_close(actual.double(), expected, rtol=tolerance, atol=tolerance)
            periodic_cutoff_cases += 1
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Planning, variable-sized allocation, preparation and evaluation all
        # execute on the current nondefault stream.
        cells = build_cells(library, state, values, float(state.supports.max()), refinement)
        plan = build_clusters(library, cells)
        actual = call(library, plan, compressed=True)
    torch.cuda.current_stream().wait_stream(stream)
    expected = dense(state, values)[cells.order]
    torch.testing.assert_close(actual.double(), expected, rtol=tolerance, atol=tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(passed=len(results), stream_passed=True, synthetic_codec_rows=synthetic_rows,
        key_capacity_cases=4, periodic_cutoff_cases=periodic_cutoff_cases,
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(), records=results), indent=2)+"\n")
    print(f"{len(results)} numerical/list/codec configurations and nondefault stream passed")


if __name__ == "__main__":
    main()
