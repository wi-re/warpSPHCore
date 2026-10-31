"""Private GPU row-stream planner. Encoding and readbacks are offline work."""
import ctypes as C
from dataclasses import dataclass
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'csr_compression'))
from runtime import Case, Library as ByteLibrary, descriptor, prefix, RAW32, RAW64, DELTA, CELL

BAND, NIBBLE = 4, 5
NAMES = {RAW64: 'csr64', RAW32: 'csr32', BAND: 'band', NIBBLE: 'nibble'}


class Library(ByteLibrary):
    def __init__(self, path):
        super().__init__(path)
        self.library.gaps_codec.argtypes = [C.POINTER(Case)] + [C.c_int] * 4
        self.library.gaps_interpolate.argtypes = [C.POINTER(Case)] + [C.c_int] * 7
        self.library.gaps_codec.restype = self.library.gaps_interpolate.restype = C.c_int

    def codec(self, request, encoding, chunk, items=1, action=2):
        request.stream = torch.cuda.current_stream().cuda_stream
        self.check(self.library.gaps_codec(C.byref(request), encoding, chunk, items, action))

    def run(self, request, encoding, chunk, items=1, batch=1, threads=128, pack=True, iterations=1):
        request.stream = torch.cuda.current_stream().cuda_stream
        self.check(self.library.gaps_interpolate(C.byref(request), encoding, chunk, items,
                                                batch, threads, pack, iterations))


@dataclass
class Plan:
    encoding: int
    chunk: int
    buffers: dict

    def storage(self):
        return {name: value.numel() * value.element_size() for name, value in self.buffers.items()}

    def request(self, tensors, n, dimension, precision):
        return descriptor(dict(tensors, **self.buffers), n, dimension=dimension, precision=precision)


def build_plan(library, indices, counts, offsets, chunk, encoding):
    sizes = torch.empty_like(counts)
    temporary = dict(indices=indices, counts=counts, row_offsets=offsets, meta_counts=sizes)
    request = descriptor(temporary, len(counts))
    library.codec(request, encoding, chunk, action=0)
    row_blocks = prefix(sizes)
    byte_count = int(row_blocks[-1])
    buffers = dict(counts=counts, row_blocks=row_blocks,
                   codes=torch.empty(byte_count, device=counts.device, dtype=torch.uint8))
    request = descriptor(dict(temporary, **buffers), len(counts))
    library.codec(request, encoding, chunk, action=1)
    return Plan(encoding, chunk, buffers)
