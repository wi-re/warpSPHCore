"""Borrowed-buffer CUDA wrappers for the isolated lossless CSR experiment.

Compression is an explicit planning step. Its size readbacks and transient CSR
inputs never enter interpolation. No validation/debug switch is added to an op.
"""
import ctypes as C
from dataclasses import dataclass
import torch

RAW64, RAW32, DELTA, CELL = range(4)
NAMES = {RAW64: 'csr64', RAW32: 'csr32', DELTA: 'delta', CELL: 'cell'}


class Case(C.Structure):
    _fields_ = [(name, C.c_int) for name in ('dimension', 'precision', 'queries', 'references')]
    _fields_ += [(name, C.c_void_p) for name in (
        'stream', 'positions', 'supports', 'masses', 'densities', 'values', 'lower', 'upper',
        'periodic', 'row_offsets', 'indices', 'counts', 'row_blocks', 'particle_bases',
        'codes', 'bases', 'masks', 'meta_counts', 'meta_offsets', 'extras', 'packed', 'output', 'decoded')]


class Library:
    def __init__(self, path):
        self.library = C.CDLL(str(path))
        self.library.compression_cell_bases.argtypes = [C.c_int, C.c_void_p, C.c_void_p, C.c_void_p]
        self.library.compression_codec.argtypes = [C.POINTER(Case), C.c_int, C.c_int, C.c_int]
        self.library.compression_interpolate.argtypes = [C.POINTER(Case)] + [C.c_int]*6
        self.library.compression_interpolate_vec4.argtypes = [C.POINTER(Case)] + [C.c_int]*6
        for name in ('compression_cell_bases', 'compression_codec', 'compression_interpolate', 'compression_interpolate_vec4'):
            getattr(self.library, name).restype = C.c_int

    @staticmethod
    def check(result):
        if result:
            raise RuntimeError(f'CUDA runtime returned error {result}')

    def codec(self, request, encoding, width, action):
        request.stream = torch.cuda.current_stream().cuda_stream
        self.check(self.library.compression_codec(C.byref(request), encoding, width, action))

    def run(self, request, encoding, width, batch=1, threads=128, pack=True, iterations=1, vector=False):
        request.stream = torch.cuda.current_stream().cuda_stream
        function = self.library.compression_interpolate_vec4 if vector else self.library.compression_interpolate
        self.check(function(C.byref(request), encoding, width, batch, threads, pack, iterations))


def descriptor(tensors, queries, references=None, dimension=1, precision=0):
    request = Case(dimension, precision, queries, queries if references is None else references)
    request.stream = torch.cuda.current_stream().cuda_stream
    for name, tensor in tensors.items():
        setattr(request, name, tensor.data_ptr())
    return request


def prefix(counts):
    return torch.cat((torch.zeros(1, device=counts.device, dtype=torch.int64),
                      counts.cumsum(0, dtype=torch.int64)))


@dataclass
class Plan:
    encoding: int
    width: int
    buffers: dict
    blocks: int
    payloads: int

    def storage(self):
        return {name: tensor.numel()*tensor.element_size() for name, tensor in self.buffers.items()}

    def request(self, tensors, queries, dimension, precision):
        # CSR indices/offsets and the particle base map are deliberately absent
        # from the retained format. Only verification supplies CSR offsets again.
        return descriptor(dict(tensors, **self.buffers), queries,
                          dimension=dimension, precision=precision)


def build_plan(library, indices, counts, offsets, width, encoding, cell_offsets=None,
               reference_count=None):
    """Two-pass GPU encoder; prefix sizing/synchronization is planning overhead."""
    row_blocks = prefix((counts.long()+width-1)//width)
    blocks = int(row_blocks[-1])
    temporary = dict(indices=indices, counts=counts, row_offsets=offsets, row_blocks=row_blocks,
                     meta_counts=torch.empty(blocks, device=counts.device, dtype=torch.int32))
    if encoding == CELL:
        particle_bases = torch.empty(reference_count, device=counts.device, dtype=torch.int32)
        library.check(library.library.compression_cell_bases(
            len(cell_offsets)-1, cell_offsets.data_ptr(), particle_bases.data_ptr(),
            torch.cuda.current_stream().cuda_stream))
        temporary['particle_bases'] = particle_bases
    request = descriptor(temporary, len(counts))
    library.codec(request, encoding, width, 0)
    meta_offsets = prefix(temporary['meta_counts'])
    payloads = int(meta_offsets[-1])
    # These private formats use uint32 directories. The benchmark's largest
    # streams are well below 4 GiB; wider directories are a future extension.
    if payloads >= 2**32:
        raise OverflowError('Experiment requires a wider payload directory')
    buffers = dict(counts=counts, row_blocks=row_blocks,
                   codes=torch.empty(blocks*width, device=counts.device, dtype=torch.uint8),
                   meta_offsets=meta_offsets.to(torch.int32),
                   extras=torch.empty(payloads, device=counts.device, dtype=torch.int32))
    buffers['bases' if encoding == DELTA else 'masks'] = torch.empty(
        blocks, device=counts.device, dtype=torch.int32)
    request = descriptor(dict(temporary, **buffers), len(counts))
    library.codec(request, encoding, width, 1)
    return Plan(encoding, width, buffers, blocks, payloads)
