"""GPU-only planning and borrowed-buffer wrappers for the cluster experiment.

Variable-size allocations synchronize during planning. Interpolation only reads
the completed plan: no data validation or host copies occur in its execution path.
"""
import ctypes as C
from dataclasses import dataclass, replace
from pathlib import Path
import sys

import torch

import importlib.util
_spec = importlib.util.spec_from_file_location("cell_tiles_runtime", Path(__file__).resolve().parents[1] / "cell_tiles/runtime.py")
cell = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cell
_spec.loader.exec_module(cell)


class ClusterCase(C.Structure):
    _fields_ = [
        *((name, C.c_int) for name in (
            "dimension", "precision", "particles", "queries", "sources", "groups",
            "query_size", "source_size", "group_size")),
        ("stream", C.c_void_p),
        *((name, C.c_void_p) for name in (
            "positions", "supports", "masses", "densities", "values",
            "query_begin", "query_end", "query_cell", "source_begin", "source_end",
            "source_cell_offsets", "group_first", "group_end", "group_cell",
            "query_group", "query_bit", "neighbors", "images", "query_bounds", "source_bounds",
            "counts", "offsets", "entries", "masks", "byte_counts", "byte_offsets",
            "compressed", "image_codes", "packed", "output")),
    ]


class Library(cell.Library):
    def __init__(self, path):
        super().__init__(path)
        for name, ints in (("cluster_bounds", 0), ("cluster_decode", 0), ("cluster_build", 2),
                           ("cluster_compress", 1), ("cluster_interpolate", 5)):
            function = getattr(self.library, name)
            function.argtypes = [C.POINTER(ClusterCase), *([C.c_int] * ints)]
            function.restype = C.c_int

    def invoke(self, name, request, *args):
        request.stream = self.stream
        self.check(getattr(self.library, name)(C.byref(request), *args))

    def run(self, request, mode=0, compressed=False, warps=4, prepare=True, iterations=1):
        self.invoke("cluster_interpolate", request, mode, int(compressed), warps,
                    int(prepare), iterations)


def build_cells(library, state, values, max_support, refinement=8):
    """Preserve coarse Morton cells, but sort inside each at a finer resolution.

    A power-of-two refinement means dropping low Morton bits recovers the coarse
    cell key exactly. This is local refinement, not the paper's Hilbert/octree.
    """
    import math
    grid = math.floor(1 / max_support)
    if refinement not in (1, 2, 4, 8, 16) or not 3 <= grid < 2**21:
        raise ValueError("Toy requires a power-of-two refinement and a valid 21-bit coarse grid")
    # Keep the coarse resolution exact when a very fine 1D problem leaves fewer
    # low key bits available. Reduce only the optional within-cell refinement.
    while grid * refinement >= 2**21:
        refinement //= 2
    keys = library.keys(state.positions, grid * refinement)
    order = torch.argsort(keys, stable=True)
    dimension = state.positions.shape[1]
    coarse = keys[order] >> (dimension * (refinement.bit_length() - 1))
    ordered = replace(state, **{name: getattr(state, name)[order] for name in
                               ("positions", "supports", "masses", "densities", "kinds")})
    unique, counts = torch.unique_consecutive(coarse, return_counts=True)
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    neighbors, images = library.neighbors(unique, dimension, grid)
    source_counts = torch.where(neighbors >= 0, counts[neighbors.long().clamp_min(0)], 0)
    prefix = torch.cat([counts.new_zeros((len(counts), 1)), source_counts.cumsum(1)], dim=1).int()
    plan = cell.Plan(ordered, values[order], order, unique, counts, offsets,
                     neighbors, images, prefix, grid)
    plan.refinement = refinement
    return plan


def prefix(counts):
    return torch.cat([counts.new_zeros(1), counts.cumsum(0)])


@dataclass
class Plan:
    cells: object
    query_size: int
    source_size: int
    group_size: int
    exact: bool
    buffers: dict
    descriptor: ClusterCase

    def memory_bytes(self, compressed=False):
        # Exclude state copies, transient bounding boxes/counts, and raw lists
        # when reporting the representation actually read by interpolation.
        names = ("query_begin", "query_end", "query_group", "query_bit", "source_begin", "source_end",
                 "offsets", "masks")
        names += ("byte_offsets", "compressed", "image_codes") if compressed else ("entries",)
        return sum(self.buffers[k].numel() * self.buffers[k].element_size() for k in names)

    def request(self, packed, output):
        request = ClusterCase.from_buffer_copy(self.descriptor)
        request.packed, request.output = packed.data_ptr(), output.data_ptr()
        return request


def build_clusters(library, cells, query_size=8, source_size=4, group_size=8,
                   exact=True, compression=True):
    if query_size not in (4, 8) or source_size not in (4, 8) or group_size not in (1, 8):
        raise ValueError("Toy tile sizes are 4/8; query list groups are 1/8")
    queries, sources = cells.tasks(query_size), cells.tasks(source_size)
    q_counts = (cells.counts + query_size - 1) // query_size
    s_counts = (cells.counts + source_size - 1) // source_size
    g_counts = (q_counts + group_size - 1) // group_size
    q_first, s_first, g_first = prefix(q_counts), prefix(s_counts), prefix(g_counts)
    groups = int(g_first[-1])
    device, dtype = cells.values.device, cells.values.dtype
    group_cell = torch.repeat_interleave(torch.arange(len(g_counts), device=device),
                                         g_counts, output_size=groups)
    local = torch.arange(groups, device=device) - g_first[group_cell]
    first = q_first[group_cell] + local * group_size
    end = torch.minimum(first + group_size, q_first[group_cell + 1])
    q_cell = queries.cells.long()
    q_local = torch.arange(len(queries.cells), device=device) - q_first[q_cell]
    # Ascending occupied cell IDs imply ascending source cluster IDs, as needed
    # by delta compression. Image shifts follow the exact same permutation.
    neighbor_sort = torch.argsort(torch.where(cells.neighbors >= 0, cells.neighbors,
                                              torch.iinfo(torch.int32).max), dim=1)
    neighbors = torch.gather(cells.neighbors, 1, neighbor_sort)
    images = torch.gather(cells.images, 1,
                          neighbor_sort[:, :, None].expand_as(cells.images))
    buffers = dict(query_begin=queries.begin, query_end=queries.end, query_cell=queries.cells,
                   source_begin=sources.begin, source_end=sources.end,
                   source_cell_offsets=s_first, group_first=first.int(), group_end=end.int(),
                   group_cell=group_cell.int(), query_group=(g_first[q_cell]+q_local//group_size).int(),
                   query_bit=(q_local % group_size).int(), neighbors=neighbors, images=images,
                   query_bounds=torch.empty((len(q_cell), 2*cells.state.positions.shape[1]+1), device=device, dtype=dtype),
                   source_bounds=torch.empty((len(sources.cells), 2*cells.state.positions.shape[1]+1), device=device, dtype=dtype),
                   counts=torch.empty(groups, device=device, dtype=torch.int64))
    request = ClusterCase()
    for name, value in dict(dimension=cells.state.positions.shape[1],
                            precision=int(dtype == torch.float64), particles=len(cells.order),
                            queries=len(q_cell), sources=len(sources.cells), groups=groups,
                            query_size=query_size, source_size=source_size, group_size=group_size).items():
        setattr(request, name, value)
    for name in ("positions", "supports", "masses", "densities"):
        setattr(request, name, getattr(cells.state, name).data_ptr())
    request.values = cells.values.data_ptr()

    def bind(**new):
        buffers.update(new)
        for name, tensor in new.items():
            setattr(request, name, tensor.data_ptr())

    bind(**buffers)
    library.invoke("cluster_bounds", request)
    library.invoke("cluster_build", request, 0, int(exact))
    offsets = prefix(buffers["counts"])
    entries = int(offsets[-1])
    bind(offsets=offsets, entries=torch.empty(entries, device=device, dtype=torch.int32),
         masks=torch.empty(entries, device=device, dtype=torch.uint8))
    library.invoke("cluster_build", request, 1, int(exact))
    if compression:
        bind(byte_counts=torch.empty(groups, device=device, dtype=torch.int64))
        library.invoke("cluster_compress", request, 0)
        byte_offsets = prefix(buffers["byte_counts"])
        total_bytes = int(byte_offsets[-1])
        bind(byte_offsets=byte_offsets, compressed=torch.empty(total_bytes, device=device, dtype=torch.uint8),
             image_codes=torch.empty(entries, device=device, dtype=torch.uint8))
        library.invoke("cluster_compress", request, 1)
    return Plan(cells, query_size, source_size, group_size, exact, buffers, request)


def call(library, plan, mode=0, compressed=False, warps=4):
    output = torch.empty_like(plan.cells.values)
    packed = output.new_empty((2*output.shape[1], len(output)))
    library.run(plan.request(packed, output), mode, compressed, warps)
    return output
