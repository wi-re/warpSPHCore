"""Experimental Morton plan and borrowed-buffer CUDA wrapper; no public API."""
from dataclasses import dataclass, replace
import ctypes as C
import math

import torch


class TileCase(C.Structure):
    _fields_ = [
        *((name, C.c_int) for name in ("dimension", "precision", "particles", "cells", "tasks", "grid")),
        ("stream", C.c_void_p),
        *((name, C.c_void_p) for name in (
            "positions", "supports", "masses", "densities", "values", "cell_offsets",
            "neighbors", "images", "prefix", "task_cells", "task_begin", "task_end", "packed", "output")),
    ]


class Library:
    def __init__(self, path):
        self.library = C.CDLL(str(path.resolve()))
        self.library.tile_keys.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int,
                                          C.c_int, C.c_void_p, C.c_void_p]
        self.library.tile_neighbors.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int,
                                               C.c_void_p, C.c_void_p, C.c_void_p]
        self.library.tile_interpolate.argtypes = [C.POINTER(TileCase), *([C.c_int] * 5)]
        for name in ("tile_keys", "tile_neighbors", "tile_interpolate"):
            getattr(self.library, name).restype = C.c_int

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f"CUDA runtime error {status}")

    @property
    def stream(self):
        return torch.cuda.current_stream().cuda_stream

    def keys(self, positions, grid):
        n, dimension = positions.shape
        result = torch.empty(n, dtype=torch.int64, device=positions.device)
        self.check(self.library.tile_keys(positions.data_ptr(), n, dimension,
            int(positions.dtype == torch.float64), grid, result.data_ptr(), self.stream))
        return result

    def neighbors(self, keys, dimension, grid):
        result = torch.empty((len(keys), 3 ** dimension), dtype=torch.int32, device=keys.device)
        images = torch.empty((*result.shape, dimension), dtype=torch.int32, device=keys.device)
        self.check(self.library.tile_neighbors(keys.data_ptr(), len(keys), dimension,
            grid, result.data_ptr(), images.data_ptr(), self.stream))
        return result, images

    def interpolate(self, request, mode, block, width, prepare=True, iterations=1):
        self.check(self.library.tile_interpolate(C.byref(request), mode, block, width,
                                                 int(prepare), iterations))


@dataclass
class Tasks:
    cells: torch.Tensor
    begin: torch.Tensor
    end: torch.Tensor


@dataclass
class Plan:
    state: object
    values: torch.Tensor
    order: torch.Tensor
    keys: torch.Tensor
    counts: torch.Tensor
    offsets: torch.Tensor
    neighbors: torch.Tensor
    images: torch.Tensor
    prefix: torch.Tensor
    grid: int

    def tasks(self, capacity):
        """Split overfull cells; no occupancy bound, padding or dropped particles."""
        tiles = (self.counts + capacity - 1) // capacity
        total = int(tiles.sum())  # Variable-sized planning allocation, outside interpolation.
        cells = torch.repeat_interleave(torch.arange(len(tiles), device=tiles.device),
                                        tiles, output_size=total)
        first_task = tiles.cumsum(0) - tiles
        local_tile = torch.arange(total, device=tiles.device) - first_task[cells]
        begin = self.offsets[cells] + local_tile * capacity
        end = torch.minimum(begin + capacity, self.offsets[cells + 1])
        return Tasks(cells.int(), begin.int(), end.int())

    def candidate_pairs(self):
        neighbors = self.neighbors.long()
        source_counts = torch.where(neighbors >= 0, self.counts[neighbors.clamp_min(0)], 0)
        return int((self.counts * source_counts.sum(1)).sum())

    def storage_bytes(self):
        return sum(t.numel() * t.element_size() for t in
                   (self.order, self.keys, self.counts, self.offsets, self.neighbors, self.images, self.prefix))


def sort_state(library, state, values, max_support):
    """Unit periodic box, finite in-box positions, Gather support <= cell width.

    Sorting, indexing, neighbor-cell lookup and state permutation stay on the GPU.
    Torch's variable-size unique/repeat operations can synchronize the host.
    """
    grid = int(math.floor(1 / max_support))
    if not 3 <= grid < 2 ** 21:
        raise ValueError("Toy requires 3 <= grid size < 2**21 in every dimension")
    keys = library.keys(state.positions, grid)
    order = torch.argsort(keys, stable=True)
    sorted_keys = keys[order]
    ordered = replace(state, **{
        name: getattr(state, name)[order] for name in
        ("positions", "supports", "masses", "kinds", "densities")})
    return ordered, values[order], order, sorted_keys, grid


def build_plan(library, state, values, max_support):
    ordered, sorted_values, order, sorted_keys, grid = sort_state(library, state, values, max_support)
    unique, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    neighbors, images = library.neighbors(unique, state.positions.shape[1], grid)
    source_counts = torch.where(neighbors >= 0, counts[neighbors.long().clamp_min(0)], 0)
    prefix = torch.cat([source_counts.new_zeros((len(counts), 1)), source_counts.cumsum(1)], dim=1).int()
    return Plan(ordered, sorted_values, order, unique, counts, offsets,
                neighbors, images, prefix, grid)


def request(library, plan, tasks, packed, output):
    a = TileCase()
    a.dimension = plan.state.positions.shape[1]
    a.precision = int(plan.state.positions.dtype == torch.float64)
    a.particles = len(plan.order)
    a.cells = len(plan.counts)
    a.tasks = len(tasks.cells)
    a.grid = plan.grid
    a.stream = library.stream
    for name in ("positions", "supports", "masses", "densities"):
        setattr(a, name, getattr(plan.state, name).data_ptr())
    buffers = dict(values=plan.values, cell_offsets=plan.offsets, neighbors=plan.neighbors, images=plan.images, prefix=plan.prefix,
                   task_cells=tasks.cells, task_begin=tasks.begin, task_end=tasks.end,
                   packed=packed, output=output)
    for name, tensor in buffers.items():
        setattr(a, name, tensor.data_ptr())
    return a


def call(library, plan, tasks, mode, block, width):
    """Allocation-inclusive call with a reusable geometric plan, like fixed CSR."""
    packed = torch.empty((2 * plan.state.positions.shape[1], len(plan.order)),
                         dtype=plan.values.dtype, device=plan.values.device)
    output = torch.empty_like(plan.values)
    library.interpolate(request(library, plan, tasks, packed, output), mode, block, width)
    return output
