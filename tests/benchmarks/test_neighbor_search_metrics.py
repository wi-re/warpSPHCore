"""Candidate visits count rejected particles as well as accepted neighbors."""

import itertools
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize('shape,periodic', [((1,), (True,)), ((2, 2), (True, True)),
    ((4, 3), (True, False)), ((3, 2, 4), (True, False, True))])
def test_candidate_visits_match_independent_cell_sets(monkeypatch, shape, periodic):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'scripts/benchmarks'))
    from bench_neighbor_search import candidateVisits
    # Sparse, nonuniform cell populations expose both empty cells and multiplicity.
    cells = [(0,) * len(shape)] + [cell for index, cell in enumerate(itertools.product(*[range(c) for c in shape]))
             for _ in range(index % 3)]
    positions = torch.tensor(cells, dtype=torch.float32) + .25
    ids = [sum(cell[d] * math.prod(shape[:d]) for d in range(len(shape))) for cell in cells]
    occupied = sorted(set(ids))
    table = torch.tensor([[cell, 0, ids.count(cell)] for cell in occupied], dtype=torch.int64)
    offsets = list(itertools.product((-1, 0, 1), repeat=len(shape)))
    grid = SimpleNamespace(numCells=torch.tensor(shape), qMin=torch.zeros(len(shape)), hCell=1.,
                           sortedCellTable=table, cellOffsets=torch.tensor(offsets))
    got = candidateVisits(positions, grid, torch.tensor(periodic))
    expected = []
    for cell in cells:
        visited = set()
        for offset in offsets:
            target = tuple((cell[d] + offset[d]) % shape[d] if periodic[d] else cell[d] + offset[d]
                           for d in range(len(shape)))
            if all(0 <= target[d] < shape[d] for d in range(len(shape))):
                visited.add(target)
        expected.append(sum(other in visited for other in cells))
    torch.testing.assert_close(got, torch.tensor(expected), rtol=0, atol=0)
