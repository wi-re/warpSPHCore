"""Primal CSR index-width experiment; production sources are never patched."""
import ast
import importlib
from pathlib import Path
import sys
from typing import Any
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments/neighbor_major'))
import layout
from warpSPHCore.dataTypes import domainData, gridData
from warpSPHCore.radiusSearch.grid_util import laneSlice


@wp.struct
class Adjacency32:
    neighborList: wp.array(dtype=wp.int32)
    neighborOffsets: wp.array(dtype=wp.int32)
    numNeighbors: wp.array(dtype=wp.int32)


@wp.func
def csrRange(i: wp.int32, o: wp.int32, lane: wp.int32, lanes: wp.int32,
             useAdjacency: wp.bool, adjacencyState: Any, gridState: gridData,
             queryState: Any, domainState: domainData):
    return laneSlice(adjacencyState.neighborOffsets[i], adjacencyState.numNeighbors[i], lane, lanes, True)


def operator(name):
    original = Path(importlib.import_module('warpSPHCore.coreOperations.wp_' + name).__file__).read_text()
    tree = ast.parse(original)
    hits = dict(adjacency=0, array=0, select=0, range=0)
    class Transform(ast.NodeTransformer):
        def visit_arg(self, node):
            if node.arg == 'adjacencyState':
                node.annotation = ast.Name(id='Any', ctx=ast.Load()); hits['adjacency'] += 1
            if node.arg == 'offsetArray':
                node.annotation = ast.parse('wp.array(dtype=Any)', mode='eval').body; hits['array'] += 1
            return node
        def visit_IfExp(self, node):
            if ast.unparse(node) == 'adjacencyState.neighborList if useAdjacency else gridState.sortIndex':
                hits['select'] += 1
                return node.body
            return self.generic_visit(node)
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == 'getIndexRangeLane':
                node.func.id = 'csrRange'; hits['range'] += 1
            return self.generic_visit(node)
    Transform().visit(tree)
    assert hits == dict(adjacency=3, array=1, select=1, range=1), hits
    tree.body.insert(0, ast.ImportFrom(module=__name__, names=[ast.alias(name='csrRange')], level=0))
    return layout._load(tree, 'warpSPHCore.coreOperations', 'index_width_' + name, original)


def collector(bits):
    original = Path(importlib.import_module('warpSPHCore.radiusSearch.compactHash.wp_collectNeighbors').__file__).read_text()
    tree = ast.parse(original)
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'radiusSearchCollectCompactHashMap')
    hits = dict(arg=0, store=0)
    class Transform(ast.NodeTransformer):
        def visit_arg(self, node):
            if node.arg in ('edge_i', 'edge_j'):
                node.annotation = ast.parse(f'wp.array(dtype=wp.int{bits})', mode='eval').body; hits['arg'] += 1
            return node
        def visit_Assign(self, node):
            if any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id in ('edge_i','edge_j') for t in node.targets):
                node.value = ast.Call(func=ast.parse(f'wp.int{bits}', mode='eval').body, args=[node.value], keywords=[])
                hits['store'] += 1
            return node
    Transform().visit(kernel)
    assert hits == dict(arg=2, store=2), hits
    module, info = layout._load(tree, 'warpSPHCore.radiusSearch.compactHash', f'index_width_collect{bits}', original)
    return module.radiusSearchCollectCompactHashMap, info
