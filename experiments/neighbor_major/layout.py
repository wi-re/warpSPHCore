"""Isolated primal neighbor-major experiment. Never imported by production.

Generated kernels change indexing only; source transforms fail on unexpected
production code. All generated sources/hashes are retained under /tmp.
"""

import ast
from dataclasses import dataclass, replace
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from warpSPHCore.dataTypes import adjacencyData, gridData, domainData
from warpSPHCore.enumTypes import SupportScheme
from warpSPHCore.autograd.stream import currentTorchWarpStream
from warpSPHCore.radiusSearch.compactHash.wp_countNeighbors import radiusSearchCountNeighborsCompactHashMap
from warpSPHCore.radiusSearch.compactHash import buildCompactHashMap


@dataclass
class Neighbors:
    """Experiment-only J and counts; no implicit CSR conversion or I array.

    paddedJ: CSR flat storage or flattened [max_neighbors, n_queries] storage.
    edgeOffsets: CSR prefix; an unused empty int32 tensor for neighbor-major.
    """
    paddedJ: torch.Tensor
    numNeighbors: torch.Tensor
    edgeOffsets: torch.Tensor
    numRows: int
    numCols: int
    numEdges: int
    layout: str
    maximum: int
    overflow: torch.Tensor

    @property
    def storageBytes(self):
        return sum(t.numel() * t.element_size() for t in
                   (self.paddedJ, self.numNeighbors, self.edgeOffsets, self.overflow))


@wp.func
def neighborMajorRange(i: wp.int32, o: wp.int32, lane: wp.int32, lanes: wp.int32,
                       useAdjacency: wp.bool, adjacencyState: adjacencyData,
                       gridState: gridData, queryState: Any, domainState: domainData):
    n = wp.int64(queryState.positions.shape[0])
    count = adjacencyState.numNeighbors[i]
    chunk = (count + lanes - 1) // lanes
    start = lane * chunk
    length = wp.max(wp.int32(0), wp.min(chunk, count - start))
    return wp.int64(i) + wp.int64(start) * n, length


def _load(tree, package, label, original):
    generated = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    sha = hashlib.sha256(generated.encode()).hexdigest()
    folder = Path(tempfile.gettempdir()) / 'warpsphcore-neighbor-major' / sha[:16]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (label + '.py')
    path.write_text(generated)
    name = package + '._neighbor_major_' + label + '_' + sha[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        spec.loader.exec_module(loaded)
    return sys.modules[name], dict(production_sha256=hashlib.sha256(original.encode()).hexdigest(),
                                   generated_sha256=sha, generated_path=str(path))


_COLLECT = {}

def collector(layout):
    if layout in _COLLECT:
        return _COLLECT[layout]
    if layout not in ('csr', 'neighbor-major'):
        raise ValueError(layout)
    module = importlib.import_module('warpSPHCore.radiusSearch.compactHash.wp_collectNeighbors')
    original = Path(module.__file__).read_text()
    tree = ast.parse(original)
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'radiusSearchCollectCompactHashMap')
    kernel.name = 'collectExperimental'
    kernel.args.args.append(ast.arg(arg='overflow', annotation=ast.parse('wp.array(dtype=wp.int32)', mode='eval').body))
    hits = dict(i=0, j=0, begin=0, step=0)

    class ChangeIndex(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == 'edge_i' for t in node.targets):
                hits['i'] += 1
                return None
            if any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == 'edge_j' for t in node.targets):
                hits['j'] += 1
                return ast.If(test=ast.parse('edge_index < edge_j.shape[0]', mode='eval').body, body=[node],
                              orelse=ast.parse('wp.atomic_max(overflow, 0, 1)').body)
            if any(isinstance(t, ast.Name) and t.id == 'edge_offset' for t in node.targets):
                hits['begin'] += 1
                node.value = ast.parse('wp.int64(i)' if layout == 'neighbor-major' else 'wp.int64(edge_offsets[i])', mode='eval').body
            return node

        def visit_AugAssign(self, node):
            if isinstance(node.target, ast.Name) and node.target.id == 'edge_index':
                hits['step'] += 1
                node.value = ast.parse('wp.int64(N)' if layout == 'neighbor-major' else 'wp.int64(1)', mode='eval').body
            return node

    ChangeIndex().visit(kernel)
    if hits != dict(i=1, j=1, begin=1, step=1):
        raise RuntimeError(f'review collect transform: {hits}')
    loaded, info = _load(tree, 'warpSPHCore.radiusSearch.compactHash', 'collect_' + layout.replace('-', '_'), original)
    _COLLECT[layout] = loaded.collectExperimental, info
    return _COLLECT[layout]


_OPERATORS = {}
BACKENDS = dict(density='_computeSPHDensity_stateBackend', interpolate='_computeSPHInterpolant_stateBackend',
                gradient='_computeSPHGradient_stateBackend', divergence='_computeSPHDivergence_stateBackend',
                curl='_computeSPHCurl_stateBackend', laplacian='_computeSPHLaplacian_stateBackend',
                covariance='_computeSPHCovariance_stateBackend')

def operatorModule(name, layout):
    module = importlib.import_module('warpSPHCore.coreOperations.wp_' + name)
    if layout == 'csr':
        return module, None
    if layout != 'neighbor-major': raise ValueError(layout)
    if name in _OPERATORS: return _OPERATORS[name]
    original = Path(module.__file__).read_text()
    tree = ast.parse(original)
    inner = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.endswith('_Func_i'))
    for argument in inner.args.args:
        if argument.arg == 'beginIndex':
            argument.annotation = ast.parse('wp.int64', mode='eval').body
    inner.args.args.append(ast.arg(arg='neighborStride', annotation=ast.parse('wp.int64', mode='eval').body))
    hits = dict(index=0, call=0, range=0)

    class ChangeOperator(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == 'jj' for t in node.targets):
                if ast.unparse(node.value) != 'beginIndex + neighborIndex': raise RuntimeError('unexpected neighbor index')
                hits['index'] += 1
                node.value = ast.parse('wp.int64(beginIndex) + wp.int64(neighborIndex) * neighborStride', mode='eval').body
            return self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == inner.name:
                hits['call'] += 1
                node.args.append(ast.parse('wp.int64(queryState.positions.shape[0])', mode='eval').body)
            if isinstance(node.func, ast.Name) and node.func.id == 'getIndexRangeLane':
                hits['range'] += 1
                node.func.id = 'neighborMajorRange'
            return self.generic_visit(node)

    ChangeOperator().visit(tree)
    if hits != dict(index=1, call=1, range=1): raise RuntimeError(f'review operator transform: {name}, {hits}')
    # Imported by the generated module through this experiment's unique name;
    # no original module, registry or spec is mutated.
    tree.body.insert(0, ast.ImportFrom(module=__name__, names=[ast.alias(name='neighborMajorRange')], level=0))
    loaded, info = _load(tree, 'warpSPHCore.coreOperations', 'op_' + name, original)
    specName = '_' + name.upper() + '_SPEC'
    setattr(loaded, specName, replace(getattr(loaded, specName), jvp=None))
    _OPERATORS[name] = loaded, info
    return _OPERATORS[name]


def commonInputs(grid, queryPositions, querySupports, periodic):
    flags = periodic.cpu().tolist()
    wrapped = torch.stack([(queryPositions[:,d] - grid.qMin[d]).remainder(grid.qMax[d]-grid.qMin[d]) + grid.qMin[d]
                           if flags[d] else queryPositions[:,d] for d in range(queryPositions.shape[1])], dim=1)
    values = [wrapped,querySupports,grid.sortedPositions,grid.sortedSupports,grid.hashTable,grid.sortedCellTable,
              grid.qMin,grid.hCell,grid.D,grid.cellOffsets,grid.numCells,grid.qMax,grid.qMin,periodic,wp.uint32(grid.mode_uint)]
    return [wp.from_torch(x) if isinstance(x,torch.Tensor) else x for x in values], wrapped


def collectInto(grid, common, neighbors):
    kernel, _ = collector(neighbors.layout)
    neighbors.overflow.zero_()
    args = common + [wp.from_torch(neighbors.numNeighbors),wp.from_torch(neighbors.edgeOffsets),wp.from_torch(grid.sortIndex),
                     wp.from_torch(neighbors.paddedJ),wp.from_torch(neighbors.paddedJ),wp.from_torch(neighbors.overflow)]
    wp.launch(kernel,dim=neighbors.numRows,inputs=args,device=neighbors.paddedJ.device.type,
              stream=currentTorchWarpStream(neighbors.paddedJ.device))


def search(grid, queryPositions, querySupports, periodic, layout, *, capacity=None, checked=True):
    """Direct grid -> experimental CSR-J or neighbor-major J (never CSR -> transpose).

    capacity optionally bounds max neighbors for neighbor-major overflow tests.
    Fixed geometry is the benchmark scope. Recount before any topology change.
    """
    common, wrapped = commonInputs(grid,queryPositions,querySupports,periodic)
    n = queryPositions.shape[0]
    counts = torch.empty(n,dtype=torch.int32,device=queryPositions.device)
    wp.launch(radiusSearchCountNeighborsCompactHashMap,n,inputs=common+[wp.from_torch(counts)],
              device=queryPositions.device.type,stream=currentTorchWarpStream(queryPositions.device))
    stats=torch.stack((counts.sum(),counts.max().long() if n else counts.new_zeros((),dtype=torch.int64))).cpu().tolist()
    edges,maximum=int(stats[0]),int(stats[1])
    if layout == 'csr':
        if edges >= 2**31: raise ValueError('production CSR offsets are int32; geometry exceeds experiment limit')
        offsets=torch.zeros(n,dtype=torch.int32,device=counts.device)
        if n: offsets[1:]=counts[:-1].cumsum(0,dtype=torch.int32)
        size=edges
    elif layout == 'neighbor-major':
        allocated=maximum if capacity is None else capacity
        if allocated<0: raise ValueError('capacity must be nonnegative')
        offsets=torch.empty(0,dtype=torch.int32,device=counts.device)
        size=n*allocated
    else: raise ValueError(layout)
    neighbors=Neighbors(torch.full((size,),-1,dtype=torch.int64,device=counts.device),counts,offsets,n,grid.sortIndex.shape[0],
                        edges,layout,maximum,torch.zeros(1,dtype=torch.int32,device=counts.device))
    collectInto(grid,common,neighbors)
    if checked and bool(neighbors.overflow.cpu()[0]): raise OverflowError('neighbor storage capacity exceeded; output discarded')
    return neighbors


def makeOperation(name, topology, particles, properties, domain, values, **corrections):
    """Primal-only matched backend calls. No production dispatch/registry mutation."""
    if not isinstance(topology,Neighbors): raise TypeError('experiment requires its explicit Neighbors type')
    for t in (particles.positions,particles.supports,particles.masses,particles.densities,values):
        if isinstance(t,torch.Tensor) and (t.requires_grad or torch.autograd.forward_ad.unpack_dual(t).tangent is not None):
            raise NotImplementedError('neighbor-major experiment is primal-only; AD is not qualified')
    module,_=operatorModule(name,topology.layout)
    backend=getattr(module,BACKENDS[name])
    if name=='density':
        if corrections: raise ValueError('density does not take corrections')
        return lambda:backend(particles,particles,domain,properties,topology)
    if name=='covariance':
        return lambda:backend(particles,properties,domain,adjacency=topology,referenceParticles=particles,**corrections)
    if name=='interpolate':
        return lambda:backend(particles,particles,domain,properties,topology,values,**corrections)
    return lambda:backend(particles,particles,domain,properties,topology,values,values,**corrections)
