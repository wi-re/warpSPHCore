"""Benchmark split I/J collection without changing production dispatch defaults.

Prefer the production J-only specialization. For historical sources without it,
generate a prototype by removing exactly one edge_i store and retain its hashes.
The generic row-fill helper permits independent lane-count/width experiments.
"""

import ast
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from typing import Any

import warp as wp


@wp.kernel
def expandRows(counts: wp.array(dtype=wp.int32), offsets: wp.array(dtype=Any),
               rows: wp.array(dtype=Any), lanes: int):
    row, lane = wp.tid()
    p = wp.int64(lane)
    while p < wp.int64(counts[row]):
        rows[offsets[row] + offsets.dtype(p)] = rows.dtype(row)
        p += wp.int64(lanes)


def loadJOnly():
    module = importlib.import_module('warpSPHCore.radiusSearch.compactHash.wp_collectNeighbors')
    source = Path(module.__file__).read_text()
    if hasattr(module, 'radiusSearchCollectJOnly'):
        return module.radiusSearchCollectJOnly, dict(
            production_sha256=hashlib.sha256(source.encode()).hexdigest(),
            transform='production compile-time writeI=False specialization; no source transform')
    tree = ast.parse(source)
    removed = 0
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'radiusSearchCollectCompactHashMap':
            node.name = 'radiusSearchCollectJOnly'
            class RemoveRowStore(ast.NodeTransformer):
                def visit_Assign(self, statement):
                    nonlocal removed
                    if any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == 'edge_i'
                           for t in statement.targets):
                        removed += 1
                        return None
                    return statement
            RemoveRowStore().visit(node)
    if removed != 1:
        raise RuntimeError(f'expected one production I store, found {removed}; review prototype transform')
    generated = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    digest = hashlib.sha256(generated.encode()).hexdigest()
    folder = Path(tempfile.gettempdir()) / 'warpsphcore-split-collect' / digest[:16]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'collect_jonly.py'
    path.write_text(generated)
    name = 'warpSPHCore.radiusSearch.compactHash._bench_jonly_' + digest[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        spec.loader.exec_module(loaded)
    return sys.modules[name].radiusSearchCollectJOnly, dict(
        production_sha256=hashlib.sha256(source.encode()).hexdigest(), generated_sha256=digest,
        generated_path=str(path), transform='remove exactly one edge_i subscript assignment; preserve ABI and all other kernel statements')


def launchRows(counts, offsets, rows, n, lanes, stream=None):
    wp.launch(expandRows, dim=(n, lanes), inputs=[counts, offsets, rows, lanes],
              block_dim=128, device=counts.device, stream=stream)


@contextmanager
def splitSearch(jOnly, lanes):
    """Redirect only this module's collect launch during benchmark public calls."""
    module = importlib.import_module('warpSPHCore.radiusSearch.compactHash.search')
    original = module.wp
    kernel = module.radiusSearchCollectCompactHashMap

    class SplitLaunch:
        def __getattr__(self, name):
            return getattr(original, name)

        def launch(self, target, *args, **kwargs):
            if target is not kernel:
                return original.launch(target, *args, **kwargs)
            original.launch(jOnly, *args, **kwargs)
            inputs = kwargs['inputs']
            launchRows(inputs[15], inputs[16], inputs[18], kwargs['dim'], lanes, kwargs.get('stream'))

    module.wp = SplitLaunch()
    try:
        yield
    finally:
        module.wp = original
