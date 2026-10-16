"""Count-validated capacity without an atomic overflow branch in each edge store.

Fixed-geometry contract: use freshly counted metadata before collection/replay.
Only experiment globals are temporarily redirected; production is untouched.
"""

import ast
from contextlib import contextmanager
from pathlib import Path

import layout as experiment

ORIGINAL_COLLECTOR=experiment.collector
ORIGINAL_COLLECT_INTO=experiment.collectInto
KERNELS={}


def collector(storage):
    if storage in KERNELS:return KERNELS[storage]
    _,parent=ORIGINAL_COLLECTOR(storage)
    original=Path(parent['generated_path']).read_text()
    tree=ast.parse(original)
    hits=0
    class RemoveEdgeGuard(ast.NodeTransformer):
        def visit_If(self,node):
            nonlocal hits
            if ast.unparse(node.test)=='edge_index < edge_j.shape[0]':
                hits+=1
                return node.body
            return self.generic_visit(node)
    RemoveEdgeGuard().visit(tree)
    if hits!=1:raise RuntimeError('review count-validated collector transform')
    module,info=experiment._load(tree,'warpSPHCore.radiusSearch.compactHash','count_validated_'+storage.replace('-','_'),original)
    info['checked_parent_generated_sha256']=parent['generated_sha256']
    info['production_sha256']=parent['production_sha256']
    info['capacity_contract']='host total/max from the same count pass checked before launch; fixed geometry/counts for bare replay'
    KERNELS[storage]=module.collectExperimental,info
    return KERNELS[storage]


def collectInto(grid,common,neighbors):
    size=neighbors.paddedJ.numel()
    if neighbors.layout=='csr':
        if neighbors.numEdges>size:raise OverflowError('CSR capacity insufficient before launch')
    elif neighbors.layout=='neighbor-major':
        if neighbors.numRows and (size%neighbors.numRows or neighbors.maximum>size//neighbors.numRows):
            raise OverflowError('neighbor-major capacity insufficient before launch')
    else:raise ValueError(neighbors.layout)
    ORIGINAL_COLLECT_INTO(grid,common,neighbors)


@contextmanager
def collectors():
    previous=(experiment.collector,experiment.collectInto)
    experiment.collector,experiment.collectInto=collector,collectInto
    try:yield
    finally:experiment.collector,experiment.collectInto=previous
