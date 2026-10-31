"""Compatibility entry points; operation policy lives with its owning module.

Explicit imports from native.api continue to request the native implementation,
including in reference-backend qualification running in a Warp process.
"""
from functools import partial
from ..enumTypes import KernelFunctions, WarpOperation
from ._autograd import REVERSE_KERNELS
from ..coreOperations.native import sphOperation as _operation
from ..radiusSearch.native import radiusSearch as _radius

warpOperation = partial(_operation, backend='native')
sphOperation = warpOperation
radiusSearch = partial(_radius, backend='native')

def nativeCapabilities():
    """Declared native scope; no runtime initialisation or compilation."""
    return dict(operations=tuple(op.name.lower() for op in WarpOperation if op != WarpOperation.Custom),
                kernels=tuple(k.name for k in KernelFunctions), dimensions=(1, 2, 3),
                precision=('float32', 'float64'), execution=('cpu', 'cuda', 'hip'),
                experimental_execution=('hip',),
                autograd=True, autograd_order=1, forward_ad=False, higher_order=False,
                reverse_kernels=tuple(k.name for k in KernelFunctions if k.value in REVERSE_KERNELS),
                field_ranks=(0, 1, 2), deterministic_adjoint=False, radius_search='all_pairs', graph_capture=False,
                graph_capture_fixed_adjacency=('interpolate_primal',),
                warp_required=False)
