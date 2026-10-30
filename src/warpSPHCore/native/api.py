"""Compatibility entry points; operation policy lives with its owning module.

Explicit imports from native.api continue to request the native implementation,
including in reference-backend qualification running in a Warp process.
"""
from functools import partial
from ..coreOperations.native import sphOperation as _operation
from ..radiusSearch.native import radiusSearch as _radius

warpOperation = partial(_operation, backend='native')
sphOperation = warpOperation
radiusSearch = partial(_radius, backend='native')

def nativeCapabilities():
    """Declared pilot scope; no runtime initialisation or compilation."""
    return dict(operations=('density', 'scalar_difference_gradient'),
                kernels=('Wendland2',), dimensions=(1, 2, 3),
                precision=('float32', 'float64'), execution=('cpu', 'cuda', 'hip'),
                experimental_execution=('hip',),
                autograd=False, radius_search='all_pairs', graph_capture=False,
                warp_required=False)
