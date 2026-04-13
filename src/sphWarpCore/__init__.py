"""
sphWarpCore — Smoothed Particle Hydrodynamics core library built on NVIDIA Warp and PyTorch.

Public API (flat imports):

    from sphWarpCore.util   import castTorchToWarp, castWarpToTorch, castTorchToWarpAsBuiltins
    from sphWarpCore.autograd import warpWrapper, WarpFunctionWrapper
    from sphWarpCore.math   import computeDistance, minimumImageDistance, ...
    from sphWarpCore.radius import radiusSearchCompactHashMap, radiusNaive, AdjacencyList, ...
    from sphWarpCore.ops    import computeDensity_warpBackend, computeSPHInterpolant_warpBackend, ...

Or import subpackages directly:

    from sphWarpCore.utils       import ...
    from sphWarpCore.mathutil    import ...
    from sphWarpCore.kernels     import ...
    from sphWarpCore.radiusSearch import ...
    from sphWarpCore.operations  import ...
"""

from . import util
from . import autograd
from . import math
from . import radius
from . import ops

# Convenience re-exports of the most commonly used symbols
from .util import (
    castTorchToWarp,
    castWarpToTorch,
    castTorchToWarpAsBuiltins,
    clearDummyTensorCache,
    getCachedDummyTensor,
    getCachedIdentityMatrices,
)
from .autograd import warpWrapper, WarpFunctionWrapper
from .radius import (
    AdjacencyList,
    DomainDescription,
    PointCloud,
    radiusSearchCompactHashMap,
    radiusNaive,
    convertModeToUint,
)
from .ops import (
    computeDensity_warpBackend,
    computeSPHInterpolant_warpBackend,
)

__version__ = "0.1.0"

__all__ = [
    # sub-modules
    "util",
    "autograd",
    "math",
    "radius",
    "ops",
    # util
    "castTorchToWarp",
    "castWarpToTorch",
    "castTorchToWarpAsBuiltins",
    "clearDummyTensorCache",
    "getCachedDummyTensor",
    "getCachedIdentityMatrices",
    # autograd
    "warpWrapper",
    "WarpFunctionWrapper",
    # radius
    "AdjacencyList",
    "DomainDescription",
    "PointCloud",
    "radiusSearchCompactHashMap",
    "radiusNaive",
    "convertModeToUint",
    # ops
    "computeDensity_warpBackend",
    "computeSPHInterpolant_warpBackend",
]
