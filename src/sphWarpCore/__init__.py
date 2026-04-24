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


from . import radius
from . import ops

# Convenience re-exports of the most commonly used symbols

from .autograd import WarpFunctionWrapper
from .radius import (
    AdjacencyList,
    DomainDescription,
    PointCloud,
    radiusSearchCompactHashMap,
    CompactHashMap,
    
    buildCompactHashMap,
    buildVerletList,
    updateNeighborsVerlet,
    filterVerletList,
)
from .ops import (
    sphOperation_warp,
    warpOperation
)
from .state import (
    ParticleState,
    OperationProperties,
    CRKState,
    GradHState,
    RenormalizationState
)

from .crk.crk_wrapper import computeCRKFactors
from .renorm.wp_covariance import computeRenormalizationMatrices

from .enumTypes import (
    KernelFunctions,
    SupportScheme,
    OperationDirection,
    WarpOperation,
    GradientScheme,
    LaplacianScheme,
    ParticleType
)
from .math import (volumeToSupport)

__version__ = "0.2.0"

__all__ = [
    "radius",
    "ops",
    "AdjacencyList",
    "CompactHashMap",
    "DomainDescription",
    "PointCloud",
    "radiusSearchCompactHashMap",
    "sphOperation_warp",
    "WarpFunctionWrapper",
    "buildCompactHashMap",
    "buildVerletList",
    "updateNeighborsVerlet",
    "filterVerletList",
    "computeCRKFactors",
    "computeRenormalizationMatrices",    
    "ParticleState",
    "OperationProperties",
    "CRKState",
    "GradHState",
    "RenormalizationState",
    "KernelFunctions",
    "SupportScheme",
    "OperationDirection",
    "GradientScheme",
    "LaplacianScheme",
    "WarpOperation",
    "volumeToSupport",
    "ParticleType"
]
