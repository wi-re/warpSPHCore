"""
sphWarpCore — Smoothed Particle Hydrodynamics core library built on NVIDIA Warp and PyTorch.

Public API (flat imports):

    from sphWarpCore.util       import castTorchToWarp, castWarpToTorch, castTorchToWarpAsBuiltins
    from sphWarpCore.autograd   import warpWrapper, WarpFunctionWrapper
    from sphWarpCore.math       import computeDistance, minimumImageDistance, ...
    from sphWarpCore.radius     import radiusSearchCompactHashMap, radiusNaive, AdjacencyList, ...
    from sphWarpCore.operations import sphOperation_warp, warpOperation

Or import subpackages directly:

    from sphWarpCore.utils          import ...
    from sphWarpCore.kernels        import ...
    from sphWarpCore.radiusSearch   import ...
    from sphWarpCore.coreOperations import ...
"""


from . import radius
from .type_config import scalar_t, scalar, dim_t, get_type_config,  get_torch_precision

# Convenience re-exports of the most commonly used symbols

from .autograd import WarpFunctionWrapper
from .radius import (
    # AdjacencyList,
    # DomainDescription,
    # PointCloud,
    radiusSearchCompactHashMap,
    # CompactHashMap,
    
    buildCompactHashMap,
    buildVerletList,
    updateNeighborsVerlet,
    filterVerletList,
)
from .operations import (
    sphOperation_warp,
    warpOperation
)
# from .state import (
#     ParticleState,
#     OperationProperties,
#     CRKState,
#     GradHState,
#     RenormalizationState
# )

from .crk import *
from .renorm import computeRenormalizationMatrices

from .enumTypes import *
from .utils.support import (volumeToSupport, n_h_to_nH)

# from .warp_state import (
#     adjacencyData,
#     gridData,
#     domainData,
#     getParticle,
#     getL_i, getVolume_i, getVolume_j,
#     getGradH_i, getGradH_j,
#     getCRK_i
# )

from .warp_state_util import parseArguments, extractStateInfo, warpWrapper2

from .utils.wp_util import (zero_like_warp,
                            checkDirectionality_i, checkDirectionality_j, getCachedDummyTensor, castTorchToWarpAsBuiltins)

from .kernels import *

from .math import computeDistanceVec, safe_sqrt
from .utils import computePairwiseSupport
from .utils.wp_autograd import launch_kernel, warpWrapper, StateAwareWarpFunction
from .math import *
from .dataTypes import *


from .types import scalar_t, vec_t, mat_t, vecArray_t, matArray_t, intArray_t, scalarArray_t

__version__ = "0.4.5"

__all__ = [
    "radius",
    # "ops",
    "scalar_t",
    "scalar",
    "dim_t",
    "get_type_config",
    # "AdjacencyList",
    # "CompactHashMap",
    # "DomainDescription",
    # "PointCloud",
    "radiusSearchCompactHashMap",
    "sphOperation_warp",
    "WarpFunctionWrapper",
    "buildCompactHashMap",
    "buildVerletList",
    "updateNeighborsVerlet",
    "filterVerletList",
    "computeCRKFactors",
    "computeRenormalizationMatrices",
    # "ParticleState",
    # "OperationProperties",
    # "CRKState",
    # "GradHState",
    # "RenormalizationState",
    # "KernelFunctions",
    # "SupportScheme",
    # "OperationDirection",
    # "GradientScheme",
    # "LaplacianScheme",
    # "WarpOperation",
    "warpOperation",
    "volumeToSupport",
    "n_h_to_nH",
    # "ParticleType",
    # "adjacencyData",
    # "gridData",
    # "domainData",
    "getParticle",
    "getL_i", "getVolume_i", "getVolume_j",
    "getGradH_i", "getGradH_j",
    "getCRK_i",
    "parseArguments",
    'zero_like_warp',
    'checkDirectionality_i', 'checkDirectionality_j', 'getCachedDummyTensor',
    'castTorchToWarpAsBuiltins',
    'computeKernelCRK', 'computeKernelGradientCRK', 'sphKernel', 'sphKernelGradient',
    'computeDistanceVec', 'safe_sqrt', 'computePairwiseSupport', 'iPow',
    'launch_kernel', 'warpWrapper',
    'StateAwareWarpFunction', 'extractStateInfo', 'warpWrapper2',
    'matmul',
    'scalar_t', 'vec_t', 'mat_t', 'vecArray_t', 'matArray_t', 'intArray_t', 'scalarArray_t', 'get_torch_precision',
    'sphKernelScale', 'sphKernel_xi'
]

__all__.extend(kernels.__all__)
__all__.extend(math.__all__)
__all__.extend(crk.__all__)
__all__.extend(enumTypes.__all__)
__all__.extend(dataTypes.__all__)