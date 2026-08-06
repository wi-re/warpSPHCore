"""
sphWarpCore — Smoothed Particle Hydrodynamics core library built on NVIDIA Warp and PyTorch.

Public API (flat imports):

    from sphWarpCore.util       import castTorchToWarp, castWarpToTorch, castTorchToWarpAsBuiltins
    from sphWarpCore.autograd   import warpWrapper, WarpFunctionWrapper
    from sphWarpCore.math       import computeDistance, minimumImageDistance, ...
    from sphWarpCore.radius     import radiusSearchCompactHashMap, radiusNaive, ...
    from sphWarpCore.dataTypes  import AdjacencyList, ParticleState, DomainDescription, ...
    from sphWarpCore.operations import sphOperation_warp, warpOperation

Or import subpackages directly:

    from sphWarpCore.utils          import ...
    from sphWarpCore.kernels        import ...
    from sphWarpCore.radiusSearch   import ...
    from sphWarpCore.coreOperations import ...
"""


# from . import radius
from .type_config import scalar_t, scalar, dim_t, get_type_config,  get_torch_precision

# Convenience re-exports of the most commonly used symbols

# from .autograd import WarpFunctionWrapper
from .radiusSearch import *

from .operations import (
    sphOperation_warp,
    warpOperation
)

from .crk import *
from .renorm import computeRenormalizationMatrices

from .enumTypes import *

# from .warp_state_util import parseArguments, extractStateInfo, warpWrapper2

from .kernels import *

from .math import computeDistanceVec, safe_sqrt
from .util import *
from .math import *
from .dataTypes import *
from .pinv import pinv_warp


from .types import scalar_t, vec_t, mat_t, vecArray_t, matArray_t, intArray_t, scalarArray_t

__version__ = "0.4.5"

__all__ = [
    # "radius",
    "scalar_t",
    "scalar",
    "dim_t",
    "get_type_config",
    # "radiusSearchCompactHashMap",
    # "sphOperation_warp",
    # "WarpFunctionWrapper",
    # "buildCompactHashMap",
    # "buildVerletList",
    # "updateNeighborsVerlet",
    # "filterVerletList",
    "computeCRKFactors",
    "computeRenormalizationMatrices",
    "warpOperation",
    # "volumeToSupport",
    # "n_h_to_nH",
    # "getParticle",
    # "getL_i", "getVolume_i", "getVolume_j",
    # "getGradH_i", "getGradH_j",
    # "getCRK_i",
    # "parseArguments",
    # 'zero_like_warp',
    # 'checkDirectionality_i', 'checkDirectionality_j', 
    # 'getCachedDummyTensor',
    # 'castTorchToWarpAsBuiltins',
    'computeKernelCRK', 'computeKernelGradientCRK', 'sphKernel', 'sphKernelGradient',
    # 'computeDistanceVec', 'safe_sqrt', 'computePairwiseSupport', 'iPow',
    # 'launch_kernel', 'warpWrapper',
    # 'StateAwareWarpFunction', 'extractStateInfo', 'warpWrapper2',
    # 'matmul',
    'scalar_t', 'vec_t', 'mat_t', 'vecArray_t', 'matArray_t', 'intArray_t', 'scalarArray_t', 'get_torch_precision',
    # 'sphKernelScale', 'sphKernel_xi',
    # 'checkOffset',
    'pinv_warp'
]

__all__.extend(kernels.__all__)
__all__.extend(math.__all__)
__all__.extend(crk.__all__)
__all__.extend(enumTypes.__all__)
__all__.extend(dataTypes.__all__)
__all__.extend(radiusSearch.__all__)
__all__.extend(util.__all__)
from .autograd import *
__all__.extend(autograd.__all__)

