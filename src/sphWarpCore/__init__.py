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
from .type_config import scalar_t, scalar, dim_t, get_type_config,  get_torch_precision

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

from .warp_state import (
    adjacencyData,
    gridData,
    domainData,
    getParticle,
    getL_i, getVolume_i, getVolume_j,
    getGradH_i, getGradH_j,
    getCRK_i
)

from .warp_state_util import parseArguments, extractStateInfo, warpWrapper2

from .utils.wp_util import (zero_like_warp,
                            checkDirectionality_i, checkDirectionality_j, getCachedDummyTensor, castTorchToWarpAsBuiltins)

from .kernels.wp_kernel import eval_kernelScale, computeKernelCRK, computeKernelGradientCRK, sphKernel, sphKernelGradient, eval_k, eval_C_d
from .mathutil.wp_math import computeDistanceVec, safe_sqrt
from .kernels.utils import computePairwiseSupport, iPow
from .utils.wp_autograd import launch_kernel, warpWrapper, StateAwareWarpFunction
from .mathutil.wp_math import matmul

from .types import scalar_t, vec_t, mat_t, vecArray_t, matArray_t, intArray_t, scalarArray_t

__version__ = "0.4.1"

__all__ = [
    "radius",
    "ops",
    "scalar_t",
    "scalar",
    "dim_t",
    "get_type_config",
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
    "warpOperation",
    "volumeToSupport",
    "ParticleType",
    "adjacencyData",
    "gridData",
    "domainData",
    "getParticle",
    "getL_i", "getVolume_i", "getVolume_j",
    "getGradH_i", "getGradH_j",
    "getCRK_i",
    "parseArguments",
    'zero_like_warp',
    'checkDirectionality_i', 'checkDirectionality_j', 'getCachedDummyTensor',
    'castTorchToWarpAsBuiltins',
    'eval_kernelScale', 'computeKernelCRK', 'computeKernelGradientCRK', 'sphKernel', 'sphKernelGradient', 'eval_k', 'eval_C_d',
    'computeDistanceVec', 'safe_sqrt', 'computePairwiseSupport', 'iPow',
    'launch_kernel', 'warpWrapper',
    'StateAwareWarpFunction', 'extractStateInfo', 'warpWrapper2',
    'matmul',
    'scalar_t', 'vec_t', 'mat_t', 'vecArray_t', 'matArray_t', 'intArray_t', 'scalarArray_t', 'get_torch_precision'

]
