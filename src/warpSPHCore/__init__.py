"""
warpSPHCore — Smoothed Particle Hydrodynamics core library built on NVIDIA Warp and PyTorch.

Public API (flat imports):

    from warpSPHCore.util       import castTorchToWarp, castWarpToTorch, castTorchToWarpAsBuiltins
    from warpSPHCore.autograd   import warpWrapper, WarpFunctionWrapper
    from warpSPHCore.math       import computeDistance, minimumImageDistance, ...
    from warpSPHCore.radius     import radiusSearchCompactHashMap, radiusNaive, ...
    from warpSPHCore.dataTypes  import AdjacencyList, ParticleState, DomainDescription, ...
    from warpSPHCore.operations import sphOperation_warp, warpOperation

Or import subpackages directly:

    from warpSPHCore.utils          import ...
    from warpSPHCore.kernels        import ...
    from warpSPHCore.radiusSearch   import ...
    from warpSPHCore.coreOperations import ...
"""

# Convenience re-exports of the most commonly used symbols

# Imported first, before anything else: a zero-dependency leaf module many
# submodules need (see profiling.py's docstring for the circular-import
# hazard this avoids -- importing it anywhere else first can reenter a
# still-initializing package).
from . import profiling as _profiling  # noqa: F401

submodules = []
# Type related submodules
from .type_config import *
submodules.append(type_config)
from .dataTypes import *
submodules.append(dataTypes)
from .enumTypes import *
submodules.append(enumTypes)

# General utility submodules
from .util import *
submodules.append(util)
from .math import *
submodules.append(math)

# Autograd submodules
from .autograd import *
submodules.append(autograd)

# SPH submodules
from .kernels import *
submodules.append(kernels)
from .radiusSearch import *
submodules.append(radiusSearch)
from .operations import *
submodules.append(operations)

# SPH correction submodules
from .crk import *
submodules.append(crk)
from .renorm import *
submodules.append(renorm)

# Specific math operations
from .pinv import *
submodules.append(pinv)

# Set version
__version__ = "0.5.0"

__all__ = []

for submodule in submodules:
    __all__.extend(getattr(submodule, '__all__', []))