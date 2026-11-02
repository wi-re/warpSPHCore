"""Native C++ execution, selected before import with WARPSPHCORE_BACKEND=native."""
from .._models import (ParticleState, ParticleTangentState, PointCloud,
                       DomainDescription, OperationProperties, AdjacencyList,
                       CapacityPolicy, DEFAULT_CAPACITY, CRKState, CRKTangentState,
                       GradHState, RenormalizationState, RenormalizationTangentState)
from ..enumTypes import *
from ..enumTypes import __all__ as _enums
from .api import warpOperation, sphOperation, radiusSearch, nativeCapabilities
from ._build import build, get_include

__all__ = list(_enums) + ['ParticleState', 'ParticleTangentState', 'PointCloud',
    'DomainDescription', 'OperationProperties', 'AdjacencyList', 'CapacityPolicy',
    'DEFAULT_CAPACITY', 'warpOperation', 'sphOperation', 'radiusSearch',
    'nativeCapabilities', 'build', 'get_include', 'CRKState', 'CRKTangentState',
    'GradHState', 'RenormalizationState', 'RenormalizationTangentState']

# Resolve module-owned APIs lazily so importing crk.native under the Warp
# backend can use native infrastructure without re-entering its own exports.
_CRK_EXPORTS = ['computeCRKVolume', 'computeCRKMoments', 'computeCRKDensity',
                'computeCRKTerms', 'computeCRKFactors', 'CRKPreparation', 'prepareCRK']
__all__ += _CRK_EXPORTS


def __getattr__(name):
    if name in _CRK_EXPORTS:
        from importlib import import_module
        return getattr(import_module('..crk.native', __name__), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
