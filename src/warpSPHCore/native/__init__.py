"""Native C++ execution, selected before import with WARPSPHCORE_BACKEND=native."""
from .._models import (ParticleState, ParticleTangentState, PointCloud,
                       DomainDescription, OperationProperties, AdjacencyList,
                       CapacityPolicy, DEFAULT_CAPACITY, CRKState, CRKTangentState,
                       GradHState, RenormalizationState, RenormalizationTangentState)
from ..enumTypes import *
from ..enumTypes import __all__ as _enums
from .api import warpOperation, warpOperationJVP, sphOperation, radiusSearch, nativeCapabilities
from ._build import build, get_include

__all__ = list(_enums) + ['ParticleState', 'ParticleTangentState', 'PointCloud',
    'DomainDescription', 'OperationProperties', 'AdjacencyList', 'CapacityPolicy',
    'DEFAULT_CAPACITY', 'warpOperation', 'sphOperation', 'radiusSearch',
    'warpOperationJVP', 'nativeCapabilities', 'build', 'get_include', 'CRKState', 'CRKTangentState',
    'GradHState', 'RenormalizationState', 'RenormalizationTangentState']

# Resolve module-owned APIs lazily so importing crk.native under the Warp
# backend can use native infrastructure without re-entering its own exports.
_CRK_EXPORTS = ['computeCRKFactorsJVP','computeCRKVolumeGeometryJVP','computeCRKMomentsGeometryJVP','computeCRKVolume', 'computeCRKMoments', 'computeCRKDensity',
                'computeCRKTerms', 'computeCRKFactors', 'CRKPreparation', 'prepareCRK']
_LINALG_EXPORTS = ['applyPinv', 'pinvSymmetric', 'symmetricPinv3', 'loadPadded3x3',
    'bicg', 'bicgstab', 'cg', 'gmres', 'minres', 'richardson', 'LinearOperator',
    'RestrictedSystem', 'NullspaceSystem', 'NullspaceRhs', 'NullspaceResiduals',
    'DiagonalPreconditioner', 'BlockJacobiPreconditioner', 'CGWorkspace',
    'BiCGStabWorkspace', 'SolveResult', 'Status', 'CheckSchedule',
    'RESIDUAL_STATISTICS', 'StationaryResult', 'residualStatistic', 'runStationary']
_POLY_EXPORTS = ['PolyFit', 'monomialExponents', 'interfacePairs', 'minimumImage', 'Reconstructor', 'smoothnessGram']
_RENORM_EXPORTS = ['computeRenormalizationMatrices', 'computeRenormalizationMatricesJVP']
_METADATA_EXPORTS = ['OperatorSpec','SPHContext','Corrections','EMPTY_CORRECTIONS','OutputSpec',
    'ExtraSpec','ExtraKind','RowRole','ShapeOf','ThreadSpec','JVPSpec','launchOperator',
    'compileGlue','compileGlueEnabled','markDynamic','setCompileGlue']
_PINV_EXPORTS = ['pinv_warp', 'pinv1x1', 'pinv2x2_warpBackend']
__all__ += _CRK_EXPORTS + _LINALG_EXPORTS + _RENORM_EXPORTS + _PINV_EXPORTS + _POLY_EXPORTS + _METADATA_EXPORTS


def __getattr__(name):
    if name in _CRK_EXPORTS:
        from importlib import import_module
        return getattr(import_module('..crk.native', __name__), name)
    for exports, module in ((_LINALG_EXPORTS, '..linalg'),
                            (_RENORM_EXPORTS, '..renorm'), (_PINV_EXPORTS, '..pinv'),
                            (_POLY_EXPORTS, '..polyfit'), (_METADATA_EXPORTS, '..autograd')):
        if name in exports:
            from importlib import import_module
            return getattr(import_module(module, __name__), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

from ._topology import CSRTopology
from ._field import linearizeFields, FieldLinearization, FieldCotangents
__all__ += ['CSRTopology', 'linearizeFields', 'FieldLinearization', 'FieldCotangents']

from ..radiusSearch.native import (buildVerletList, filterVerletList, verletStaleFlag,
                                  verletUsage, verletValidityMetrics, deferVerletChecks)
__all__ += ["buildVerletList", "filterVerletList", "verletStaleFlag", "verletUsage",
            "verletValidityMetrics", "deferVerletChecks"]

# Host dtype settings remain available to callers building shared chunk policies.
from ..type_config import (get_type_config,get_precision,get_dim,get_torch_precision,
                           get_torch_index_dtype,get_torch_offset_dtype,to_torch,to_numpy)
from ..radiusSearch import (OwnedRowChunk,ChunkedAdjacency,RowChunk,RowChunkGeneration,
    RowChunkPlan,RowChunkPolicy,buildRowChunkPlan,radiusSearchRowChunks_,radiusSearchRowChunks,
    buildChunkedVerletList_,buildChunkedVerletList,filterChunkedVerletList_,filterChunkedVerletList,
    chunkedVerletStaleFlag_,ChunkedPairField,rowChunkCheckpointRecipe,rowChunkCheckpointState,restoreRowChunkCheckpointState)
__all__ += ['get_type_config','get_precision','get_dim','get_torch_precision',
    'get_torch_index_dtype','get_torch_offset_dtype','to_torch','to_numpy',
    'OwnedRowChunk','ChunkedAdjacency','RowChunk','RowChunkGeneration','RowChunkPlan','RowChunkPolicy',
    'buildRowChunkPlan','radiusSearchRowChunks_','radiusSearchRowChunks','buildChunkedVerletList_',
    'buildChunkedVerletList','filterChunkedVerletList_','filterChunkedVerletList','chunkedVerletStaleFlag_',
    'ChunkedPairField','rowChunkCheckpointRecipe','rowChunkCheckpointState','restoreRowChunkCheckpointState']

from ..coreOperations.native.nt_density_hvp import computeSPHDensityPositionHVP
__all__ += ['computeSPHDensityPositionHVP']

from ..coreOperations.native.nt_density_hvp import warpOperationHVP
from .._operation_flat import sphOperation_warp
from ..sampling import sampleDensestLattice,LatticeSample
from ..kernels.native import evaluateKernel,KernelEvaluation
__all__ += ['warpOperationHVP','sphOperation_warp','sampleDensestLattice','LatticeSample',
            'evaluateKernel','KernelEvaluation']

from ..math.prime import getNextPrime
__all__ += ['getNextPrime']
