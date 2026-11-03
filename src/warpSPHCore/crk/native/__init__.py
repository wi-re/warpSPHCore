"""Native CRK primals; importing this module does not initialize Warp."""
from .nt_volume import computeCRKVolume
from .nt_moments import computeCRKMoments
from .nt_density import computeCRKDensity
from .nt_factors import CRKPreparation, prepareCRK, computeCRKFactors
from ..crk_terms import computeCRKTerms, computeCRKTermsWarp

# Preserve the module-level entry names used by the existing Warp scaffolding.
_computeCRKVolume_stateBackend = computeCRKVolume
_computeCRKMoments_stateBackend = computeCRKMoments
_computeCRKDensity_stateBackend = computeCRKDensity

__all__ = ['computeCRKVolume', 'computeCRKMoments', 'computeCRKDensity',
           'computeCRKTerms', 'computeCRKTermsWarp', 'computeCRKFactors',
           'CRKPreparation', 'prepareCRK']

from .nt_jvp import computeCRKFactorsJVP,computeCRKVolumeGeometryJVP,computeCRKMomentsGeometryJVP
__all__ += ['computeCRKFactorsJVP','computeCRKVolumeGeometryJVP','computeCRKMomentsGeometryJVP']
