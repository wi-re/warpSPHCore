"""Common module-local wrapper for Warp and C++ operator implementations.

Only an explicitly selected Warp implementation imports the wp_* modules.
Native callers never enter the Warp argument conversion or autograd bridge.
"""
from ..._backend import resolve_backend as _resolve_backend
from ..._models import ParticleState, DomainDescription, OperationProperties
from ...enumTypes import WarpOperation


def computeDensity(queryParticles, referenceParticles, domain, operationProperties,
                   adjacency=None, *, backend=None):
    if _resolve_backend(backend) == 'warp':
        from .wp_density import _computeSPHDensity_stateBackend
        return _computeSPHDensity_stateBackend(queryParticles, referenceParticles, domain,
                                               operationProperties, adjacency)
    from .nt_density import computeDensity as native_density
    return native_density(queryParticles, referenceParticles, domain, operationProperties, adjacency)


def computeGradient(queryParticles, referenceParticles, domain, operationProperties, adjacency,
                    queryValues, referenceValues, queryVolumes=None, referenceVolumes=None,
                    crkState=None, gradHState=None, renormalizationState=None, *, backend=None):
    if _resolve_backend(backend) == 'warp':
        from .wp_gradient import _computeSPHGradient_stateBackend
        return _computeSPHGradient_stateBackend(
            queryParticles, referenceParticles, domain, operationProperties, adjacency,
            queryValues, referenceValues, queryVolumes=queryVolumes, referenceVolumes=referenceVolumes,
            crkState=crkState, gradHState=gradHState, renormalizationState=renormalizationState)
    from .nt_gradient import computeGradient as native_gradient
    return native_gradient(queryParticles, referenceParticles, domain, operationProperties, adjacency,
                            queryValues, referenceValues, queryVolumes, referenceVolumes,
                            crkState, gradHState, renormalizationState)


def sphOperation(queryParticles: ParticleState, operationProperties: OperationProperties,
                 domain: DomainDescription, queryValues=None, referenceValues=None,
                 queryVolumes=None, referenceVolumes=None, adjacency=None,
                 referenceParticles=None, preScatteredQuantities=None, crkState=None,
                 gradHState=None, renormalizationState=None, consistentDivergence=False,
                 covarianceReturnNumNeighbors=False, *, backend=None):
    """Common call surface; the selected backend owns supported-feature policy."""
    selected = _resolve_backend(backend)
    if selected == 'warp':
        from ...operations import warpOperation
        return warpOperation(
            queryParticles, operationProperties, domain, queryValues, referenceValues,
            queryVolumes, referenceVolumes, adjacency, referenceParticles,
            preScatteredQuantities, crkState, gradHState, renormalizationState,
            consistentDivergence, covarianceReturnNumNeighbors)
    if not isinstance(operationProperties, OperationProperties):
        raise TypeError('operationProperties must be an OperationProperties')
    if (any(x is not None for x in (queryVolumes, referenceVolumes, preScatteredQuantities,
                                    crkState, gradHState, renormalizationState))
            or consistentDivergence or covarianceReturnNumNeighbors):
        raise NotImplementedError('Corrections and specialised options are not supported by the native pilot')
    reference = referenceParticles if referenceParticles is not None else queryParticles
    referenceValues = referenceValues if referenceValues is not None else queryValues
    if operationProperties.operation == WarpOperation.Density:
        return computeDensity(queryParticles, reference, domain, operationProperties, adjacency, backend='native')
    if operationProperties.operation == WarpOperation.Gradient:
        return computeGradient(queryParticles, reference, domain, operationProperties, adjacency,
                               queryValues, referenceValues, backend='native')
    raise NotImplementedError('Native pilot supports Density and Gradient only')


warpOperation = sphOperation
