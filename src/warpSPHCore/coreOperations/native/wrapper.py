"""Common module-local wrapper for Warp and C++ operator implementations.

Only an explicitly selected Warp implementation imports the wp_* modules.
Native callers use their own Torch bridge and module-owned native kernels.
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
    from ...radiusSearch.chunked import ChunkedAdjacency
    if isinstance(adjacency,ChunkedAdjacency):
        from dataclasses import replace
        return sphOperation(queryParticles,replace(operationProperties,operation=WarpOperation.Density),
                            domain,adjacency=adjacency,referenceParticles=referenceParticles,backend='native')
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
    from ...radiusSearch.chunked import ChunkedAdjacency
    if isinstance(adjacency,ChunkedAdjacency):
        from dataclasses import replace
        return sphOperation(queryParticles,replace(operationProperties,operation=WarpOperation.Gradient),
                            domain,queryValues,referenceValues,queryVolumes,referenceVolumes,adjacency,referenceParticles,
                            crkState=crkState,gradHState=gradHState,renormalizationState=renormalizationState,backend='native')
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
    if operationProperties.calibrateNormalization:
        from dataclasses import replace
        from ...native._options import normalization_factor
        factor = normalization_factor(operationProperties, domain.dim)
        result = sphOperation(queryParticles, replace(operationProperties, calibrateNormalization=False),
            domain, queryValues, referenceValues, queryVolumes, referenceVolumes, adjacency,
            referenceParticles, preScatteredQuantities, crkState, gradHState, renormalizationState,
            consistentDivergence, covarianceReturnNumNeighbors, backend='native')
        if isinstance(result, tuple):
            return tuple(value * factor if value.is_floating_point() else value for value in result)
        return result * factor
    reference = referenceParticles if referenceParticles is not None else queryParticles
    referenceValues = referenceValues if referenceValues is not None else queryValues
    operation = operationProperties.operation
    # The public Warp dispatcher only forwards each option to its consumer.
    # Resolve these aliases before query slicing and ignore unread bundles.
    covarianceReturnNumNeighbors = covarianceReturnNumNeighbors if operation == WarpOperation.Covariance else False
    consistentDivergence = consistentDivergence if operation == WarpOperation.Divergence else False
    if operation == WarpOperation.Density:
        queryValues = referenceValues = None
        queryVolumes = referenceVolumes = crkState = gradHState = renormalizationState = None
        preScatteredQuantities = None
    elif operation == WarpOperation.Covariance:
        queryValues = referenceValues = gradHState = None
        preScatteredQuantities = None
    elif operation == WarpOperation.Interpolate:
        queryValues = gradHState = renormalizationState = None
    if preScatteredQuantities is not None:
        raise NotImplementedError('Pre-scattered quantities are unsupported; pass query/reference values')
    from ...radiusSearch.chunked import ChunkedAdjacency
    if isinstance(adjacency,ChunkedAdjacency):
        from ...native._row_dispatch import standard
        return standard(queryParticles,reference,domain,operationProperties,adjacency,
                        queryValues,referenceValues,queryVolumes,referenceVolumes,crkState,
                        gradHState,renormalizationState,consistentDivergence,covarianceReturnNumNeighbors)
    # Density's Warp equation consumes none of the optional corrections.
    # Frontends may pass their common correction bundle to every operation.
    if operationProperties.operation == WarpOperation.Density and not (consistentDivergence or covarianceReturnNumNeighbors):
        return computeDensity(queryParticles, reference, domain, operationProperties, adjacency, backend='native')
    if (operationProperties.operation == WarpOperation.Laplacian and operationProperties.positiveDivergence) or consistentDivergence or covarianceReturnNumNeighbors or any(value is not None for value in (queryVolumes, referenceVolumes,
                                           crkState, gradHState, renormalizationState)):
        from .nt_corrected import compute
        return compute(queryParticles, reference, domain, operationProperties, adjacency,
                       queryValues, referenceValues, queryVolumes, referenceVolumes,
                       crkState, gradHState, renormalizationState,
                       consistent_divergence=consistentDivergence,
                       return_neighbor_counts=covarianceReturnNumNeighbors)
    if operationProperties.operation == WarpOperation.Gradient:
        return computeGradient(queryParticles, reference, domain, operationProperties, adjacency,
                               queryValues, referenceValues, backend='native')
    from ...enumTypes import WarpOperation as Op
    if operationProperties.operation == Op.Interpolate:
        from .nt_interpolate import compute
    elif operationProperties.operation == Op.Divergence:
        from .nt_divergence import compute
    elif operationProperties.operation == Op.Curl:
        from .nt_curl import compute
    elif operationProperties.operation == Op.Laplacian:
        from .nt_laplacian import compute
    elif operationProperties.operation == Op.Covariance:
        from .nt_covariance import compute
    else:
        raise NotImplementedError('Custom native operations require their own wrapper')
    return compute(queryParticles, reference, domain, operationProperties, adjacency, queryValues, referenceValues)


warpOperation = sphOperation
