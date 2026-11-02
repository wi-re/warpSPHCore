"""Native preparation boundary shared by the module's three pair passes."""
from ...enumTypes import KernelFunctions
from ...native._checks import check_execution
from ...native._inputs import prepare_geometry, prepare_adjacency


def prepare(query, reference, domain, properties, adjacency, *values):
    reference = query if reference is None else reference
    if not isinstance(properties.kernel, KernelFunctions):
        raise TypeError('kernel must be a KernelFunctions')
    if properties.calibrateNormalization or properties.positiveDivergence:
        raise NotImplementedError('CRK preparation does not apply normalization calibration or positive divergence')
    # Only inputs consumed by the pass enter its derivative policy. Geometry
    # and volumes are primal-only until their manual derivatives are ported.
    check_execution(query.positions, reference.positions, query.supports,
                    reference.supports, domain.min, domain.max, *values)
    geometry = prepare_geometry(query, reference, domain)
    adjacency = prepare_adjacency(query, reference, domain, properties, adjacency)
    inputs = [*geometry[:4], reference.masses, query.kinds, reference.kinds,
              *geometry[4:], adjacency.edgeOffsets, adjacency.numNeighbors,
              adjacency.paddedJ, None, None, None, None]
    return inputs, (properties.supportMode.value, properties.operationMode.value,
                    properties.kernel.value)
