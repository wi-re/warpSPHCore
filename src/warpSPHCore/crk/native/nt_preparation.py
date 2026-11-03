"""Native preparation boundary shared by the module's three pair passes."""
from ...enumTypes import KernelFunctions
from ...native._checks import check_execution
from ...native._topology import CSRTopology
from ...native._inputs import prepare_geometry, prepare_adjacency


def prepare(query, reference, domain, properties, adjacency, *values):
    reference = query if reference is None else reference
    if not isinstance(properties.kernel, KernelFunctions):
        raise TypeError('kernel must be a KernelFunctions')
    if properties.calibrateNormalization or properties.positiveDivergence:
        raise NotImplementedError('CRK preparation does not apply normalization calibration or positive divergence')
    # Only consumed inputs enter the first-order differentiation policy.
    check_execution(query.positions, reference.positions, query.supports,
                    reference.supports, domain.min, domain.max, *values, allow_reverse=True, allow_forward=True)
    geometry = prepare_geometry(query, reference, domain, allow_reverse=True, allow_forward=True)
    topology = adjacency if isinstance(adjacency, CSRTopology) else None
    adjacency = prepare_adjacency(query, reference, domain, properties, adjacency)
    topology = topology if topology is not None else CSRTopology(adjacency)
    inputs = [*geometry[:4], reference.masses, query.kinds, reference.kinds,
              *geometry[4:], adjacency.edgeOffsets, adjacency.numNeighbors,
              adjacency.paddedJ, None, None, None, None]
    return inputs, (properties.supportMode.value, properties.operationMode.value,
                    properties.kernel.value), topology
