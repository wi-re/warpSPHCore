"""Reusable native wrapper preparation for core and frontend operators.

Tensor shape/device/content validation remains in C++; operator-specific field
requirements and derivative policy belong to each nt_<operator> entry point.
"""
from dataclasses import replace
from .._models import AdjacencyList
from ..enumTypes import KernelFunctions
import torch
from ._checks import check_execution


def prepare_geometry(query, reference, domain, *, allow_reverse=False, allow_forward=False):
    """Pack particle/domain geometry shared by search and pair operators."""
    if domain.dim != query.positions.shape[-1]:
        raise ValueError('domain.dim must match the positions dimension')
    tensors = (query.positions, reference.positions, query.supports, reference.supports,
               domain.min, domain.max, domain.periodic)
    check_execution(*tensors, allow_reverse=allow_reverse, allow_forward=allow_forward)
    return tensors


def prepare_operator_inputs(query, reference, domain, properties, adjacency,
                            *, fields=(), corrections=()):
    """Prepare the shared native pair-operator tensor layout and CSR adjacency.

    fields contains the operator's query/reference scalar values and reference
    density, or is empty for an operator that does not consume those slots.
    """
    if not isinstance(properties.kernel, KernelFunctions):
        raise TypeError('kernel must be a KernelFunctions')
    if (any(x is not None for x in corrections) or properties.calibrateNormalization
            or (properties.operation.name == "Laplacian" and properties.positiveDivergence)):
        raise NotImplementedError('Use the corrected native entry point for correction and specialised options')
    geometry = prepare_geometry(query, reference, domain, allow_reverse=True, allow_forward=True)
    check_execution(reference.masses, *fields, reference.densities, query.densities, allow_reverse=True, allow_forward=True)
    adjacency = prepare_adjacency(query, reference, domain, properties, adjacency)
    if not fields:
        empty = query.positions.new_empty(0)
        fields = (empty, empty, empty)
    return [*geometry[:4], reference.masses, query.kinds, reference.kinds,
            *geometry[4:], adjacency.edgeOffsets, adjacency.numNeighbors,
            adjacency.paddedJ, *fields, query.densities if query.densities is not None else query.positions.new_empty(0)]


def prepare_adjacency(query, reference, domain, properties, adjacency):
    """Resolve only discrete topology; callers own their AD/input policy."""
    from ..radiusSearch.native import radiusSearch
    from ._topology import CSRTopology
    if isinstance(adjacency, CSRTopology): adjacency = adjacency.adjacency
    if adjacency is None:
        # Neighbour topology is discrete; differentiate the fixed CSR primal.
        with torch.no_grad():
            query_geometry = replace(query, positions=query.positions.detach(), supports=query.supports.detach())
            reference_geometry = replace(reference, positions=reference.positions.detach(), supports=reference.supports.detach())
            fixed_domain = replace(domain, min=domain.min.detach(), max=domain.max.detach())
            adjacency = radiusSearch(query_geometry, fixed_domain, reference_geometry, properties.supportMode, backend='native')
    from ..radiusSearch.chunked import ChunkedAdjacency
    if isinstance(adjacency,ChunkedAdjacency):
        adjacency.assertStorageValid()
        if adjacency.numRows!=len(query.positions) or adjacency.numCols!=len(reference.positions):
            raise ValueError('adjacency populations must match the particle states')
        return adjacency
    if not isinstance(adjacency, AdjacencyList):
        raise TypeError('Native operators require a Torch CSR AdjacencyList or ChunkedAdjacency')
    if adjacency.numRows != len(query.positions) or adjacency.numCols != len(reference.positions):
        raise ValueError('adjacency populations must match the particle states')
    return adjacency
