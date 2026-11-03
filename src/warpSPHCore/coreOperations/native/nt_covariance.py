"""Native covariance primal entry point."""
from ...native._inputs import prepare_operator_inputs


def compute(query, reference, domain, properties, adjacency, queryValues=None, referenceValues=None):
    empty = query.positions.new_empty(0)
    if reference.densities is None:
        raise ValueError('covariance requires reference densities')
    if properties.operation.name != 'Covariance' and referenceValues is None:
        raise ValueError('covariance requires reference values')
    inputs = prepare_operator_inputs(query, reference, domain, properties, adjacency,
        fields=(queryValues if queryValues is not None else empty,
                referenceValues if referenceValues is not None else empty, reference.densities))
    from .nt_covariance_backward import apply
    return apply(inputs, properties)
