"""Complete independent field directions for the explicit JVP API."""


def field_directions(query_count, reference_count, query_values, reference_values,
                     tangent_query, tangent_reference):
    """An omitted direction is zero, even when primal field roles alias."""
    if tangent_query is None:
        template = query_values if query_values is not None else reference_values
        if template is None:
            template = tangent_reference
        tangent_query = template.new_zeros((query_count, *template.shape[1:]))
    if tangent_reference is None:
        template = reference_values if reference_values is not None else query_values
        if template is None:
            template = tangent_query
        tangent_reference = template.new_zeros((reference_count, *template.shape[1:]))
    return tangent_query, tangent_reference
