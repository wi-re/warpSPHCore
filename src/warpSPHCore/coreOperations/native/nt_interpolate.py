"""Native interpolation entry point, including packed fields and search inputs."""


def compute(query, reference, domain, properties, adjacency, queryValues=None, referenceValues=None):
    from .nt_corrected import compute as corrected
    return corrected(query, reference, domain, properties, adjacency, queryValues, referenceValues)
