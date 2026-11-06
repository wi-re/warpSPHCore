"""Native Laplacian boundary, including packed and clipped field maps."""
def compute(query,reference,domain,properties,adjacency,queryValues=None,referenceValues=None):
    from .nt_corrected import compute as corrected
    return corrected(query,reference,domain,properties,adjacency,queryValues,referenceValues)
