"""Native curl boundary with bounded packed-field products."""
def compute(query,reference,domain,properties,adjacency,queryValues=None,referenceValues=None):
    from .nt_corrected import compute as corrected
    return corrected(query,reference,domain,properties,adjacency,queryValues,referenceValues)
