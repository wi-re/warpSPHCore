"""Native covariance entry point with complete first-order ownership."""

def compute(query,reference,domain,properties,adjacency,queryValues=None,referenceValues=None):
    from .nt_corrected import compute as corrected
    return corrected(query,reference,domain,properties,adjacency,None,None)
