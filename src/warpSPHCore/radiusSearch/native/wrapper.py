"""Backend selection at the radius-search module boundary.

High-level callers import from this subpackage, not a backend implementation.
Both backends use compact hashing and minimum-image distances. The native
backend owns a reusable C++ index and emits canonical CSR.
"""
from ..._backend import resolve_backend as _resolve_backend
from ..._models import ParticleState, DomainDescription, AdjacencyList
from ...enumTypes import SupportScheme

def radiusSearch(queryParticles: ParticleState, domain: DomainDescription,
                 referenceParticles: ParticleState = None,
                 mode: SupportScheme = SupportScheme.Gather, *, backend=None) -> AdjacencyList:
    """Compact-hash radius search, returning minimum-image CSR neighborhoods.

    Native search preserves ascending original reference IDs. IDs and offsets
    are int64, counts int32. This Python entry builds an index per call; the
    standalone C++ API can retain one across count/collect and later queries.
    """
    selected = _resolve_backend(backend)
    if selected == 'warp':
        from ..compactHash import radiusSearchCompactHashMap
        return radiusSearchCompactHashMap(queryParticles, domain, referenceParticles=referenceParticles, mode=mode)
    from .nt_radius import radiusSearch as native_radius
    return native_radius(queryParticles, domain, referenceParticles, mode)
