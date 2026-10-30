"""Backend selection at the radius-search module boundary.

High-level callers import from this subpackage, not a backend implementation.
The Warp backend retains its compact-hash implementation; the native pilot
currently provides exact all-pairs CSR construction.
"""
from ..._backend import resolve_backend as _resolve_backend
from ..._models import ParticleState, DomainDescription, AdjacencyList
from ...enumTypes import SupportScheme

def radiusSearch(queryParticles: ParticleState, domain: DomainDescription,
                 referenceParticles: ParticleState = None,
                 mode: SupportScheme = SupportScheme.Gather, *, backend=None) -> AdjacencyList:
    """Exact native all-pairs radius search (O(Nq*Nr)), returning CSR.

    Intended for correctness and small problems in this pilot. This is not a
    compact-hash replacement; scalable spatial indexing is a later milestone.
    Particle IDs and offsets are int64; counts remain int32. No padding/reuse.
    """
    selected = _resolve_backend(backend)
    if selected == 'warp':
        from ..compactHash import radiusSearchCompactHashMap
        return radiusSearchCompactHashMap(queryParticles, domain, referenceParticles=referenceParticles, mode=mode)
    from .nt_radius import radiusSearch as native_radius
    return native_radius(queryParticles, domain, referenceParticles, mode)
