"""`StateBundle`: persistent, refreshable Warp struct instances that replace
the per-call closure `arg_extract.py`'s `build_fn` otherwise rebuilds from
scratch every launch (warpier_fields.md Step F).

**No-grad path only -- this is not a gating nicety, it is a correctness
requirement.** Verified directly against warp 1.16.0: `wp.Tape` does not
snapshot a struct's field values at launch time -- it holds a live reference
to the (mutable) struct object and re-reads its fields lazily, at
`tape.backward()` time. A minimal repro (two `wp.launch` calls sharing one
mutable struct, the struct's array field reassigned between them, backward
run only on the first tape) showed the *second* call's array ending up in
the *first* call's gradient, and the first call's own gradient reading as
zero. Sharing a mutable `StateBundle` across grad-requiring calls would
silently corrupt gradients any time a call's backward is deferred past a
later call that reuses the bundle -- which is completely ordinary PyTorch
usage (build a graph across several ops, call `.backward()` once), not an
edge case. So: bundles are reused and refreshed in place only when nothing
in the call requires grad; a grad-requiring call always gets a fresh,
call-local set of structs (`arg_extract.py`'s original per-call construction
path, unchanged), exactly mirroring Step D's no-grad-only gate for view
reuse and for the same underlying reason -- shared mutable state is safe
only where there is no deferred tape to corrupt it.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..dataTypes.adjacency_t import adjacencyData, gridData
from ..dataTypes.domain_t import domainData
from ..dataTypes.field_t import ExecutionMode
from ..dataTypes.kernelState_t import kernelState
from ..util.fieldRegistry import structFor

# wa-index -> (bundle attribute, struct field name), in the same order as
# arg_extract.py's flat tensor index layout (0-35). Indices not listed here
# (33-35 are handled by name below; adjacency/grid halves are mutually
# exclusive per call but both structs are always present on the bundle) are
# either scalars (handled separately in refresh(), since they legitimately
# change every call) or constants fixed at bundle-construction time (D/dim).
_ARRAY_FIELDS = [
    (0, "queryParticle", "positions"),
    (2, "queryParticle", "supports"),
    (4, "queryParticle", "masses"),
    (6, "queryParticle", "densities"),
    (8, "queryParticle", "kinds"),
    (1, "referenceParticle", "positions"),
    (3, "referenceParticle", "supports"),
    (5, "referenceParticle", "masses"),
    (7, "referenceParticle", "densities"),
    (9, "referenceParticle", "kinds"),
    (10, "correction", "renormalizationMatrices"),
    (11, "correction", "queryOmegas"),
    (12, "correction", "referenceOmegas"),
    (13, "correction", "queryVolumes"),
    (14, "correction", "referenceVolumes"),
    (15, "correction", "queryA"),
    (16, "correction", "queryB"),
    (17, "correction", "queryGradA"),
    (18, "correction", "queryGradB"),
    (19, "correction", "referenceA"),
    (20, "correction", "referenceB"),
    (21, "correction", "referenceGradA"),
    (22, "correction", "referenceGradB"),
    (23, "adjacency", "neighborList"),
    (24, "adjacency", "neighborOffsets"),
    (25, "adjacency", "numNeighbors"),
    (26, "grid", "sortIndex"),
    (27, "grid", "qMin"),
    (28, "grid", "qMax"),
    (29, "grid", "numCells"),
    (30, "grid", "hashTable"),
    (31, "grid", "cellTable"),
    (32, "grid", "cellOffsets"),
    (33, "domain", "domainMin"),
    (34, "domain", "domainMax"),
    (35, "domain", "periodicity"),
]


class StateBundle:
    """Persistent Warp struct instances for one `dim`, refreshed in place.

    `refresh` writes an array field only when the incoming `wp.array` object
    differs (by identity) from what is already there -- on a call sharing
    most of its state with the previous one (Section 2.3: adjacency, domain,
    and every disabled-correction null field are typically unchanged call to
    call within a step), most of the ~34 candidate writes are skipped
    entirely.
    """

    __slots__ = (
        "queryParticle",
        "referenceParticle",
        "domain",
        "adjacency",
        "grid",
        "correction",
        "kernelProperties",
        "_last_wa",
    )

    def __init__(self, dim: int, mode: ExecutionMode):
        ParticleSoA = structFor("particleDataSoA", dim, mode)
        CorrData = structFor("correctionData", dim, mode)

        self.queryParticle = ParticleSoA()
        self.referenceParticle = ParticleSoA()
        self.domain = domainData()
        self.adjacency = adjacencyData()
        self.grid = gridData()
        self.correction = CorrData()
        self.kernelProperties = kernelState()

        # Fixed for the lifetime of this bundle (== dim, which is the cache
        # key -- see getStateBundle): assigned once here, never touched by
        # refresh().
        self.domain.dim = dim
        self.grid.D = dim

        self._last_wa: Optional[List] = None

    def refresh(self, wa: List, cfg: Dict) -> None:
        last = self._last_wa
        for idx, bundle_attr, field_name in _ARRAY_FIELDS:
            new_val = wa[idx]
            if last is None or last[idx] is not new_val:
                setattr(getattr(self, bundle_attr), field_name, new_val)
        self._last_wa = wa

        # Scalars: always assign. Cheap (plain attribute writes on a struct
        # already holding a value of the same type -- no array-wrapper
        # machinery involved), and several of these legitimately change
        # every call (grid_hCell tracks the *current* adjacency's cell size).
        self.grid.hCell = cfg["grid_hCell"]
        self.grid.numOffsets = cfg["grid_numOffsets"]
        self.correction.useGradientRenormalization = cfg["useGradientRenormalization"]
        self.correction.useGradHTerms = cfg["useGradHTerms"]
        self.correction.useVolume = cfg["useVolumes"]
        self.correction.useCRK = cfg["useCRK"]
        self.kernelProperties.kernelFunction = cfg["kernel_int"]
        self.kernelProperties.supportMode = cfg["mode_uint"]
        self.kernelProperties.gradientMode = cfg["gradientMode_int"]
        self.kernelProperties.laplacianMode = cfg["laplacianMode_int"]
        self.kernelProperties.positiveDivergenceMode = cfg["positiveDivergence"]
        self.kernelProperties.divergenceMode = cfg["divergenceMode"]
        self.kernelProperties.operationMode = cfg["opInt"]


# dim in {1, 2, 3}: at most three bundles ever exist, so a plain dict with no
# eviction is sufficient -- no LRU needed for a cache this small.
#
# Keyed on dim alone, not (dim, mode): every mode currently registered in
# fieldRegistry._STRUCT_TABLE resolves NONE and REVERSE to the *same*
# struct classes (Step B), so mode cannot yet select a different bundle.
# ExecutionMode.FORWARD raises in structFor() rather than silently handing
# back a REVERSE-shaped struct, so this does not need to guard against it
# separately. Widening the key to include mode is a one-line change
# (`_BUNDLE_CACHE[(dim, mode)]`) whenever Phase 6 registers FORWARD-specific
# struct rows that actually differ -- see Step G's readiness audit.
_BUNDLE_CACHE: Dict[int, StateBundle] = {}


def getStateBundle(dim: int, mode: ExecutionMode = ExecutionMode.REVERSE) -> StateBundle:
    bundle = _BUNDLE_CACHE.get(dim)
    if bundle is None:
        bundle = StateBundle(dim, mode)
        _BUNDLE_CACHE[dim] = bundle
    return bundle


def clearStateBundleCache() -> None:
    """Test/debug hook: drop every cached bundle."""
    _BUNDLE_CACHE.clear()
