"""Process-wide backend selection, resolved before importing runtime types."""
import os
BACKEND = os.environ.get("WARPSPHCORE_BACKEND", "warp").strip().lower()
if BACKEND not in ("warp", "native"):
    raise ValueError("WARPSPHCORE_BACKEND must be 'warp' or 'native'")


# A Warp frontend may retain its search/custom kernels while exercising the
# native core. Resolve once at import; unsupported inputs fail in that backend.
OPERATORS = os.environ.get("WARPSPHCORE_OPERATORS", BACKEND).strip().lower()
if OPERATORS not in ("warp", "native"):
    raise ValueError("WARPSPHCORE_OPERATORS must be 'warp' or 'native'")
if BACKEND == "native" and OPERATORS != "native":
    raise ValueError("A native-only process requires native operators")


def resolve_backend(backend=None):
    """Resolve an optional module-local override without importing a runtime."""
    selected = BACKEND if backend is None else backend
    if selected not in ('warp', 'native'):
        raise ValueError("backend must be 'warp' or 'native'")
    if BACKEND == 'native' and selected == 'warp':
        raise ValueError('Select the Warp backend before importing the package')
    return selected
