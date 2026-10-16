"""Multi-lane ("tiled") neighbour loops -- the small-problem GPU fast path.

Every neighbour-loop operator kernel is written thread-per-particle: one
thread walks particle ``i``'s whole neighbour list serially. With ~100
neighbours per particle that loop is a chain of ~100 dependent, uncoalesced
gathers, and on a problem small enough to fit the GPU in a single wave (up
to a few hundred thousand particles on a large part) the kernel's run time
is that chain's latency, *independent of the particle count*. Measured on the
Marrone 3.1 dam break (RTX PRO 6000, nx=70, 10.9k particles, ~101 neighbours):
the interpolation kernel took 535 us at 4.7k, 10.9k and 28.5k particles alike.

An ``OperatorSpec`` may therefore carry a ``tiledKernel``: the same physics
launched as ``dim=[N, lanes]`` with ``block_dim=lanes``, where each of the
``lanes`` threads of a block walks a contiguous slice of particle ``i``'s
neighbour range (``getIndexRangeLane``) and the partial sums are combined with
a deterministic ``wp.tile_sum``. Same interpolation kernel at 32 lanes: 30 us
at 10.9k particles (17.7x), 201 us vs 591 us at 85k, 666 us vs 1138 us at
283k -- so it is the default at every size measured, not only small ones.

The only numerical difference is float summation order (~1e-7 relative).

``WARPSPHCORE_NEIGHBOR_LANES`` (or :func:`setNeighborLanes`) picks the lane
count; ``1`` (or ``0``) restores the thread-per-particle kernels everywhere.
Without an explicit process setting, qualified kernels default to 32 lanes;
the trial curl and Laplacian variants retain one lane pending consumer and
large-problem qualification. OperatorSpec.defaultNeighborLanes carries that
choice; ordinary and prepared launchers honor it alike.
Lane counts must divide the 32-thread warp evenly to keep blocks full, so
only powers of two up to 32 are accepted. A process-wide setting, read at
launch time, so it can be flipped between calls (A/B timing, bisection).
"""

from __future__ import annotations

import os
from typing import Optional

__all__ = ['neighborLanes', 'setNeighborLanes', 'DEFAULT_NEIGHBOR_LANES']

DEFAULT_NEIGHBOR_LANES = 32
_VALID = (1, 2, 4, 8, 16, 32)

_override: Optional[int] = None


def _validate(lanes: int) -> int:
    lanes = max(1, int(lanes))
    if lanes not in _VALID:
        raise ValueError(f'neighbour lanes must be one of {_VALID} (0 means 1), got {lanes}')
    return lanes


def _fromEnv() -> Optional[int]:
    raw = os.environ.get('WARPSPHCORE_NEIGHBOR_LANES')
    return None if raw is None or raw == '' else _validate(int(raw))


_envLanes = _fromEnv()


def neighborLanes(*, default: int = DEFAULT_NEIGHBOR_LANES) -> int:
    """Resolve explicit process overrides before an operator's default.

    Trial kernels may default to one lane until consumer qualification. An
    explicit environment value or setNeighborLanes still selects their tiled
    variant. Calling without arguments retains the existing 32-lane default.
    """
    if _override is not None:
        return _override
    return _envLanes if _envLanes is not None else _validate(default)


def setNeighborLanes(lanes: Optional[int]) -> None:
    """Override the lane count for this process; ``None`` reverts to the
    environment / per-operator default."""
    global _override
    _override = None if lanes is None else _validate(lanes)
