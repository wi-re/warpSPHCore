from .._backend import BACKEND

if BACKEND == 'native':
    from .native import radiusSearch
    __all__ = ['radiusSearch']
else:
    from .naive.radius_naive import (
        radiusNaive,
        radiusNaiveFixed,
    )

    from .grid_util import checkOffset, getIndexRange, getIndexRangeLane, laneSubRange, laneSlice, laneSum

    from .small.wp_radius_small import warp_radius_search_small
    from .options import NEIGHBOR_COLLECTION_MODES, NEIGHBOR_QUERY_ORDERS


    __all__ = [
        "radiusNaive",
        "radiusNaiveFixed",
        "checkOffset",
        'getIndexRange',
        'getIndexRangeLane',
        'laneSubRange',
        'laneSlice',
        'laneSum',
        'warp_radius_search_small',
        'NEIGHBOR_COLLECTION_MODES',
        'NEIGHBOR_QUERY_ORDERS',
    ]

    from .compactHash import *
    __all__.extend(compactHash.__all__)

    from .verlet import *
    __all__.extend(verlet.__all__)

    # Opt-in storage APIs; existing flat search remains the default.
    from .rowChunks import RowChunk, RowChunkGeneration, RowChunkPlan, RowChunkPolicy, buildRowChunkPlan
    from . import chunked
    from .chunked import *
    __all__.extend(['RowChunk', 'RowChunkGeneration', 'RowChunkPlan', 'RowChunkPolicy', 'buildRowChunkPlan'])
    __all__.extend(chunked.__all__)
    from .pairField import ChunkedPairField
    __all__.append('ChunkedPairField')

    from .chunkedState import rowChunkCheckpointRecipe, rowChunkCheckpointState, restoreRowChunkCheckpointState
    __all__.extend(['rowChunkCheckpointRecipe', 'rowChunkCheckpointState', 'restoreRowChunkCheckpointState'])
