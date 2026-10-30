from .._models import PointCloud, ParticleState, ParticleTangentState
from .._backend import BACKEND

if BACKEND == "warp":
    import warp as wp
    from warp.types import vector
    from ..type_config import *

    @wp.struct
    class particleDataSoA_1:
        positions: wp.array(dtype=vector(length=1, dtype = scalar_t)) # type: ignore
        supports: wp.array(dtype = scalar_t) # type: ignore
        masses: wp.array(dtype = scalar_t) # type: ignore
        densities: wp.array(dtype = scalar_t) # type: ignore
        kinds: wp.array(dtype = wp.int32) # type: ignore
    @wp.struct
    class particleDataSoA_2:
        positions: wp.array(dtype=vector(length=2, dtype = scalar_t)) # type: ignore
        supports: wp.array(dtype = scalar_t) # type: ignore
        masses: wp.array(dtype = scalar_t) # type: ignore
        densities: wp.array(dtype = scalar_t) # type: ignore
        kinds: wp.array(dtype = wp.int32) # type: ignore
    @wp.struct
    class particleDataSoA_3:
        positions: wp.array(dtype=vector(length=3, dtype = scalar_t)) # type: ignore
        supports: wp.array(dtype = scalar_t) # type: ignore
        masses: wp.array(dtype = scalar_t) # type: ignore
        densities: wp.array(dtype = scalar_t) # type: ignore
        kinds: wp.array(dtype = wp.int32) # type: ignore


    @wp.struct
    class WarpParticle_1:
        position: vector(length=1, dtype = scalar_t) # type: ignore
        support: scalar_t
        mass: scalar_t
        density: scalar_t
        kind: wp.int32

    @wp.struct
    class WarpParticle_2:
        position: vector(length=2, dtype = scalar_t) # type: ignore
        support: scalar_t
        mass: scalar_t
        density: scalar_t
        kind: wp.int32

    @wp.struct
    class WarpParticle_3:
        position: vector(length=3, dtype = scalar_t) # type: ignore
        support: scalar_t
        mass: scalar_t
        density: scalar_t
        kind: wp.int32
