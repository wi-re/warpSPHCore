from dataclasses import dataclass
import torch
import warp as wp
from typing import NamedTuple, Union, Tuple, List, Optional, Any
from warp.types import vector, matrix
from ..type_config import *


# @torch.jit.script
@dataclass#(slots=True)
class PointCloud:
    """
    A named tuple containing the positions of the particles and the number of particles.
    """
    positions: torch.Tensor
    supports: torch.Tensor

    def __ne__(self, other: 'PointCloud') -> bool:
        return not self.__eq__(other)

@dataclass
class ParticleState:
    positions: torch.Tensor # shape [N,D] # D is the dimensionality of the simulation, e.g. 2 for 2D, 3 for 3D. This tensor holds the spatial coordinates of each particle in the simulation. For periodic domains these positions can be provided in the original unwrapped form, and the kernel computations will handle the periodicity based on the domain information provided separately.
    supports: torch.Tensor # Effective radius or smoothing length for each particle, which can be used to determine the influence radius for neighbor interactions. This is a crucial property for SPH computations as it defines the spatial extent of each particle's influence. Shape [N]
    masses: torch.Tensor # Shape [N]
    kinds: torch.Tensor # Particle types/categories, used e.g. for directional interactions or multi-material simulations. Shape [N], int32. Required: every operator kernel indexes SoA.kinds[i] unconditionally (see util/stateUtil.py's getParticleData), and OperationDirection.AllToAll used to paper over a missing kinds tensor with an N-sized zero dummy -- see warpier_fields.md Section 2.5.
    densities: Optional[torch.Tensor] = None # Densities are optional to allow for density computations which do not require pre-computed densities.

@dataclass
class ParticleTangentState:
    """Bundled JVP tangent counterpart to `ParticleState`, minus `kinds`
    (categorical, no tangent). Used by `warpOperationJVP`'s geometry-tangent
    path (`_jvpCommon.py`'s `launchGeometryJVP`, every `wp_<op>JVP.py`'s
    `computeSPH<Op>GeometryJVP`) in place of the loose parallel
    `tangentQuery*`/`tangentReference*` tensor kwargs it replaced
    (`warpier_tier2_correction_jvp_plan.md` phase a1)."""
    positions: torch.Tensor # [N,D]
    supports: torch.Tensor # [N]
    masses: torch.Tensor # [N]
    densities: Optional[torch.Tensor] = None # [N]

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
