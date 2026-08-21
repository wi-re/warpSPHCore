from dataclasses import dataclass
import torch
import warp as wp
from typing import NamedTuple, Union, Tuple, List, Optional, Any
from warp.types import vector, matrix
from ..enumTypes import *
from ..type_config import *

@dataclass
class CRKState:
    A: torch.Tensor # shape [N]
    B: torch.Tensor # shape [N,D]
    gradA: torch.Tensor # shape [N,D]
    gradB: torch.Tensor # shape [N,D,D]

@dataclass
class GradHState:
    queryOmegas: torch.Tensor # shape [N]
    referenceOmegas : Optional[torch.Tensor] = None # shape [N], if None then queryOmegas are used as referenceOmegas

@dataclass
class RenormalizationState:
    renormalizationMatrices: torch.Tensor # shape [N,D,D]

@dataclass
class CRKTangentState:
    """Bundled JVP tangent counterpart to `CRKState` (`warpier_tier2_correction_jvp_plan.md`
    phase a2). No `Optional` fields, unlike `ParticleTangentState` -- a caller
    supplying CRK correction tangent support supplies all four."""
    A: torch.Tensor        # shape [N]
    B: torch.Tensor        # shape [N,D]
    gradA: torch.Tensor    # shape [N,D]
    gradB: torch.Tensor    # shape [N,D,D]

@dataclass
class RenormalizationTangentState:
    """Bundled JVP tangent counterpart to `RenormalizationState`
    (`warpier_tier2_correction_jvp_plan.md` phase a2)."""
    renormalizationMatrices: torch.Tensor  # shape [N,D,D]


@wp.struct
class correctionData_1:
    useGradientRenormalization: wp.bool
    renormalizationMatrices: wp.array(dtype=matrix(shape=(1,1), dtype=scalar_t)) # type: ignore
    useVolume: wp.bool
    queryVolumes: wp.array(dtype = scalar_t) # type: ignore
    referenceVolumes: wp.array(dtype = scalar_t) # type: ignore
    useGradHTerms: wp.bool
    queryOmegas: wp.array(dtype = scalar_t)  # type: ignore
    referenceOmegas: wp.array(dtype = scalar_t) # type: ignore
    useCRK: wp.bool
    queryA: wp.array(dtype = scalar_t)  # type: ignore
    queryB: wp.array(dtype=vector(length=1, dtype=scalar_t))  # type: ignore
    queryGradA: wp.array(dtype = vector(length=1, dtype=scalar_t))  # type: ignore
    queryGradB: wp.array(dtype=matrix(shape=(1,1), dtype=scalar_t)) # type: ignore
    referenceA: wp.array(dtype = scalar_t)  # type: ignore
    referenceB: wp.array(dtype=vector(length=1, dtype=scalar_t))  # type: ignore
    referenceGradA: wp.array(dtype = vector(length=1, dtype=scalar_t))  # type: ignore
    referenceGradB: wp.array(dtype=matrix(shape=(1,1), dtype=scalar_t)) # type: ignore

@wp.struct
class correctionData_2:
    useGradientRenormalization: wp.bool
    renormalizationMatrices: wp.array(dtype=matrix(shape=(2,2), dtype=scalar_t)) # type: ignore
    useVolume: wp.bool
    queryVolumes: wp.array(dtype = scalar_t) # type: ignore
    referenceVolumes: wp.array(dtype = scalar_t) # type: ignore
    useGradHTerms: wp.bool
    queryOmegas: wp.array(dtype = scalar_t) # type: ignore
    referenceOmegas: wp.array(dtype = scalar_t) # type: ignore
    useCRK: wp.bool
    queryA: wp.array(dtype = scalar_t)  # type: ignore
    queryB: wp.array(dtype=vector(length=2, dtype=scalar_t))  # type: ignore
    queryGradA: wp.array(dtype=vector(length=2, dtype=scalar_t))  # type: ignore
    queryGradB: wp.array(dtype=matrix(shape=(2,2), dtype=scalar_t)) # type: ignore
    referenceA: wp.array(dtype = scalar_t)  # type: ignore
    referenceB: wp.array(dtype=vector(length=2, dtype=scalar_t))  # type: ignore
    referenceGradA: wp.array(dtype = vector(length=2, dtype=scalar_t))  # type: ignore
    referenceGradB: wp.array(dtype=matrix(shape=(2,2), dtype=scalar_t)) # type: ignore

@wp.struct
class correctionData_3:
    useGradientRenormalization: wp.bool
    renormalizationMatrices: wp.array(dtype=matrix(shape=(3,3), dtype=scalar_t)) # type: ignore
    useVolume: wp.bool
    queryVolumes: wp.array(dtype = scalar_t)  # type: ignore
    referenceVolumes: wp.array(dtype = scalar_t) # type: ignore
    useGradHTerms: wp.bool
    queryOmegas: wp.array(dtype = scalar_t) # type: ignore
    referenceOmegas: wp.array(dtype = scalar_t) # type: ignore
    useCRK: wp.bool
    queryA: wp.array(dtype=scalar_t)  # type: ignore
    queryB: wp.array(dtype=vector(length=3, dtype=scalar_t))  # type: ignore
    queryGradA: wp.array(dtype=vector(length=3, dtype=scalar_t))  # type: ignore
    queryGradB: wp.array(dtype=matrix(shape=(3,3), dtype=scalar_t)) # type: ignore
    referenceA: wp.array(dtype=scalar_t)  # type: ignore
    referenceB: wp.array(dtype=vector(length=3, dtype=scalar_t))  # type: ignore
    referenceGradA: wp.array(dtype=vector(length=3, dtype=scalar_t))  # type: ignore
    referenceGradB: wp.array(dtype=matrix(shape=(3,3), dtype=scalar_t)) # type: ignore

# Parallel tangent counterpart to correctionData_{dim} -- a complete field-for-field
# mirror (renorm/volume/grad-H/CRK), minus the useX bool flags, matching how
# ParticleTangentState mirrors ParticleState field-for-field (`warpier_tier2_correction_jvp_plan.md`
# phase a2). Deliberately a separate struct rather than extending correctionData_{dim} itself:
# that struct is shared with every primal production kernel, so adding unused tangent fields to
# it would grow every non-JVP kernel's ABI for no reason. Gating (whether these arrays are
# actually read) is driven entirely by the paired correctionData_{dim}'s own useVolume/
# useGradHTerms/useGradientRenormalization/useCRK flags -- there is no separate enable flag here.
# Every field is present even where nothing consumes it yet (grad-H has no JVP wiring at all,
# `warpOperationJVP` still rejects `gradHState` outright) so that a downstream caller adding a new
# correction's tangent support never needs to widen this struct's shape or re-touch every JVP
# kernel's ABI a second time -- deliberately the same "canonical, unconditionally-present" shape
# `correctionData` itself already has across every JVP kernel (Density/Interpolate included, even
# though most never read it).
@wp.struct
class correctionTangentData_1:
    renormalizationMatrices: wp.array(dtype=matrix(shape=(1,1), dtype=scalar_t)) # type: ignore
    queryVolumes: wp.array(dtype = scalar_t) # type: ignore
    referenceVolumes: wp.array(dtype = scalar_t) # type: ignore
    queryOmegas: wp.array(dtype = scalar_t) # type: ignore
    referenceOmegas: wp.array(dtype = scalar_t) # type: ignore
    queryA: wp.array(dtype = scalar_t)  # type: ignore
    queryB: wp.array(dtype=vector(length=1, dtype=scalar_t))  # type: ignore
    queryGradA: wp.array(dtype = vector(length=1, dtype=scalar_t))  # type: ignore
    queryGradB: wp.array(dtype=matrix(shape=(1,1), dtype=scalar_t)) # type: ignore
    referenceA: wp.array(dtype = scalar_t)  # type: ignore
    referenceB: wp.array(dtype=vector(length=1, dtype=scalar_t))  # type: ignore
    referenceGradA: wp.array(dtype = vector(length=1, dtype=scalar_t))  # type: ignore
    referenceGradB: wp.array(dtype=matrix(shape=(1,1), dtype=scalar_t)) # type: ignore

@wp.struct
class correctionTangentData_2:
    renormalizationMatrices: wp.array(dtype=matrix(shape=(2,2), dtype=scalar_t)) # type: ignore
    queryVolumes: wp.array(dtype = scalar_t) # type: ignore
    referenceVolumes: wp.array(dtype = scalar_t) # type: ignore
    queryOmegas: wp.array(dtype = scalar_t) # type: ignore
    referenceOmegas: wp.array(dtype = scalar_t) # type: ignore
    queryA: wp.array(dtype = scalar_t)  # type: ignore
    queryB: wp.array(dtype=vector(length=2, dtype=scalar_t))  # type: ignore
    queryGradA: wp.array(dtype=vector(length=2, dtype=scalar_t))  # type: ignore
    queryGradB: wp.array(dtype=matrix(shape=(2,2), dtype=scalar_t)) # type: ignore
    referenceA: wp.array(dtype = scalar_t)  # type: ignore
    referenceB: wp.array(dtype=vector(length=2, dtype=scalar_t))  # type: ignore
    referenceGradA: wp.array(dtype = vector(length=2, dtype=scalar_t))  # type: ignore
    referenceGradB: wp.array(dtype=matrix(shape=(2,2), dtype=scalar_t)) # type: ignore

@wp.struct
class correctionTangentData_3:
    renormalizationMatrices: wp.array(dtype=matrix(shape=(3,3), dtype=scalar_t)) # type: ignore
    queryVolumes: wp.array(dtype = scalar_t) # type: ignore
    referenceVolumes: wp.array(dtype = scalar_t) # type: ignore
    queryOmegas: wp.array(dtype = scalar_t) # type: ignore
    referenceOmegas: wp.array(dtype = scalar_t) # type: ignore
    queryA: wp.array(dtype=scalar_t)  # type: ignore
    queryB: wp.array(dtype=vector(length=3, dtype=scalar_t))  # type: ignore
    queryGradA: wp.array(dtype=vector(length=3, dtype=scalar_t))  # type: ignore
    queryGradB: wp.array(dtype=matrix(shape=(3,3), dtype=scalar_t)) # type: ignore
    referenceA: wp.array(dtype=scalar_t)  # type: ignore
    referenceB: wp.array(dtype=vector(length=3, dtype=scalar_t))  # type: ignore
    referenceGradA: wp.array(dtype = vector(length=3, dtype=scalar_t))  # type: ignore
    referenceGradB: wp.array(dtype=matrix(shape=(3,3), dtype=scalar_t)) # type: ignore

@wp.struct
class ParticleCorrectionData_1:
    renormalizationMatrix: matrix(shape=(1,1), dtype=scalar_t)
    volume: scalar_t
    omega: scalar_t
    A: scalar_t
    B: vector(length=1, dtype=scalar_t)
    gradA: vector(length=1, dtype=scalar_t)
    gradB: matrix(shape=(1,1), dtype=scalar_t)

@wp.struct
class ParticleCorrectionData_2:
    renormalizationMatrix: matrix(shape=(2,2), dtype=scalar_t)
    volume: scalar_t
    omega: scalar_t
    A: scalar_t
    B: vector(length=2, dtype=scalar_t)
    gradA: vector(length=2, dtype=scalar_t)
    gradB: matrix(shape=(2,2), dtype=scalar_t)

@wp.struct
class ParticleCorrectionData_3:
    renormalizationMatrix: matrix(shape=(3,3), dtype=scalar_t)
    volume: scalar_t
    omega: scalar_t
    A: scalar_t
    B: vector(length=3, dtype=scalar_t)
    gradA: vector(length=3, dtype=scalar_t)
    gradB: matrix(shape=(3,3), dtype=scalar_t)

# Per-query-particle tangent bundle returned by getParticleCorrectionTangentData_i, a complete
# field-for-field mirror of ParticleCorrectionData_{dim} (including volume/omega, even though
# nothing consumes the omega field yet -- grad-H has no JVP wiring at all).
@wp.struct
class ParticleCorrectionTangentData_1:
    renormalizationMatrix: matrix(shape=(1,1), dtype=scalar_t)
    volume: scalar_t
    omega: scalar_t
    A: scalar_t
    B: vector(length=1, dtype=scalar_t)
    gradA: vector(length=1, dtype=scalar_t)
    gradB: matrix(shape=(1,1), dtype=scalar_t)

@wp.struct
class ParticleCorrectionTangentData_2:
    renormalizationMatrix: matrix(shape=(2,2), dtype=scalar_t)
    volume: scalar_t
    omega: scalar_t
    A: scalar_t
    B: vector(length=2, dtype=scalar_t)
    gradA: vector(length=2, dtype=scalar_t)
    gradB: matrix(shape=(2,2), dtype=scalar_t)

@wp.struct
class ParticleCorrectionTangentData_3:
    renormalizationMatrix: matrix(shape=(3,3), dtype=scalar_t)
    volume: scalar_t
    omega: scalar_t
    A: scalar_t
    B: vector(length=3, dtype=scalar_t)
    gradA: vector(length=3, dtype=scalar_t)
    gradB: matrix(shape=(3,3), dtype=scalar_t)