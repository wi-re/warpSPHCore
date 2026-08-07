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