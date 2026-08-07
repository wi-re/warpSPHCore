import warp as wp
import torch 
from ...type_config import *
from typing import NamedTuple, Union, Tuple, List, Optional, Any
from warp.types import vector, matrix
from ...math import *
from ...util import *

# Convert Warp arrays back to PyTorch tensors using wp.to_torch() for direct GPU access
from ...dataTypes import *
from ...enumTypes import *
from .util import _minimum_image_delta

# @torch.jit.script # jit script is deprecated :/
def _verlet_validity_metrics(
        queryPositions: torch.Tensor,
        referencePositions: torch.Tensor,
        priorQueryPositions: torch.Tensor,
        priorReferencePositions: torch.Tensor,
        querySupports: torch.Tensor,
        referenceSupports: torch.Tensor,
        supports_a: torch.Tensor,
        supports_b: torch.Tensor,
    periodicity: torch.Tensor,
    domainMin: torch.Tensor,
    domainMax: torch.Tensor,
        verletScale: float,
        support_case: int):
    # Stored supports in the Verlet adjacency were scaled by `verletScale` during build.
    priorQuerySupports = supports_a / verletScale
    priorReferenceSupports = supports_b / verletScale

    delta_a = _minimum_image_delta(queryPositions, priorQueryPositions, periodicity, domainMin, domainMax)
    delta_b = _minimum_image_delta(referencePositions, priorReferencePositions, periodicity, domainMin, domainMax)
    distance_a_max = torch.linalg.vector_norm(delta_a, dim=-1).amax()
    distance_b_max = torch.linalg.vector_norm(delta_b, dim=-1).amax()
    maxDistance = distance_a_max + distance_b_max

    querySupportDeltaMax = torch.abs(priorQuerySupports - querySupports).amax()
    referenceSupportDeltaMax = torch.abs(priorReferenceSupports - referenceSupports).amax()

    queryMinSupport = torch.minimum(priorQuerySupports.amin(), querySupports.amin())
    referenceMinSupport = torch.minimum(priorReferenceSupports.amin(), referenceSupports.amin())

    if support_case == 0:
        supportFactor = querySupportDeltaMax
        minSupport = queryMinSupport
    elif support_case == 1:
        supportFactor = referenceSupportDeltaMax
        minSupport = referenceMinSupport
    else:
        supportFactor = torch.maximum(querySupportDeltaMax, referenceSupportDeltaMax)
        minSupport = torch.minimum(queryMinSupport, referenceMinSupport)

    supportBuffer = (verletScale - 1.0) * minSupport
    # Motion and support drift both consume the same Verlet buffer budget.
    budgetUse = maxDistance + supportFactor
    shouldRebuild = budgetUse > supportBuffer
    return shouldRebuild, maxDistance, supportFactor, minSupport, supportBuffer
