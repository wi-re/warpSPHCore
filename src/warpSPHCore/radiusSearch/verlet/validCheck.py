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
from ...autograd.compileGlue import compileGlue

# @torch.jit.script # jit script is deprecated :/
@compileGlue
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

    # Query and reference are usually the same particle set (the same
    # tensors): the reference half is then the identical computation, reused
    # (bitwise the same numbers, half the kernels of a check that runs up to
    # three times per step).
    sameSet = (queryPositions is referencePositions and priorQueryPositions is priorReferencePositions
               and querySupports is referenceSupports and supports_a is supports_b)

    delta_a = _minimum_image_delta(queryPositions, priorQueryPositions, periodicity, domainMin, domainMax)
    distance_a_max = torch.linalg.vector_norm(delta_a, dim=-1).amax()
    if sameSet:
        distance_b_max = distance_a_max
    else:
        delta_b = _minimum_image_delta(referencePositions, priorReferencePositions, periodicity, domainMin, domainMax)
        distance_b_max = torch.linalg.vector_norm(delta_b, dim=-1).amax()
    maxDistance = distance_a_max + distance_b_max

    querySupportDeltaMax = torch.abs(priorQuerySupports - querySupports).amax()
    queryMinSupport = torch.minimum(priorQuerySupports.amin(), querySupports.amin())
    if sameSet:
        referenceSupportDeltaMax, referenceMinSupport = querySupportDeltaMax, queryMinSupport
    else:
        referenceSupportDeltaMax = torch.abs(priorReferenceSupports - referenceSupports).amax()
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
