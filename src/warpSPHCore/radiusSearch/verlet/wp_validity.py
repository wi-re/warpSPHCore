"""The Verlet validity check as two warp launches (it was ~44 torch ops per call, 180 calls per 30 CompSPH steps).

`verletValidityMetrics` returns what `validCheck._verlet_validity_metrics` returns -- the same decision from the
same quantities -- with the reductions done by atomic max / min on a 6-slot device array:

    slot 0/1  max |minimum-image displacement| of the query / reference positions since the list was built
    slot 2/3  max |prior support - support| of the query / reference set
    slot 4/5  min over (prior support, support) of the query / reference set

(maxima are stored negated so one `atomic_min` initial value of +inf serves all slots). A one-thread kernel then
combines them into the rebuild flag. The torch version stays as the reference the test compares against.

`verletScale` is a float (a launch argument: the original path, bitwise unchanged) or a 0-d / 1-element DEVICE tensor (read by the kernels from
device memory: a scale computed on the device, e.g. a reach that depends on the particles, needs no host read to be used).
"""

import warp as wp
import torch
from typing import Any
from warp.types import vector

from ...type_config import *
from ...math import *
from ...util import *
from ...dataTypes import *
from ...autograd.cache import getCachedDummyTensor


@wp.kernel
def _validityReduce(
        positions: wp.array(dtype=vector(dtype=scalar_t, length=Any)),          # type: ignore
        priorPositions: wp.array(dtype=vector(dtype=scalar_t, length=Any)),     # type: ignore
        supports: wp.array(dtype=scalar_t), priorSupportsScaled: wp.array(dtype=scalar_t),   # type: ignore
        domainMin: wp.array(dtype=scalar_t), domainMax: wp.array(dtype=scalar_t),            # type: ignore
        periodicity: wp.array(dtype=wp.bool),                                                # type: ignore
        verletScale: scalar_t, scaleArr: wp.array(dtype=scalar_t), useArr: wp.int32, slot: wp.int32,
        acc: wp.array(dtype=scalar_t)):                                                      # type: ignore
    i = wp.tid()
    domainState = domainData()
    domainState.domainMin = domainMin
    domainState.domainMax = domainMax
    domainState.periodicity = periodicity
    domainState.dim = wp.int32(domainMin.shape[0])
    d = computeDistanceVec(positions[i], priorPositions[i], domainState)
    dist = wp.sqrt(wp.dot(d, d))
    scale = verletScale
    if useArr == 1:
        scale = scaleArr[0]
    prior = priorSupportsScaled[i] / scale
    h = supports[i]
    wp.atomic_min(acc, slot, -dist)                                  # max distance, negated
    wp.atomic_min(acc, slot + 2, -wp.abs(prior - h))                 # max support change, negated
    wp.atomic_min(acc, slot + 4, wp.min(prior, h))                   # min support


@wp.kernel
def _validityDecide(acc: wp.array(dtype=scalar_t), verletScale: scalar_t, scaleArr: wp.array(dtype=scalar_t), useArr: wp.int32,
                    supportCase: wp.int32,   # type: ignore
                    rebuildSlack: scalar_t, sameSet: wp.int32,
                    out: wp.array(dtype=scalar_t), flag: wp.array(dtype=wp.bool)):                   # type: ignore
    distA = -acc[0]
    distB = -acc[1]
    if sameSet == 1:
        distB = distA
    dSupA = -acc[2]
    dSupB = -acc[3]
    minA = acc[4]
    minB = acc[5]
    if sameSet == 1:
        dSupB = dSupA
        minB = minA
    maxDistance = distA + distB
    supportFactor = dSupA
    minSupport = minA
    if supportCase == 1:
        supportFactor = dSupB
        minSupport = minB
    elif supportCase == 2:
        supportFactor = wp.max(dSupA, dSupB)
        minSupport = wp.min(minA, minB)
    scale = verletScale
    if useArr == 1:
        scale = scaleArr[0]
    buffer = (scale - scalar_t(1.0)) * minSupport
    out[0] = maxDistance
    out[1] = supportFactor
    out[2] = minSupport
    out[3] = buffer
    flag[0] = (maxDistance + supportFactor) > buffer * (scalar_t(1.0) - rebuildSlack)


def verletValidityMetrics(queryPositions, referencePositions, priorQueryPositions, priorReferencePositions,
                          querySupports, referenceSupports, supports_a, supports_b,
                          periodicity, domainMin, domainMax, verletScale, support_case, rebuildSlack=0.0):
    """``(shouldRebuild, maxDistance, supportFactor, minSupport, supportBuffer)``, all 0-d tensors (as
    `_verlet_validity_metrics`)."""
    device = queryPositions.device
    if (queryPositions.shape != priorQueryPositions.shape or
            referencePositions.shape != priorReferencePositions.shape):
        zero = querySupports.new_zeros(())
        return torch.ones((), dtype=torch.bool, device=device), zero, zero, zero, zero
    if queryPositions.shape[0] == 0 or referencePositions.shape[0] == 0:
        # There are no pairs to invalidate while either population stays
        # empty. Population-count changes are handled by the rebuild caller.
        zero = querySupports.new_zeros(())
        flag = torch.zeros((), dtype=torch.bool, device=device)
        return flag, zero, zero, zero, zero
    sameSet = (queryPositions is referencePositions and priorQueryPositions is priorReferencePositions
               and querySupports is referenceSupports and supports_a is supports_b)
    acc = torch.full((6,), float('inf'), dtype=querySupports.dtype, device=device)
    out = torch.empty((4,), dtype=querySupports.dtype, device=device)
    flag = torch.empty((1,), dtype=torch.bool, device=device)
    wpDevice = wp.device_from_torch(device)
    stream = wp.stream_from_torch(torch.cuda.current_stream(device)) if device.type == 'cuda' else None
    accW = wp.from_torch(acc)
    if isinstance(verletScale, torch.Tensor):
        scaleArr, useArr, scaleValue = wp.from_torch(verletScale.detach().reshape(1).to(querySupports.dtype).contiguous()), 1, 1.0
    else:
        scaleArr, useArr, scaleValue = wp.from_torch(getCachedDummyTensor((1,), dtype=querySupports.dtype, device=device)), 0, verletScale
    dom = (castTorchToWarp(domainMin), castTorchToWarp(domainMax), castTorchToWarp(periodicity))
    sets = [(queryPositions, priorQueryPositions, querySupports, supports_a, 0)]
    if not sameSet:
        sets.append((referencePositions, priorReferencePositions, referenceSupports, supports_b, 1))
    for pos, prior, sup, supPrior, slot in sets:
        wp.launch(_validityReduce, dim=pos.shape[0], device=wpDevice, stream=stream, inputs=[
            castTorchToWarpAsBuiltins(pos), castTorchToWarpAsBuiltins(prior),
            castTorchToWarp(sup), castTorchToWarp(supPrior), *dom,
            scalar_t(scaleValue), scaleArr, wp.int32(useArr), wp.int32(slot), accW])
    wp.launch(_validityDecide, dim=1, device=wpDevice, stream=stream, inputs=[
        accW, scalar_t(scaleValue), scaleArr, wp.int32(useArr), wp.int32(support_case), scalar_t(rebuildSlack), wp.int32(1 if sameSet else 0),
        wp.from_torch(out), wp.from_torch(flag)])
    maxDistance, supportFactor, minSupport, supportBuffer = out.unbind(0)
    return flag.reshape(()), maxDistance, supportFactor, minSupport, supportBuffer
