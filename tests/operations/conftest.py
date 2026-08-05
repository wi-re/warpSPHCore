import math

import pytest
import torch
import warp as wp

from sphWarpCore import (
    ParticleState,
    OperationProperties,
    radiusSearchCompactHashMap,
    warpOperation,
)
from sphWarpCore.crk import computeCRKFactors
from sphWarpCore.enumTypes import (
    GradientScheme,
    KernelFunctions,
    LaplacianScheme,
    OperationDirection,
    SupportScheme,
    WarpOperation,
)
from sphWarpCore.renorm import computeRenormalizationMatrices
from sphWarpCore.state import RenormalizationState
from sphWarpCore.util import generateNeighborTestData


@pytest.fixture(scope="session", autouse=True)
def _init_warp_once():
    # Warp must be initialized once before kernel launches.
    wp.init()


@pytest.fixture(scope="session", params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


@pytest.fixture(scope="session")
def kernel():
    # Wendland2 is used in most repo notebooks and examples.
    return KernelFunctions.Wendland2


@pytest.fixture(scope="session")
def particle_case(device):
    dim = 2
    nx = 32
    target_neighbors = 55

    positions, supports, _, domain, dx = generateNeighborTestData(
        nx, target_neighbors, dim, True, device
    )

    masses = torch.full((positions.shape[0],), dx ** dim, device=device, dtype=positions.dtype)
    kinds = torch.zeros(positions.shape[0], device=device, dtype=torch.int32)

    particles = ParticleState(
        positions=positions.contiguous(),
        supports=supports.contiguous(),
        masses=masses.contiguous(),
        densities=None,
        kinds=kinds,
    )

    adjacency = radiusSearchCompactHashMap(
        particles,
        domain,
        mode=SupportScheme.SuperSymmetric,
    )

    densities = warpOperation(
        particles,
        OperationProperties(
            kernel=KernelFunctions.Wendland2,
            operation=WarpOperation.Density,
            supportMode=SupportScheme.Gather,
            operationMode=OperationDirection.AllToAll,
        ),
        domain,
        adjacency=adjacency,
    )
    particles.densities = densities

    return {
        "particles": particles,
        "domain": domain,
        "adjacency": adjacency,
        "dx": dx,
        "dim": dim,
    }


def op(
    case,
    operation,
    query_values=None,
    reference_values=None,
    gradient_mode=GradientScheme.Difference,
    laplacian_mode=LaplacianScheme.Default,
    support_mode=SupportScheme.Gather,
    operation_mode=OperationDirection.AllToAll,
    crk_state=None,
    renorm_state=None,
    consistent_divergence=False,
    traversal="adjacency",
):
    particles = case["particles"]
    if traversal == "adjacency":
        adjacency = case["adjacency"]
    elif traversal == "grid":
        # adjacency=None routes sphOperation_warp -> sphOperation_warp_grid,
        # building a CompactHashMap on the fly for this call.
        adjacency = None
    else:
        raise ValueError(f"Unknown traversal mode: {traversal!r}")
    return warpOperation(
        particles,
        OperationProperties(
            kernel=KernelFunctions.Wendland2,
            operation=operation,
            supportMode=support_mode,
            operationMode=operation_mode,
            gradientMode=gradient_mode,
            laplacianMode=laplacian_mode,
        ),
        case["domain"],
        adjacency=adjacency,
        queryValues=query_values,
        referenceValues=reference_values,
        crkState=crk_state,
        renormalizationState=renorm_state,
        consistentDivergence=consistent_divergence,
    )


def interior_mask(case, band_cells=3.0):
    x = case["particles"].positions
    domain = case["domain"]
    dx = case["dx"]
    band = float(band_cells) * float(dx)

    mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    for axis in range(x.shape[1]):
        mask = mask & (x[:, axis] > domain.min[axis] + band)
        mask = mask & (x[:, axis] < domain.max[axis] - band)
    return mask


def linear_scalar_field(case, ax=5.0, by=3.0, c=0.0):
    x = case["particles"].positions
    return ax * x[:, 0] + by * x[:, 1] + c


def linear_vector_field(case, a=2.0, b=-1.0, c=4.0, d=3.0):
    x = case["particles"].positions
    vx = a * x[:, 0] + b * x[:, 1]
    vy = c * x[:, 0] + d * x[:, 1]
    return torch.stack((vx, vy), dim=1)


def matrix_field(case):
    x = case["particles"].positions
    m = torch.empty((x.shape[0], 2, 2), dtype=x.dtype, device=x.device)
    m[:, 0, 0] = 1.2 * x[:, 0]
    m[:, 0, 1] = -0.3 * x[:, 1]
    m[:, 1, 0] = 0.5 * x[:, 1]
    m[:, 1, 1] = -0.9 * x[:, 0]
    return m


def crk_state(case, kernel):
    _, _, crk = computeCRKFactors(
        queryParticles=case["particles"],
        domain=case["domain"],
        kernel=kernel,
        operationMode=OperationDirection.AllToAll,
        adjacency=case["adjacency"],
    )
    return crk


def renorm_state(case, kernel):
    result = computeRenormalizationMatrices(
        queryParticles=case["particles"],
        operationProperties=OperationProperties(
            kernel=kernel,
            operation=WarpOperation.Gradient,
            supportMode=SupportScheme.Gather,
            operationMode=OperationDirection.AllToAll,
            gradientMode=GradientScheme.Difference,
        ),
        domain=case["domain"],
        adjacency=case["adjacency"],
        returnEigVals=False,
    )
    if isinstance(result, RenormalizationState):
        return result
    # Defensive fallback if API changes and returns tuple in future.
    if isinstance(result, tuple):
        return result[-1]
    raise AssertionError("Unexpected renormalization return type")


def mean_abs_error(actual, expected, mask):
    diff = torch.abs(actual - expected)
    if diff.ndim > 1:
        diff = torch.sum(diff, dim=tuple(range(1, diff.ndim))) / float(math.prod(diff.shape[1:]))
    return torch.mean(diff[mask]).item()
