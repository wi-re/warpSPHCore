"""Public CSR dispatch and its existing backward, against independent equations."""
from dataclasses import replace

import pytest
import torch
import warpSPHCore as sph

from _support.operators import dense, fixture


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('rank', [0, 1, 2])
@pytest.mark.parametrize('operation', [sph.WarpOperation.Density,
                                      sph.WarpOperation.Interpolate,
                                      sph.WarpOperation.Gradient])
@pytest.mark.parametrize('filtered', [False, True])
@pytest.mark.parametrize('index_dtype,offset_dtype', [
    (torch.int32, torch.int32), (torch.int32, torch.int64),
    (torch.int64, torch.int32), (torch.int64, torch.int64)])
def test_prepared_csr_public_primal_and_backward(
        device, dtype, dimension, rank, operation, filtered, index_dtype, offset_dtype):
    check_public_csr(device, dtype, dimension, rank, operation, filtered,
        index_dtype, offset_dtype, sph.KernelFunctions.Wendland2,
        sph.SupportScheme.Gather, sph.GradientScheme.Difference)


def check_public_csr(device, dtype, dimension, rank, operation, filtered,
                     index_dtype, offset_dtype, kernel, support, scheme):
    query, reference, domain, qfield, rfield = fixture(device, dimension, rank)
    # Reuse the independent FP64 oracle on the actual rounded inputs. Include
    # geometry/domain gradients so a new primal cannot silently break the
    # established backward even though this change adds no new AD registration.
    values = [query.positions, reference.positions, query.supports, reference.supports,
              reference.masses, query.densities, reference.densities,
              qfield, rfield, domain.min, domain.max]
    # Cusp/singular kernel backward support is unchanged by primal migration.
    backward = kernel.value in {0, 1, 2, 20, 30, 31, 32, 33, 34, 36, 41}
    inputs = [value.to(dtype).detach().requires_grad_(backward) for value in values]
    with torch.no_grad():
        inputs[0][:, 0].add_(6)
        inputs[1][:, 0].sub_(4)
    query = replace(query, positions=inputs[0], supports=inputs[2],
                    masses=query.masses.to(dtype), densities=inputs[5])
    reference = replace(reference, positions=inputs[1], supports=inputs[3],
                        masses=inputs[4], densities=inputs[6])
    domain = replace(domain, min=inputs[9], max=inputs[10])
    qfield, rfield = inputs[7:9]

    # Planning is explicit and outside the operator. Reverse reference order
    # within rows, leave an unused prefix, and vary offset/ID widths separately.
    displacement = query.positions.detach()[:, None] - reference.positions.detach()[None]
    length = (domain.max - domain.min).detach()
    displacement = torch.where(domain.periodic,
        torch.remainder(displacement + length / 2, length) - length / 2, displacement)
    hi, hj = query.supports.detach()[:, None], reference.supports.detach()[None]
    radius = {11: hi, 12: hj, 13: (hi + hj) / 2,
              14: torch.maximum(hi, hj), 15: torch.maximum(hi, hj)}[support.value]
    membership = displacement.norm(dim=-1) < radius
    counts = membership.sum(1).to(torch.int32)
    _, reversed_ids = torch.where(membership.flip(1))
    ids = (len(reference.positions) - 1 - reversed_ids).to(index_dtype)
    offsets = (counts.long().cumsum(0) - counts + 5).to(offset_dtype)
    adjacency = sph.AdjacencyList(None,
        torch.cat([ids.new_zeros(5), ids]), counts, offsets,
        len(query.positions), len(reference.positions), numEdges=ids.numel())
    properties = sph.OperationProperties(kernel,
        operation=operation, gradientMode=scheme,
        supportMode=support,
        operationMode=(sph.OperationDirection.FluidToBoundary if filtered
                       else sph.OperationDirection.TrueAllToToAll))
    actual = sph.warpOperation(query, properties, domain, qfield, rfield,
        referenceParticles=reference, adjacency=adjacency)

    q64 = replace(query, positions=inputs[0].double(), supports=inputs[2].double(),
                  masses=query.masses.double(), densities=inputs[5].double())
    r64 = replace(reference, positions=inputs[1].double(), supports=inputs[3].double(),
                  masses=inputs[4].double(), densities=inputs[6].double())
    d64 = replace(domain, min=inputs[9].double(), max=inputs[10].double())
    expected = dense(q64, r64, d64, properties, qfield.double(), rfield.double(),
                     differentiable=True)
    tolerance = 3e-4 if dtype == torch.float32 else 3e-9
    torch.testing.assert_close(actual.double(), expected, rtol=tolerance, atol=tolerance)
    if not backward:
        return
    cotangent = torch.cos(torch.arange(actual.numel(), device=device, dtype=dtype)).reshape_as(actual)
    gradients = torch.autograd.grad(actual, inputs, cotangent, allow_unused=True)
    expected_gradients = torch.autograd.grad(expected, inputs, cotangent.double(), allow_unused=True)
    for source, left, right in zip(inputs, gradients, expected_gradients):
        left = torch.zeros_like(source) if left is None else left
        right = torch.zeros_like(source) if right is None else right
        torch.testing.assert_close(left, right, rtol=tolerance, atol=tolerance)


@pytest.mark.extended
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('operation', [sph.WarpOperation.Density,
                                      sph.WarpOperation.Interpolate,
                                      sph.WarpOperation.Gradient])
@pytest.mark.parametrize('support', list(sph.SupportScheme)[:-1])
@pytest.mark.parametrize('kernel', list(sph.KernelFunctions))
def test_general_csr_kernel_and_support_coverage(device, dtype, dimension,
                                                operation, support, kernel):
    # Exhaust kernel/support/dimension/precision combinations while rotating
    # rank, filtering and independently chosen index widths. The prepared test
    # above separately exhausts all combinations of those layout dimensions.
    case = kernel.value + support.value + dimension + operation.value
    check_public_csr(device, dtype, dimension, case % 3, operation, bool(case % 2),
        torch.int64 if case % 4 < 2 else torch.int32,
        torch.int64 if case % 2 else torch.int32, kernel, support,
        sph.GradientScheme.Difference)


@pytest.mark.extended
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('rank', [0, 1, 2])
@pytest.mark.parametrize('support', list(sph.SupportScheme)[:-1])
@pytest.mark.parametrize('scheme', list(sph.GradientScheme))
def test_general_csr_gradient_schemes(device, dtype, dimension, rank, support, scheme):
    kernels = [sph.KernelFunctions.Wendland2, sph.KernelFunctions.CubicSpline,
               sph.KernelFunctions.Wendland4, sph.KernelFunctions.Poly6]
    case = dimension + rank + support.value + scheme.value
    check_public_csr(device, dtype, dimension, rank, sph.WarpOperation.Gradient,
        bool(case % 2), torch.int32 if case % 2 else torch.int64,
        torch.int64 if case % 3 else torch.int32, kernels[case % len(kernels)],
        support, scheme)
