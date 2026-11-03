"""Rotation and finite-difference oracles for retained/truncated eigenspaces."""

import pytest
import torch
import warp as wp

from warpSPHCore import (
    DomainDescription, KernelFunctions, OperationProperties, ParticleState,
    ParticleTangentState, SupportScheme, WarpOperation, radiusSearchCompactHashMap,
)
from warpSPHCore.renorm import (
    computeRenormalizationMatrices, computeRenormalizationMatricesJVP, _symmetric_pinv_jvp,
)
from warpSPHCore.type_config import scalar_t


DTYPE = wp.dtype_to_torch(scalar_t)


@pytest.mark.parametrize('eigenvalues', [
    [2., 2.], [2., 0.], [2., 1e-6], [-2., -1e-6],
    [2., 2., 2.], [2., 0., 0.], [2., 1., 0.],
    [2., 1e-6, 5e-7], [2., 1., 1e-6],
])
def test_selected_spectral_derivative_against_finite_difference(device, eigenvalues):
    # Independent spectral oracle includes a nonzero truncated eigenvalue;
    # float64 isolates this algebra from Warp's process precision.
    dim = len(eigenvalues)
    dtype = torch.float64
    seed = torch.arange(dim * dim, dtype=dtype, device=device).reshape(dim, dim).sin()
    basis, _ = torch.linalg.qr(seed + torch.eye(dim, dtype=dtype, device=device))
    C = basis @ torch.diag(torch.tensor(eigenvalues, dtype=dtype, device=device)) @ basis.T
    dC = (seed + seed.T) * .15

    def selected_inverse(matrix):
        eig, vectors = torch.linalg.eigh(matrix)
        keep = eig.abs() > 1e-6 * eig.abs().max()
        inverse = torch.where(keep, 1 / torch.where(keep, eig, 1.), 0.)
        return (vectors * inverse) @ vectors.T

    L = selected_inverse(C)
    actual = _symmetric_pinv_jvp(C[None], L[None], dC[None])[0]
    eps = 1e-7
    expected = (selected_inverse(C + eps * dC) - selected_inverse(C - eps * dC)) / (2 * eps)
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize('dim', [1, 2, 3])
def test_repeated_full_rank_derivative_remains_differentiable(device, dim):
    C = (torch.eye(dim, dtype=torch.float64, device=device) * 2).unsqueeze(0).requires_grad_()
    dC = torch.ones_like(C, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda c, dc: _symmetric_pinv_jvp(c, torch.linalg.inv(c), dc), (C, dC))


@pytest.mark.parametrize('dim', [2, 3])
def test_half_precision_local_solve_preserves_tangent_dtype(device, dim):
    C = torch.eye(dim, dtype=torch.float16, device=device)[None] * 2
    L = torch.eye(dim, dtype=torch.float16, device=device)[None] * .5
    dC = torch.ones_like(C)
    result = _symmetric_pinv_jvp(C, L, dC)
    torch.testing.assert_close(result, torch.full_like(C, -.25), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA capture')
def test_projected_derivative_capture_replays_changed_tangent():
    C = torch.diag(torch.tensor([2., 0.], device='cuda'))[None]
    L = torch.diag(torch.tensor([.5, 0.], device='cuda'))[None]
    tangent = torch.tensor([[[0., .3], [.3, 0.]]], device='cuda')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _symmetric_pinv_jvp(C, L, tangent)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = _symmetric_pinv_jvp(C, L, tangent)
    for factor in (1., 2., -1.):
        tangent.copy_(torch.tensor([[[0., .3], [.3, 0.]]], device='cuda') * factor)
        graph.replay()
        torch.testing.assert_close(output, tangent * .25)


@pytest.mark.parametrize('traversal', ['csr', 'grid', 'implicit'])
@pytest.mark.parametrize('thin', [False, True])
def test_rotating_truncated_neighborhood(device, traversal, thin):
    # Seven neighbors avoid the low-count fallback. The thin case has a small
    # NONZERO discarded eigenvalue: the constant-rank MP identity alone would
    # still miss the denominator involving that discarded eigenvalue.
    x = torch.linspace(-.3, .3, 7, dtype=DTYPE, device=device)
    y = x.square() * (1e-3 if thin else 0)
    positions = torch.stack((x, y), dim=-1)
    rotation = positions.new_tensor([[0., -1.], [1., 0.]])
    tangent = positions @ rotation.T
    h = torch.full_like(x, 1.2)
    particles = ParticleState(positions=positions, supports=h,
                              masses=torch.ones_like(x), densities=torch.ones_like(x),
                              kinds=torch.zeros(7, dtype=torch.int32, device=device))
    domain = DomainDescription(min=positions.new_full((2,), -2),
                               max=positions.new_full((2,), 2),
                               periodic=torch.zeros(2, dtype=torch.bool, device=device), dim=2)
    props = OperationProperties(kernel=KernelFunctions.Wendland2,
                                operation=WarpOperation.Gradient,
                                supportMode=SupportScheme.Gather)
    adjacency = radiusSearchCompactHashMap(particles, domain)
    traversal_arg = {'csr': adjacency, 'grid': adjacency.hashMap, 'implicit': None}[traversal]
    C, eig, state, dstate = computeRenormalizationMatricesJVP(
        particles, props, domain, ParticleTangentState(positions=tangent, supports=None, masses=None),
        referenceTangentState=ParticleTangentState(positions=tangent, supports=None, masses=None),
        adjacency=traversal_arg)
    L = state.renormalizationMatrices
    dL = dstate.renormalizationMatrices
    assert bool((eig[:, 1].abs() < 1e-6 * eig[:, 0].abs()).all())
    expected = rotation @ L - L @ rotation
    tol = 3e-5 if DTYPE == torch.float32 else 2e-10
    torch.testing.assert_close(dL, expected, rtol=tol, atol=tol)

    # Reverse-mode dot product from the actual correction construction.
    pos = positions.detach().requires_grad_()
    particles.positions = pos
    primal = computeRenormalizationMatrices(
        particles, props, domain, adjacency=traversal_arg, returnEigVals=False)
    cotangent = torch.arange(L.numel(), dtype=DTYPE, device=device).reshape_as(L).sin()
    vjp, = torch.autograd.grad((primal.renormalizationMatrices * cotangent).sum(), pos)
    torch.testing.assert_close((dL * cotangent).sum(), (vjp * tangent).sum(),
                               rtol=tol, atol=tol)


@pytest.mark.parametrize('count', [1, 3])
def test_low_neighbor_correction_has_zero_tangent(device, count):
    positions = torch.zeros((count, 2), dtype=DTYPE, device=device)
    h = torch.ones(count, dtype=DTYPE, device=device)
    p = ParticleState(positions=positions, supports=h, masses=h, densities=h,
                      kinds=torch.zeros(count, dtype=torch.int32, device=device))
    domain = DomainDescription(min=h.new_full((2,), -2), max=h.new_full((2,), 2),
                               periodic=torch.zeros(2, dtype=torch.bool, device=device), dim=2)
    _, _, state, tangent = computeRenormalizationMatricesJVP(
        p, OperationProperties(kernel=KernelFunctions.Wendland2,
                               operation=WarpOperation.Gradient), domain,
        ParticleTangentState(positions=torch.ones_like(positions), supports=h, masses=None))
    torch.testing.assert_close(state.renormalizationMatrices,
                               torch.eye(2, dtype=DTYPE, device=device).expand(count, 2, 2))
    assert torch.count_nonzero(tangent.renormalizationMatrices) == 0


@pytest.mark.parametrize('rank', [1, 2])
@pytest.mark.parametrize('traversal', ['csr', 'grid', 'implicit'])
def test_rotating_3d_retained_subspace(device, rank, traversal):
    if device.type != 'cuda':
        pytest.skip('3D qualification runs on CUDA')
    axis = torch.linspace(-.3, .3, 3, dtype=DTYPE, device=device)
    xy = torch.cartesian_prod(axis, axis)
    positions = torch.cat((xy, torch.zeros_like(xy[:, :1])), -1)
    if rank == 1:
        positions[:, 1] = 0
    rotation = positions.new_tensor([[0., 0., 1.], [0., 0., 0.], [-1., 0., 0.]])
    h = positions.new_full((len(positions),), 1.2)
    p = ParticleState(positions=positions, supports=h, masses=h, densities=torch.ones_like(h),
                      kinds=torch.zeros(len(h), dtype=torch.int32, device=device))
    domain = DomainDescription(min=h.new_full((3,), -2), max=h.new_full((3,), 2),
                               periodic=torch.zeros(3, dtype=torch.bool, device=device), dim=3)
    adjacency = radiusSearchCompactHashMap(p, domain)
    adj = {'csr': adjacency, 'grid': adjacency.hashMap, 'implicit': None}[traversal]
    tangent = ParticleTangentState(positions=positions @ rotation.T, supports=None, masses=None)
    _, _, state, derivative = computeRenormalizationMatricesJVP(
        p, OperationProperties(kernel=KernelFunctions.Wendland2, operation=WarpOperation.Gradient),
        domain, tangent, referenceTangentState=tangent, adjacency=adj)
    L = state.renormalizationMatrices
    torch.testing.assert_close(derivative.renormalizationMatrices, rotation @ L - L @ rotation,
                               rtol=3e-5 if DTYPE == torch.float32 else 2e-10,
                               atol=3e-5 if DTYPE == torch.float32 else 2e-10)
