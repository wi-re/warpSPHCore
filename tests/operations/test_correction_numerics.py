"""Fixed-branch selected inverse AD and augmented CRK moment degeneracy."""

from dataclasses import replace
from itertools import permutations

import pytest
import torch

from warpSPHCore import (DomainDescription, ParticleState, ParticleTangentState,
    OperationProperties, WarpOperation, KernelFunctions, SupportScheme,
    GradientScheme, OperationDirection, radiusSearchCompactHashMap, warpOperation, warpOperationJVP)
from warpSPHCore.pinv.symmetric import selectedSymmetricInverse, _symmetric_pinv_jvp
from warpSPHCore.crk.crk_terms import computeCRKTermsWarp
from warpSPHCore.crk import computeCRKFactors, computeCRKFactorsJVP
from warpSPHCore.renorm import computeRenormalizationMatrices, computeRenormalizationMatricesJVP
from warpSPHCore.type_config import get_torch_precision


@pytest.mark.parametrize('spectrum', list(permutations([1., -3., 2.])))
def test_3d_inverse_diagnostics_keep_sign_and_order_by_magnitude(device, spectrum):
    from warpSPHCore.pinv import pinv_warp
    matrix = torch.diag(torch.tensor(spectrum, dtype=get_torch_precision(), device=device))[None].requires_grad_()
    _, eigenvalues = pinv_warp(matrix, torch.tensor([9], dtype=torch.int32, device=device), symmetric=True)
    torch.testing.assert_close(eigenvalues, matrix.new_tensor([[-3., 2., 1.]]), rtol=0, atol=0)
    gradient, = torch.autograd.grad(eigenvalues.sum(), matrix)
    torch.testing.assert_close(gradient, torch.eye(3, dtype=matrix.dtype, device=device)[None], rtol=0, atol=0)


def spectral_inverse(matrix):
    eigenvalues, basis = torch.linalg.eigh((matrix + matrix.mT) * .5)
    keep = eigenvalues.abs() > eigenvalues.abs().amax(-1, keepdim=True) * 1e-6
    inverse = torch.where(keep, 1 / torch.where(keep, eigenvalues, 1.), 0.)
    return (basis * inverse.unsqueeze(-2)) @ basis.mT


@pytest.mark.parametrize('spectrum', [[2., 2., 2.], [2., 0., 0.], [2., 1., 0.],
    [2., 1.8e-6, 9e-7], [2., 1., 1.8e-6], [-2., -1., -1.8e-6], [0., 0., 0.]])
def test_selected_inverse_primal_reverse_and_dual(device, spectrum):
    from torch.autograd import forward_ad
    dtype = torch.float64  # Tiny Torch algebra is independent of Warp precision.
    seed = torch.arange(9, dtype=dtype, device=device).reshape(3, 3).cos()
    basis, _ = torch.linalg.qr(seed + torch.eye(3, dtype=dtype, device=device))
    matrix = (basis @ torch.diag(seed.new_tensor(spectrum)) @ basis.T)[None].requires_grad_()
    tangent = (seed + seed.T)[None] * .1
    cotangent = seed.sin()[None]
    actual = selectedSymmetricInverse(matrix)
    torch.testing.assert_close(actual, spectral_inverse(matrix), rtol=2e-12, atol=2e-12)
    derivative = _symmetric_pinv_jvp(matrix, actual, tangent)
    reverse, = torch.autograd.grad((actual * cotangent).sum(), matrix)
    torch.testing.assert_close((reverse * tangent).sum(), (derivative * cotangent).sum(), rtol=2e-11, atol=2e-11)
    # Directional spectral oracle; eps stays far from the retained-set cutoff.
    if any(spectrum):
        eps = 1e-8
        expected = (spectral_inverse(matrix + eps*tangent) - spectral_inverse(matrix - eps*tangent)) / (2*eps)
        torch.testing.assert_close(derivative, expected, rtol=2e-6, atol=2e-7)
    else:
        # The all-zero relative-cutoff branch has no full neighborhood on
        # which rank stays zero; only its selected constant extension is tested.
        assert torch.count_nonzero(derivative) == 0
    with forward_ad.dual_level():
        primal, dual = forward_ad.unpack_dual(selectedSymmetricInverse(
            forward_ad.make_dual(matrix.detach(), tangent)))
        torch.testing.assert_close(primal, actual, rtol=0, atol=0)
        torch.testing.assert_close(dual, derivative, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize('spectrum', [[2., 2., 2.], [2., 1.8e-6, 9e-7], [2., 1., 1.8e-6]])
def test_selected_inverse_first_and_second_reverse_gradcheck(device, spectrum):
    matrix = torch.diag(torch.tensor(spectrum, dtype=torch.float64, device=device))[None].requires_grad_()
    # General matrix perturbations are symmetrized by the forward definition.
    assert torch.autograd.gradcheck(selectedSymmetricInverse, (matrix,), eps=1e-8, atol=2e-7, rtol=2e-6)
    assert torch.autograd.gradgradcheck(selectedSymmetricInverse, (matrix,), eps=1e-8, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('schur', [0., -1e-8, 4. * torch.finfo(torch.float64).eps, .2])
@pytest.mark.parametrize('scale', [1e-3, 1., 1e3])
def test_augmented_moment_guard_and_branch_derivatives(device, dim, schur, scale):
    dtype = torch.float64
    m0 = torch.tensor([1.+schur], dtype=dtype, device=device) * scale
    m1 = torch.zeros(1, dim, dtype=dtype, device=device); m1[:, 0] = scale
    m2 = torch.eye(dim, dtype=dtype, device=device)[None] * scale
    moments = tuple(t.requires_grad_() for t in (m0, m1, m2, torch.ones_like(m1)*.02,
                                                torch.ones_like(m2)*.03, torch.ones(1, dim, dim, dim, dtype=dtype, device=device)*.01))
    count = torch.tensor([9], device=device, dtype=torch.int32)
    def f(*args):
        return computeCRKTermsWarp(*args, count, torch.ones_like(m0))
    values, tangents = torch.autograd.functional.jvp(f, moments, tuple(torch.ones_like(t)*.07 for t in moments), create_graph=True)
    assert all(torch.isfinite(t).all() for t in (*values, *tangents))
    if schur < .1:
        torch.testing.assert_close(values[0], torch.ones_like(m0), rtol=0, atol=0)
        assert all(torch.count_nonzero(t) == 0 for t in (*values[1:], *tangents))
        grads = torch.autograd.grad(sum(t.sum() for t in values), moments)
        assert all(torch.isfinite(g).all() and torch.count_nonzero(g) == 0 for g in grads)
    else:
        torch.testing.assert_close(values[0], torch.full_like(m0, 1/(schur*scale)), rtol=1e-12, atol=1e-12)
        assert torch.autograd.gradcheck(f, moments)


def particle_state(positions, kind=0):
    n = len(positions)
    h = positions.new_full((n,), 1.3)
    return ParticleState(positions=positions, supports=h, masses=torch.ones_like(h),
        densities=torch.ones_like(h), kinds=torch.full((n,), kind, dtype=torch.int32, device=positions.device))


def domain_for(positions):
    dim = positions.shape[-1]
    return DomainDescription(positions.new_full((dim,), -2), positions.new_full((dim,), 2),
                             torch.zeros(dim, dtype=torch.bool, device=positions.device), dim)


@pytest.mark.parametrize('rank', [1, 2])
@pytest.mark.parametrize('traversal', ['csr', 'grid', 'implicit'])
def test_3d_thin_renorm_rotation_backward_and_reverse_over_jvp(device, rank, traversal):
    if device.type != 'cuda':
        pytest.skip('3D SPH qualification runs on CUDA')
    dtype = get_torch_precision()
    blocks = []
    for axis in range(3):
        xyz = torch.zeros(4, 3, dtype=dtype, device=device)
        xyz[:, axis] = xyz.new_tensor([-.3, -.15, .15, .3]) * (1 if axis < rank else .0006)
        blocks.append(xyz)
    r = particle_state(torch.cat(blocks))
    q = particle_state(r.positions.new_tensor([[-.03, 0, 0], [.03, 0, 0]]))
    domain = domain_for(q.positions)
    adjacency = radiusSearchCompactHashMap(q, domain, referenceParticles=r)
    adj = {'csr': adjacency, 'grid': adjacency.hashMap, 'implicit': None}[traversal]
    rotation = r.positions.new_tensor([[0., -.7, .6], [.7, 0., -.4], [-.6, .4, 0.]])
    props = OperationProperties(operation=WarpOperation.Gradient, kernel=KernelFunctions.Wendland2)

    def f(angle, jvp=False):
        basis = torch.matrix_exp(angle * rotation)
        query = replace(q, positions=q.positions @ basis.T)
        ref = replace(r, positions=r.positions @ basis.T)
        kwargs = dict(adjacency=adj, referenceParticles=ref)
        if not jvp:
            return computeRenormalizationMatrices(query, props, domain, **kwargs,
                returnEigVals=False).renormalizationMatrices
        _, eig, state, tangent = computeRenormalizationMatricesJVP(query, props, domain,
            ParticleTangentState(query.positions @ rotation.T, None, None),
            referenceTangentState=ParticleTangentState(ref.positions @ rotation.T, None, None), **kwargs)
        assert bool((eig[:, rank:].abs() < 1e-6 * eig[:, :1].abs()).all())
        assert bool((eig[:, rank:].abs() > 0).all())
        return state.renormalizationMatrices, tangent.renormalizationMatrices

    angle = q.positions.new_tensor(.17, requires_grad=True)
    L, dL = f(angle, True)
    expected = rotation @ L - L @ rotation
    tol = 4e-5 if dtype == torch.float32 else 2e-10
    torch.testing.assert_close(dL, expected, rtol=tol, atol=tol)
    cot = torch.arange(L.numel(), dtype=dtype, device=device).reshape_as(L).sin()
    reverse, = torch.autograd.grad((f(angle)*cot).sum(), angle)
    torch.testing.assert_close(reverse, (dL*cot).sum(), rtol=tol, atol=tol)
    if dtype == torch.float64:
        assert torch.autograd.gradcheck(f, (angle,), eps=1e-5, atol=2e-9, rtol=2e-8)
        assert torch.autograd.gradcheck(lambda a: f(a, True)[1], (angle,), eps=1e-5, atol=2e-9, rtol=2e-8)


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('traversal', ['csr', 'grid', 'implicit'])
def test_affine_reference_plane_has_constant_fallback(device, dim, traversal):
    if dim == 3 and device.type == 'cpu':
        pytest.skip('3D SPH qualification runs on CUDA')
    dtype = get_torch_precision()
    axis = torch.linspace(-.3, .3, 3 if dim == 3 else 9, dtype=dtype, device=device)
    plane = torch.cartesian_prod(axis, axis) if dim == 3 else axis[:, None]
    r = particle_state(torch.cat((plane, plane.new_full((len(plane), 1), .18)), -1).requires_grad_())
    q = particle_state(torch.zeros(2, dim, dtype=dtype, device=device, requires_grad=True))
    domain = domain_for(q.positions)
    adjacency = radiusSearchCompactHashMap(q, domain, referenceParticles=r)
    adj = {'csr': adjacency, 'grid': adjacency.hashMap, 'implicit': None}[traversal]
    tq = ParticleTangentState(q.positions.detach().sin()+.02, None, None)
    tr = ParticleTangentState(r.positions.detach().cos()*.03, None, None)
    _, _, state = computeCRKFactors(q, domain, KernelFunctions.Wendland2, adjacency=adj, referenceState=r)
    _, _, _, tangent = computeCRKFactorsJVP(q, domain, KernelFunctions.Wendland2, tq,
        adjacency=adj, referenceState=r, referenceTangentState=tr)
    torch.testing.assert_close(state.A, torch.ones_like(state.A), rtol=0, atol=0)
    assert all(torch.count_nonzero(t) == 0 for t in (state.B, state.gradA, state.gradB, *vars(tangent).values()))
    for value in vars(state).values():
        grads = torch.autograd.grad(value.sum(), (q.positions, r.positions), retain_graph=True)
        assert all(torch.isfinite(g).all() and torch.count_nonzero(g) == 0 for g in grads)


# Enum names describe source -> destination. This fixture chooses an admitted
# source/destination for each rule independently of the Warp filtering helper.
PAIRS = [(2,2), (0,0), (0,1), (1,0), (1,1), (0,2), (2,0), (1,2), (2,1),
         (0,3), (0,2), (0,0), (0,1), (0,1), (1,0)]


@pytest.mark.parametrize('direction', list(OperationDirection))
@pytest.mark.parametrize('traversal', ['csr', 'grid', 'implicit'])
def test_directed_reference_volume_policy_reproduction_and_ad(device, direction, traversal):
    from warpSPHCore.crk.crk_volume import _computeCRKVolume_stateBackend
    from warpSPHCore.crk.crk_moments import _computeCRKMoments_stateBackend
    dtype = get_torch_precision()
    source, destination = PAIRS[direction.value]
    axis = torch.linspace(-.3, .3, 3, dtype=dtype, device=device)
    r = particle_state(torch.cat((torch.cartesian_prod(axis, axis),
                                  axis.new_tensor([[.18, .22], [.1, -.25]]))), source)
    r.kinds[-2:] = 1 if source == 2 else 2
    q = particle_state(axis.new_tensor([[-.03, .02], [.04, -.01]]), destination)
    domain = domain_for(q.positions)
    adjacency = radiusSearchCompactHashMap(q, domain, referenceParticles=r)
    adj = {'csr': adjacency, 'grid': adjacency.hashMap, 'implicit': None}[traversal]
    mode = OperationDirection.TrueAllToToAll
    dq, dr = q.positions.cos()*.02, r.positions.sin()*.04+.01
    dhq, dhr = q.supports*.01, r.supports*.015

    def f(step, jvp=False):
        query = replace(q, positions=q.positions+step*dq, supports=q.supports+step*dhq)
        ref = replace(r, positions=r.positions+step*dr, supports=r.supports+step*dhr)
        kwargs = dict(operationMode=direction, referenceVolumeMode=mode, adjacency=adj, referenceState=ref)
        if jvp:
            return computeCRKFactorsJVP(query, domain, KernelFunctions.Wendland2,
                ParticleTangentState(dq, dhq, None), referenceTangentState=ParticleTangentState(dr, dhr, None), **kwargs)
        return computeCRKFactors(query, domain, KernelFunctions.Wendland2, **kwargs)

    step = axis.new_zeros((), requires_grad=True)
    volume, dv, state, tangent = f(step, True)
    _, density, primal = f(step)
    assert torch.isfinite(density).all() and bool((volume > 0).all())
    if direction.value in (2, 3, 5, 6, 7, 8, 10, 12):
        # Inheriting a cross-category rule rejects reference-to-reference
        # query rows. Keep that legacy default explicit; the opt-in geometric
        # estimate above makes a reproducing directed stencil possible.
        old_volume, _, old = computeCRKFactors(q, domain, KernelFunctions.Wendland2,
            operationMode=direction, referenceState=r, adjacency=adj)
        torch.testing.assert_close(old_volume, volume, rtol=0, atol=0)
        torch.testing.assert_close(old.A, torch.ones_like(old.A), rtol=0, atol=0)
        assert torch.count_nonzero(old.B) == 0
    weights = _computeCRKVolume_stateBackend(r, OperationProperties(kernel=KernelFunctions.Wendland2,
        supportMode=SupportScheme.Gather, operationMode=mode), domain)
    m0, m1, m2, *_ = _computeCRKMoments_stateBackend(q, OperationProperties(kernel=KernelFunctions.Wendland2,
        supportMode=SupportScheme.Scatter, operationMode=direction), domain,
        referenceParticles=r, queryVolumes=volume.detach(), referenceVolumes=weights, adjacency=adj)
    tol = 4e-5 if dtype == torch.float32 else 2e-10
    torch.testing.assert_close(state.A*(m0+(state.B*m1).sum(-1)), torch.ones_like(m0), rtol=tol, atol=tol)
    torch.testing.assert_close(state.A[:,None]*(m1+(m2@state.B[:,:,None]).squeeze(-1)), torch.zeros_like(m1), rtol=tol, atol=tol)
    coefficient = axis.new_tensor([.6, -.4])
    for operation in (WarpOperation.Interpolate, WarpOperation.Gradient):
        result = warpOperation(q, OperationProperties(operation=operation, kernel=KernelFunctions.Wendland2,
            supportMode=SupportScheme.Scatter, operationMode=direction, gradientMode=GradientScheme.Naive), domain,
            adjacency=adj, referenceParticles=r, referenceVolumes=weights, crkState=state,
            queryValues=1.2+q.positions@coefficient, referenceValues=1.2+r.positions@coefficient)
        expected = 1.2+q.positions@coefficient if operation == WarpOperation.Interpolate else coefficient.expand_as(q.positions)
        torch.testing.assert_close(result, expected, rtol=tol*4, atol=tol*4)
    for name in vars(state):
        value, derivative = getattr(primal, name), getattr(tangent, name)
        torch.testing.assert_close(value, getattr(state,name), rtol=tol, atol=tol)
        cot = torch.arange(value.numel(), dtype=dtype, device=device).reshape_as(value).cos()
        reverse, = torch.autograd.grad((value*cot).sum(), step, retain_graph=True)
        torch.testing.assert_close(reverse, (derivative*cot).sum(), rtol=tol*4, atol=tol*4)
    eps = .01 if dtype == torch.float32 else 1e-5
    plus, minus = f(step+eps)[2], f(step-eps)[2]
    for name in vars(state):
        fd = (getattr(plus,name)-getattr(minus,name))/(2*eps)
        torch.testing.assert_close(getattr(tangent,name), fd,
            rtol=4e-3 if dtype == torch.float32 else 2e-7, atol=3e-4 if dtype == torch.float32 else 2e-8)


def test_explicit_reference_volume_mode_with_aliased_population(device):
    axis = torch.linspace(-.3, .3, 3, dtype=get_torch_precision(), device=device)
    p = particle_state(torch.cartesian_prod(axis, axis))
    domain = domain_for(p.positions)
    adj = radiusSearchCompactHashMap(p, domain)
    dp, dh = p.positions.sin()*.03, p.supports*.02
    tangent = ParticleTangentState(dp, dh, None)
    mode = OperationDirection.TrueAllToToAll
    step = axis.new_zeros((), requires_grad=True)
    query = replace(p, positions=p.positions+step*dp, supports=p.supports+step*dh)
    volume, dvolume, state, dstate = computeCRKFactorsJVP(query, domain, KernelFunctions.Wendland2,
        tangent, referenceTangentState=tangent, referenceVolumeMode=mode, adjacency=adj)
    primal_volume, _, primal = computeCRKFactors(query, domain, KernelFunctions.Wendland2,
        referenceVolumeMode=mode, adjacency=adj)
    tol = 2e-4 if axis.dtype == torch.float32 else 2e-10
    for value, derivative in zip((primal_volume, *vars(primal).values()), (dvolume, *vars(dstate).values())):
        reverse, = torch.autograd.grad(value.sum(), step, retain_graph=True)
        torch.testing.assert_close(reverse, derivative.sum(), rtol=tol, atol=tol)
