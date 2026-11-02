"""Correction reproduction, nullable input and explicit AD-boundary tests."""
from dataclasses import replace
import sys

import pytest
import torch
import warpSPHCore as sph
from _support.operators import fixture
from test_corrected_primal import correction_fixture, oracle


def corrected_fixture(device, operation=sph.WarpOperation.Gradient):
    q, r, domain, qf, rf = fixture(device, 2, 1)
    crk, (qv, rv), omega, R = correction_fixture(q, r)
    prop = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=operation,
        gradientMode=sph.GradientScheme.Difference,
        operationMode=sph.OperationDirection.TrueAllToToAll)
    options = dict(crkState=crk, queryVolumes=qv, referenceVolumes=rv)
    if operation == sph.WarpOperation.Gradient:
        options.update(gradHState=omega, renormalizationState=R)
    return q, r, domain, qf, rf, prop, options


@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('rank', [0, 1, 2])
@pytest.mark.parametrize('scheme', [sph.GradientScheme.Naive, sph.GradientScheme.Difference])
def test_crk_exact_linear_reproduction(device, dimension, rank, scheme):
    generator = torch.Generator().manual_seed(3344)
    dtype = torch.float64
    def random(*size):return torch.rand(size, generator=generator, dtype=dtype)
    rx = random(41, dimension) - .5
    qx = (random(3, dimension) - .5) * .25
    volume = .05 + random(41) * .1
    h = 1.5
    C = [1.25, 2.228169203286535, 3.3422538049298023][dimension - 1]
    power, slope = (3, 3) if dimension == 1 else (4, 4)
    # Solve the augmented polynomial moment system independently in Torch.
    # Differentiate the solve to obtain spatial derivatives of A and B;
    # neither the production factor solver nor its derivative code is used.
    def coefficients(x):
        dx = x - rx
        ratio = dx.norm(dim=-1) / h
        w = C / h**dimension * (1 - ratio)**power * (1 + slope * ratio)
        basis = torch.cat((torch.ones(41, 1, dtype=dtype), dx), dim=1)
        moments = torch.einsum('n,ni,nj->ij', volume * w, basis, basis)
        rhs = torch.zeros(dimension + 1, dtype=dtype); rhs[0] = 1
        solution = torch.linalg.solve(moments, rhs)
        return torch.cat((solution[:1], solution[1:] / solution[0]))
    factors, derivatives = [], []
    for x in qx:
        factors.append(coefficients(x))
        derivatives.append(torch.autograd.functional.jacobian(coefficients, x))
    factors, derivatives = torch.stack(factors), torch.stack(derivatives)
    crk = sph.CRKState(factors[:, 0].contiguous().to(device), factors[:, 1:].contiguous().to(device),
        derivatives[:, 0].contiguous().to(device), derivatives[:, 1:].transpose(1, 2).contiguous().to(device))
    def state(x):
        x = x.to(device)
        return sph.ParticleState(x, x.new_full((len(x),), h), x.new_ones(len(x)),
            torch.zeros(len(x), device=device, dtype=torch.int32), x.new_ones(len(x)))
    q, r = state(qx), state(rx)
    domain = sph.DomainDescription(q.positions.new_full((dimension,), -2),
        q.positions.new_full((dimension,), 2), torch.zeros(dimension, device=device, dtype=torch.bool), dimension)
    fields = dimension**rank
    offset, slopes = random(fields), random(fields, dimension) - .5
    qf = (offset + qx @ slopes.T).reshape(len(qx), *([dimension] * rank)).to(device)
    rf = (offset + rx @ slopes.T).reshape(len(rx), *([dimension] * rank)).to(device)
    prop = sph.OperationProperties(sph.KernelFunctions.Wendland2,
        gradientMode=scheme, operationMode=sph.OperationDirection.TrueAllToToAll)
    kwargs = dict(referenceParticles=r, referenceVolumes=volume.to(device), crkState=crk)
    actual = sph.warpOperation(q, prop, domain, qf, rf, **kwargs)
    torch.testing.assert_close(actual, qf, atol=2e-12, rtol=2e-12)
    result = sph.warpOperation(q, replace(prop, operation=sph.WarpOperation.Gradient), domain, qf, rf, **kwargs)
    expected = slopes.to(device).reshape(*([dimension] * rank), dimension).expand_as(result)
    torch.testing.assert_close(result, expected, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize('operation', [sph.WarpOperation.Interpolate, sph.WarpOperation.Gradient])
def test_volume_override_can_omit_mass_and_density(device, operation):
    q, r, d, qf, rf, prop, kw = corrected_fixture(device, operation)
    expected = oracle(q, r, d, prop, qf, rf, **kw)
    q, r = replace(q, masses=None, densities=None), replace(r, masses=None, densities=None)
    actual = sph.warpOperation(q, prop, d, qf, rf, referenceParticles=r, **kw)
    torch.testing.assert_close(actual, expected, rtol=3e-10, atol=3e-10)
    if operation == sph.WarpOperation.Gradient:
        with pytest.raises(RuntimeError, match='reference masses|reference densities'):
            sph.warpOperation(q, replace(prop, gradientMode=sph.GradientScheme.Symmetric), d,
                              qf, rf, referenceParticles=r, **kw)


@pytest.mark.parametrize('operation', [sph.WarpOperation.Interpolate, sph.WarpOperation.Gradient])
def test_corrected_self_pair(device, operation):
    x = torch.zeros(1, 2, device=device, dtype=torch.float64)
    state = sph.ParticleState(x, x.new_ones(1), x.new_ones(1), torch.zeros(1, device=device, dtype=torch.int32), x.new_ones(1))
    domain = sph.DomainDescription(x.new_full((2,), -1), x.new_ones(2), torch.zeros(2, device=device, dtype=torch.bool), 2)
    crk = sph.CRKState(x.new_tensor([2.]), x.new_tensor([[.3, -.4]]),
        x.new_tensor([[.7, .2]]), x.new_tensor([[[.1, .8], [-.3, .6]]]))
    prop = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=operation,
        gradientMode=sph.GradientScheme.Naive, operationMode=sph.OperationDirection.TrueAllToToAll)
    actual = sph.warpOperation(state, prop, domain, x.new_tensor([3.]), crkState=crk)
    expected = (3 * 2 * 2.228169203286535 if operation == sph.WarpOperation.Interpolate else
                3 * 2.228169203286535 * (2 * crk.B + crk.gradA))
    torch.testing.assert_close(actual, torch.as_tensor(expected, device=device, dtype=actual.dtype).expand_as(actual))


@pytest.mark.parametrize('sizes', [(0, 5), (7, 0), (0, 0)])
@pytest.mark.parametrize('operation', [sph.WarpOperation.Interpolate, sph.WarpOperation.Gradient])
def test_corrected_empty_populations(device, sizes, operation):
    q, r, d, qf, rf, prop, kw = corrected_fixture(device, operation)
    nq, nr = sizes
    def trim(s, n):return replace(s, **{name: getattr(s, name)[:n] for name in ('positions','supports','masses','densities','kinds')})
    q, r = trim(q, nq), trim(r, nr)
    kw['crkState'] = sph.CRKState(**{name: getattr(kw['crkState'], name)[:nq] for name in ('A','B','gradA','gradB')})
    kw['queryVolumes'], kw['referenceVolumes'] = kw['queryVolumes'][:nq], kw['referenceVolumes'][:nr]
    if operation == sph.WarpOperation.Gradient:
        omega = kw['gradHState']; kw['gradHState'] = sph.GradHState(omega.queryOmegas[:nq], omega.referenceOmegas[:nr])
        kw['renormalizationState'] = kw['renormalizationState'].renormalizationMatrices[:nq]
    out = sph.warpOperation(q, prop, d, qf[:nq], rf[:nr], referenceParticles=r, **kw)
    assert out.shape[0] == nq and out.count_nonzero() == 0


@pytest.mark.parametrize('operation', [sph.WarpOperation.Interpolate, sph.WarpOperation.Gradient])
@pytest.mark.parametrize('slot', ['positions', 'values', 'A', 'B', 'gradA', 'gradB', 'volume', 'omega', 'renorm'])
def test_corrected_ad_is_explicit(device, operation, slot):
    if operation == sph.WarpOperation.Interpolate and slot in ('gradA','gradB','omega','renorm'):
        pytest.skip('unused by interpolation')
    q, r, d, qf, rf, prop, kw = corrected_fixture(device, operation)
    if slot == 'positions':q.positions.requires_grad_()
    elif slot == 'values':rf.requires_grad_()
    elif slot in ('A','B','gradA','gradB'):getattr(kw['crkState'], slot).requires_grad_()
    elif slot == 'volume':kw['referenceVolumes'].requires_grad_()
    elif slot == 'omega':kw['gradHState'].referenceOmegas.requires_grad_()
    else:kw['renormalizationState'].renormalizationMatrices.requires_grad_()
    with pytest.raises(NotImplementedError, match='Corrected native primals do not support autograd'):
        sph.warpOperation(q, prop, d, qf, rf, referenceParticles=r, **kw)
    with torch.no_grad():
        out = sph.warpOperation(q, prop, d, qf, rf, referenceParticles=r, **kw)
    assert not out.requires_grad


def test_interpolation_uses_only_query_A_B(device):
    q, r, d, qf, rf, prop, kw = corrected_fixture(device, sph.WarpOperation.Interpolate)
    expected = sph.warpOperation(q, prop, d, qf, rf, referenceParticles=r, **kw)
    kw['queryVolumes'] = kw['queryVolumes'].requires_grad_()  # Explicit reference volume overrides this.
    crk = kw['crkState'];crk.gradA.requires_grad_();crk.gradB.requires_grad_()
    reference_crk = sph.CRKState(r.supports.requires_grad_(), r.positions.requires_grad_())
    # Query CRK derivatives and the independent reference coefficient state are
    # unused. Reference *geometry* must remain non-differentiable for this call.
    reference_crk = sph.CRKState(reference_crk.A.detach().clone().requires_grad_(), reference_crk.B.detach().clone().requires_grad_())
    r = replace(r, supports=r.supports.detach(), positions=r.positions.detach())
    kw['crkState'] = (sph.CRKState(crk.A, crk.B), reference_crk)
    actual = sph.warpOperation(q, prop, d, qf, rf, referenceParticles=r, **kw)
    torch.testing.assert_close(actual, expected)
    assert 'warp' not in sys.modules


@pytest.mark.parametrize('invalid', ['B_shape','A_dtype','volume_shape','renorm_contiguous','missing_derivatives'])
def test_corrected_metadata_validation(device, invalid):
    q, r, d, qf, rf, prop, kw = corrected_fixture(device)
    if invalid == 'B_shape':kw['crkState'].B = kw['crkState'].B[:, :1]
    elif invalid == 'A_dtype':kw['crkState'].A = kw['crkState'].A.float()
    elif invalid == 'volume_shape':kw['referenceVolumes'] = kw['referenceVolumes'][:-1]
    elif invalid == 'renorm_contiguous':kw['renormalizationState'] = kw['renormalizationState'].renormalizationMatrices.transpose(1,2)
    else:kw['crkState'].gradA = None
    with pytest.raises((ValueError, RuntimeError), match='shape|dtype|axes|contiguous|gradA'):
        sph.warpOperation(q, prop, d, qf, rf, referenceParticles=r, **kw)


@pytest.mark.parametrize('slot', ['positions', 'values', 'A', 'volume', 'omega', 'renorm'])
def test_corrected_forward_ad_is_explicit(device, slot):
    q, r, d, qf, rf, prop, kw = corrected_fixture(device)
    with torch.autograd.forward_ad.dual_level():
        def dual(value):return torch.autograd.forward_ad.make_dual(value, torch.ones_like(value))
        if slot == 'positions':q = replace(q, positions=dual(q.positions))
        elif slot == 'values':rf = dual(rf)
        elif slot == 'A':kw['crkState'].A = dual(kw['crkState'].A)
        elif slot == 'volume':kw['referenceVolumes'] = dual(kw['referenceVolumes'])
        elif slot == 'omega':kw['gradHState'].referenceOmegas = dual(kw['gradHState'].referenceOmegas)
        else:kw['renormalizationState'] = dual(kw['renormalizationState'].renormalizationMatrices)
        with pytest.raises(NotImplementedError, match='forward-mode'):
            sph.warpOperation(q, prop, d, qf, rf, referenceParticles=r, **kw)


def test_corrected_module_local_gradient(device):
    from warpSPHCore.coreOperations.native import computeGradient
    q, r, d, qf, rf, prop, kw = corrected_fixture(device)
    expected = sph.warpOperation(q, prop, d, qf, rf, referenceParticles=r, **kw)
    actual = computeGradient(q, r, d, prop, None, qf, rf, **kw)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('operation', [sph.WarpOperation.Interpolate, sph.WarpOperation.Gradient])
def test_query_volume_fallback_for_aliased_population(device, operation):
    q, _, d, qf, _ = fixture(device, 2, 0)
    q = replace(q, masses=None, densities=None)
    volume = q.supports.square() * .2
    prop = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=operation,
        gradientMode=sph.GradientScheme.Difference)
    expected = sph.warpOperation(q, prop, d, qf, referenceVolumes=volume)
    actual = sph.warpOperation(q, prop, d, qf, queryVolumes=volume)
    torch.testing.assert_close(actual, expected)
