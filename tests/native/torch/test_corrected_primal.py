"""Corrected primals against independent differentiated kernel equations."""
from dataclasses import replace

import pytest
import torch
import warpSPHCore as sph
from _support.operators import fixture, normalization
from _support.kernels import shape


def correction_fixture(q, r):
    generator = torch.Generator().manual_seed(20261103)
    def random(*size):
        return torch.rand(size, generator=generator, dtype=torch.float64).to(q.positions)
    n, m, d = len(q.positions), len(r.positions), q.positions.shape[1]
    state = sph.CRKState(.7 + random(n), random(n, d) - .5,
                         random(n, d) - .5, random(n, d, d) - .5)
    volumes = (.1 + random(n), .1 + random(m))
    omega = sph.GradHState(.6 + random(n), .8 + random(m))
    renorm = torch.eye(d).to(q.positions)[None] + .3 * (random(n, d, d) - .5)
    return state, volumes, omega, sph.RenormalizationState(renorm)


def oracle(q, r, domain, prop, qf, rf, *, referenceVolumes=None, crkState=None,
           gradHState=None, renormalizationState=None, **unused):
    # Each pair gets its own virtual query displacement. Differentiate the
    # corrected scalar kernel with spatially varying A/B; no production CRK
    # formula or hand-coded gradient enters this oracle.
    dtype = torch.float64
    d = domain.dim
    qx, rx = q.positions.double(), r.positions.double()
    delta = torch.zeros((len(qx), len(rx), d), device=qx.device, dtype=dtype, requires_grad=True)
    dx = qx[:, None] + delta - rx[None]
    length = (domain.max - domain.min).double()
    dx = torch.where(domain.periodic, torch.remainder(dx + length / 2, length) - length / 2, dx)
    distance = dx.norm(dim=-1)
    hi, hj = q.supports.double()[:, None], r.supports.double()[None]
    support = prop.supportMode.value
    h = {11: hi, 12: hj, 13: (hi + hj) / 2,
         14: torch.maximum(hi, hj), 15: torch.maximum(hi, hj)}[support]
    def kernel(radius):
        ratio = distance / radius
        active = (radius > 0) & (ratio <= 1)
        if prop.kernel == sph.KernelFunctions.AdhesionKernel:
            active = active & (ratio > .5)
            ratio = torch.where(active, ratio, .75)
        radial = shape(ratio, prop.kernel, d)
        return torch.where(active, normalization(prop.kernel, d) * radial / radius**d, 0.)
    w = (kernel(hi) + kernel(hj)) / 2 if support in (14, 15) else kernel(h)
    if crkState is not None:
        state = crkState[0] if isinstance(crkState, tuple) else crkState
        A, B = state.A.double()[:, None], state.B.double()[:, None]
        if prop.operation == sph.WarpOperation.Gradient:
            A = A + (state.gradA.double()[:, None] * delta).sum(-1)
            B = B + torch.einsum('ngc,nmg->nmc', state.gradB.double(), delta)
        w = A * (1 + (B * dx).sum(-1)) * w
    gradient = torch.autograd.grad(w.sum(), delta)[0]
    active = distance.detach() < h
    if prop.operationMode == sph.OperationDirection.FluidToBoundary:
        active = active & (q.kinds == 1)[:, None] & (r.kinds == 0)[None]
    elif prop.operationMode != sph.OperationDirection.TrueAllToToAll:
        raise AssertionError('oracle direction not implemented')
    volume = (referenceVolumes.double() if referenceVolumes is not None
              else r.masses.double() / r.densities.double())
    field_shape = rf.shape[1:]
    fj = rf.double().reshape(len(rx), -1)
    if prop.operation == sph.WarpOperation.Interpolate:
        return torch.einsum('ij,jf->if', w.detach() * active * volume, fj).reshape(len(qx), *field_shape)
    if gradHState is not None:
        omega = (gradHState.referenceOmegas if isinstance(gradHState, sph.GradHState)
                 else gradHState[1] if isinstance(gradHState, tuple) else gradHState)
        fj = fj / omega.double()[:, None]
    fi = qf.double().reshape(len(qx), -1)
    mode = prop.gradientMode
    if mode == sph.GradientScheme.Naive:
        weighted = fj[None] * volume[:, None]
    elif mode == sph.GradientScheme.Symmetric:
        rhoi, rhoj = q.densities.double(), r.densities.double()
        weighted = r.masses.double()[None, :, None] * rhoi[:, None, None] * (
            fi[:, None] / rhoi[:, None, None]**2 + fj[None] / rhoj[None, :, None]**2)
    else:
        combined = fj[None] - fi[:, None] if mode == sph.GradientScheme.Difference else fj[None] + fi[:, None]
        weighted = combined * volume[:, None]
    if renormalizationState is not None:
        R = (renormalizationState.renormalizationMatrices if isinstance(renormalizationState, sph.RenormalizationState)
             else renormalizationState)
        gradient = torch.einsum('nab,nmb->nma', R.double(), gradient)
    return torch.einsum('nmf,nmd->nfd', weighted, gradient * active[..., None]).reshape(len(qx), *field_shape, d)


def check_case(device, dtype, dimension, rank, operation, mode, support, scheme,
               kernel, index_dtype, offset_dtype, filtered):
    q, r, domain, qf, rf = fixture(device, dimension, rank)
    def cast_state(s):
        return replace(s, **{name: getattr(s, name).to(dtype) for name in
                            ('positions', 'supports', 'masses', 'densities')})
    q, r = cast_state(q), cast_state(r)
    q.positions[:, 0].add_(6); r.positions[:, 0].sub_(4)
    domain = replace(domain, min=domain.min.to(dtype), max=domain.max.to(dtype))
    qf, rf = qf.to(dtype), rf.to(dtype)
    state, (qv, rv), omega, R = correction_fixture(q, r)
    kw = {}
    if mode in ('crk', 'crk_volume', 'all'): kw['crkState'] = state
    if mode in ('volume', 'crk_volume', 'all'): kw.update(queryVolumes=qv, referenceVolumes=rv)
    if mode in ('omega', 'all'): kw['gradHState'] = omega
    if mode in ('renorm', 'all'): kw['renormalizationState'] = R
    prop = sph.OperationProperties(kernel, operation=operation, supportMode=support,
        gradientMode=scheme, operationMode=(sph.OperationDirection.FluidToBoundary if filtered
                                          else sph.OperationDirection.TrueAllToToAll))
    dx = q.positions[:, None] - r.positions[None]
    length = domain.max - domain.min
    dx = torch.where(domain.periodic, torch.remainder(dx + length / 2, length) - length / 2, dx)
    hi, hj = q.supports[:, None], r.supports[None]
    radius = {11: hi, 12: hj, 13: (hi + hj) / 2,
              14: torch.maximum(hi, hj), 15: torch.maximum(hi, hj)}[support.value]
    membership = dx.norm(dim=-1) < radius
    counts = membership.sum(1).int()
    _, ids = torch.where(membership.flip(1))
    ids = (len(r.positions) - 1 - ids).to(index_dtype)
    graph = sph.AdjacencyList(None, torch.cat((ids.new_zeros(3), ids)), counts,
        (counts.long().cumsum(0) - counts + 3).to(offset_dtype), len(q.positions), len(r.positions), numEdges=ids.numel())
    actual = sph.warpOperation(q, prop, domain, qf, rf, referenceParticles=r, adjacency=graph, **kw)
    expected = oracle(q, r, domain, prop, qf, rf, **kw)
    tolerance = 4e-4 if dtype == torch.float32 else 3e-9
    torch.testing.assert_close(actual.double(), expected, rtol=tolerance, atol=tolerance)
    assert not actual.requires_grad


MODES = [(sph.WarpOperation.Interpolate, mode) for mode in ('crk', 'volume', 'crk_volume')] + [
    (sph.WarpOperation.Gradient, mode) for mode in ('crk', 'volume', 'crk_volume', 'omega', 'renorm', 'all')]


@pytest.mark.parametrize('operation,mode', MODES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('rank', [0, 1, 2])
@pytest.mark.parametrize('index_dtype,offset_dtype', [(torch.int32, torch.int32), (torch.int32, torch.int64),
                                                   (torch.int64, torch.int32), (torch.int64, torch.int64)])
def test_corrected_layouts(device, operation, mode, dtype, dimension, rank, index_dtype, offset_dtype):
    case = dimension + rank + MODES.index((operation, mode))
    check_case(device, dtype, dimension, rank, operation, mode,
        list(sph.SupportScheme)[:5][case % 5], list(sph.GradientScheme)[case % 4],
        [sph.KernelFunctions.Wendland2, sph.KernelFunctions.CubicSpline, sph.KernelFunctions.Wendland4][case % 3],
        index_dtype, offset_dtype, bool(case % 2))


@pytest.mark.extended
@pytest.mark.parametrize('operation', [sph.WarpOperation.Interpolate, sph.WarpOperation.Gradient])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('kernel', list(sph.KernelFunctions))
@pytest.mark.parametrize('support', list(sph.SupportScheme)[:5])
def test_corrected_kernel_supports(device, operation, dtype, dimension, kernel, support):
    case = kernel.value + support.value + dimension
    check_case(device, dtype, dimension, case % 3, operation,
        'crk_volume' if operation == sph.WarpOperation.Interpolate else 'all', support,
        list(sph.GradientScheme)[case % 4], kernel,
        torch.int64 if case % 2 else torch.int32, torch.int32 if case % 2 else torch.int64, bool(case % 2))


@pytest.mark.extended
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('rank', [0, 1, 2])
@pytest.mark.parametrize('scheme', list(sph.GradientScheme))
@pytest.mark.parametrize('support', list(sph.SupportScheme)[:5])
def test_corrected_gradient_schemes(device, dtype, dimension, rank, scheme, support):
    check_case(device, dtype, dimension, rank, sph.WarpOperation.Gradient, 'all', support, scheme,
        sph.KernelFunctions.Wendland2, torch.int32, torch.int64, False)
