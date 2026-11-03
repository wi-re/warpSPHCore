"""Native correctness gates. Run in a fresh WARPSPHCORE_BACKEND=native process."""
import importlib.abc
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

if os.environ.get('WARPSPHCORE_BACKEND') != 'native':
    pytest.skip('native backend is qualified in a separate process', allow_module_level=True)

import warpSPHCore as sph


def case(device, dtype=torch.float32, dim=2, nq=11, nr=9):
    gen = torch.Generator().manual_seed(178)
    def tensor(shape):
        return torch.rand(shape, generator=gen, dtype=dtype).to(device)
    def particles(n):
        return sph.ParticleState(positions=(tensor((n, dim)) * 8 - 4).contiguous(),
                                 supports=tensor((n,)) * .8 + .4,
                                 masses=tensor((n,)) + .2,
                                 kinds=(torch.arange(n, device=device) % 4).int(),
                                 densities=tensor((n,)) + .8)
    q, r = particles(nq), particles(nr)
    d = sph.DomainDescription(q.positions.new_full((dim,), -1),
                              q.positions.new_full((dim,), 1),
                              torch.tensor([i % 2 == 0 for i in range(dim)], device=device), dim)
    return q, r, d, tensor((nq,)), tensor((nr,))


def reference(q, r, d, support, direction, qf, rf):
    # An independent tensor implementation, including the unsimplified radial
    # derivative. Operates on all pairs rather than the native CSR.
    length = d.max - d.min
    qx = torch.where(d.periodic, torch.remainder(q.positions - d.min, length) + d.min, q.positions)
    rx = torch.where(d.periodic, torch.remainder(r.positions - d.min, length) + d.min, r.positions)
    dx = qx[:, None] - rx[None]
    dx = torch.where(d.periodic, torch.remainder(dx + length / 2, length) - length / 2, dx)
    dist = dx.norm(dim=-1)
    hi, hj = q.supports[:, None], r.supports[None]
    effective = {11: lambda: hi, 12: lambda: hj, 13: lambda: (hi + hj) / 2,
                 14: lambda: torch.maximum(hi, hj), 15: lambda: torch.maximum(hi, hj)}[support.value]()
    pairs = dist < effective
    def kernel(h):
        safe = h.clamp_min(torch.finfo(h.dtype).tiny)
        z = dist / safe
        normal = (5 / 4 if d.dim == 1 else 7 / torch.pi if d.dim == 2 else 21 / (2 * torch.pi))
        power = 3 if d.dim == 1 else 4
        multiplier = 3 if d.dim == 1 else 4
        value = normal * (1 - z).clamp_min(0).pow(power) * (1 + multiplier * z) / safe.pow(d.dim)
        dk = (-12 * z * (1-z).clamp_min(0).square() if d.dim == 1
              else -20 * z * (1-z).clamp_min(0).pow(3))
        grad = normal * dk[..., None] / safe[..., None].pow(d.dim + 1)
        grad = grad * dx / dist.clamp_min(torch.finfo(dist.dtype).tiny)[..., None]
        return (torch.where(h > 0, value, 0), torch.where((h > 0)[..., None], grad, 0))
    if support.value in (14, 15):
        wi, gi = kernel(hi); wj, gj = kernel(hj)
        w, g = (wi + wj) / 2, (gi + gj) / 2
    else:
        w, g = kernel(effective)
    # Query/reference roles are the existing API's category convention.
    role_sets = {
        0: (None, None), 9: (None, (0, 1, 3)),
        1: ((0,), (0,)), 2: ((1,), (0,)), 3: ((0,), (1,)), 4: ((1,), (1,)),
        5: ((2,), (0,)), 6: ((0,), (2,)), 7: ((2,), (1,)), 8: ((1,), (2,)),
        10: ((2,), (0, 1, 3)), 11: ((0,), (0, 1, 3)), 12: ((1,), (0, 1, 3)),
        13: ((0, 1, 3), (0,)), 14: ((0, 1, 3), (1,)),
    }
    qs, rs = role_sets[direction.value]
    active = pairs.clone()
    if qs is not None: active &= torch.isin(q.kinds, q.kinds.new_tensor(qs))[:, None]
    if rs is not None: active &= torch.isin(r.kinds, r.kinds.new_tensor(rs))[None]
    density = (w * r.masses * active).sum(-1)
    gradient = (g * ((rf[None] - qf[:, None]) * (r.masses / r.densities) * active)[..., None]).sum(1)
    return pairs, density, gradient


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('support', list(sph.SupportScheme)[:-1])
@pytest.mark.parametrize('direction', [sph.OperationDirection.TrueAllToToAll,
                                       sph.OperationDirection.AllToAll,
                                       sph.OperationDirection.FluidToBoundary])
def test_native_against_tensor_oracle(device, dtype, dim, support, direction):
    q, r, d, qf, rf = case(device, dtype, dim)
    pairs, density, gradient = reference(q, r, d, support, direction, qf, rf)
    a = sph.radiusSearch(q, d, r, support)
    i, j = a.livePairs()
    expected_i, expected_j = pairs.nonzero(as_tuple=True)
    assert torch.equal(i, expected_i) and torch.equal(j, expected_j)
    prop = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=sph.WarpOperation.Density,
                                    supportMode=support, operationMode=direction)
    rho = sph.warpOperation(q, prop, d, adjacency=a, referenceParticles=r)
    grad = sph.warpOperation(q, replace(prop, operation=sph.WarpOperation.Gradient,
                                        gradientMode=sph.GradientScheme.Difference),
                             d, qf, rf, adjacency=a, referenceParticles=r)
    tol = 3e-5 if dtype == torch.float32 else 3e-12
    torch.testing.assert_close(rho, density, rtol=tol, atol=tol)
    torch.testing.assert_close(grad, gradient, rtol=tol, atol=tol)
    assert 'warp' not in sys.modules


@pytest.mark.parametrize('direction', list(sph.OperationDirection))
def test_all_directions(device, direction):
    q, r, d, qf, rf = case(device)
    _, rho, grad = reference(q, r, d, sph.SupportScheme.Gather, direction, qf, rf)
    prop = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=sph.WarpOperation.Density,
                                    operationMode=direction)
    torch.testing.assert_close(sph.warpOperation(q, prop, d, referenceParticles=r), rho, rtol=3e-5, atol=3e-5)
    gp = replace(prop, operation=sph.WarpOperation.Gradient, gradientMode=sph.GradientScheme.Difference)
    torch.testing.assert_close(sph.warpOperation(q, gp, d, qf, rf, referenceParticles=r), grad, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize('nq,nr', [(0, 0), (0, 4), (4, 0)])
def test_empty_populations(device, nq, nr):
    q, r, d, qf, rf = case(device, nq=nq, nr=nr)
    a = sph.radiusSearch(q, d, r)
    assert a.numEdges == 0
    prop = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=sph.WarpOperation.Density)
    assert sph.warpOperation(q, prop, d, adjacency=a, referenceParticles=r).shape == (nq,)
    grad = sph.warpOperation(q, replace(prop, operation=sph.WarpOperation.Gradient,
                                        gradientMode=sph.GradientScheme.Difference),
                             d, qf, rf, adjacency=a, referenceParticles=r)
    torch.testing.assert_close(grad, q.positions.new_zeros(nq, d.dim))


@pytest.mark.parametrize('index_dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('offset_dtype', [torch.int32, torch.int64])
def test_superset_csr_padding_self_pairs_and_constant_gradient(device, index_dtype, offset_dtype):
    q, _, d, _, _ = case(device)
    q.kinds.zero_()
    n = len(q.positions)
    a = sph.AdjacencyList(None, torch.arange(n, device=device).repeat(n),
                           torch.full((n,), n, dtype=torch.int32, device=device),
                           torch.arange(n, device=device) * n, n, n, numEdges=n*n)
    # This all-pairs CSR deliberately includes out-of-support references.
    # Tail padding must never be read; counts/offsets retain their meanings.
    a.paddedJ = torch.cat((a.paddedJ.to(index_dtype), torch.full((7,), -1, dtype=index_dtype, device=device)))
    a.edgeOffsets = a.edgeOffsets.to(offset_dtype)
    prop = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=sph.WarpOperation.Gradient,
                                    gradientMode=sph.GradientScheme.Difference)
    y = sph.warpOperation(q, prop, d, torch.ones_like(q.masses), adjacency=a)
    torch.testing.assert_close(y, torch.zeros_like(q.positions), atol=0, rtol=0)
    density_prop = replace(prop, operation=sph.WarpOperation.Density)
    _, expected_density, _ = reference(q, q, d, sph.SupportScheme.Gather,
                                        sph.OperationDirection.AllToAll,
                                        torch.ones_like(q.masses), torch.ones_like(q.masses))
    torch.testing.assert_close(sph.warpOperation(q, density_prop, d, adjacency=a),
                               expected_density, rtol=3e-5, atol=3e-5)
    assert a.cooRows().numel() == a.paddedJ.numel()
    assert (a.cooRows()[-7:] == -1).all()


def test_metadata_failures_are_explicit(device):
    q, r, d, qf, rf = case(device)
    prop = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=sph.WarpOperation.Density)
    with pytest.raises(TypeError, match='kernel'):
        sph.warpOperation(q, replace(prop, kernel=1234), d)
    with pytest.raises(NotImplementedError, match='Custom'):
        sph.warpOperation(q, replace(prop, operation=sph.WarpOperation.Custom), d)
    with pytest.raises(ValueError, match='positive n_h'):
        sph.warpOperation(q, replace(prop, calibrateNormalization=True, n_h=0), d)
    assert sph.warpOperation(replace(q, positions=q.positions.clone().requires_grad_()), prop, d).requires_grad
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(q.positions, torch.zeros_like(q.positions))
        result=sph.warpOperation(replace(q, positions=dual), prop, d)
        assert torch.isfinite(torch.autograd.forward_ad.unpack_dual(result).tangent).all()
    with pytest.raises(RuntimeError, match='contiguous'):
        sph.radiusSearch(replace(q, positions=q.positions.T.contiguous().T), d)



@pytest.mark.gpu
def test_current_gpu_stream_and_cpu_parity():
    if not torch.cuda.is_available(): pytest.skip('requires a real GPU')
    q, r, d, qf, rf = case('cpu', torch.float64)
    prop = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=sph.WarpOperation.Density)
    expected = sph.warpOperation(q, prop, d, referenceParticles=r)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        def move(p):
            return replace(p, **{name: getattr(p, name).cuda() for name in
                                  ('positions', 'supports', 'masses', 'kinds', 'densities')})
        qg, rg = move(q), move(r)
        dg = replace(d, min=d.min.cuda(), max=d.max.cuda(), periodic=d.periodic.cuda())
        rg.masses.mul_(2)
        got = sph.warpOperation(qg, prop, dg, referenceParticles=rg)
        assert torch.cuda.current_stream() == stream
    stream.synchronize()
    torch.testing.assert_close(got.cpu(), 2 * expected, rtol=3e-12, atol=3e-12)


@pytest.mark.cpu
def test_imports_with_warp_blocked(tmp_path):
    # Reject attempted imports, rather than merely checking sys.modules after.
    (tmp_path / 'sitecustomize.py').write_text('''
import importlib.abc, sys
class NoWarp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "warp" or fullname.startswith("warp."):
            raise ImportError("Warp import blocked by native qualification")
sys.meta_path.insert(0, NoWarp())
''')
    root = Path(__file__).resolve().parents[3]
    package_root = Path(sph.__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(tmp_path), str(package_root), os.environ.get('PYTHONPATH', ''))))
    subprocess.run([sys.executable, str(root / 'scripts/examples/native_core.py'), '--device', 'cpu'],
                   env=env, check=True, capture_output=True, text=True)


def test_zero_support_and_inference_with_grad_inputs(device):
    q, _, d, _, _ = case(device)
    q.supports.zero_()
    q.positions.requires_grad_()
    p = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=sph.WarpOperation.Density)
    with torch.no_grad():
        a = sph.radiusSearch(q, d)
        result = sph.warpOperation(q, p, d, adjacency=a)
    assert a.numEdges == 0 and not result.requires_grad
    torch.testing.assert_close(result, torch.zeros_like(q.masses), rtol=0, atol=0)


@pytest.mark.cpu
def test_parallel_cpu_rows_match_serial_and_oracle():
    q, r, d, qf, rf = case('cpu', torch.float64, dim=3, nq=129, nr=67)
    prop = sph.OperationProperties(sph.KernelFunctions.Wendland2,
                                    operation=sph.WarpOperation.Gradient,
                                    gradientMode=sph.GradientScheme.Difference,
                                    operationMode=sph.OperationDirection.TrueAllToToAll)
    _, _, expected = reference(q, r, d, prop.supportMode, prop.operationMode, qf, rf)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        serial = sph.warpOperation(q, prop, d, qf, rf, referenceParticles=r)
        torch.set_num_threads(4)
        parallel = sph.warpOperation(q, prop, d, qf, rf, referenceParticles=r)
    finally:
        torch.set_num_threads(threads)
    torch.testing.assert_close(parallel, serial, rtol=0, atol=0)
    torch.testing.assert_close(parallel, expected, rtol=3e-12, atol=3e-12)
