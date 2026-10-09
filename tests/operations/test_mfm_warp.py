"""Warp-kernel MFM / MFV backend (`warpSPHCore.mfm.MeshlessWarp`, particle-centred
kernels over the CSR neighbourhood) against the pure-torch reference
(`MeshlessGeometry`, `mfmRates`, `signalTimestep`): same geometry, gradients,
closure, rates and time step up to floating-point summation order, exact
pairwise antisymmetry (conservation), no pair list."""

import pytest
import torch

from test_mfm import GAMMA, KERNEL, _lattice
from warpSPHCore import DomainDescription, ParticleState
from warpSPHCore.enumTypes import KernelFunctions
from warpSPHCore.mfm import MeshlessGeometry, MeshlessWarp, mfmRates, signalTimestep


def _random_state(P, seed=3):
    N = P.positions.shape[0]
    gen = torch.Generator().manual_seed(seed)
    rnd = lambda *s: torch.rand(*s, generator=gen, dtype=torch.float64).to(P.positions.dtype).to(P.positions.device)
    return 1 + 0.3 * rnd(N), rnd(N, P.positions.shape[1]) - 0.5, 1 + 0.3 * rnd(N)


@pytest.mark.parametrize("jitter", [0.0, 0.3])
def test_geometry_gradients_and_closure_match_the_torch_reference(device, jitter):
    P, dom = _lattice(device, 14, jitter=jitter)
    g = MeshlessGeometry.build(P, dom, KERNEL)
    w = MeshlessWarp.build(P.positions, P.supports, dom, KERNEL)
    assert torch.allclose(g.omega, w.omega, rtol=2e-5)
    assert torch.allclose(g.Einv, w.Einv, rtol=1e-4, atol=1e-4 * g.Einv.abs().max().item())
    assert torch.allclose(g.cond, w.cond, rtol=1e-4)
    rho, vel, pres = _random_state(P)
    G = g.gradient(torch.cat([rho[:, None], vel, pres[:, None]], 1))
    gr, gv, gp = w.gradients(rho, vel, pres)
    scale = G.abs().max().item()
    assert (G[:, 0] - gr).abs().max() < 1e-4 * scale
    assert (G[:, 1:3] - gv).abs().max() < 1e-4 * scale
    assert (G[:, 3] - gp).abs().max() < 1e-4 * scale
    # the paper's closure residual, relative to the face norm, agrees (it is the quantity the projection removes)
    g0 = MeshlessGeometry.build(P, dom, KERNEL, closure="none")
    a, faceNorm = w.residual()
    ref = g0.closure().norm() / g0.A.norm()
    assert (a.norm() / faceNorm).item() == pytest.approx(ref.item(), rel=1e-3, abs=1e-6)
    if jitter > 0:
        assert w.lam.abs().max() > 0


@pytest.mark.parametrize("mode", ["MFM", "MFV"])
@pytest.mark.parametrize("order, dt", [(1, 0.0), (2, 0.0), (2, 1e-3)])
def test_rates_match_the_torch_reference(device, mode, order, dt):
    P, dom = _lattice(device, 14, jitter=0.3)
    g = MeshlessGeometry.build(P, dom, KERNEL)
    w = MeshlessWarp.build(P.positions, P.supports, dom, KERNEL)
    rho, vel, pres = _random_state(P)
    ref, _ = mfmRates(g, rho, vel, pres, GAMMA, dt=dt, mode=mode, order=order)
    got = w.rates(rho, vel, pres, GAMMA, dt=dt, mode=mode, order=order)
    assert got.shape == ref.shape
    assert (ref - got).norm() < 1e-4 * ref.norm()
    # the flux is bitwise identical on both sides of a pair: columns sum to zero to round-off
    assert got.sum(0).abs().max() < 1e-4 * got.abs().max()
    if mode == "MFM":
        assert got[:, 0].abs().max() == 0


def test_uniform_flow_is_preserved_by_the_warp_backend(device):
    P, dom = _lattice(device, 16, jitter=0.3)
    w = MeshlessWarp.build(P.positions, P.supports, dom, KERNEL)
    N = P.positions.shape[0]
    rho = torch.full((N,), 1.2, dtype=P.positions.dtype, device=device)
    vel = torch.tensor([0.6, -0.4], dtype=P.positions.dtype, device=device).expand(N, 2).clone()
    pres = torch.full((N,), 0.9, dtype=P.positions.dtype, device=device)
    for mode in ("MFM", "MFV"):
        r = w.rates(rho, vel, pres, GAMMA, dt=1e-3, mode=mode)
        assert r.abs().max() < 1e-3 * 0.9 * (w.volume / w.particles.supports).max().item() * 10


def test_timestep_matches_the_torch_reference(device):
    P, dom = _lattice(device, 14, jitter=0.2)
    g = MeshlessGeometry.build(P, dom, KERNEL)
    w = MeshlessWarp.build(P.positions, P.supports, dom, KERNEL)
    rho, vel, pres = _random_state(P)
    assert torch.allclose(signalTimestep(g, rho, vel, pres, GAMMA, 0.2), w.timestep(rho, vel, pres, GAMMA, 0.2), rtol=1e-4)


def test_three_dimensional_rates_match(device):
    n = 6
    ax = torch.arange(n, dtype=torch.float64)
    pts = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3) / n
    gen = torch.Generator().manual_seed(5)
    pts = (pts + (torch.rand(pts.shape, generator=gen, dtype=torch.float64) - 0.5) * 0.2 / n) % 1.0
    N, dx, dt_ = pts.shape[0], 1.0 / n, torch.float32
    dom = DomainDescription(torch.zeros(3, dtype=dt_, device=device), torch.ones(3, dtype=dt_, device=device),
                            torch.ones(3, dtype=torch.bool, device=device), 3)
    pos = pts.to(dt_).to(device)
    h = torch.full((N,), 2.6 * dx, dtype=dt_, device=device)
    P = ParticleState(positions=pos, supports=h, masses=torch.full((N,), dx ** 3, dtype=dt_, device=device),
                      densities=torch.ones(N, dtype=dt_, device=device), kinds=torch.zeros(N, dtype=torch.int32, device=device))
    g = MeshlessGeometry.build(P, dom, KERNEL)
    w = MeshlessWarp.build(pos, h, dom, KERNEL)
    rho, vel, pres = _random_state(P)
    for mode in ("MFM", "MFV"):
        ref, _ = mfmRates(g, rho, vel, pres, GAMMA, dt=1e-3, mode=mode)
        got = w.rates(rho, vel, pres, GAMMA, dt=1e-3, mode=mode)
        assert (ref - got).norm() < 1e-4 * ref.norm()
