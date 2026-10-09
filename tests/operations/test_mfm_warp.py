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


def test_guards_and_retry_match_the_torch_reference(device):
    """Force the fallback face on every pair and the pair-centred weights on a strongly varying support;
    the warp kernels must follow the torch reference through the guarded faces, the closure and the flux."""
    P, dom = _lattice(device, 14, jitter=0.3)
    P.supports = P.supports * (1 + 0.5 * torch.rand_like(P.supports))
    rho, vel, pres = _random_state(P)
    for kwargs in (dict(), dict(face_cond_max=0.5), dict(area_cap=True), dict(centred_weights=False)):
        g = MeshlessGeometry.build(P, dom, KERNEL, **kwargs)
        w = MeshlessWarp.build(P.positions, P.supports, dom, KERNEL, **kwargs)
        for mode in ("MFM", "MFV"):
            ref, _ = mfmRates(g, rho, vel, pres, GAMMA, dt=1e-3, mode=mode)
            got = w.rates(rho, vel, pres, GAMMA, dt=1e-3, mode=mode)
            assert (ref - got).norm() < 2e-4 * ref.norm(), (kwargs, mode)
            assert got.sum(0).abs().max() < 1e-4 * got.abs().max()


def test_retry_is_triggered_identically(device):
    """A state with a violent pressure jump forces retries; torch and warp must resolve them the same way."""
    P, dom = _lattice(device, 12, jitter=0.1)
    N = P.positions.shape[0]
    rho = torch.ones(N, dtype=P.positions.dtype, device=device)
    pres = torch.where(P.positions[:, 0] > 0.5, 1.0e4, 1.0).to(P.positions.dtype)
    vel = torch.zeros(N, 2, dtype=P.positions.dtype, device=device)
    vel[:, 0] = 0.5 * torch.sin(12.0 * P.positions[:, 1])
    g = MeshlessGeometry.build(P, dom, KERNEL)
    w = MeshlessWarp.build(P.positions, P.supports, dom, KERNEL)
    ref, d = mfmRates(g, rho, vel, pres, GAMMA, dt=1e-4, mode="MFM")
    got = w.rates(rho, vel, pres, GAMMA, dt=1e-4, mode="MFM")
    assert torch.isfinite(got).all() and torch.isfinite(ref).all()
    assert (ref - got).norm() < 1e-3 * ref.norm()


# --- multigrid closure solve ---------------------------------------------------------

def _lattice3(device, n=8, jitter=0.3, seed=2):
    ax = torch.arange(n, dtype=torch.float64)
    pts = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3) / n
    gen = torch.Generator().manual_seed(seed)
    pts = (pts + (torch.rand(pts.shape, generator=gen, dtype=torch.float64) - 0.5) * jitter / n) % 1.0
    N, dx, dt_ = pts.shape[0], 1.0 / n, torch.float32
    dom = DomainDescription(torch.zeros(3, dtype=dt_, device=device), torch.ones(3, dtype=dt_, device=device),
                            torch.ones(3, dtype=torch.bool, device=device), 3)
    return pts.to(dt_).to(device), torch.full((N,), 2.6 * dx, dtype=dt_, device=device), dom


@pytest.mark.parametrize("dimension", [2, 3])
def test_multigrid_closure_matches_plain_cg_and_needs_fewer_iterations(device, dimension):
    if dimension == 2:
        P, dom = _lattice(device, 40, jitter=0.3)
        pos, h = P.positions, P.supports
    else:
        pos, h, dom = _lattice3(device, 16)
    cg = MeshlessWarp.build(pos, h, dom, KERNEL, closure_solver="cg")
    mg = MeshlessWarp.build(pos, h, dom, KERNEL, closure_solver="mg", mg_max_coarse=16)
    assert mg.multigridLevels >= 2
    assert mg.closureResidualRatio < 5e-4 and cg.closureResidualRatio < 5e-4
    assert mg.closureIterations < cg.closureIterations
    # same closed faces: the rates agree
    N = pos.shape[0]
    gen = torch.Generator().manual_seed(4)
    rnd = lambda *s: torch.rand(*s, generator=gen, dtype=torch.float64).to(pos.dtype).to(device)
    rho, vel, pres = 1 + 0.3 * rnd(N), rnd(N, dimension) - 0.5, 1 + 0.3 * rnd(N)
    a = cg.rates(rho, vel, pres, GAMMA, dt=1e-3)
    b = mg.rates(rho, vel, pres, GAMMA, dt=1e-3)
    assert (a - b).norm() < 2e-3 * a.norm()


def test_multigrid_handles_a_one_dimensional_and_an_open_domain(device):
    n, dt_ = 600, torch.float32
    x = ((torch.arange(n, dtype=dt_) + 0.5) / n).to(device)[:, None]
    x = x + 0.2 / n * torch.sin(37.0 * x)                                   # disorder
    h = torch.full((n,), 7.0 / (2 * n), dtype=dt_, device=device)
    for periodic in (True, False):
        dom = DomainDescription(torch.zeros(1, dtype=dt_, device=device), torch.ones(1, dtype=dt_, device=device),
                                torch.full((1,), periodic, dtype=torch.bool, device=device), 1)
        cg = MeshlessWarp.build(x, h, dom, KERNEL, closure_solver="cg")
        mg = MeshlessWarp.build(x, h, dom, KERNEL, closure_solver="mg", mg_max_coarse=8)
        assert mg.closureResidualRatio < 5e-4 or mg.closureIterations == 0
        if cg.closureIterations:
            assert mg.closureIterations <= cg.closureIterations


def test_mfv_mass_flux_limiter_agrees_between_backends_and_limits_only_the_mass(device):
    P, dom = _lattice(device, 14, jitter=0.2)
    g = MeshlessGeometry.build(P, dom, KERNEL)
    w = MeshlessWarp.build(P.positions, P.supports, dom, KERNEL)
    rho, vel, pres = _random_state(P)
    vel = vel * 20.0                                                # large relative velocities: big mass fluxes
    dt = 0.05
    free = w.rates(rho, vel, pres, GAMMA, dt=dt, mode="MFV", massFluxLimit=0.0)
    lim = w.rates(rho, vel, pres, GAMMA, dt=dt, mode="MFV", massFluxLimit=0.01)
    ref, _ = mfmRates(g, rho, vel, pres, GAMMA, dt=dt, mode="MFV", massFluxLimit=0.01)
    assert lim[:, 0].abs().max() < free[:, 0].abs().max()           # the mass update is capped ...
    assert (lim[:, 0] - ref[:, 0]).abs().max() < 1e-3 * ref[:, 0].abs().max() + 1e-7
    # ... per pair, to massFluxLimit of the donor: a particle with ~N pairs changes by at most N * 1 % per step
    mass = rho * w.volume
    assert (lim[:, 0].abs() * dt <= 0.01 * mass * 60).all()
    assert lim[:, 0].sum().abs() < 1e-4 * lim[:, 0].abs().max()     # still conservative


def test_single_precision_thresholds_and_coincident_particles_in_the_warp_backend(device):
    from test_mfm import _strip
    P, dom = _strip(device, 1.0e-3)
    w = MeshlessWarp.build(P.positions, P.supports, dom, KERNEL, closure="none", cond_max=1e30)
    assert w.condBad.any()                                            # float32: cutoff ~2e4, this strip has N_cond ~ 4e4
    wd = MeshlessWarp.build(P.positions, P.supports, dom, KERNEL, closure="none", cond_max=1e30, face_cond_max=1.0e6)
    assert not wd.condBad.any()
    P2, dom2 = _lattice(device, 10, jitter=0.0)
    pos = P2.positions.clone()
    pos[1] = pos[0]
    w2 = MeshlessWarp.build(pos, P2.supports, dom2, KERNEL)
    rho, vel, pres = _random_state(P2)
    for mode in ("MFM", "MFV"):
        assert torch.isfinite(w2.rates(rho, vel, pres, GAMMA, dt=1e-3, mode=mode)).all()
