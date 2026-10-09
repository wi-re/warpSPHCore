"""Meshless finite-mass / finite-volume backend (`warpSPHCore.mfm`, Hopkins 2015):
kernel weights against the warp operators, the matrix-gradient / effective-face
identities, face-vector closure, conservation of the flux assembly, the HLLC
star state, the limiters, and a short end-to-end shock tube. Periodic 2-D
jittered lattice and a 1-D tube; torch throughout, so every test also runs on CPU.
"""

import math

import pytest
import torch

from warpSPHCore import (DomainDescription, OperationProperties, ParticleState,
                         radiusSearchCompactHashMap, warpOperation)
from warpSPHCore.enumTypes import (KernelFunctions, OperationDirection,
                                   SupportScheme, WarpOperation)
from warpSPHCore.mfm import (MeshlessGeometry, SUPPORTED_KERNELS, conserved, faceFlux,
                             kernelWeight, mfmRates, pairLimit, primitives, signalTimestep,
                             slopeLimiter, starState, conditionBeta)

KERNEL = KernelFunctions.Wendland2
GAMMA = 1.4


def _lattice(device, n, jitter=0.3, seed=0, periodic=True, dtype=torch.float32, support=3.0):
    g = torch.Generator().manual_seed(seed)
    ax = torch.arange(n, dtype=torch.float64)
    pts = torch.stack(torch.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2) / n
    pts = pts + (torch.rand(pts.shape, generator=g, dtype=torch.float64) - 0.5) * jitter / n
    pts = pts % 1.0
    N = pts.shape[0]
    dx = 1.0 / n
    domain = DomainDescription(torch.zeros(2, dtype=dtype, device=device),
                               torch.ones(2, dtype=dtype, device=device),
                               torch.full((2,), periodic, dtype=torch.bool, device=device), 2)
    P = ParticleState(positions=pts.to(dtype).to(device),
                      supports=torch.full((N,), support * dx, dtype=dtype, device=device),
                      masses=torch.full((N,), dx * dx, dtype=dtype, device=device),
                      densities=torch.ones(N, dtype=dtype, device=device),
                      kinds=torch.zeros(N, dtype=torch.int32, device=device))
    return P, domain


# --- kernel weights ----------------------------------------------------------------

@pytest.mark.parametrize("kernel", SUPPORTED_KERNELS)
def test_kernel_weights_match_the_warp_density_operator(device, kernel):
    P, dom = _lattice(device, 18, support=3.2)
    g = MeshlessGeometry.build(P, dom, kernel)
    adj = radiusSearchCompactHashMap(P, dom, mode=SupportScheme.SuperSymmetric)
    ones = ParticleState(positions=P.positions, supports=P.supports,
                         masses=torch.ones_like(P.supports), densities=torch.ones_like(P.supports),
                         kinds=P.kinds)
    omega = warpOperation(ones, OperationProperties(
        kernel=kernel, operation=WarpOperation.Density, supportMode=SupportScheme.Gather,
        operationMode=OperationDirection.AllToAll), dom, adjacency=adj)
    assert torch.allclose(g.omega, omega, rtol=2e-5)


def test_unsupported_kernel_is_refused():
    with pytest.raises(NotImplementedError):
        kernelWeight(torch.zeros(1), torch.ones(1), 2, KernelFunctions.Gaussian)


# --- geometry ----------------------------------------------------------------------

def test_matrix_gradient_is_exact_for_linear_fields(device):
    P, dom = _lattice(device, 20, periodic=False)
    g = MeshlessGeometry.build(P, dom, KERNEL)
    x = P.positions
    G = torch.tensor([[1.3, -0.4], [0.7, 0.2]], dtype=x.dtype, device=device)
    f = torch.stack([x @ torch.tensor([0.9, -1.7], dtype=x.dtype, device=device) + 0.3, x @ G[0], x @ G[1]], dim=1)
    gr = g.gradient(f)                                                    # (N, 3, 2)
    ref = torch.stack([torch.tensor([0.9, -1.7], dtype=x.dtype, device=device).expand(len(x), 2), G[0].expand(len(x), 2),
                       G[1].expand(len(x), 2)], dim=1)
    ok = ~g.deficient
    assert ok.float().mean() > 0.8
    assert torch.allclose(gr[ok], ref[ok], atol=2e-4)


def test_face_vectors_are_antisymmetric_by_construction(device):
    """Every conserved quantity sums to zero over the particles whatever the fluxes are."""
    P, dom = _lattice(device, 16)
    g = MeshlessGeometry.build(P, dom, KERNEL)
    flux = torch.randn(g.numPairs, 4, dtype=P.positions.dtype, device=device)
    rates = g.divergence(flux)
    assert rates.sum(0).abs().max() < 1e-4 * rates.abs().max()


def test_closure_projection_closes_disordered_faces(device):
    P, dom = _lattice(device, 20, jitter=0.4)
    paper = MeshlessGeometry.build(P, dom, KERNEL, closure="none")
    closed = MeshlessGeometry.build(P, dom, KERNEL, closure="project")
    rel = lambda g: (g.closure().norm(dim=1).pow(2).mean().sqrt() / g.A.norm(dim=1).mean()).item()
    assert rel(paper) > 0.2                       # the paper's face vector does not close on disordered sets
    assert rel(closed) < 1e-3
    assert torch.allclose(paper.A0, closed.A0)


def test_closure_is_a_noop_on_a_regular_lattice(device):
    P, dom = _lattice(device, 16, jitter=0.0)
    paper = MeshlessGeometry.build(P, dom, KERNEL, closure="none")
    closed = MeshlessGeometry.build(P, dom, KERNEL, closure="project")
    assert paper.closure().abs().max() < 1e-4 * paper.A.abs().max()
    assert torch.allclose(paper.A, closed.A, atol=1e-6 * paper.A.abs().max().item())


def test_linear_flux_divergence_keeps_its_consistency_after_closure(device):
    """sum_j A_ij . F(x_ij) = V tr(G) for a linear flux F = G (x - x_i): the consistency the matrix
    gradient guarantees. The closure projection must not spoil it (it changes the sum by O(noise))."""
    P, dom = _lattice(device, 24, jitter=0.3)
    Gm = torch.tensor([[0.7, 0.3], [-0.2, 0.5]], dtype=P.positions.dtype, device=device)
    for closure, tol in (("none", 0.05), ("project", 0.05)):
        g = MeshlessGeometry.build(P, dom, KERNEL, closure=closure)
        frac = g.faceFraction()
        Fi = torch.einsum("ab,pb->pa", Gm, frac[:, None] * g.d)
        Fj = torch.einsum("ab,pb->pa", Gm, -(1 - frac)[:, None] * g.d)
        out = -torch.zeros(g.N, dtype=g.d.dtype, device=device).index_add(0, g.i, (g.A * Fi).sum(-1)) \
            + torch.zeros(g.N, dtype=g.d.dtype, device=device).index_add(0, g.j, (g.A * Fj).sum(-1))
        exact = -g.volume * torch.trace(Gm)
        assert ((out - exact).pow(2).mean().sqrt() / exact.abs().mean()) < tol


def test_geometry_is_differentiable_in_positions(device):
    P, dom = _lattice(device, 10)
    pos = P.positions.clone().requires_grad_(True)
    P2 = ParticleState(positions=pos, supports=P.supports, masses=P.masses, densities=P.densities, kinds=P.kinds)
    g = MeshlessGeometry.build(P2, dom, KERNEL)
    f = torch.sin(6.0 * pos[:, 0])
    (g.gradient(f).pow(2).sum() + g.A.pow(2).sum()).backward()
    assert torch.isfinite(pos.grad).all() and pos.grad.abs().max() > 0


# --- Riemann solver ----------------------------------------------------------------

def _star(rl, ul, pl, rr, ur, pr):
    t = lambda v: torch.tensor([v], dtype=torch.float64)
    z = torch.zeros(1, 1, dtype=torch.float64)
    SL, SR, Ss, Ps = starState(t(rl), t(ul), z, t(pl), t(rr), t(ur), z, t(pr), GAMMA)
    return Ss.item(), Ps.item()


def test_hllc_star_state_reduces_to_the_acoustic_solution_for_weak_jumps():
    """Weak waves: P* = (P_L + P_R)/2 - rho c (u_R - u_L)/2, S* = (u_L + u_R)/2 + (P_L - P_R)/(2 rho c)."""
    rl, ul, pl, rr, ur, pr = 1.0, 0.01, 1.0, 1.003, -0.01, 1.004
    Ss, Ps = _star(rl, ul, pl, rr, ur, pr)
    rho_c = 0.5 * (rl + rr) * math.sqrt(GAMMA * 0.5 * (pl + pr) / (0.5 * (rl + rr)))
    assert Ps == pytest.approx(0.5 * (pl + pr) - 0.5 * rho_c * (ur - ul), rel=2e-3)
    assert Ss == pytest.approx(0.5 * (ul + ur) + (pl - pr) / (2 * rho_c), abs=2e-3)


def test_hllc_sod_star_state_is_positive_and_between_the_states():
    """HLLC's wave model collapses the rarefaction, so P*, S* are not the exact 0.303 / 0.927 (it gives
    ~0.198 / 0.68 with Roe speeds, the well-known behaviour); they must still be physical."""
    Ss, Ps = _star(1.0, 0.0, 1.0, 0.125, 0.0, 0.1)
    assert 0.1 < Ps < 1.0 and 0.0 < Ss < 1.5


def test_flux_of_identical_states_is_the_physical_flux_in_both_modes():
    dim = 2
    rho, P = torch.tensor([1.3]), torch.tensor([0.8])
    v = torch.tensor([[0.4, -0.7]])
    n = torch.tensor([[0.6, 0.8]])
    vf = torch.tensor([[0.1, 0.2]])                      # arbitrary frame velocity
    vb = v - vf
    u = (vb * n).sum(-1)
    vt = vb - u[:, None] * n
    for mode in ("MFV", "MFM"):
        flux, Ss, Ps = faceFlux(rho, u, vt, P, rho, u, vt, P, GAMMA, mode, n, vf)
        # MFV: the Euler flux relative to the face, which moves with vf (lab-frame quantities)
        vn = (v * n).sum(-1)
        ub = u
        E = P / (GAMMA - 1) + 0.5 * rho * (v * v).sum(-1)
        phys = torch.cat([(rho * ub)[:, None], (rho * ub)[:, None] * v + P[:, None] * n, (E * ub + P * vn)[:, None]], -1)
        if mode == "MFV":
            assert torch.allclose(flux, phys, atol=1e-5)
        else:  # MFM moves the face with the contact: no mass flux, momentum flux P n, energy flux P (v_face . n)
            assert flux[0, 0] == 0
            assert torch.allclose(flux[0, 1:3], P[0] * n[0], atol=1e-6)
            assert flux[0, 3] == pytest.approx(float(P[0] * ((vf * n).sum() + Ss[0])), rel=1e-5)
        assert Ps[0] == pytest.approx(0.8, rel=1e-5)


def test_mfm_flux_is_galilean_invariant():
    t = lambda *a: torch.tensor([a[0]], dtype=torch.float64)
    n = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    z = torch.zeros(1, 2, dtype=torch.float64)
    res = []
    for boost in (0.0, 3.0):
        vf = torch.tensor([[boost, 0.0]], dtype=torch.float64)
        flux, Ss, Ps = faceFlux(t(1.0), t(0.2), z, t(1.0), t(0.4), t(-0.1), z, t(0.3), GAMMA, "MFM", n, vf)
        res.append((flux, Ss, Ps))
    assert torch.allclose(res[0][1], res[1][1]) and torch.allclose(res[0][2], res[1][2])
    assert torch.allclose(res[0][0][0, :2], res[1][0][0, :2])                      # mass, momentum
    assert res[1][0][0, 3] - res[0][0][0, 3] == pytest.approx(3.0 * float(res[0][2][0]), rel=1e-9)   # energy flux shifts by v . P n


# --- limiters ----------------------------------------------------------------------

def test_pair_limiter_keeps_the_face_value_between_the_neighbours():
    g = torch.Generator().manual_seed(1)
    a, b = torch.rand(4000, dtype=torch.float64, generator=g) * 4 - 2, torch.rand(4000, dtype=torch.float64, generator=g) * 4 - 2
    raw = a + (torch.rand(4000, dtype=torch.float64, generator=g) * 6 - 3) * (b - a)       # wild unlimited reconstruction
    lim = pairLimit(a, b, raw, torch.full_like(a, 0.5))
    lo, hi = torch.minimum(a, b), torch.maximum(a, b)
    slack = 0.5 * (hi - lo) + 1e-12                                                          # psi_1 = 1/2 allows a bounded overshoot
    assert (lim >= lo - slack).all() and (lim <= hi + slack).all()
    # the sign of a positive field cannot flip
    pa, pb = a.abs() + 0.1, b.abs() + 0.1
    assert (pairLimit(pa, pb, pa + (pb - pa) * (-5.0), torch.full_like(pa, 0.5)) > 0).all()
    # a perfectly linear profile is left alone
    mid = 0.5 * (a + b)
    assert torch.allclose(pairLimit(a, b, mid, torch.full_like(a, 0.5)), mid)


def test_slope_limiter_is_one_on_linear_data_and_bounds_the_face_values(device):
    P, dom = _lattice(device, 18, jitter=0.2, periodic=False)
    g = MeshlessGeometry.build(P, dom, KERNEL)
    x = P.positions
    lin = (x @ torch.tensor([0.7, -0.3], dtype=x.dtype, device=device))[:, None]
    beta = conditionBeta(g.cond)
    a = slopeLimiter(g, lin, g.gradient(lin), beta)
    interior = (x > 0.25).all(1) & (x < 0.75).all(1)
    assert a[interior].min() > 0.99
    step = (x[:, 0] > 0.5).to(x.dtype)[:, None]
    a = slopeLimiter(g, step, g.gradient(step), beta)
    assert a.min() >= 0 and a.max() <= 1 and a[:, 0].min() < 1               # the jump is limited somewhere


# --- rates -------------------------------------------------------------------------

def _state(P, dom, rho=1.0, vel=(0.0, 0.0), press=1.0):
    N = P.positions.shape[0]
    g = MeshlessGeometry.build(P, dom, KERNEL)
    r = torch.full((N,), rho, dtype=P.positions.dtype, device=P.positions.device)
    v = torch.tensor(vel, dtype=P.positions.dtype, device=P.positions.device).expand(N, 2).clone()
    p = torch.full((N,), press, dtype=P.positions.dtype, device=P.positions.device)
    return g, r, v, p


@pytest.mark.parametrize("mode", ["MFM", "MFV"])
@pytest.mark.parametrize("jitter", [0.0, 0.3])
def test_uniform_flow_is_preserved(device, mode, jitter):
    P, dom = _lattice(device, 16, jitter=jitter)
    g, r, v, p = _state(P, dom, rho=1.2, vel=(0.6, -0.4), press=0.9)
    rates, _ = mfmRates(g, r, v, p, GAMMA, dt=1e-3, mode=mode)
    # sum_j A_ij = 0 after the closure, so a uniform state has no net flux
    scale = (p[0] * g.A.norm(dim=-1).mean() / (r[0] * g.volume.mean())).item()
    assert rates.abs().max() < 1e-3 * p[0].item() * g.A.norm(dim=-1).max().item()


@pytest.mark.parametrize("mode", ["MFM", "MFV"])
def test_rates_conserve_mass_momentum_and_energy(device, mode):
    P, dom = _lattice(device, 14, jitter=0.3)
    g = MeshlessGeometry.build(P, dom, KERNEL)
    N = P.positions.shape[0]
    gen = torch.Generator().manual_seed(3)
    rnd = lambda *s: torch.rand(*s, generator=gen, dtype=torch.float64).to(P.positions.dtype).to(device)
    r, p = 1.0 + 0.5 * rnd(N), 1.0 + 0.5 * rnd(N)
    v = rnd(N, 2) - 0.5
    rates, _ = mfmRates(g, r, v, p, GAMMA, dt=1e-3, mode=mode)
    assert rates.sum(0).abs().max() < 1e-4 * rates.abs().max()
    if mode == "MFM":
        assert rates[:, 0].abs().max() == 0                                       # finite MASS: no mass flux


def test_signal_timestep_is_positive_and_scales_with_the_support(device):
    P, dom = _lattice(device, 14)
    g, r, v, p = _state(P, dom)
    dt = signalTimestep(g, r, v, p, GAMMA, 0.2)
    c = math.sqrt(GAMMA)
    assert dt.min() > 0
    assert dt.min().item() == pytest.approx(2 * 0.2 * P.supports[0].item() / (2 * c), rel=1e-3)


def test_conserved_primitive_roundtrip():
    m = torch.tensor([0.7, 1.1], dtype=torch.float64)
    rho = torch.tensor([1.3, 0.4], dtype=torch.float64)
    v = torch.tensor([[0.2, -0.1], [1.5, 0.3]], dtype=torch.float64)
    p = torch.tensor([0.9, 2.2], dtype=torch.float64)
    Q = conserved(m, rho, v, p, GAMMA)
    r2, v2, p2 = primitives(Q, m / rho, GAMMA)
    assert torch.allclose(r2, rho) and torch.allclose(v2, v) and torch.allclose(p2, p)


# --- end-to-end 1-D shock tube -----------------------------------------------------

@pytest.mark.parametrize("mode", ["MFM", "MFV"])
def test_periodic_sod_tube_conserves_and_resolves_the_star_state(device, mode):
    n, dt_, nngb = 200, torch.float32, 7.0
    x = ((torch.arange(n, dtype=dt_) + 0.5) / n).to(device)
    high = (x >= 0.25) & (x < 0.75)
    rho = torch.where(high, 1.0, 0.125).to(dt_)
    p = torch.where(high, 1.0, 0.1).to(dt_)
    v = torch.zeros(n, 1, dtype=dt_, device=device)
    mass = rho / n
    dom = DomainDescription(torch.zeros(1, dtype=dt_, device=device), torch.ones(1, dtype=dt_, device=device),
                            torch.ones(1, dtype=torch.bool, device=device), 1)
    h = torch.full((n,), nngb / (2.0 * n), dtype=dt_, device=device)
    pos, Q = x[:, None], conserved(mass, rho, v, p, GAMMA)
    Q0, t = Q.sum(0).clone(), 0.0
    kinds = torch.zeros(n, dtype=torch.int32, device=device)
    while t < 0.1 - 1e-9:
        P = ParticleState(positions=pos, supports=h, masses=mass, densities=None, kinds=kinds)
        g = MeshlessGeometry.build(P, dom, KERNEL)
        r, vel, pr = primitives(Q, g.volume, GAMMA)
        dt = min(float(signalTimestep(g, r, vel, pr, GAMMA, 0.2).min()), 0.1 - t)
        rates, _ = mfmRates(g, r, vel, pr, GAMMA, dt=dt, mode=mode)
        Qn = Q + dt * rates
        pos = torch.remainder(pos + 0.5 * dt * (vel + Qn[:, 1:-1] / Qn[:, :1]), 1.0)
        h = (nngb * g.volume / 2.0)
        Q, t = Qn, t + dt
    assert (Q.sum(0) - Q0).abs().max() < 2e-5                                       # conservation to round-off
    P = ParticleState(positions=pos, supports=h, masses=mass, densities=None, kinds=kinds)
    g = MeshlessGeometry.build(P, dom, KERNEL)
    r, vel, pr = primitives(Q, g.volume, GAMMA)
    assert torch.isfinite(r).all() and r.min() > 0.1 and r.max() < 1.05            # no overshoot, no vacuum
    # the star-region plateau behind the right-moving shock/contact: u* = 0.927, P* = 0.303
    xs = pos[:, 0]
    plateau = (xs > 0.78) & (xs < 0.82) if False else ((xs - 0.75 > 0.02) & (xs - 0.75 < 0.08))
    assert vel[plateau, 0].mean().item() == pytest.approx(0.927, abs=0.08)
    assert pr[plateau].mean().item() == pytest.approx(0.303, abs=0.05)


# --- extras ------------------------------------------------------------------------

def test_a_custom_star_state_plugs_into_the_mfm_flux(device):
    P, dom = _lattice(device, 12)
    g, r, v, p = _state(P, dom, press=2.0)
    calls = []

    def acoustic(rL, uL, vtL, PL, rR, uR, vtR, PR):
        calls.append(rL.shape[0])
        return 0.5 * (uL + uR), 0.5 * (PL + PR)

    rates, d = mfmRates(g, r, v, p, GAMMA, dt=1e-3, mode="MFM", starFn=acoustic)
    assert calls and d["Pstar"].allclose(torch.full_like(d["Pstar"], 2.0), atol=1e-5)


def test_three_dimensional_uniform_flow_and_conservation(device):
    n = 6
    ax = torch.arange(n, dtype=torch.float64)
    pts = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3) / n
    gen = torch.Generator().manual_seed(5)
    pts = (pts + (torch.rand(pts.shape, generator=gen, dtype=torch.float64) - 0.5) * 0.2 / n) % 1.0
    N, dx = pts.shape[0], 1.0 / n
    dt_ = torch.float32
    dom = DomainDescription(torch.zeros(3, dtype=dt_, device=device), torch.ones(3, dtype=dt_, device=device),
                            torch.ones(3, dtype=torch.bool, device=device), 3)
    P = ParticleState(positions=pts.to(dt_).to(device), supports=torch.full((N,), 2.6 * dx, dtype=dt_, device=device),
                      masses=torch.full((N,), dx ** 3, dtype=dt_, device=device),
                      densities=torch.ones(N, dtype=dt_, device=device), kinds=torch.zeros(N, dtype=torch.int32, device=device))
    g = MeshlessGeometry.build(P, dom, KERNEL)
    assert g.dim == 3 and g.A.shape[1] == 3 and g.Einv.shape[1:] == (3, 3)
    rho = torch.full((N,), 1.1, dtype=dt_, device=device)
    vel = torch.tensor([0.3, -0.2, 0.5], dtype=dt_, device=device).expand(N, 3).clone()
    press = torch.full((N,), 0.7, dtype=dt_, device=device)
    for mode in ("MFM", "MFV"):
        rates, _ = mfmRates(g, rho, vel, press, GAMMA, dt=1e-3, mode=mode)
        assert rates.shape == (N, 5)
        assert rates.abs().max() < 1e-3 * 0.7 * g.A.norm(dim=-1).max().item()
    # conservation with random states
    rho, press = 1 + 0.3 * torch.rand(N, dtype=dt_, device=device), 1 + 0.3 * torch.rand(N, dtype=dt_, device=device)
    vel = torch.rand(N, 3, dtype=dt_, device=device) - 0.5
    rates, _ = mfmRates(g, rho, vel, press, GAMMA, dt=1e-3, mode="MFV")
    assert rates.sum(0).abs().max() < 1e-4 * rates.abs().max()


def test_forward_mode_through_the_geometry(device):
    import torch.autograd.forward_ad as fwAD
    P, dom = _lattice(device, 10)
    tang = torch.randn_like(P.positions) * 1e-3
    f = torch.sin(6.0 * P.positions[:, 0])
    with fwAD.dual_level():
        pos = fwAD.make_dual(P.positions, tang)
        P2 = ParticleState(positions=pos, supports=P.supports, masses=P.masses, densities=P.densities, kinds=P.kinds)
        g = MeshlessGeometry.build(P2, dom, KERNEL)
        t = fwAD.unpack_dual(g.gradient(f)).tangent
    assert t is not None and torch.isfinite(t).all() and t.abs().max() > 0


# --- GIZMO guards ------------------------------------------------------------------

def test_kernel_derivative_matches_autograd():
    from warpSPHCore.mfm import kernelDerivative
    r = torch.linspace(0.02, 0.98, 40, dtype=torch.float64, requires_grad=True)
    h = torch.tensor(1.0, dtype=torch.float64)
    for kernel in SUPPORTED_KERNELS:
        for dim in (1, 2, 3):
            g, = torch.autograd.grad(kernelWeight(r, h, dim, kernel).sum(), r)
            d = kernelDerivative(r.detach(), h, dim, kernel)
            assert torch.allclose(g, d, rtol=1e-9, atol=1e-12), (kernel, dim)


def test_fallback_face_is_used_where_the_matrix_is_ill_conditioned_or_the_face_points_backwards(device):
    P, dom = _lattice(device, 14, jitter=0.3)
    plain = MeshlessGeometry.build(P, dom, KERNEL, guards=False, closure="none")
    forced = MeshlessGeometry.build(P, dom, KERNEL, face_cond_max=0.5, closure="none")      # every row "ill-conditioned"
    assert not plain.fallback.any() and forced.fallback.all()
    # SPH-style face: along the pair axis, positive, antisymmetric by construction
    cosd = (forced.A * forced.d).sum(-1) / (forced.A.norm(dim=-1) * forced.d.norm(dim=-1)).clamp_min(1e-30)
    assert (cosd > 0.999999).all()
    # the unguarded sums keep working: a faked negative A.d is replaced
    g = MeshlessGeometry.build(P, dom, KERNEL, closure="none")
    assert (((g.A * g.d).sum(-1)) >= 0).all()


def test_guards_leave_well_conditioned_faces_untouched(device):
    P, dom = _lattice(device, 14, jitter=0.2)
    a = MeshlessGeometry.build(P, dom, KERNEL, guards=False)
    b = MeshlessGeometry.build(P, dom, KERNEL)
    assert not b.fallback.any()
    assert torch.allclose(a.A, b.A, atol=1e-7)


def test_centred_weights_and_area_cap(device):
    P, dom = _lattice(device, 14, jitter=0.0)
    P.supports = P.supports * (1 + 0.6 * torch.rand_like(P.supports, generator=None))        # strongly varying support
    g = MeshlessGeometry.build(P, dom, KERNEL, closure="none")
    g_plain = MeshlessGeometry.build(P, dom, KERNEL, closure="none", centred_weights=False)
    assert torch.isfinite(g.A).all() and torch.isfinite(g_plain.A).all()
    capped = MeshlessGeometry.build(P, dom, KERNEL, closure="none", area_cap=True)
    expected = 2 * 3.141592653589793 * (capped.volume ** 0.5)
    assert (capped.A.norm(dim=-1) <= torch.minimum(expected[capped.i], expected[capped.j]) * 1.0001).all()


def test_riemann_retry_falls_back_to_first_order_states_and_then_to_zero_jump():
    from warpSPHCore.mfm import robustFaceFlux
    f = lambda v: torch.tensor([v], dtype=torch.float64)
    z = torch.zeros(1, 1, dtype=torch.float64)
    n, vf = torch.ones(1, 1, dtype=torch.float64), torch.zeros(1, 1, dtype=torch.float64)
    good = (f(1.0), f(0.1), z, f(1.0), f(1.0), f(-0.1), z, f(1.0))
    # primary states with a wild (but positive) pressure far above the limit trigger the first-order states
    wild = (f(1.0), f(0.1), z, f(1.0e6), f(1.0), f(-0.1), z, f(1.0e6))
    flux, Ss, Ps, stage = robustFaceFlux(wild, good, GAMMA, "MFM", n, vf, f(2.0))
    assert stage.item() == 1 and 1.0 < Ps.item() < 1.5          # the compressive first-order star pressure, not 1e6
    # a star-pressure function that is invalid for the first-order states too -> zero velocity jump
    calls = []

    def star(rL, uL, vtL, PL, rR, uR, vtR, PR):
        calls.append(uL.clone())
        bad = (uL.abs() > 0) | (uR.abs() > 0)
        return torch.zeros_like(PL), torch.where(bad, torch.full_like(PL, float("nan")), PL)

    flux, Ss, Ps, stage = robustFaceFlux(wild, good, GAMMA, "MFM", n, vf, f(2.0), starFn=star)
    assert stage.item() == 2 and torch.isfinite(Ps).all() and Ps.item() == pytest.approx(1.0)
    # valid states never retry
    assert robustFaceFlux(good, good, GAMMA, "MFV", n, vf, f(2.0))[3].item() == 0


def test_rates_report_which_pairs_needed_the_retry(device):
    P, dom = _lattice(device, 12, jitter=0.3)
    g, r, v, p = _state(P, dom)
    _, d = mfmRates(g, r, v, p, GAMMA, dt=1e-3, mode="MFM")
    assert (d["stage"] == 0).all()                                    # a smooth uniform state never retries


# --- precision-aware thresholds ------------------------------------------------------

def test_precision_defaults():
    from warpSPHCore.mfm.precision import faceCondMax, pinvAbove, pinvRtol
    assert faceCondMax(torch.float64) == 1.0e6                       # GIZMO's cutoff, tuned for double
    assert 1.0e4 < faceCondMax(torch.float32) < 4.0e4                # ~ 1e-3 / (0.4 eps): the measured error law
    assert pinvRtol(torch.float64) == 1.0e-12 and pinvRtol(torch.float32) > 1.0e-6
    assert pinvAbove(torch.float32) < pinvAbove(torch.float64)


def _strip(device, squeeze, dtype=torch.float32):
    """A thin, rotated strip of particles: N_cond ~ 0.4 / squeeze^2."""
    import math
    n = 40
    ax, ay = torch.arange(n, dtype=torch.float64), torch.arange(7, dtype=torch.float64)
    pts = torch.stack(torch.meshgrid(ax, ay, indexing="ij"), -1).reshape(-1, 2) / n
    gen = torch.Generator().manual_seed(1)
    pts = pts + (torch.rand(pts.shape, generator=gen, dtype=torch.float64) - 0.5) * 0.3 / n
    pts[:, 1] *= squeeze
    th = 0.5
    R = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], dtype=torch.float64)
    pts = pts @ R.T
    N = pts.shape[0]
    dom = DomainDescription(torch.full((2,), -3.0, dtype=dtype, device=device), torch.full((2,), 3.0, dtype=dtype, device=device),
                            torch.zeros(2, dtype=torch.bool, device=device), 2)
    P = ParticleState(positions=pts.to(dtype).to(device), supports=torch.full((N,), 3.5 / n, dtype=dtype, device=device),
                      masses=torch.ones(N, dtype=dtype, device=device), densities=torch.ones(N, dtype=dtype, device=device),
                      kinds=torch.zeros(N, dtype=torch.int32, device=device))
    return P, dom


def test_the_face_fallback_threshold_follows_the_precision(device):
    P, dom = _strip(device, 1.0e-3)                                  # N_cond ~ 4e4 (rotated): fine for GIZMO's 1e6, not in float32
    g32 = MeshlessGeometry.build(P, dom, KERNEL, closure="none", cond_max=1e30)
    assert g32.cond.median() > 2.1e4
    assert g32.fallback.any()                                        # float32 default cutoff 2e4
    gold = MeshlessGeometry.build(P, dom, KERNEL, closure="none", cond_max=1e30, face_cond_max=1.0e6)
    assert not gold.fallback.any()                                   # the double-precision value would have trusted it


def test_coincident_particles_do_not_produce_nan_in_single_precision(device):
    P, dom = _lattice(device, 10, jitter=0.0)
    pos = P.positions.clone()
    pos[1] = pos[0]                                                  # r = 0 pair
    P2 = ParticleState(positions=pos, supports=P.supports, masses=P.masses, densities=P.densities, kinds=P.kinds)
    g = MeshlessGeometry.build(P2, dom, KERNEL)
    r, v, p = _state(P2, dom)[1:]
    rates, _ = mfmRates(g, r, v, p, GAMMA, dt=1e-3, mode="MFV")
    assert torch.isfinite(g.A).all() and torch.isfinite(rates).all()
