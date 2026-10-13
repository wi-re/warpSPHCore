"""Warp-kernel MFM / MFV backend: O(N) memory, no pair list.

:class:`MeshlessWarp` is the warp counterpart of ``MeshlessGeometry`` +
``mfmRates`` (see ``wp_mfm.py`` for the kernels): the per-particle
reductions (moments, gradients, limiter factors, the closure matvec and the
flux sum) run in particle-centred kernels over the CSR neighbour list; only
O(N) work stays in torch (the ``dim x dim`` matrix inverses, the condition
number, the vector algebra of the closure conjugate-gradient iterations).

Same options and same results as the torch reference up to floating-point
summation order (``tests/operations/test_mfm_warp.py`` compares them); the
torch path remains the differentiable one -- these kernels are forward only.
"""

from __future__ import annotations

from typing import Optional

import torch

from ..dataTypes import DomainDescription, OperationProperties, ParticleState
from ..enumTypes import KernelFunctions, OperationDirection, SupportScheme
from .geometry import invertMoments
from .precision import faceCondMax
from .limiters import conditionBeta
from .multigrid import CellMultigrid
from ..type_config import scalar_t
from .fastLaunch import PreparedLauncher
from .wp_mfm import (_CLOSURE_RES_SPEC, _CLOSURE_W_SPEC, _COARSE_STENCIL_SPEC, _FLUX_SPEC, _GRADIENTS_SPEC,
                     _LIMITER_SPEC, _MOMENTS_SPEC, _TIMESTEP_SPEC, _ctx, mfmClosureMatvecWarp)
from .wp_closure import mfmClosureMatvecCSR

__all__ = ["MeshlessWarp"]

#: the closure CG reads its device-side stopping flag every this many iterations (see `_solveClosure`)
_CG_CHECK = 4
#: on CUDA, the closure CG reads the stopping flag of the iteration (or graph replay) this many back, without
#: blocking on the newer ones
_CG_LAG = 1
#: on CUDA (multigrid closure on an adjacency), a solve still running after this many eager iterations replays
#: the rest as a CUDA graph of `_CG_GRAPH_CHUNK` iterations, captured once per closure workspace (None: eager)
_CG_GRAPH_AFTER = 1
_CG_GRAPH_CHUNK = 1
#: spare neighbour-list capacity of a closure workspace (a larger adjacency gets a new workspace)
_WS_CAPACITY = 1.25
#: closure workspaces kept (one per grid / particle count; the SSPRK stages share one)
_WS_MAX = 2
#: outer passes of the bounded closure's lagged-secant iteration (`closure_cap` > 0)
_CAP_OUTER = 8


class _ClosureWorkspace:
    """Fixed-address buffers for the closure solve of one particle count and multigrid grid: the CSR matvec
    (neighbour list and weights with spare capacity), the multigrid hierarchy, the fine diagonal, the CG target
    and loop state, and the CUDA graph of `_CG_GRAPH_CHUNK` CG iterations captured on them. Each solve copies
    its adjacency / weights / hierarchy in (``load``), so one capture serves every later solve: the launches of
    an iteration (~100, mostly the V-cycle's small torch ops) cost one graph replay of host time instead of
    ~1 ms. Same ops on the same values, so bitwise the eager result."""

    def __init__(self, adjacency, kappa, mg, like):
        self.mv = mfmClosureMatvecCSR(adjacency, kappa, capacity=int(adjacency.paddedJ.shape[0] * _WS_CAPACITY) + 1024)
        self.mg = mg                                     # the first hierarchy's tensors are the static ones
        self.diag = torch.empty(like.shape[0], dtype=like.dtype, device=like.device)
        self.target = torch.empty((), dtype=like.dtype, device=like.device)
        self.S = None
        self.graph = None
        self.signature = None

    def load(self, adjacency, kappa, mg) -> bool:
        if not self.mv.assign(adjacency, kappa) or not self.mg.sameGrid(mg):
            return False
        if mg is not self.mg:
            self.mg.assign(mg)
        return True

    def capture(self, iteration, S, chunk):
        """Capture `chunk` iterations of ``iteration`` on the static state (allocated on first use; ``S`` is
        copied in). Captured on a side stream with ``capture_begin`` / ``capture_end`` directly
        (``torch.cuda.graph`` adds a device sync, a gc pass and an allocator flush). The warp kernels inside
        launch on torch's current stream, i.e. the capture stream; they are loaded by the eager iterations."""
        self.S = [t.clone() for t in S]
        self.graph = None
        graph = torch.cuda.CUDAGraph()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            graph.capture_begin(capture_error_mode="thread_local")
            try:
                out = self.S
                for _ in range(chunk):
                    out = iteration(out)
                for t, o in zip(self.S, out):
                    t.copy_(o)
            finally:
                graph.capture_end()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = graph
        self._captured = (out, side)   # the tensors (and stream) the capture used stay alive with the graph
        self.signature = (chunk, self.mg.cholOk, tuple((t.shape, t.dtype) for t in S))


_WORKSPACES: "dict" = {}


def _closureWorkspace(adjacency, kappa, mg, like) -> _ClosureWorkspace:
    key = (str(like.device), like.dtype, like.shape[0], like.shape[1], tuple(mg.n0), tuple(mg.periodic), mg.nLevels)
    ws = _WORKSPACES.get(key)
    if ws is not None and ws.load(adjacency, kappa, mg):
        return ws
    ws = _ClosureWorkspace(adjacency, kappa, mg, like)
    _WORKSPACES.pop(key, None)
    while len(_WORKSPACES) >= _WS_MAX:
        _WORKSPACES.pop(next(iter(_WORKSPACES)))
    _WORKSPACES[key] = ws
    return ws


class MeshlessWarp:
    """Per-particle meshless geometry on the CSR neighbourhood. Build with :meth:`build`.

    Attributes (all ``(N, ...)`` torch tensors): ``omega``, ``volume`` (``1/omega``),
    ``Einv`` (``Ehat^-1``), ``cond`` (Hopkins ``N_cond``), ``deficient``,
    ``num_nbrs`` and ``lam`` (the closure potential, zero when ``closure="none"``)."""

    def __init__(self, particles: ParticleState, props: OperationProperties, domain: DomainDescription,
                 adjacency, kernel: KernelFunctions, dim: int):
        self.particles, self.props, self.domain, self.adjacency = particles, props, domain, adjacency
        self.kernel, self.dim = kernel, dim
        # every kernel of this geometry runs on the same particles / adjacency: the operator ABI prefix is
        # built once (fastLaunch.py) instead of on every launch
        self._fast = PreparedLauncher(_ctx(particles, props, domain, adjacency))

    @classmethod
    def build(cls, positions: torch.Tensor, supports: torch.Tensor, domain: DomainDescription,
              kernel: KernelFunctions, adjacency=None, cond_max: float = 1.0e3, closure: str = "project",
              closure_power: float = 1.0, rtol: Optional[float] = None, cg_tol: Optional[float] = None,
              cg_iters: int = 200, lam0: Optional[torch.Tensor] = None, guards: bool = True,
              face_cond_max: Optional[float] = None, centred_weights: bool = True, area_cap: bool = False,
              closure_solver: str = "auto", mg_min_particles: int = 2048,
              mg_max_coarse: int = 512, lam_cap: float = 1e3, closure_cap: float = 0.0) -> "MeshlessWarp":
        if closure not in ("project", "none"):
            raise ValueError("closure must be 'project' or 'none'")
        N, dim = positions.shape
        ones = torch.ones(N, dtype=positions.dtype, device=positions.device)
        P = ParticleState(positions=positions, supports=supports, masses=ones, densities=ones,
                          kinds=torch.zeros(N, dtype=torch.int32, device=positions.device))
        if adjacency is None:
            from ..radiusSearch import radiusSearchCompactHashMap
            adjacency = radiusSearchCompactHashMap(P, domain, mode=SupportScheme.SuperSymmetric)
        props = OperationProperties(kernel=kernel, supportMode=SupportScheme.Gather,
                                    operationMode=OperationDirection.AllToAll)
        self = cls(P, props, domain, adjacency, kernel, dim)

        omega, Ehat, cnt = self._fast.launch(_MOMENTS_SPEC)
        h = supports
        scale = (omega * h ** 2)[:, None, None]
        Es = Ehat / scale
        Es_inv, cond = invertMoments(Es, rtol)
        self.omega, self.num_nbrs = omega, cnt
        self.Einv = Es_inv / scale
        self.cond = cond
        self.deficient = (cnt < dim + 1) | (self.cond > cond_max) | ~torch.isfinite(self.cond)
        self.volume = 1.0 / omega
        self.closure_power = closure_power
        self.guards, self.centred, self.areaCap = bool(guards), int(bool(guards and centred_weights)), int(bool(guards and area_cap))
        if face_cond_max is None:
            face_cond_max = faceCondMax(positions.dtype)
        self.condBad = ((self.cond > face_cond_max) | ~torch.isfinite(self.cond)).to(torch.int32) if guards else torch.zeros_like(cnt)
        self.lam = torch.zeros(N, dim, dtype=positions.dtype, device=positions.device)
        self.closureResidualRatio = 0.0
        self.closureIterations = 0
        self.closure_cap = float(closure_cap)
        self.closureOuterIterations = 0
        self.closureSaturated = 0.0
        if closure == "project":
            self.close(closure_power=closure_power, cg_tol=cg_tol, cg_iters=cg_iters, lam0=lam0,
                       closure_solver=closure_solver, mg_min_particles=mg_min_particles,
                       mg_max_coarse=mg_max_coarse, lam_cap=lam_cap, closure_cap=closure_cap)
        return self

    def close(self, closure_power: float = 1.0, cg_tol: Optional[float] = None, cg_iters: int = 200,
              lam0: Optional[torch.Tensor] = None, closure_solver: str = "auto", mg_min_particles: int = 2048,
              mg_max_coarse: int = 512, lam_cap: float = 1e3, closure_cap: Optional[float] = None) -> "MeshlessWarp":
        """Solve the closure projection on this geometry (in place; what ``build(closure="project")`` does after
        the moments). Lets a geometry built with ``closure="none"`` -- e.g. for the timestep, which never reads
        ``lam`` -- be reused as the step's geometry at the same positions and supports instead of rebuilt."""
        self.closure_power = closure_power
        N = self.volume.shape[0]
        useMg = closure_solver == "mg" or (closure_solver == "auto" and N >= mg_min_particles)
        if closure_cap is not None:
            self.closure_cap = float(closure_cap)
        self.lam = self._solveClosure(cg_tol, cg_iters, lam0, useMg, mg_max_coarse, lam_cap=lam_cap)
        self.closureOuterIterations = 1
        self.closureSaturated = 0.0
        if self.closure_cap > 0.0:
            # bounded closure: the face correction is kappa clip(dlam, cap). Lagged secant weights
            # kappa min(1, cap / |dlam|) turn each outer pass into the same linear graph-Laplacian solve
            # (warm-started); with no face over the cap this loop does nothing (the solve above is the
            # exact unbounded closure, bitwise).
            for _ in range(_CAP_OUTER):
                scale, over = self._capScale(self.lam, self.closure_cap)
                self.closureSaturated = float(over)
                if over == 0.0:
                    break
                self.lam = self._solveClosure(cg_tol, cg_iters, self.lam, useMg, mg_max_coarse, lam_cap=lam_cap,
                                              capScale=scale)
                self.closureOuterIterations += 1
            else:
                _, over = self._capScale(self.lam, self.closure_cap)
                self.closureSaturated = float(over)
        return self

    def _capScale(self, lam: torch.Tensor, cap: float):
        """Per CSR entry ``min(1, cap / |lam_j - lam_i|)`` (symmetric in the pair) and the fraction of entries
        whose potential difference exceeds ``cap``."""
        adj = self.adjacency
        N = lam.shape[0]
        i = torch.repeat_interleave(torch.arange(N, device=lam.device), adj.numNeighbors.long())
        mag = (lam[adj.livePairs()[1].long()] - lam[i]).norm(dim=-1)
        scale = torch.clamp(cap / mag.clamp_min(1e-30), max=1.0).to(lam.dtype)
        return scale, (mag > cap).to(lam.dtype).mean()

    # ------------------------------------------------------------------
    def residual(self):
        """``(sum_j A_ij, face norm)`` of the paper's face vector: the ``(N, dim)`` closure residual and the global
        norm ``sqrt(sum_pairs |A|^2)`` it is measured against."""
        a, a2 = self._fast.launch(_CLOSURE_RES_SPEC, vol=self.volume, Einv=self.Einv, condBad=self.condBad,
                                  centred=int(self.centred), areaCap=int(self.areaCap))
        return a, torch.sqrt(a2.sum() / 2.0)

    def _solveClosure(self, tol: Optional[float], iters: int, lam0: Optional[torch.Tensor] = None,
                      multigrid: bool = False, mgMaxCoarse: int = 512, lam_cap: float = 1e3,
                      capScale: Optional[torch.Tensor] = None) -> torch.Tensor:
        """CG for ``sum_j kappa_ij (lam_i - lam_j) = sum_j A_ij`` (the torch reference's graph-Laplacian problem).

        Divergence guard (mirrors the torch reference's ``_closeFaces``): on a degenerate graph the
        Laplacian is near-singular (a coalesced pair or a kernel-edge bridge edge has ``kappa ~ 0``),
        and the CG can break down in the working precision -- the residual *grows* and the potential
        reaches ``1e14`` and up (measured on the KH 256 MFM run, COMPRESSIBLE_SPH_PLAN wave 2 D) -- in
        which case the correction ``kappa (lam_j - lam_i)`` replaces the physical face with a spurious
        force of the same magnitude. The minimum-residual iterate is tracked; a solve that never
        improved on the initial residual (or went non-finite) applies no correction beyond the warm
        start, and the potential is capped at ``lam_cap`` so the correction of a pair stays bounded by
        ``2 lam_cap |A_ij|`` (a correction, never a replacement of the face)."""
        a, faceNorm = self.residual()
        if tol is None:
            tol = 1e-6 if a.dtype == torch.float64 else 1e-4
        floor = 1e-12 if a.dtype == torch.float64 else 1e-5
        a = a - a.mean(0, keepdim=True)
        bnorm = a.norm()
        if not bool(bnorm > floor * faceNorm):
            return torch.zeros_like(a)
        target = tol * bnorm
        kappa = self._fast.launch(_CLOSURE_W_SPEC, vol=self.volume, Einv=self.Einv, condBad=self.condBad,
                                  centred=int(self.centred), areaCap=int(self.areaCap), power=scalar_t(self.closure_power))
        if capScale is not None:
            kappa = kappa * capScale
        # preconditioned CG; the diagonal sum_j kappa_ij comes out of the first matvec. The preconditioner is
        # Jacobi, or one V-cycle of the cell-aggregation multigrid (multigrid.py) for large N
        # (a warm start `lam0`, e.g. the previous step's potential, removes part of the work in a time loop).
        # The matvec launches directly on the CSR arrays (bitwise the operator's result, without its
        # per-launch host cost); the operator path remains for a grid (non-adjacency) traversal.
        x0 = torch.zeros_like(a) if lam0 is None else lam0.clone()
        csr = hasattr(self.adjacency, 'edgeOffsets')
        ws = None
        if multigrid:
            mg = CellMultigrid(self.particles.positions, self.particles.supports.max(), self.domain, self.dim,
                               maxCoarse=mgMaxCoarse)
            n, per = mg.ncells, mg.periodic
            S = self._fast.launch(_COARSE_STENCIL_SPEC, kappa=kappa, cellId=mg.cellId, n0=int(n[0]), n1=int(n[1]),
                                  n2=int(n[2]), p0=int(per[0]), p1=int(per[1]), p2=int(per[2]))
            mg.setWeights(S)
            self.multigridLevels = mg.nLevels
            if csr and a.is_cuda and _CG_GRAPH_AFTER is not None:
                ws = _closureWorkspace(self.adjacency, kappa, mg, a)
                mg = ws.mg
        if ws is not None:
            mv = ws.mv
        elif csr:
            mv = mfmClosureMatvecCSR(self.adjacency, kappa)
        else:
            mv = lambda p: mfmClosureMatvecWarp(self.particles, self.props, self.domain, self.adjacency, kappa,
                                                p.contiguous())
        Ax0, diag = mv(x0)
        if ws is not None:
            ws.diag.copy_(diag)
            ws.target.copy_(target)
            diag, target = ws.diag, ws.target
        if multigrid:
            precond = lambda r: mg.vcycle(r, lambda v: mv(v)[0], diag)
        else:
            minv = 1.0 / torch.where(diag > 0, diag, torch.ones_like(diag))[:, None]
            precond = lambda r: minv * r
        x = x0
        r = a - Ax0
        rnorm0 = r.norm()
        self.closureResidualRatio = rnorm0 / bnorm
        if bool(rnorm0 <= target):
            return x.clamp(-lam_cap, lam_cap)
        z = precond(r)
        p = z.clone()
        rz = (r * z).sum()
        rs0 = (r * r).sum()
        best_x, best_rn = x0, rs0
        # The loop runs without a host sync per iteration: the stopping tests (breakdown, convergence) are
        # evaluated on the device and freeze the iterate (``torch.where``) from the iteration they fire, so the
        # result is bitwise the stop-at-the-first-test loop's (the frozen iterations' work is discarded). On
        # CUDA each iteration copies its "done" flag to pinned host memory behind an event, and the host reads
        # the flag of the iteration `_CG_LAG` back -- waiting only for that one, so the GPU always has the
        # next iterations queued (no idle bubble per check) and at most `_CG_LAG` iterations are wasted. On
        # the CPU the flag is read every `_CG_CHECK` iterations.
        lagged = a.is_cuda
        if lagged:
            flags = torch.zeros(_CG_LAG + 1, dtype=torch.bool, pin_memory=True)
            events = [torch.cuda.Event() for _ in range(_CG_LAG + 1)]
        done = torch.zeros((), dtype=torch.bool, device=a.device)
        conv = torch.zeros((), dtype=torch.bool, device=a.device)
        conv_rn = rs0
        count = torch.zeros((), dtype=torch.int32, device=a.device)

        def iteration(S):
            x, r, p, rz, best_x, best_rn, done, conv, conv_rn, count = S
            Ap, _ = mv(p)
            pAp = (p * Ap).sum()
            done = done | ~(pAp > 1e-30 * rz)                    # breakdown (or null-space direction)
            act = ~done
            al = rz / pAp
            x = torch.where(act, x + al * p, x)
            r = torch.where(act, r - al * Ap, r)
            count = count + act.to(torch.int32)
            rn = (r * r).sum()
            better = act & (rn < best_rn)
            best_x = torch.where(better, x, best_x)
            best_rn = torch.where(better, rn, best_rn)
            cv = act & (rn.sqrt() <= target)
            conv = conv | cv
            conv_rn = torch.where(cv, rn, conv_rn)
            done = done | cv
            act = ~done
            z = precond(r)
            rzn = (r * z).sum()
            p = torch.where(act, z + rzn / rz * p, p)
            rz = torch.where(act, rzn, rz)
            return [x, r, p, rz, best_x, best_rn, done, conv, conv_rn, count]

        S = [x, r, p, rz, best_x, best_rn, done, conv, conv_rn, count]
        it, replaying, check = 0, False, 0
        while it < iters:
            if not replaying and ws is not None and it >= _CG_GRAPH_AFTER and iters - it >= _CG_GRAPH_CHUNK:
                # a longer solve: the rest as replays of the workspace's CUDA graph (captured on its first use)
                signature = (_CG_GRAPH_CHUNK, mg.cholOk, tuple((t.shape, t.dtype) for t in S))
                if ws.graph is None or ws.signature != signature:
                    ws.capture(iteration, S, _CG_GRAPH_CHUNK)
                for t, v in zip(ws.S, S):
                    t.copy_(v)
                S, replaying = ws.S, True
            if replaying and iters - it >= _CG_GRAPH_CHUNK:
                ws.graph.replay()
                it += _CG_GRAPH_CHUNK
            else:                                                # (the tail before the cap runs eagerly)
                S = iteration(S)
                it += 1
            if lagged:
                slot = check % (_CG_LAG + 1)
                flags[slot].copy_(S[6], non_blocking=True)
                events[slot].record()
                if check >= _CG_LAG:
                    old = (check - _CG_LAG) % (_CG_LAG + 1)
                    events[old].synchronize()
                    if bool(flags[old]):
                        break
                check += 1
            elif it % _CG_CHECK == 0 and bool(S[6]):
                break
        # the workspace's state is overwritten by the next solve: keep what is returned or read later
        x, r, p, rz, best_x, best_rn, done, conv, conv_rn, count = (t.clone() for t in S) if replaying else S
        self.closureIterations = count
        if bool(conv):
            self.closureResidualRatio = conv_rn.sqrt() / bnorm
            return x.clamp(-lam_cap, lam_cap)
        # not converged: a solve that never improved on the initial residual (or went
        # non-finite) applies no correction beyond the warm start, and the cap keeps
        # the correction a correction
        if not bool(torch.isfinite(best_rn)) or not bool(best_rn < rs0):
            self.closureResidualRatio = r.norm() / bnorm
            return x0.clamp(-lam_cap, lam_cap)
        self.closureResidualRatio = best_rn.sqrt() / bnorm
        return best_x.clamp(-lam_cap, lam_cap)

    # closure diagnostics: kept as device tensors by the solve (no sync), converted on access
    @property
    def closureResidualRatio(self) -> float:
        return float(self._closureResidualRatio)

    @closureResidualRatio.setter
    def closureResidualRatio(self, value):
        self._closureResidualRatio = value

    @property
    def closureIterations(self) -> int:
        return int(self._closureIterations)

    @closureIterations.setter
    def closureIterations(self, value):
        self._closureIterations = value

    # ------------------------------------------------------------------
    def gradients(self, rho: torch.Tensor, vel: torch.Tensor, pres: torch.Tensor):
        """Matrix least-squares gradients ``(grad rho (N,d), grad v (N,d,d), grad P (N,d))``."""
        return self._fast.launch(_GRADIENTS_SPEC, rho=rho.contiguous(), vel=vel.contiguous(), pres=pres.contiguous(),
                                 Einv=self.Einv, deficient=self.deficient.to(torch.int32))

    def rates(self, rho: torch.Tensor, vel: torch.Tensor, pres: torch.Tensor, gamma: float, dt: float = 0.0,
              mode: str = "MFM", order: int = 2, beta_min: float = 1.0, beta_max: float = 2.0,
              cond_crit: float = 100.0, psi1: float = 0.5, psi2: float = 0.25, timeCentredFrame: bool = True,
              retry: bool = True, massFluxLimit: float = 0.1, faceGuard: int = 0):
        """``dQ/dt`` of shape ``(N, 2 + dim)`` for ``Q = (m, m v, m e_tot)``, as ``scheme.mfmRates``."""
        N, dim = vel.shape
        rho, vel, pres = rho.contiguous(), vel.contiguous(), pres.contiguous()
        if order >= 2:
            gRho, gVel, gP = self.gradients(rho, vel, pres)
            beta = conditionBeta(self.cond, beta_min, beta_max, cond_crit)
            aRho, aVel, aP = self._fast.launch(_LIMITER_SPEC, rho=rho, vel=vel, pres=pres, gRho=gRho, gVel=gVel,
                                               gP=gP, beta=beta)
        else:
            gRho = torch.zeros(N, dim, dtype=rho.dtype, device=rho.device)
            gVel = torch.zeros(N, dim, dim, dtype=rho.dtype, device=rho.device)
            gP = torch.zeros_like(gRho)
            aRho, aP = torch.ones_like(rho), torch.ones_like(rho)
            aVel = torch.ones_like(vel)
        dm, dp, de = self._fast.launch(
            _FLUX_SPEC, rho=rho, vel=vel, pres=pres, gRho=gRho, gVel=gVel, gP=gP, aRho=aRho, aVel=aVel, aP=aP,
            vol=self.volume, Einv=self.Einv, lam=self.lam, condBad=self.condBad, gamma=scalar_t(gamma), dt=scalar_t(dt),
            mode=int(0 if mode.upper() == "MFM" else 1), order=int(order), psi1=scalar_t(psi1), psi2=scalar_t(psi2),
            power=scalar_t(self.closure_power), timeCentred=int(timeCentredFrame), centred=int(self.centred),
            areaCap=int(self.areaCap), retry=int(retry), massLimit=scalar_t(massFluxLimit),
            faceGuard=int(faceGuard), closureCap=scalar_t(self.closure_cap))
        return torch.cat([dm[:, None], dp, de[:, None]], dim=-1)

    def timestep(self, rho: torch.Tensor, vel: torch.Tensor, pres: torch.Tensor, gamma: float, cfl: float = 0.2):
        """Per-particle CFL time step ``2 C h_i / v_sig,i`` (Eq. 24-25)."""
        return self._fast.launch(_TIMESTEP_SPEC, rho=rho.contiguous(), vel=vel.contiguous(), pres=pres.contiguous(),
                                 gamma=scalar_t(gamma), cfl=scalar_t(cfl))
