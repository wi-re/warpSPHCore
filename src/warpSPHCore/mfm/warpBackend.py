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
from .limiters import conditionBeta
from .wp_mfm import (mfmClosureMatvecWarp, mfmClosureResidualWarp, mfmFluxWarp, mfmGradientsWarp,
                     mfmLimiterWarp, mfmMomentsWarp, mfmTimestepWarp)

__all__ = ["MeshlessWarp"]


class MeshlessWarp:
    """Per-particle meshless geometry on the CSR neighbourhood. Build with :meth:`build`.

    Attributes (all ``(N, ...)`` torch tensors): ``omega``, ``volume`` (``1/omega``),
    ``Einv`` (``Ehat^-1``), ``cond`` (Hopkins ``N_cond``), ``deficient``,
    ``num_nbrs`` and ``lam`` (the closure potential, zero when ``closure="none"``)."""

    def __init__(self, particles: ParticleState, props: OperationProperties, domain: DomainDescription,
                 adjacency, kernel: KernelFunctions, dim: int):
        self.particles, self.props, self.domain, self.adjacency = particles, props, domain, adjacency
        self.kernel, self.dim = kernel, dim

    @classmethod
    def build(cls, positions: torch.Tensor, supports: torch.Tensor, domain: DomainDescription,
              kernel: KernelFunctions, adjacency=None, cond_max: float = 1.0e3, closure: str = "project",
              closure_power: float = 1.0, rtol: float = 1.0e-12, cg_tol: Optional[float] = None,
              cg_iters: int = 200, lam0: Optional[torch.Tensor] = None) -> "MeshlessWarp":
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

        omega, Ehat, cnt = mfmMomentsWarp(P, props, domain, adjacency)
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
        self.lam = torch.zeros(N, dim, dtype=positions.dtype, device=positions.device)
        self.closureResidualRatio = 0.0
        self.closureIterations = 0
        if closure == "project":
            self.lam = self._solveClosure(cg_tol, cg_iters, lam0)
        return self

    # ------------------------------------------------------------------
    def residual(self):
        """``(sum_j A_ij, face norm)`` of the paper's face vector: the ``(N, dim)`` closure residual and the global
        norm ``sqrt(sum_pairs |A|^2)`` it is measured against."""
        a, a2 = mfmClosureResidualWarp(self.particles, self.props, self.domain, self.adjacency, self.volume, self.Einv)
        return a, torch.sqrt(a2.sum() / 2.0)

    def _solveClosure(self, tol: Optional[float], iters: int, lam0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """CG for ``sum_j kappa_ij (lam_i - lam_j) = sum_j A_ij`` (the torch reference's graph-Laplacian problem)."""
        a, faceNorm = self.residual()
        if tol is None:
            tol = 1e-6 if a.dtype == torch.float64 else 1e-4
        floor = 1e-12 if a.dtype == torch.float64 else 1e-5
        a = a - a.mean(0, keepdim=True)
        bnorm = a.norm()
        if not bool(bnorm > floor * faceNorm):
            return torch.zeros_like(a)
        target = tol * bnorm
        mv = lambda p: mfmClosureMatvecWarp(self.particles, self.props, self.domain, self.adjacency, self.volume,
                                            self.Einv, p.contiguous(), self.closure_power)
        # Jacobi-preconditioned CG; the diagonal sum_j kappa_ij comes out of the first matvec
        # (a warm start `lam0`, e.g. the previous step's potential, removes most of the work in a time loop)
        x0 = torch.zeros_like(a) if lam0 is None else lam0.clone()
        Ax0, diag = mv(x0)
        minv = 1.0 / torch.where(diag > 0, diag, torch.ones_like(diag))[:, None]
        x = x0
        r = a - Ax0
        self.closureResidualRatio = float(r.norm() / bnorm)
        if bool(r.norm() <= target):
            return x
        z = minv * r
        p = z.clone()
        rz = (r * z).sum()
        self.closureIterations = 0
        for _ in range(iters):
            Ap, _ = mv(p)
            pAp = (p * Ap).sum()
            if not bool(pAp > 1e-30 * rz):
                break
            al = rz / pAp
            x = x + al * p
            r = r - al * Ap
            self.closureIterations += 1
            if bool(r.norm() <= target):
                break
            z = minv * r
            rzn = (r * z).sum()
            p = z + rzn / rz * p
            rz = rzn
        self.closureResidualRatio = float(r.norm() / bnorm)
        return x

    # ------------------------------------------------------------------
    def gradients(self, rho: torch.Tensor, vel: torch.Tensor, pres: torch.Tensor):
        """Matrix least-squares gradients ``(grad rho (N,d), grad v (N,d,d), grad P (N,d))``."""
        return mfmGradientsWarp(self.particles, self.props, self.domain, self.adjacency, rho.contiguous(),
                                vel.contiguous(), pres.contiguous(), self.Einv, self.deficient.to(torch.int32))

    def rates(self, rho: torch.Tensor, vel: torch.Tensor, pres: torch.Tensor, gamma: float, dt: float = 0.0,
              mode: str = "MFM", order: int = 2, beta_min: float = 1.0, beta_max: float = 2.0,
              cond_crit: float = 100.0, psi1: float = 0.5, psi2: float = 0.25, timeCentredFrame: bool = True):
        """``dQ/dt`` of shape ``(N, 2 + dim)`` for ``Q = (m, m v, m e_tot)``, as ``scheme.mfmRates``."""
        N, dim = vel.shape
        rho, vel, pres = rho.contiguous(), vel.contiguous(), pres.contiguous()
        if order >= 2:
            gRho, gVel, gP = self.gradients(rho, vel, pres)
            beta = conditionBeta(self.cond, beta_min, beta_max, cond_crit)
            aRho, aVel, aP = mfmLimiterWarp(self.particles, self.props, self.domain, self.adjacency,
                                            rho, vel, pres, gRho, gVel, gP, beta)
        else:
            gRho = torch.zeros(N, dim, dtype=rho.dtype, device=rho.device)
            gVel = torch.zeros(N, dim, dim, dtype=rho.dtype, device=rho.device)
            gP = torch.zeros_like(gRho)
            aRho, aP = torch.ones_like(rho), torch.ones_like(rho)
            aVel = torch.ones_like(vel)
        dm, dp, de = mfmFluxWarp(self.particles, self.props, self.domain, self.adjacency, rho, vel, pres,
                                 gRho, gVel, gP, aRho, aVel, aP, self.volume, self.Einv, self.lam,
                                 gamma, dt, 0 if mode.upper() == "MFM" else 1, order, psi1, psi2,
                                 self.closure_power, int(timeCentredFrame))
        return torch.cat([dm[:, None], dp, de[:, None]], dim=-1)

    def timestep(self, rho: torch.Tensor, vel: torch.Tensor, pres: torch.Tensor, gamma: float, cfl: float = 0.2):
        """Per-particle CFL time step ``2 C h_i / v_sig,i`` (Eq. 24-25)."""
        return mfmTimestepWarp(self.particles, self.props, self.domain, self.adjacency, rho.contiguous(),
                               vel.contiguous(), pres.contiguous(), gamma, cfl)
