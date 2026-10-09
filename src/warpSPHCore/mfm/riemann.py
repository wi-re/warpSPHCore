"""HLLC Riemann solver and the MFM / MFV face fluxes (Hopkins 2015, App. A).

The face problem is one-dimensional along the face normal ``n`` (the unit
vector of the effective face area ``A_ij``). States are primitive
``(rho, u, v_t, P)`` with ``u`` the normal velocity and ``v_t`` the tangential
velocity *vector* (it is passive: advected with the contact), already boosted
to the rest frame of the quadrature point (Eq. A1). Everything is torch and
vectorised over pairs.

Wave speeds follow the paper's fallback chain: Roe-averaged (Einfeldt)
estimates first; where the resulting star pressure is not positive, the Toro
PVRS-based estimates; where that fails too, Rusanov's single speed. (The paper
adds an exact solver as the last resort; it is not implemented here -- a pair
that fails all three is a vacuum-adjacent state.)

* **MFV** keeps the face at rest in that frame, so the full HLLC flux
  (including the mass flux) is used.
* **MFM** moves the face with the contact (``S*``): the mass flux is zero by
  construction, the momentum flux is ``P* n`` and, in the lab frame, the energy
  flux is ``P* (S* + v_frame . n)``. This needs only ``(P*, S*)`` of the
  Riemann problem, so any approximate solver that provides them can replace
  HLLC (``starState`` is the single place to change).

Fluxes are returned per unit face area in the *lab* frame (the de-boost of
Eq. A8), ordered ``(mass, momentum[dim], energy)``.
"""

from __future__ import annotations

from typing import Tuple

import torch

__all__ = ["starState", "faceFlux", "robustFaceFlux"]


def _energy(rho, u, vt, P, gamma):
    """Total energy per volume of the boosted state."""
    return P / (gamma - 1.0) + 0.5 * rho * (u * u + (vt * vt).sum(-1))


def _speeds(kind: str, rhoL, uL, vtL, PL, aL, rhoR, uR, vtR, PR, aR, gamma):
    if kind == "roe":
        R = torch.sqrt(rhoR / rhoL)
        HL = (_energy(rhoL, uL, vtL, PL, gamma) + PL) / rhoL
        HR = (_energy(rhoR, uR, vtR, PR, gamma) + PR) / rhoR
        ut = (uL + R * uR) / (1 + R)
        vtt = (vtL + R[..., None] * vtR) / (1 + R[..., None])
        H = (HL + R * HR) / (1 + R)
        a2 = (gamma - 1.0) * (H - 0.5 * (ut * ut + (vtt * vtt).sum(-1)))
        at = torch.sqrt(a2.clamp_min(0.0))
        return torch.minimum(uL - aL, ut - at), torch.maximum(uR + aR, ut + at)
    if kind == "pvrs":
        rbar, abar = 0.5 * (rhoL + rhoR), 0.5 * (aL + aR)
        pstar = (0.5 * (PL + PR) - 0.5 * (uR - uL) * rbar * abar).clamp_min(0.0)

        def q(P):
            return torch.where(pstar > P,
                               torch.sqrt(1.0 + (gamma + 1.0) / (2.0 * gamma) * (pstar / P - 1.0)),
                               torch.ones_like(P))
        return uL - aL * q(PL), uR + aR * q(PR)
    s = torch.maximum(uL.abs() + aL, uR.abs() + aR)             # rusanov
    return -s, s


def _starSpeedPressure(SL, SR, rhoL, uL, PL, rhoR, uR, PR):
    num = PR - PL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)
    den = rhoL * (SL - uL) - rhoR * (SR - uR)
    Ss = num / den
    Ps = PL + rhoL * (SL - uL) * (Ss - uL)
    return Ss, Ps


def starState(rhoL, uL, vtL, PL, rhoR, uR, vtR, PR, gamma) -> Tuple[torch.Tensor, ...]:
    """``(S_L, S_R, S*, P*)`` of the HLLC model, with the wave-speed fallback chain."""
    aL = torch.sqrt(gamma * PL / rhoL)
    aR = torch.sqrt(gamma * PR / rhoR)
    best = None
    for kind in ("roe", "pvrs", "rusanov"):
        SL, SR = _speeds(kind, rhoL, uL, vtL, PL, aL, rhoR, uR, vtR, PR, aR, gamma)
        Ss, Ps = _starSpeedPressure(SL, SR, rhoL, uL, PL, rhoR, uR, PR)
        cand = (SL, SR, Ss, Ps)
        if best is None:
            best = cand
            continue
        bad = ~((best[3] > 0) & torch.isfinite(best[3]) & torch.isfinite(best[2]))
        best = tuple(torch.where(bad, c, b) for b, c in zip(best, cand))
    return best


def faceFlux(rhoL, uL, vtL, PL, rhoR, uR, vtR, PR, gamma: float, mode: str,
             normal: torch.Tensor, vframe: torch.Tensor, starFn=None):
    """Lab-frame flux through the face per unit area.

    ``rho*, u*, P*`` are ``(P,)``; ``vt*`` ``(P, dim)`` (tangential velocity of
    the boosted state); ``normal`` ``(P, dim)`` the unit face normal pointing
    from the left (``i``) to the right (``j``) state; ``vframe`` ``(P, dim)``
    the velocity of the quadrature point. Returns ``(flux, Sstar, Pstar)`` with
    ``flux`` of shape ``(P, 2 + dim)`` ordered (mass, momentum, energy).

    ``starFn(rhoL, uL, vtL, PL, rhoR, uR, vtR, PR) -> (S*, P*)`` replaces the
    HLLC star state in ``mode="MFM"`` (the MFM flux needs nothing else): any
    solver, or an equation of state other than the ideal gas, plugs in here."""
    if starFn is not None:
        if mode.upper() != "MFM":
            raise ValueError("starFn is only meaningful for mode='MFM'")
        Ss, Ps = starFn(rhoL, uL, vtL, PL, rhoR, uR, vtR, PR)
        SL = SR = None
    else:
        SL, SR, Ss, Ps = starState(rhoL, uL, vtL, PL, rhoR, uR, vtR, PR, gamma)
    vn = (vframe * normal).sum(-1)
    if mode.upper() == "MFM":
        zero = torch.zeros_like(Ps)
        mom = Ps[:, None] * normal
        energy = Ps * (Ss + vn)
        return torch.cat([zero[:, None], mom, energy[:, None]], dim=-1), Ss, Ps
    if mode.upper() != "MFV":
        raise ValueError("mode must be 'MFM' or 'MFV'")

    # --- full HLLC flux in the boosted frame (Toro 10.4) ------------------------------
    def side(rho, u, vt, P, S):
        E = _energy(rho, u, vt, P, gamma)
        Fm = rho * u
        Fn = rho * u * u + P
        Ft = Fm[:, None] * vt
        Fe = (E + P) * u
        coef = rho * (S - u) / (S - Ss)
        Ustar = (coef, coef * Ss, coef[:, None] * vt,
                 coef * (E / rho + (Ss - u) * (Ss + P / (rho * (S - u)))))
        U = (rho, rho * u, rho[:, None] * vt, E)
        F = (Fm, Fn, Ft, Fe)
        Fstar = tuple(f + S[..., None] * (us - uu) if f.dim() == 2 else f + S * (us - uu)
                      for f, us, uu in zip(F, Ustar, U))
        return F, Fstar

    FL, FsL = side(rhoL, uL, vtL, PL, SL)
    FR, FsR = side(rhoR, uR, vtR, PR, SR)
    sel = lambda a, b, c, d, t: torch.where(
        (SL >= 0).view(-1, *([1] * (a.dim() - 1))), a,
        torch.where((Ss >= 0).view(-1, *([1] * (a.dim() - 1))), b,
                    torch.where((SR > 0).view(-1, *([1] * (a.dim() - 1))), c, d)))
    Fm = sel(FL[0], FsL[0], FsR[0], FR[0], None)
    Fn = sel(FL[1], FsL[1], FsR[1], FR[1], None)
    Ft = sel(FL[2], FsL[2], FsR[2], FR[2], None)
    Fe = sel(FL[3], FsL[3], FsR[3], FR[3], None)
    mom = Fn[:, None] * normal + Ft                      # momentum flux vector in the boosted frame
    # de-boost (Eq. A8)
    mom_lab = mom + vframe * Fm[:, None]
    e_lab = Fe + (vframe * mom).sum(-1) + 0.5 * (vframe * vframe).sum(-1) * Fm
    return torch.cat([Fm[:, None], mom_lab, e_lab[:, None]], dim=-1), Ss, Ps


def robustFaceFlux(primary, firstOrder, gamma: float, mode: str, normal: torch.Tensor, vframe: torch.Tensor,
                   pressureLimit: torch.Tensor, starFn=None, retry: bool = True):
    """:func:`faceFlux` with GIZMO's failure handling (``hydro_core_meshless.h``): if the star pressure of the
    reconstructed states ``primary`` is not positive, not finite or exceeds ``1.4 * pressureLimit``, the pair
    is re-solved with the first-order (particle) states ``firstOrder``; if that is still invalid, with a zero
    velocity jump (both sides at rest in the face frame, the particles' densities and pressures). States are
    tuples ``(rhoL, uL, vtL, PL, rhoR, uR, vtR, PR)``; returns ``(flux, Sstar, Pstar, stage)`` with ``stage`` 0,
    1 or 2 the stage that was used."""
    flux, Ss, Ps = faceFlux(*primary, gamma, mode, normal, vframe, starFn)
    stage = torch.zeros(Ps.shape, dtype=torch.int8, device=Ps.device)
    if not retry:
        return flux, Ss, Ps, stage
    bad = ~((Ps > 0) & torch.isfinite(Ps) & torch.isfinite(Ss) & (Ps <= 1.4 * pressureLimit)) | ~torch.isfinite(flux).all(-1)
    if not bool(bad.any()):
        return flux, Ss, Ps, stage
    idx = torch.nonzero(bad).flatten()
    sub = lambda t: t[idx]
    f2, S2, P2 = faceFlux(*[sub(t) for t in firstOrder], gamma, mode, sub(normal), sub(vframe), starFn)
    bad2 = ~((P2 > 0) & torch.isfinite(P2) & torch.isfinite(S2)) | ~torch.isfinite(f2).all(-1)
    rL, uL, vtL, pL, rR, uR, vtR, pR = [sub(t) for t in firstOrder]
    zero = torch.zeros_like(uL)
    f3, S3, P3 = faceFlux(rL, zero, torch.zeros_like(vtL), pL, rR, zero, torch.zeros_like(vtR), pR, gamma, mode,
                          sub(normal), sub(vframe), starFn)
    pick3 = bad2[:, None]
    f2 = torch.where(pick3, f3, f2)
    S2 = torch.where(bad2, S3, S2)
    P2 = torch.where(bad2, P3, P2)
    flux = flux.clone(); Ss = Ss.clone(); Ps = Ps.clone()
    flux[idx], Ss[idx], Ps[idx] = f2, S2, P2
    stage[idx] = torch.where(bad2, 2, 1).to(torch.int8)
    return flux, Ss, Ps, stage
