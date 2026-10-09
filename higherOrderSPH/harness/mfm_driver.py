"""Reference time stepper for the core MFM / MFV backend (`warpSPHCore.mfm`).

Deliberately minimal -- the production solver is the frontend's. One global
time step (`signalTimestep`), the single-stage MUSCL-Hancock update of Hopkins
(2015) Eq. 22-23 with the conserved variables of every particle advanced by the
face-flux divergence, the particles then drifted with the time-centred velocity
and the supports re-derived from the effective volumes (`h = (N_ngb V / C_nu)^(1/nu)`,
lagged one step). Used by `run_mfm_sod.py` and the other MFM evidence scripts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from warpSPHCore import ParticleState
from warpSPHCore.mfm import (MeshlessGeometry, conserved, mfmRates, primitives,
                             signalTimestep)

BALL = {1: 2.0, 2: math.pi, 3: 4.0 * math.pi / 3.0}


@dataclass
class MFMState:
    pos: torch.Tensor
    mass: torch.Tensor
    Q: torch.Tensor
    h: torch.Tensor
    t: float = 0.0


def supports_from_volume(V, nngb, dim):
    return (nngb * V / BALL[dim]) ** (1.0 / dim)


CLOSURE_POWER = 1.0


def geometry(state: MFMState, domain, kernel, closure="project"):
    P = ParticleState(positions=state.pos, supports=state.h, masses=state.mass,
                      densities=None, kinds=torch.zeros(state.pos.shape[0], dtype=torch.int32,
                                                        device=state.pos.device))
    return MeshlessGeometry.build(P, domain, kernel, closure=closure, closure_power=CLOSURE_POWER)


def init_state(pos, mass, rho, vel, P, h, gamma):
    return MFMState(pos=pos, mass=mass, Q=conserved(mass, rho, vel, P, gamma), h=h)


def wrap(pos, domain):
    L = domain.max - domain.min
    per = domain.periodic
    return torch.where(per, domain.min + torch.remainder(pos - domain.min, L), pos)


def step(state: MFMState, domain, kernel, gamma, nngb, cfl=0.2, mode="MFM", dt=None,
         closure="project", order=2, tmax=None, **kw):
    dim = state.pos.shape[1]
    g = geometry(state, domain, kernel, closure)
    rho, vel, P = primitives(state.Q, g.volume, gamma)
    if dt is None:
        dt = float(signalTimestep(g, rho, vel, P, gamma, cfl).min())
    if tmax is not None:
        dt = min(dt, tmax - state.t)
    rates, diag = mfmRates(g, rho, vel, P, gamma, dt=dt, mode=mode, order=order, **kw)
    Qn = state.Q + dt * rates
    vel_n = Qn[:, 1:-1] / Qn[:, :1]
    pos = wrap(state.pos + 0.5 * dt * (vel + vel_n), domain)
    h = supports_from_volume(g.volume, nngb, dim)
    return MFMState(pos=pos, mass=state.mass, Q=Qn, h=h, t=state.t + dt), g, dt
