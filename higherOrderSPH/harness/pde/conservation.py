#!/usr/bin/env python3
"""Conservation diagnostics for the PDE benchmark suite (Pass 2).

Computed driver-side from a particle state (masses, velocities, positions) --
no `src/` or frontend change, consistent with the harness convention of never
re-implementing an operator. Pure torch, so it is unit-testable on CPU without
warp or the float64 env dance.

The conserved quantities and the drift convention:

* **mass** -- ``sum(m)``. SPH never changes a particle's mass, so the relative
  drift ``(m_f - m_i)/m_i`` is the clean conservation check (expect ~0 to
  machine precision).
* **momentum** -- ``sum(m v)``. Reported as a vector and its norm. Cases with
  a symmetric initial condition (TGV, Gresho) start at net momentum ~0, so a
  *relative* drift is division by ~0 and meaningless there; the meaningful
  signal is the **absolute** vector drift ``|p_f - p_0|`` (a good scheme keeps
  it at round-off). Not every IC starts at ~0 (KH's unequal-density shear
  layers carry net |p| ~ 0.23), so the final norm alone is not a drift.
* **kineticEnergy** -- ``0.5 sum(m |v|^2)``. For compressible cases the total
  energy (KE + internal) is the conserved quantity; for the first pass we track
  KE (incompressible cases like TGV have KE decay viscously, so its drift is
  the dissipation signal, not a conservation violation).
* **angularMom** -- the out-of-plane component ``sum(m (x v_y - y v_x))`` in 2D
  (the full vector in 3D), reported as ``|L_f - L_0|`` like momentum. About
  the origin, so only an invariant on open domains (see `drift`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["ConservedQuantities", "conserved", "drift", "ke_rebound",
           "KE_REBOUND_TOL"]


@dataclass
class ConservedQuantities:
    mass: float
    momentum: torch.Tensor          # (dim,) vector
    momentum_norm: float            # |momentum|
    kinetic_energy: float
    angular_momentum: torch.Tensor  # (dim,) in 3D, (1,) in 2D (z-component)
    angular_momentum_norm: float
    internal_energy: float = 0.0    # 0 for incompressible states (no IE)

    @property
    def total_energy(self) -> float:
        """KE + IE -- the conserved quantity for compressible cases. For
        incompressible states (IE = 0) this equals the kinetic energy, which
        decays viscously (a dissipation signal, not a conservation check)."""
        return self.kinetic_energy + self.internal_energy

    def as_dict(self) -> dict:
        return {
            "mass": self.mass,
            "momentum_norm": self.momentum_norm,
            "kinetic_energy": self.kinetic_energy,
            "total_energy": self.total_energy,
            "angular_momentum_norm": self.angular_momentum_norm,
        }


def conserved(masses: torch.Tensor,
              velocities: torch.Tensor,
              positions: torch.Tensor,
              internal_energies: torch.Tensor | None = None
              ) -> ConservedQuantities:
    """Conserved quantities of a particle state.

    `masses` (N,), `velocities` (N, dim), `positions` (N, dim); optional
    `internal_energies` (N,) *specific* internal energy (compressible cases).
    All on the same device/dtype. Returns a `ConservedQuantities` with scalars
    on the host (the tensors are detached and moved to CPU).
    """
    m = masses.detach()
    v = velocities.detach()
    x = positions.detach()
    dim = v.shape[1]

    mass = float(m.sum())
    momentum = (m[:, None] * v).sum(dim=0)
    momentum_norm = float(momentum.norm())
    ke = float(0.5 * (m * (v ** 2).sum(dim=-1)).sum())
    ie = (float((m * internal_energies.detach()).sum())
          if internal_energies is not None else 0.0)

    if dim >= 2:
        # Full r x v in 3D; the single out-of-plane (z) component in 2D.
        rxv = torch.stack([
            x[:, 1] * v[:, 2] - x[:, 2] * v[:, 1],
            x[:, 2] * v[:, 0] - x[:, 0] * v[:, 2],
            x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0],
        ], dim=-1) if dim == 3 else torch.stack([
            x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0],
        ], dim=-1)
        angular_momentum = (m[:, None] * rxv).sum(dim=0)
    else:
        # 1D: no out-of-plane angular momentum is defined; report zero.
        angular_momentum = torch.zeros(1, dtype=m.dtype, device=m.device)
    angular_momentum_norm = float(angular_momentum.norm())

    return ConservedQuantities(
        mass=mass,
        momentum=momentum.cpu(),
        momentum_norm=momentum_norm,
        kinetic_energy=ke,
        angular_momentum=angular_momentum.cpu(),
        angular_momentum_norm=angular_momentum_norm,
        internal_energy=ie,
    )


def drift(initial: ConservedQuantities,
          final: ConservedQuantities,
          eps: float = 1e-30) -> dict:
    """Relative drift ``(final - initial)/initial`` for the O(1) quantities
    (mass, kinetic energy, total energy), plus the **absolute** drift of the
    momentum / angular-momentum *vectors* ``|p_f - p_0|``, ``|L_f - L_0|``
    (a relative drift would divide by ~0 for the symmetric ICs). The init and
    final norms are reported too; the final norm alone is NOT a drift -- KH
    (shear layers of unequal density) and Gresho (a vortex) start with O(0.1)
    |p| / |L| that is simply carried through.

    `initial` must be measured on the actual t=0 state (see
    `run_pde.run_one`); a zero-filled stand-in makes every drift equal the
    final value.

    Angular momentum is taken about the origin; on a periodic domain it is
    not an invariant of the exact dynamics (the pairwise forces across the
    seam carry a net torque), so `angmom_drift_abs` is only a conservation
    check on open domains.

    `ke_drift` is the dissipation signal for incompressible cases (KE decays
    viscously); `total_energy_drift` is the conservation check for
    compressible cases (KE + IE is the invariant; for incompressible states it
    equals `ke_drift`).

    Returns a flat dict ready to land in a report row.
    """
    def rel(a: float, b: float) -> float:
        return (b - a) / a if abs(a) > eps else float("nan")

    return {
        "mass_drift": rel(initial.mass, final.mass),
        "ke_drift": rel(initial.kinetic_energy, final.kinetic_energy),
        "total_energy_drift": rel(initial.total_energy, final.total_energy),
        # absolute norms: a symmetric IC starts at ~0, so these are the
        # spurious-momentum / spurious-angular-momentum signal.
        "momentum_norm_init": initial.momentum_norm,
        "momentum_norm_final": final.momentum_norm,
        "angmom_norm_init": initial.angular_momentum_norm,
        "angmom_norm_final": final.angular_momentum_norm,
        "momentum_drift_abs": float(
            (final.momentum - initial.momentum).norm()),
        "angmom_drift_abs": float(
            (final.angular_momentum - initial.angular_momentum).norm()),
    }


# A KE rise above its running minimum larger than this fraction of the
# initial KE flags a KE-non-increasing case (steady / decaying, unforced).
KE_REBOUND_TOL = 1e-2


def ke_rebound(kinetic_energy) -> float:
    """Largest rise of the kinetic energy above its own running minimum,
    relative to the initial value: ``max_t (KE(t) - min_{s<=t} KE(s)) / KE(0)``.

    For an unforced steady or decaying flow (Gresho, TGV) KE may drop
    (dissipation) but must never climb back: energy flowing from internal to
    kinetic energy there is anti-dissipation (negative entropy production).
    The final-state `ke_drift` cannot see this when an early dissipative dip
    and a later spin-up partly cancel -- the CRKSPH Gresho vortex dips -3.8%
    and then gains +7.3% (nx=64), which went unnoticed until this check
    (`higher_order.md`, Phase 1). Returns nan for an empty / zero-KE series.
    """
    ke = [float(k) for k in kinetic_energy]
    if not ke or ke[0] == 0.0:
        return float("nan")
    run_min, worst = ke[0], 0.0
    for k in ke:
        run_min = min(run_min, k)
        worst = max(worst, k - run_min)
    return worst / ke[0]
