#!/usr/bin/env python3
"""Periodic minimum-image invariance check: shifting a particle by an exact
integer multiple of the periodic domain length must be physically invisible.

**Why this exists.** warpier_adjoint.md's Tier-2 forward-mode plan flags the
periodic minimum-image wrap (`computeDistanceVec` -> `minimumImageDistance`
-> `mod_warp`, `math/wp_distance.py`) as a genuine value-level discontinuity
right at a pair separation of exactly `L/2` (half the periodic axis length),
and explicitly puts it out of scope rather than trying to derive an adjoint
through it. That is a true, unavoidable non-differentiability -- but it says
nothing about whether the *wrap itself* is implemented correctly everywhere
else. This script is a cheap, independent regression test of exactly that:
away from the `L/2` boundary, translating a particle by `n*L` (any integer
`n`, per periodic axis) must not change anything -- not the forward output
of any operator, and not its gradient w.r.t. any input.

This matters for the adjoint plan specifically: every Tier-2.x script's
"keep particles well inside a non-periodic domain" workaround is only a
valid substitute for testing periodicity properly if periodicity itself is
known to be handled correctly by the *existing, already-shipped* reverse
-mode path. This script is that check, run once, independent of any single
Tier's math.

**Method.** For a small particle configuration in a periodic domain (all
pair separations relevant to the kernel's compact support kept well clear of
`L/2` -- see warpier_adjoint.md's note that this is achievable in general
whenever `h < L/2`, the standard periodic-SPH requirement anyway), build a
second configuration where a subset of particles have their RAW (unwrapped)
coordinate shifted by a different integer multiple of `L` per particle (not
a uniform shift of everyone -- that would trivially cancel and not exercise
per-particle wrap logic). Both configurations describe the exact same
physical state. For every operator/scheme tested:

  1. **Forward**: `f(shifted) == f(original)`, exactly (both paths active:
     `buildCompactHashMap` wraps positions before hashing -- confirmed by
     reading `radiusSearch/compactHash/buildHashmap.py` -- and
     `computeDistanceVec` wraps again inside the kernel evaluation itself,
     so this exercises both layers, not just one).
  2. **Backward**: the full reverse-mode Jacobian w.r.t. positions/supports/
     masses[/densities] (`torch.autograd.functional.jacobian`, the same
     reference construction `spike_forward_mode_tier1/tier2_density.py`
     use) must also match exactly. This is not automatic even if (1) holds:
     it additionally requires that the AD bridge differentiate through the
     wrap consistently for both configurations, not just evaluate the same
     answer forward.

No finite differences anywhere -- both checks are exact-equality checks on
quantities that are analytically identical between the two configurations.

    python scripts/periodic_invariance_check.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL
from warpSPHCore import OperationProperties, ParticleState, radiusSearchCompactHashMap, warpOperation
from warpSPHCore.dataTypes import DomainDescription
from warpSPHCore.enumTypes import GradientScheme, LaplacianScheme, OperationDirection, SupportScheme, WarpOperation

TOL = 1e-9  # float64, both sides exact -- round-off only


def make_periodic_domain(dim: int, L: float) -> DomainDescription:
    return DomainDescription(
        min=torch.tensor([-L / 2] * dim, dtype=DTYPE, device=DEVICE),
        max=torch.tensor([L / 2] * dim, dtype=DTYPE, device=DEVICE),
        periodic=torch.tensor([True] * dim, device=DEVICE),
        dim=dim,
    )


def line_case_periodic(n: int = 7, L: float = 2.0, h: float = 0.5):
    """7 particles spanning most of a 1D periodic box of length L, with h
    large enough that the leftmost/rightmost particles genuinely interact
    THROUGH the periodic image (not just a degenerate non-wrapping case),
    but h well under L/2 -- clear of the true q=L/2 discontinuity, per
    warpier_adjoint.md's note that h<L/2 keeps any interacting pair
    provably away from that boundary."""
    domain = make_periodic_domain(1, L)
    pos = torch.linspace(-0.9 * L / 2, 0.9 * L / 2, n, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    sup = torch.full((n,), h, dtype=DTYPE, device=DEVICE)
    mass = torch.full((n,), 1.0, dtype=DTYPE, device=DEVICE)
    density = torch.full((n,), 1.0, dtype=DTYPE, device=DEVICE) + 0.2 * torch.rand(n, dtype=DTYPE)
    shift_n = torch.tensor([0, 3, -2, 1, -4, 2, -1], dtype=DTYPE, device=DEVICE)[:n].unsqueeze(-1)
    pos_shifted = pos + shift_n * L
    return domain, pos, pos_shifted, sup, mass, density


def grid_case_2d_periodic(n_per_side: int = 3, L: float = 2.0, h: float = 0.5):
    domain = make_periodic_domain(2, L)
    coords = torch.linspace(-0.7 * L / 2, 0.7 * L / 2, n_per_side, dtype=DTYPE, device=DEVICE)
    gx, gy = torch.meshgrid(coords, coords, indexing="ij")
    pos = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    n = pos.shape[0]
    sup = torch.full((n,), h, dtype=DTYPE, device=DEVICE)
    mass = torch.full((n,), 1.0, dtype=DTYPE, device=DEVICE)
    density = torch.full((n,), 1.0, dtype=DTYPE, device=DEVICE) + 0.2 * torch.rand(n, dtype=DTYPE)
    shift_n = torch.arange(n, dtype=DTYPE, device=DEVICE).reshape(-1, 1) % 5 - 2  # -2..2, varies per particle
    shift_n = torch.stack([shift_n[:, 0], -shift_n[:, 0]], dim=1)  # distinct per-axis integer shifts
    pos_shifted = pos + shift_n * L
    return domain, pos, pos_shifted, sup, mass, density


def build_adjacency(domain, positions, supports, masses):
    kinds = torch.zeros(positions.shape[0], dtype=torch.int32, device=DEVICE)
    p = ParticleState(positions=positions.detach(), supports=supports.detach(),
                      masses=masses.detach(), densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p, domain, mode=SupportScheme.MeanSymmetric)
    return adjacency, kinds


def make_f(domain, kinds, adjacency, operation, fv=None, **props_kwargs):
    def f(pos, sup, mass, density=None):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=density, kinds=kinds)
        props = OperationProperties(kernel=KERNEL, operation=operation,
                                     supportMode=SupportScheme.MeanSymmetric,
                                     operationMode=OperationDirection.AllToAll, **props_kwargs)
        kwargs = {}
        if fv is not None:
            kwargs["queryValues"] = fv
            kwargs["referenceValues"] = fv
        return warpOperation(p, props, domain, adjacency=adjacency, **kwargs)
    return f


def check(name, a, b, atol=1e-10):
    diff = float((a - b).abs().max())
    scale = max(float(b.abs().max()), 1e-300)
    ok = diff <= atol or diff / scale <= 1e-9
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:55s} max|Δ|={diff:.3e}")
    return ok


def run_case(name, domain, pos0, pos_shifted, sup0, mass0, density0, operation, needs_density=False, fv_shape=None, **props_kwargs):
    fv = torch.randn(fv_shape, dtype=DTYPE, device=DEVICE) if fv_shape is not None else None

    ok = True
    for label, pos in (("original", pos0), ("shifted", pos_shifted)):
        adjacency, kinds = build_adjacency(domain, pos, sup0, mass0)
        f = make_f(domain, kinds, adjacency, operation, fv=fv, **props_kwargs)

        p_leaf = pos.detach().clone().requires_grad_(True)
        s_leaf = sup0.detach().clone().requires_grad_(True)
        m_leaf = mass0.detach().clone().requires_grad_(True)
        inputs = (p_leaf, s_leaf, m_leaf)
        if needs_density:
            d_leaf = density0.detach().clone().requires_grad_(True)
            inputs = inputs + (d_leaf,)

        out = f(*inputs).detach()
        jac = torch.autograd.functional.jacobian(f, inputs, vectorize=False)

        if label == "original":
            out_orig, jac_orig = out, jac
        else:
            ok &= check(f"{name}: forward(shifted)==forward(original)", out, out_orig)
            for k, (Jo, Js) in enumerate(zip(jac_orig, jac)):
                ok &= check(f"{name}: d(out)/d(input[{k}]) shifted==original", Js, Jo)
    return ok


def main():
    wp.init()
    print(__doc__.split("\n\n")[0])
    print()

    ok = True

    print("1D periodic line (L=2.0), 7 particles, some shifted by n*L:")
    domain, pos0, pos_shifted, sup0, mass0, density0 = line_case_periodic()
    ok &= run_case("Density", domain, pos0, pos_shifted, sup0, mass0, density0, WarpOperation.Density)
    ok &= run_case("Interpolate", domain, pos0, pos_shifted, sup0, mass0, density0, WarpOperation.Interpolate,
                    needs_density=True, fv_shape=(7,))
    for scheme in (GradientScheme.Naive, GradientScheme.Difference):
        ok &= run_case(f"Gradient[{scheme.name}]", domain, pos0, pos_shifted, sup0, mass0, density0,
                        WarpOperation.Gradient, needs_density=True, fv_shape=(7,), gradientMode=scheme)
    ok &= run_case("Divergence", domain, pos0, pos_shifted, sup0, mass0, density0, WarpOperation.Divergence,
                    needs_density=True, fv_shape=(7, 1))
    ok &= run_case("Laplacian[Brookshaw]", domain, pos0, pos_shifted, sup0, mass0, density0, WarpOperation.Laplacian,
                    needs_density=True, fv_shape=(7,), laplacianMode=LaplacianScheme.Brookshaw)

    print("\n2D periodic grid (L=2.0), 3x3 particles, distinct per-axis integer shifts:")
    domain2, pos0_2, pos_shifted_2, sup0_2, mass0_2, density0_2 = grid_case_2d_periodic()
    n2 = pos0_2.shape[0]
    ok &= run_case("Density (2D)", domain2, pos0_2, pos_shifted_2, sup0_2, mass0_2, density0_2, WarpOperation.Density)
    ok &= run_case("Gradient[Difference] (2D)", domain2, pos0_2, pos_shifted_2, sup0_2, mass0_2, density0_2,
                    WarpOperation.Gradient, needs_density=True, fv_shape=(n2,), gradientMode=GradientScheme.Difference)
    ok &= run_case("Divergence (2D)", domain2, pos0_2, pos_shifted_2, sup0_2, mass0_2, density0_2,
                    WarpOperation.Divergence, needs_density=True, fv_shape=(n2, 2))
    ok &= run_case("Curl (2D)", domain2, pos0_2, pos_shifted_2, sup0_2, mass0_2, density0_2,
                    WarpOperation.Curl, needs_density=True, fv_shape=(n2, 2))

    print()
    if ok:
        print("ALL PASSED -- the minimum-image wrap (buildCompactHashMap's pre-hash wrap AND")
        print("  computeDistanceVec's per-pair wrap) is translation-by-L invariant, forward AND")
        print("  reverse-mode, everywhere tested. The only known discontinuity is the analytic")
        print("  one at exactly r=L/2 (warpier_adjoint.md), which this script's h<L/2 test data")
        print("  stays provably clear of, by construction.")
    else:
        print("FAILED -- see the individual case(s) above; this would mean the periodic wrap")
        print("  itself (not just its known q=L/2 non-differentiability) is inconsistently")
        print("  implemented between buildCompactHashMap and computeDistanceVec, or between")
        print("  forward and reverse-mode -- a real bug, not the documented edge case.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
