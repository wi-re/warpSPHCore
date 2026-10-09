#!/usr/bin/env python3
"""Forward-mode (JVP) check of the order-p polynomial fit (PolyFit).

Not a gradcheck (no backward involved): it checks the *forward-mode* identity
``tangent(output) == central finite difference of the primal along the same
tangent`` for the geometry tangents (positions, supports, masses, densities)
and the field tangent at once, through the whole chain -- dedicated JVP
kernels for the moment matrix and right-hand side, ``dc = M^-1 (db - dM c)``
in torch, and torch forward AD for the 1/h^|a| read-out. MLS and LABFM
forms, gradient (p = 1, 2) and Laplacian (p = 2) outputs, on a small periodic
jittered lattice (every row has a full stencil, so the pseudo-inverse is an
inverse and differentiable).

    python scripts/gradcheck/gradcheck_polyfit_jvp_native.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import torch.autograd.forward_ad as fwAD
import warp as wp

from gradcheck_polyfit_native import KERNEL, build_case
from warpSPHCore import ParticleState
from warpSPHCore.polyfit import PolyFit


def run(name, order, constant, output) -> bool:
    p, domain, adjacency, rho = build_case()
    kinds = p.kinds
    torch.manual_seed(7)
    base = dict(pos=p.positions.clone(), sup=p.supports.clone(), mas=p.masses.clone(), den=rho.clone())
    f0 = torch.sin(3 * p.positions[:, 0]) * torch.cos(2 * p.positions[:, 1])
    tan = {k: torch.randn_like(v) for k, v in base.items()}
    tf = torch.randn_like(f0)

    def primal(d, f):
        q = ParticleState(positions=d["pos"], supports=d["sup"], masses=d["mas"], densities=d["den"], kinds=kinds)
        pf = PolyFit.build(q, domain, KERNEL, order, constant=constant, adjacency=adjacency)
        return pf.gradient(f) if output == "gradient" else pf.laplacian(f)

    with fwAD.dual_level():
        dual = {k: fwAD.make_dual(base[k], tan[k]) for k in base}
        out = primal(dual, fwAD.make_dual(f0, tf))
        _, jvp = fwAD.unpack_dual(out)

    eps = 1e-6
    plus = primal({k: base[k] + eps * tan[k] for k in base}, f0 + eps * tf)
    minus = primal({k: base[k] - eps * tan[k] for k in base}, f0 - eps * tf)
    fd = (plus - minus) / (2 * eps)
    scale = fd.abs().max().item()
    err = (jvp - fd).abs().max().item()
    ok = err < 1e-5 * max(scale, 1.0)
    print(f"{'PASS' if ok else 'FAIL'} {name}: max|jvp - fd| = {err:.2e} (scale {scale:.2e})")
    return ok


def run_reconstruction(kind: str) -> bool:
    """Geometry JVP of the TENO / WENO interface states (smooth data, so the
    stencil selection is locally constant): tangent vs central finite
    difference, with a fixed neighbour structure."""
    from warpSPHCore.polyfit import Reconstructor
    p, domain, adjacency, rho = build_case(n=16, nbrs=16)
    kinds = p.kinds
    torch.manual_seed(11)
    base = dict(pos=p.positions.clone(), sup=p.supports.clone(), mas=p.masses.clone(), den=rho.clone())
    tan = {k: 0.5 * torch.randn_like(v) for k, v in base.items()}
    tan["sup"] = torch.zeros_like(tan["sup"])      # supports only set the interface weights
    f0 = torch.sin(2 * torch.pi * base["pos"][:, 0]) * torch.cos(2 * torch.pi * base["pos"][:, 1])
    tf = torch.randn_like(f0)
    # fixed pair list (first-neighbour ring), pair vectors rebuilt from each position set
    x = base["pos"]
    d0 = x[None] - x[:, None]
    d0 = d0 - torch.round(d0)
    dx = (1.0 / 16)
    i, j = torch.nonzero((d0.norm(dim=-1) < 1.6 * dx) & (d0.norm(dim=-1) > 0), as_tuple=True)
    # one adjacency for the largest stencil, shared by every evaluation
    from warpSPHCore.radiusSearch import radiusSearchCompactHashMap
    from warpSPHCore.enumTypes import SupportScheme
    big = ParticleState(positions=base["pos"], supports=torch.full_like(base["sup"], 6.0 * dx),
                        masses=base["mas"], densities=base["den"], kinds=kinds)
    adj = radiusSearchCompactHashMap(big, domain, mode=SupportScheme.SuperSymmetric)

    def states(d, f):
        q = ParticleState(positions=d["pos"], supports=d["sup"], masses=d["mas"], densities=d["den"], kinds=kinds)
        rec = (Reconstructor.teno(q, domain, KERNEL, "O4", adjacency=adj) if kind == "teno"
               else Reconstructor.weno(q, domain, KERNEL, degree=2, adjacency=adj))
        pos = d["pos"]
        dd = pos[j] - pos[i]
        dd = dd - torch.round(dd)
        fl, fr = rec.interfaceStates(f, i, j, dd)
        return torch.cat([fl, fr])

    with fwAD.dual_level():
        dual = {k: fwAD.make_dual(base[k], tan[k]) for k in base}
        out = states(dual, fwAD.make_dual(f0, tf))
        _, jvp = fwAD.unpack_dual(out)
    # A much smaller step than for the fit itself: the unit-weight WENO stencils have a
    # hard membership cutoff (|x_ij| < h), so the output is only piecewise smooth in the
    # geometry and a 1e-6 step crosses a boundary for a few particles (32 of 4072 entries
    # off, all at once; gone at 1e-8 and below, where the JVP agrees to 4e-8).
    eps = 1e-8
    plus = states({k: base[k] + eps * tan[k] for k in base}, f0 + eps * tf)
    minus = states({k: base[k] - eps * tan[k] for k in base}, f0 - eps * tf)
    fd = (plus - minus) / (2 * eps)
    err = (jvp - fd).abs().max().item()
    scale = max(fd.abs().max().item(), 1.0)
    bad = ((jvp - fd).abs() > 1e-4 * scale).sum().item()
    print(f"   [{kind}] entries off by > 1e-4 of the scale: {bad} of {fd.numel()}")
    ok = err < 1e-5 * scale
    print(f"{'PASS' if ok else 'FAIL'} {kind.upper()} interface states: max|jvp - fd| = {err:.2e} (scale {scale:.2e})")
    return ok


def main() -> int:
    wp.init()
    cases = [
        ("MLS p=1 gradient", 1, True, "gradient"),
        ("MLS p=2 gradient", 2, True, "gradient"),
        ("MLS p=2 laplacian", 2, True, "laplacian"),
        ("LABFM p=2 gradient", 2, False, "gradient"),
        ("LABFM p=2 laplacian", 2, False, "laplacian"),
    ]
    results = [run(*c) for c in cases]
    results += [run_reconstruction("teno"), run_reconstruction("weno")]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
