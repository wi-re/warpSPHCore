#!/usr/bin/env python3
"""Step G's Tier-1 forward-mode spike (warpier_fields.md Section 3.6, Step G).

**The question this answers.** Section 3.6's Fact 2 measured that every SPH
operator is linear in the *field values* (the correction terms depend on
positions/supports/masses/densities but not on the values themselves), and
concluded that a Tier-1 JVP -- a tangent w.r.t. field values only -- is just
"re-launch the same kernel on the tangent array": zero new kernels, zero new
adjoint code. That was a linearity measurement, not a JVP test. This script
tests the actual claim end-to-end:

    JVP_v[f](qval, rval) . (dq, dr)  ==  f(dq, dr)

with every non-value input held at its primal value, checked against an
independent reference derived from the *already-verified* reverse-mode path.
If this passes, Tier-1 forward mode is a bridge, not a kernel project.

**Why the reference is `jacobian @ v` and not `torch.autograd.functional.jvp`.**
The plan's Step G text says to check against `jvp`. That does not work here and
the script proves why rather than assuming it: torch computes a forward JVP
either via `torch.autograd.forward_ad` dual tensors (which needs the custom
Function to implement a `jvp` staticmethod -- `StateAwareWarpFunction` does
not, by design; that IS Phase 6's work) or, in
`torch.autograd.functional.jvp`'s case, via the double-backward trick (which
needs the *backward* pass itself to be differentiable -- our backward reads
gradients out of a `wp.Tape` and is not). Both failure modes are probed and
reported below. The reference used instead is the full Jacobian assembled by
reverse-mode VJPs (`torch.autograd.functional.jacobian`, which only needs the
first-order backward the gradcheck suite already validates) contracted with the
tangent. On the small cases here that is exact, not approximate.

Two properties are checked per case, because linearity is what makes Tier 1
work and *homogeneity* is the half that is easy to lose:

  1. f(0, 0) == 0  -- f is linear, not merely affine. An affine offset would
     make f(dv) the wrong tangent by exactly that offset.
  2. f(dq, dr) == J @ (dq, dr) -- the JVP identity itself.

CPU only, float64, tiny cases (a 7-particle line, a 3x3 grid): the Jacobian
reference costs one backward pass per output element, so this deliberately
stays small.

    python scripts/spike_forward_mode_tier1.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, grid_case_2d, line_case, make_domain
from warpSPHCore import OperationProperties, ParticleState, warpOperation
from warpSPHCore.enumTypes import GradientScheme, LaplacianScheme, OperationDirection, SupportScheme, WarpOperation
from warpSPHCore.renorm import computeRenormalizationMatrices

TOL = 1e-10  # float64, and the reference is exact -- this is round-off only


def compute_densities(positions, supports, masses, kinds, domain, adjacency):
    """Realistic density magnitudes via the (separately gradchecked) Density op.
    Detached: this spike differentiates w.r.t. field values only -- tangents
    w.r.t. densities are Tier 2 and deliberately out of scope."""
    p = ParticleState(positions=positions.detach(), supports=supports.detach(),
                      masses=masses.detach(), densities=None, kinds=kinds)
    rho = warpOperation(
        p,
        OperationProperties(kernel=KERNEL, operation=WarpOperation.Density,
                            supportMode=SupportScheme.Gather,
                            operationMode=OperationDirection.AllToAll),
        domain, adjacency=adjacency,
    )
    return rho.detach()


def make_f(particles, dim, operation, *, gradientMode=GradientScheme.Naive,
           laplacianMode=LaplacianScheme.Brookshaw, correction=None):
    """Build f(qval, rval) -> output with every non-value input frozen at its
    primal value (detached), which is exactly Tier 1's scope."""
    positions, supports, masses = particles
    domain = make_domain(dim=dim)
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    densities = compute_densities(positions, supports, masses, kinds, domain, adjacency)

    props = OperationProperties(
        kernel=KERNEL, operation=operation, supportMode=SupportScheme.Gather,
        operationMode=OperationDirection.AllToAll,
        gradientMode=gradientMode, laplacianMode=laplacianMode,
    )

    renormState = None
    if correction == "renorm":
        p = ParticleState(positions=positions.detach(), supports=supports.detach(),
                          masses=masses.detach(), densities=densities, kinds=kinds)
        _, _, renormState = computeRenormalizationMatrices(p, props, domain, adjacency=adjacency)
        # Freeze the correction too: it is a function of positions/supports/
        # masses/densities only, which is precisely Section 3.6's Fact 2.
        renormState.renormalizationMatrices = renormState.renormalizationMatrices.detach()

    def f(qval, rval):
        p = ParticleState(positions=positions.detach(), supports=supports.detach(),
                          masses=masses.detach(), densities=densities, kinds=kinds)
        return warpOperation(p, props, domain, queryValues=qval, referenceValues=rval,
                             adjacency=adjacency, renormalizationState=renormState)

    return f


def check(name, f, value_shape):
    torch.manual_seed(0)
    qval = torch.randn(*value_shape, dtype=DTYPE, device=DEVICE, requires_grad=True)
    rval = torch.randn(*value_shape, dtype=DTYPE, device=DEVICE, requires_grad=True)
    dq = torch.randn(*value_shape, dtype=DTYPE, device=DEVICE)
    dr = torch.randn(*value_shape, dtype=DTYPE, device=DEVICE)

    # -- property 1: linear, not merely affine ---------------------------
    zero = torch.zeros(*value_shape, dtype=DTYPE, device=DEVICE)
    y0 = f(zero, zero).detach()
    homogeneous = bool(y0.abs().max() <= TOL)

    # -- Tier-1 tangent: re-launch the SAME operator on the tangents ------
    dy_tier1 = f(dq, dr).detach()

    # -- reference: reverse-mode Jacobian contracted with the tangent ------
    J_q, J_r = torch.autograd.functional.jacobian(f, (qval, rval), vectorize=False)
    out_numel = dy_tier1.numel()
    in_numel = qval.numel()
    dy_ref = (J_q.reshape(out_numel, in_numel) @ dq.reshape(in_numel)
              + J_r.reshape(out_numel, in_numel) @ dr.reshape(in_numel)).reshape(dy_tier1.shape)

    scale = max(float(dy_ref.abs().max()), 1e-300)
    err = float((dy_tier1 - dy_ref).abs().max()) / scale
    ok = homogeneous and err <= 1e-9

    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name:52s} f(0)={float(y0.abs().max()):.2e}  rel_err={err:.3e}")
    return ok


def probe_torch_forward_paths(f, value_shape):
    """Report *why* the plan's literal 'check against torch.autograd.functional.jvp'
    does not apply, instead of silently substituting a different reference."""
    print("\n--- Why the reference is jacobian@v, not torch's own JVP machinery ---")
    qval = torch.randn(*value_shape, dtype=DTYPE, device=DEVICE, requires_grad=True)
    rval = torch.randn(*value_shape, dtype=DTYPE, device=DEVICE, requires_grad=True)
    dq = torch.randn(*value_shape, dtype=DTYPE, device=DEVICE)
    dr = torch.randn(*value_shape, dtype=DTYPE, device=DEVICE)

    # The Tier-1 answer, which the checks above established is exact.
    truth = f(dq, dr).detach()

    try:
        _, tangent = torch.autograd.functional.jvp(f, (qval, rval), (dq, dr))
        tangent = tangent.detach()
        err = float((tangent - truth).abs().max()) / max(float(truth.abs().max()), 1e-300)
        verdict = "AGREES" if err <= 1e-9 else f"SILENTLY WRONG (rel_err={err:.3e})"
        print(f"  torch.autograd.functional.jvp: returned without raising -- {verdict}")
        print(f"    |tangent|max={float(tangent.abs().max()):.3e} vs truth {float(truth.abs().max()):.3e}")
        if err > 1e-9:
            print("    This is the dangerous case: strict=False means the double-backward")
            print("    trick returns a zero/garbage tangent instead of erroring, so a")
            print("    Phase 6 bridge must NOT be validated against this function.")
        try:
            torch.autograd.functional.jvp(f, (qval, rval), (dq, dr), strict=True)
            print("    strict=True: also passed")
        except Exception as exc:
            print(f"    strict=True -> {type(exc).__name__}: {str(exc)[:100]}")
    except Exception as exc:
        print(f"  torch.autograd.functional.jvp -> {type(exc).__name__}: {str(exc)[:110]}")

    import torch.autograd.forward_ad as fwAD
    try:
        with fwAD.dual_level():
            dual_q = fwAD.make_dual(qval.detach(), dq)
            dual_r = fwAD.make_dual(rval.detach(), dr)
            out = f(dual_q, dual_r)
            _, tangent = fwAD.unpack_dual(out)
        if tangent is None:
            print("  torch.autograd.forward_ad: ran, but the output carries NO tangent --")
            print("    StateAwareWarpFunction has no jvp() staticmethod, so the dual's")
            print("    tangent is silently dropped. This is exactly Phase 6's bridge work.")
        else:
            print("  torch.autograd.forward_ad: output carried a tangent (unexpected)")
    except Exception as exc:
        print(f"  torch.autograd.forward_ad -> {type(exc).__name__}: {str(exc)[:110]}")


def main():
    wp.init()
    print(__doc__.split("\n\n")[0])
    print()

    line = line_case(7)
    grid = grid_case_2d(3)
    n_line, n_grid = 7, 9

    ok = True

    print("1D line of 7 particles (scalar and vector fields):")
    ok &= check("Interpolate (scalar field)",
                make_f(line, 1, WarpOperation.Interpolate), (n_line,))
    ok &= check("Interpolate (vector field)",
                make_f(line, 1, WarpOperation.Interpolate), (n_line, 3))
    for scheme in (GradientScheme.Naive, GradientScheme.Difference,
                   GradientScheme.Summation, GradientScheme.Symmetric):
        ok &= check(f"Gradient ({scheme.name}, scalar field)",
                    make_f(line, 1, WarpOperation.Gradient, gradientMode=scheme), (n_line,))
    ok &= check("Laplacian (Brookshaw, scalar field)",
                make_f(line, 1, WarpOperation.Laplacian), (n_line,))
    ok &= check("Divergence (vector field)",
                make_f(line, 1, WarpOperation.Divergence), (n_line, 1))

    print("\n2D grid of 3x3 particles:")
    ok &= check("Interpolate (scalar field)",
                make_f(grid, 2, WarpOperation.Interpolate), (n_grid,))
    ok &= check("Gradient (Difference, scalar field)",
                make_f(grid, 2, WarpOperation.Gradient, gradientMode=GradientScheme.Difference), (n_grid,))
    ok &= check("Divergence (vector field)",
                make_f(grid, 2, WarpOperation.Divergence), (n_grid, 2))
    ok &= check("Curl (vector field)",
                make_f(grid, 2, WarpOperation.Curl), (n_grid, 2))

    print("\n2D grid, with the renormalisation correction active")
    print("(Section 3.6 Fact 2: corrections depend on positions/supports/masses/")
    print(" densities but NOT on the field values, so linearity must survive):")
    ok &= check("Gradient (Difference) + renormalisation",
                make_f(grid, 2, WarpOperation.Gradient, gradientMode=GradientScheme.Difference,
                       correction="renorm"), (n_grid,))
    ok &= check("Laplacian (Brookshaw) + renormalisation",
                make_f(grid, 2, WarpOperation.Laplacian, correction="renorm"), (n_grid,))

    probe_torch_forward_paths(make_f(line, 1, WarpOperation.Interpolate), (n_line,))

    print()
    if ok:
        print("ALL PASSED -- Tier-1 forward mode is a bridge, not a kernel project:")
        print("  the JVP w.r.t. field values IS the existing kernel re-launched on the")
        print("  tangent arrays, for every operator, scheme and correction path above.")
    else:
        print("FAILED -- at least one operator is not linear in its field values, or")
        print("  the Tier-1 identity does not hold. Step G's conclusion does not stand;")
        print("  see warpier_fields.md Section 3.6 before costing Phase 6.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
