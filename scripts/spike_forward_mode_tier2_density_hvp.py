#!/usr/bin/env python3
"""Phase 4 step 3 spike: `Hess(Density) @ v` as "a JVP of that JVP"
(`warpier_forward_mode_plan.md` Phase 4 Step 3).

**The question this answers.** `computeSPHDensityPositionHVP`
(`coreOperations/wp_densityHVP.py`, dispatched to by `warpOperationHVP`)
claims `HVP_i = sum_j m_j * H_ij @ (v_i - v_j)`, `H_ij` the already-validated
Tier-2.0 `kernels.hessian.sphKernelHessian` building block -- obtained by
differentiating `computeSPHDensityPositionJVP`'s own position-tangent
formula (`dW_ij = ∇W_ij · dx_ij`) a second time, analytically, rather than
by any torch-level `torch.func.jvp`/`forward_ad` composition (tried first;
does not work through a warp-kernel-backed function in this codebase -- see
`wp_densityHVP.py`'s module docstring). This script checks that claim
end to end against an *independent* code path:

    HVP[Density](primal, v)
        == d/dt[ JVP[Density](primal + t*v; tangentQueryPositions=e_a,
                              tangentReferencePositions=0) ]|_{t=0}
           for each coordinate direction a, finite-differenced in t.

The reference calls `computeSPHDensityPositionJVP` (Tier 2.1's own
production function, a *different* formula than the Hessian-based one this
script is checking) at `primal +/- eps*v`, central-differenced -- an
independent check because it never touches `sphKernelHessian` at all, only
the first-order JVP evaluated off the base point. Setting the *inner*
tangent to a fixed coordinate basis vector `e_a` (zero on the reference
side) isolates `(grad_i C)_a` as a plain scalar function of position, so
finite-differencing it along `v` gives exactly `HVP_i[a]`, not some other
contraction -- the algebra is spelled out in `wp_densityHVP.py`'s own
`computeSPHDensityPositionHVP` docstring.

Also cross-checked against a second, cross-repo, non-FD path: `warpSPH`'s
`tests/test_implicitShiftingHessianJVP.py` compares the same
`warpOperationHVP` output directly against `wp_implicitShifting.py`'s
hand-built `H`/`_multiplyLaplacianBlock` matvec (bit-for-bit against a
*different* implementation of the same math, no finite differences at all)
-- run that test too; it isn't duplicated here since it needs the `warpSPH`
sibling repo's sampling/config machinery.

    python scripts/spike_forward_mode_tier2_density_hvp.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, line_case, grid_case_2d, make_domain
from warpSPHCore import OperationProperties, ParticleState, warpOperationHVP
from warpSPHCore.coreOperations import computeSPHDensityPositionJVP
from warpSPHCore.enumTypes import OperationDirection, SupportScheme, WarpOperation

TOL = 1e-6  # float64, one side finite-differenced -- truncation-limited, not round-off
EPS = 1e-5


def check(name, assembled, reference):
    scale = max(float(reference.abs().max()), 1e-300)
    err = float((assembled - reference).abs().max()) / scale
    ok = err <= TOL
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:55s} rel_err={err:.3e}")
    return ok


def fd_hvp_reference(pos0, sup0, mass0, domain, adjacency, kinds, mode, v, dim, eps=EPS):
    """Independent reference: central-difference `computeSPHDensityPositionJVP`
    (a first-order-only formula) along `v`, one coordinate direction at a
    time -- see module docstring for why this reconstructs `HVP_i[a]`
    exactly rather than some other contraction."""
    n = pos0.shape[0]
    reference = torch.zeros(n, dim, dtype=DTYPE, device=DEVICE)
    zero = torch.zeros(n, dim, dtype=DTYPE, device=DEVICE)
    for a in range(dim):
        ea = torch.zeros(n, dim, dtype=DTYPE, device=DEVICE)
        ea[:, a] = 1.0

        def g(x):
            p = ParticleState(positions=x, supports=sup0, masses=mass0, densities=None, kinds=kinds)
            return computeSPHDensityPositionJVP(p, domain, KERNEL, mode, adjacency,
                                                tangentQueryPositions=ea, tangentReferencePositions=zero)

        gp = g(pos0 + eps * v)
        gm = g(pos0 - eps * v)
        reference[:, a] = (gp - gm) / (2 * eps)
    return reference


def run_case(n, dim, mode: SupportScheme, seed=0):
    torch.manual_seed(seed)
    if dim == 1:
        pos0, sup0, mass0 = line_case(n)
    else:
        pos0, sup0, mass0 = grid_case_2d(n)
    pos0, sup0, mass0 = pos0.detach(), sup0.detach(), mass0.detach()
    sup0 = sup0 * (1.0 + 0.15 * torch.linspace(-1, 1, sup0.shape[0], dtype=DTYPE))

    domain = make_domain(dim=dim)
    adjacency, kinds = build_adjacency(pos0, sup0, mass0, domain, mode=SupportScheme.KernelMeanSymmetric)

    torch.manual_seed(seed + 1000)
    v = torch.randn_like(pos0)

    p0 = ParticleState(positions=pos0, supports=sup0, masses=mass0, densities=None, kinds=kinds)
    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Density,
                                supportMode=mode, operationMode=OperationDirection.AllToAll)
    assembled = warpOperationHVP(p0, props, domain, adjacency=adjacency,
                                 tangentQueryPositions=v, tangentReferencePositions=v)
    reference = fd_hvp_reference(pos0, sup0, mass0, domain, adjacency, kinds, mode, v, dim)
    return assembled, reference


def main():
    wp.init()
    print(__doc__.split("\n\n")[0])
    print()

    ok = True

    print("Density HVP, 1D line of 7 particles, non-uniform supports:")
    # KernelMeanSymmetric deliberately excluded: sphKernelHessian
    # (kernels/hessian.py) only special-cases SuperSymmetric's two-term
    # W-average branch, not KernelMeanSymmetric's (both go through it in
    # sphKernelJVP_ij's first-order dispatch, but sphKernelHessian's own
    # `if supportMode == SuperSymmetric` check does not) -- a pre-existing
    # gap in that Tier-2.0 building block itself (also inherited silently by
    # wp_implicitShifting.py, which sidesteps it by always using Gather),
    # not something this step introduces or is scoped to fix.
    for mode in (SupportScheme.Gather, SupportScheme.Scatter, SupportScheme.MeanSymmetric, SupportScheme.SuperSymmetric):
        assembled, reference = run_case(7, 1, mode)
        ok &= check(f"Hess(Density) @ v ({mode.name})", assembled, reference)

    print("\nDensity HVP, 2D 3x3 grid, non-uniform supports:")
    for mode in (SupportScheme.Gather, SupportScheme.MeanSymmetric):
        assembled, reference = run_case(3, 2, mode)
        ok &= check(f"Hess(Density) @ v ({mode.name})", assembled, reference)

    print()
    if ok:
        print("ALL PASSED -- warpOperationHVP's analytic sphKernelHessian-based")
        print("  Hessian-vector product matches an independent finite-difference-of-")
        print("  the-first-order-JVP reference, for every single-h SupportScheme plus")
        print("  SuperSymmetric's two-term average. (KernelMeanSymmetric excluded --")
        print("  sphKernelHessian itself only special-cases SuperSymmetric, a")
        print("  pre-existing gap in that Tier-2.0 building block; see the case-list")
        print("  comment above.)")
    else:
        print("FAILED -- see the individual case(s) above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
