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
from warpSPHCore import DomainDescription, OperationProperties, ParticleState, radiusSearchCompactHashMap, warpOperation, warpOperationHVP
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


def reverse_mode_self_hessian_check(h=1.0, x0=0.3):
    """A single particle's own self-density contribution, differentiated by
    `warpOperation`'s production reverse-mode path (gradcheck-validated,
    routes `d(vectorNorm_warp)/dx` through `math/wp_normalize.py`'s manually
    -written eps-guarded adjoints, not a naive automatic one -- exactly the
    kind of `x/|x|` expression that would otherwise divide by zero at `r=0`).
    Cross-checks the chain-rule identity `computeSPHDensityPositionHVP`'s
    docstring derives (the self term's true contribution to `d^2 C_i/dx_i^2`
    is exactly zero, not `sphKernelHessian`'s `-15`-ish value at `r=0`) via a
    completely independent numerical route: two single reverse-mode backward
    passes at nearby points, finite-differenced -- never a genuine
    double-backward through the same graph, so immune to the NaN hazard a
    naive from-scratch double-backward attempt would hit differentiating a
    normalized direction a second time without the matching manual adjoint.
    """
    domain = DomainDescription(min=torch.tensor([-10.0], dtype=DTYPE, device=DEVICE),
                               max=torch.tensor([10.0], dtype=DTYPE, device=DEVICE),
                               periodic=torch.tensor([False], device=DEVICE), dim=1)
    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Density,
                                supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll)

    def self_density(x):
        positions = x.view(1, 1)
        supports = torch.tensor([h], dtype=DTYPE, device=DEVICE)
        masses = torch.tensor([1.0], dtype=DTYPE, device=DEVICE)
        kinds = torch.zeros(1, dtype=torch.int32, device=DEVICE)
        p = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
        adjacency = radiusSearchCompactHashMap(p, domain, mode=SupportScheme.Gather)
        return warpOperation(p, props, domain, adjacency=adjacency)[0]

    def rev_grad(xval):
        xg = xval.clone().requires_grad_(True)
        val = self_density(xg)
        g, = torch.autograd.grad(val, xg)
        return g

    x0 = torch.tensor(x0, dtype=DTYPE, device=DEVICE)
    grad0 = rev_grad(x0)
    eps = 1e-5
    fdHessian = (rev_grad(x0 + eps) - rev_grad(x0 - eps)) / (2 * eps)

    gradOk = float(grad0.abs()) <= 1e-10
    hessOk = float(fdHessian.abs()) <= 1e-4  # FD truncation, not round-off
    print(f"  [{'PASS' if gradOk else 'FAIL'}] {'reverse-mode self-gradient == 0':55s} value={float(grad0):.3e}")
    print(f"  [{'PASS' if hessOk else 'FAIL'}] {'FD-of-reverse-mode self-Hessian == 0 (not -15ish)':55s} value={float(fdHessian):.3e}")
    return gradOk and hessOk


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

    print("\nReverse-mode cross-check of the self-pair-drop identity:")
    ok &= reverse_mode_self_hessian_check()

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
