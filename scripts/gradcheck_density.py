#!/usr/bin/env python3
"""Reverse-mode gradient check for the Density operator.

Checks that d(density)/d(positions|supports|masses), computed by warpOperation's
reverse-mode AD, matches a central-difference numerical estimate -- the
backward-mode counterpart to scripts/operation_matrix.py, which only checks
forward values. Run as its own process:

    python scripts/gradcheck_density.py
    python scripts/gradcheck_density.py --line-n 9 --plot

--------------------------------------------------------------------------
Precision note
--------------------------------------------------------------------------
sphWarpCore's scalar precision (SPHWARPCORE_PRECISION) is a process-global
setting baked into every @wp.kernel/@wp.func at *import* time, so it cannot
be changed after sphWarpCore has been imported once in a process. A
gradcheck-quality comparison needs float64. That is why this is a
standalone script (`python scripts/gradcheck_density.py`) rather than a
pytest case living alongside the float32 forward-mode suite in
tests/operations/ -- pytest runs everything in one process, and the
forward-mode suite intentionally runs at the library's default float32.

--------------------------------------------------------------------------
Historical note: manual Jacobian, not torch.autograd.gradcheck
--------------------------------------------------------------------------
This script builds the analytical Jacobian by hand -- one fresh
forward+backward pass per output row -- and cross-checks it against a
central-difference numerical Jacobian, instead of calling
torch.autograd.gradcheck. That was originally a workaround for what looked
like a warp-lang Tape-reentrancy limitation (a second backward() against a
retained graph, which is exactly what gradcheck's default multi-output
Jacobian construction does, silently returned a wrong gradient). That bug
turned out to be in sphWarpCore's own AD bridge (WarpFunctionWrapper.backward
/ StateAwareWarpFunction.backward in wp_autograd.py: the gradient read out
of Warp wasn't cloned, and the tape wasn't zeroed afterward, so a later call
sharing the same tensor storage could read stale/aliased state) and has
since been fixed -- see warpier_core.md's "Backward-Mode (Reverse AD)
Findings" and scripts/gradcheck_density_native.py, which now passes calling
gradcheck directly with zero workarounds.

This script itself was left as-is rather than rewritten to use gradcheck,
since its manual Jacobian, closed-form single-particle check, self/non-self
breakdown, and plot are still valid, still-passing coverage on their own
merits -- not because the workaround is still needed. New gradcheck scripts
for other operators should call torch.autograd.gradcheck directly (see
gradcheck_density_native.py for the pattern), not copy this file's manual
Jacobian machinery.
"""

from __future__ import annotations

import os

os.environ.setdefault("SPHWARPCORE_PRECISION", "float64")

import argparse
import sys

import torch
import warp as wp

from _gradcheck_common import (
    DEVICE,
    DTYPE,
    KERNEL,
    WENDLAND2_C_D_1D,
    build_adjacency,
    line_case,
    make_domain,
    single_particle_case,
    wendland2_kernel_1d,
)
from sphWarpCore import OperationProperties, ParticleState, warpOperation
from sphWarpCore.enumTypes import OperationDirection, SupportScheme, WarpOperation


# --------------------------------------------------------------------------
# Pure-PyTorch reference density built on the shared Wendland2 kernel
# (SupportScheme.Gather: h_ij = h_i, non-periodic):
#   rho_i = sum_j m_j * W(|x_i - x_j|, h_i)
# --------------------------------------------------------------------------

def reference_density(positions: torch.Tensor, supports: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
    x = positions.squeeze(-1)
    r = (x.unsqueeze(1) - x.unsqueeze(0)).abs()  # r[i, j] = |x_i - x_j|
    h = supports.unsqueeze(1)  # Gather: h_ij = h_i, broadcasts over j
    W = wendland2_kernel_1d(r, h)
    return (W * masses.unsqueeze(0)).sum(dim=1)


# --------------------------------------------------------------------------
# Warp density wrapper. Clones inputs on every call -- see the module
# docstring's reentrancy note for why.
# --------------------------------------------------------------------------

def make_warp_density_fn(domain, adjacency, kinds):
    def warp_density(positions, supports, masses):
        pos_c, sup_c, mass_c = positions.clone(), supports.clone(), masses.clone()
        p = ParticleState(positions=pos_c, supports=sup_c, masses=mass_c, densities=None, kinds=kinds)
        return warpOperation(
            p,
            OperationProperties(
                kernel=KERNEL,
                operation=WarpOperation.Density,
                supportMode=SupportScheme.Gather,
                operationMode=OperationDirection.AllToAll,
            ),
            domain,
            adjacency=adjacency,
        )

    return warp_density


# --------------------------------------------------------------------------
# Manual analytical + numerical Jacobian (see reentrancy note: this
# deliberately does not use torch.autograd.gradcheck).
# --------------------------------------------------------------------------

def analytical_jacobian(fn, inputs):
    """dOutput/dInput_k for each input, one fresh forward+backward pass per output row."""
    out0 = fn(*inputs)
    n_out = out0.numel()
    jacs = [torch.zeros(n_out, inp.numel(), dtype=DTYPE) for inp in inputs]
    for k in range(n_out):
        out = fn(*inputs)  # fresh tape every row -- see module docstring
        grad_outputs = torch.zeros(n_out, dtype=DTYPE)
        grad_outputs[k] = 1.0
        grads = torch.autograd.grad(out.reshape(-1), inputs, grad_outputs=grad_outputs, retain_graph=False, allow_unused=True)
        for gi, g in enumerate(grads):
            if g is not None:
                jacs[gi][k, :] = g.reshape(-1)
    return jacs


def numerical_jacobian(fn, inputs, eps=1e-6):
    """Central-difference Jacobian. Forward-only, so unaffected by the reentrancy issue."""
    with torch.no_grad():
        out0 = fn(*inputs).reshape(-1).clone()
    n_out = out0.numel()
    jacs = [torch.zeros(n_out, inp.numel(), dtype=DTYPE) for inp in inputs]
    with torch.no_grad():
        for ii, inp in enumerate(inputs):
            flat = inp.reshape(-1)
            for j in range(flat.numel()):
                orig = flat[j].item()
                flat[j] = orig + eps
                out_p = fn(*inputs).reshape(-1).clone()
                flat[j] = orig - eps
                out_m = fn(*inputs).reshape(-1).clone()
                flat[j] = orig
                jacs[ii][:, j] = (out_p - out_m) / (2 * eps)
    return jacs


def compare(name, analytical, numerical, atol=1e-5, rtol=1e-3):
    diff = (analytical - numerical).abs()
    tol = atol + rtol * numerical.abs()
    ok = bool((diff <= tol).all())
    max_err = diff.max().item() if diff.numel() else 0.0
    status = "OK" if ok else "MISMATCH"
    print(f"  [{status}] {name}: max_abs_err={max_err:.3e}, shape={tuple(analytical.shape)}")
    if not ok:
        bad = (diff > tol).nonzero()
        print(f"    worst entries (row=output idx, col=input idx): {bad[:8].tolist()}")
        for idx in bad[:5]:
            i, j = idx.tolist()
            print(f"      out[{i}]/in[{j}]: analytical={analytical[i,j].item():.8f} numerical={numerical[i,j].item():.8f}")
    return ok


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

def run_single_particle_case(h=1.0):
    print(f"\n=== Single particle, h={h} ===")
    positions, supports, masses = single_particle_case(h)
    domain = make_domain()
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    fn = make_warp_density_fn(domain, adjacency, kinds)

    analytical = analytical_jacobian(fn, (positions, supports, masses))
    numerical = numerical_jacobian(fn, (positions, supports, masses))

    all_ok = True
    for name, a, n in zip(("d(rho)/d(position)", "d(rho)/d(support)", "d(rho)/d(mass)"), analytical, numerical):
        all_ok &= compare(name, a, n)

    # Closed-form sanity check specific to the single-particle case: the only
    # contribution is the self term. q=r/h=0 always (r=0 regardless of h),
    # so k(q)=k(0)=1 is constant in h -- only the C_d/h prefactor depends on
    # h -- and the kernel gradient at r=0 is exactly zero for a symmetric
    # kernel (wendland2_dkdq(0) = 0), so d(rho)/d(position) must be exactly 0.
    m = masses.item()
    expected_drho_dpos = 0.0
    expected_drho_dh = -m * WENDLAND2_C_D_1D / (h * h)
    expected_drho_dm = WENDLAND2_C_D_1D / h
    print("  Closed-form self-term check:")
    print(f"    d(rho)/d(position) expected {expected_drho_dpos:.6f}, got {analytical[0][0,0].item():.6f}")
    print(f"    d(rho)/d(support)  expected {expected_drho_dh:.6f}, got {analytical[1][0,0].item():.6f}")
    print(f"    d(rho)/d(mass)     expected {expected_drho_dm:.6f}, got {analytical[2][0,0].item():.6f}")
    all_ok &= abs(analytical[0][0, 0].item() - expected_drho_dpos) < 1e-8
    all_ok &= abs(analytical[1][0, 0].item() - expected_drho_dh) < 1e-6
    all_ok &= abs(analytical[2][0, 0].item() - expected_drho_dm) < 1e-6
    return all_ok


def run_line_case(n, xmin=-1.0, xmax=1.0, plot=False):
    print(f"\n=== Line of {n} particles, x in [{xmin}, {xmax}] ===")
    positions, supports, masses = line_case(n, xmin, xmax)
    domain = make_domain()
    adjacency, kinds = build_adjacency(positions, supports, masses, domain)
    fn = make_warp_density_fn(domain, adjacency, kinds)

    analytical = analytical_jacobian(fn, (positions, supports, masses))
    numerical = numerical_jacobian(fn, (positions, supports, masses))

    all_ok = True
    for name, a, n_ in zip(("d(rho)/d(position)", "d(rho)/d(support)", "d(rho)/d(mass)"), analytical, numerical):
        all_ok &= compare(name, a, n_)

    dpos = analytical[0]  # [n_out, n_in], both == n
    self_term = dpos.diagonal()
    off_diag_sum = dpos.sum(dim=1) - self_term
    print("  Self vs non-self d(rho_i)/d(x_j) breakdown (diagonal vs off-diagonal row-sum):")
    for i in range(n):
        print(f"    i={i}: self={self_term[i].item(): .6f}  sum_j!=i={off_diag_sum[i].item(): .6f}")

    if plot:
        _plot_line_case(positions, supports, masses, dpos)

    return all_ok


def _plot_line_case(positions, supports, masses, warp_dpos):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = positions.detach().squeeze(-1)
    n = x.shape[0]

    with torch.no_grad():
        ref_rho = reference_density(positions.detach(), supports.detach(), masses.detach())
    ref_grad_rows = []
    for k in range(n):
        p = positions.detach().clone().requires_grad_(True)
        rho = reference_density(p, supports.detach(), masses.detach())
        g = torch.autograd.grad(rho[k], p, retain_graph=False)[0].squeeze(-1)
        ref_grad_rows.append(g)
    ref_dpos = torch.stack(ref_grad_rows, dim=0)

    idx = list(range(n))
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    axes[0].plot(idx, warp_dpos.diagonal().tolist(), "o-", label="warp AD (self term)")
    axes[0].plot(idx, ref_dpos.diagonal().tolist(), "x--", label="pure-torch reference (self term)")
    axes[0].set_ylabel(r"$\partial \rho_i / \partial x_i$")
    axes[0].set_title("Density self-gradient: warp AD vs. independent pure-torch reference")
    axes[0].legend()

    axes[1].plot(idx, (warp_dpos.sum(dim=1) - warp_dpos.diagonal()).tolist(), "o-", label="warp AD (sum over j!=i)")
    axes[1].plot(idx, (ref_dpos.sum(dim=1) - ref_dpos.diagonal()).tolist(), "x--", label="pure-torch reference (sum over j!=i)")
    axes[1].set_xlabel("particle index i")
    axes[1].set_ylabel(r"$\sum_{j \neq i} \partial \rho_i / \partial x_j$")
    axes[1].legend()

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "gradcheck_density_line.png")
    fig.savefig(out_path, dpi=150)
    print(f"  Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--line-n", type=int, default=7, help="number of particles in the line case (default: 7)")
    parser.add_argument("--h", type=float, default=1.0, help="support radius for the single-particle case (default: 1.0)")
    parser.add_argument("--plot", action="store_true", help="save a self/non-self gradient comparison plot for the line case")
    args = parser.parse_args()

    wp.init()

    ok = True
    ok &= run_single_particle_case(h=args.h)
    ok &= run_line_case(args.line_n, plot=args.plot)

    print()
    print("PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
