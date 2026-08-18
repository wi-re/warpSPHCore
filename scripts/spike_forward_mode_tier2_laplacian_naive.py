#!/usr/bin/env python3
"""Tier 2.3 forward-mode spike: Laplacian's Naive scheme (warpier_adjoint.md).

**Why this exists.** `LaplacianScheme.Naive` is not on any performance-
relevant path -- `wp_laplacian.py`'s own comments treat Brookshaw as the
consistent estimator, and Tier 2.2 already covers it. The plan's Tier 2.3
entry accordingly recommended deferring this tier and asking whether Naive
is used anywhere production-relevant before deriving it. Asked, and the
answer was: derive it anyway, for methodological completeness of the
adjoint SPH scheme -- Naive calls `sphKernelLaplacian` directly (the actual
analytic second-derivative-of-r estimator, `d2W/dr2 + (dim-1)/r * dW/dr`
with the same eps-regularized r/dot(x,x) algebra `sphKernelHessian_` uses),
so completing its Tier-2 JVP is part of stating the adjoint of the SPH
Laplacian in general, not just the one scheme wired into every current
consumer. Naive IS nonetheless a real, wired-in `LaplacianScheme`
(`wp_laplacian.py`'s Naive branch, `gradcheck_laplacian_native.py` already
reverse-mode validates it across every `GradientScheme`) -- so this is a
genuine (if currently unused) adjoint, not a derivation of dead code.

**The two new kernel-level building blocks** (`kernels/laplacian.py`):
`sphKernelLaplacianGradient_` (`d(sphKernelLaplacian_)/dx`, vector) and
`sphKernelLaplacianDkDh_` (`d(sphKernelLaplacian_)/dh`, scalar) -- the
genuinely new kernel math the plan's Tier 2.3 entry anticipated needing
`eval_d3kdq3` for (Section C already validated it; this is its first real
consumer, one rung up the `eval_dkdq -> eval_d2kdq2 -> eval_d3kdq3` ladder
from where Section G/H's Hessian/Laplacian sit). Both are validated at the
kernel level against `wp.Tape` in `kernel_sanity_native.py` Section K
(`sphKernelLaplacian_` itself was already AD-cross-checked there against
the Hessian's trace, Section G/H) -- see that section and the two
functions' own docstrings for the closed-form derivation. This script is
the operator-level half: does the assembled Naive-Laplacian JVP, built only
from those two validated building blocks plus ordinary calculus, match the
production operator's own reverse-mode derivative end to end.

**Assembly** (`wp_laplacian.py`'s Naive branch: `laplacian_contribution =
q_ij * sphKernelLaplacian(...)`, `sphKernelLaplacian`'s own two-branch
`SupportScheme` dispatch):

    SuperSymmetric (explicitly branched):
      L_ij  = 0.5*(sphKernelLaplacian_(xij,hi) + sphKernelLaplacian_(xij,hj))
      dL_ij = 0.5*[ dot(LG(xij,hi),dxij) + LDkDh(xij,hi)*dhi
                    + dot(LG(xij,hj),dxij) + LDkDh(xij,hj)*dhj ]
    else (Gather/Scatter/MeanSymmetric/KernelMeanSymmetric/max-fallback):
      h_ij = computePairwiseSupport(hi,hj,mode), dh_ij = Tier 2.1's _pairwiseSupportTangent(...)
      L_ij  = sphKernelLaplacian_(xij, h_ij)
      dL_ij = dot(LG(xij,h_ij),dxij) + LDkDh(xij,h_ij)*dh_ij

`q_ij` (Naive gradient-mode coefficient) is exactly Tier 2.2's `B_ij` again
(`_gradient_weights`, reused verbatim) -- `wp_laplacian.py`'s `q_ij` doesn't
depend on `laplacianMode` at all, only on `gradientMode`, so this is the
SAME finding Tier 2.2 already made, just re-confirmed for a different
`laplacianMode`:

    L = Sum_j q_ij * L_ij,   dL = Sum_j (dq_ij*L_ij + q_ij*dL_ij)

**A genuine finding, not shared with Tier 2.2's Brookshaw scheme.**
`sphKernelLaplacian`'s `SupportScheme` dispatch has only TWO branches
(SuperSymmetric explicit, everything else -- including KernelMeanSymmetric
-- falling through to `computePairwiseSupport`'s own dispatch, which has no
explicit KernelMeanSymmetric branch either and so silently uses the
max-fallback `h_ij`). This is DIFFERENT from `sphKernel_ij`/
`sphKernelGradient_ij` (Tier 2.1/2.2), which both give KernelMeanSymmetric
its own explicit two-term-average branch identical in structure to
SuperSymmetric's. For the kernel *value* and *gradient*, KernelMeanSymmetric
and SuperSymmetric are provably identical (Tier 2.1/2.2's findings); for the
Naive Laplacian, they are NOT -- KernelMeanSymmetric here gets a single
max-h evaluation, SuperSymmetric gets a genuine two-term average. Checked
explicitly below (the two schemes' assembled JVPs are asserted to *differ*,
the mirror image of Tier 2.2's "assert identical" check).

    python scripts/spike_forward_mode_tier2_laplacian_naive.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys
from typing import Any

import torch
import warp as wp
from warp.types import vector

from _gradcheck_common import DEVICE, DTYPE, KERNEL, build_adjacency, grid_case_2d, line_case, make_domain
from warpSPHCore import OperationProperties, ParticleState, warpOperation
from warpSPHCore.enumTypes import GradientScheme, LaplacianScheme, OperationDirection, SupportScheme, WarpOperation
from warpSPHCore.type_config import scalar_t
from warpSPHCore.kernels.laplacian import sphKernelLaplacian_, sphKernelLaplacianGradient_, sphKernelLaplacianDkDh_
from warpSPHCore.util.support import computePairwiseSupport

TOL = 1e-9  # float64, both sides exact analytic derivatives -- round-off only

vec1_t = vector(dtype=scalar_t, length=1)
vec2_t = vector(dtype=scalar_t, length=2)


# --------------------------------------------------------------------------
# Tier 2.1's building block, reused verbatim (computePairwiseSupport's
# dispatch is unchanged by this tier).
# --------------------------------------------------------------------------

@wp.func
def _pairwiseSupportTangent(hi: scalar_t, hj: scalar_t, dhi: scalar_t, dhj: scalar_t, mode: wp.uint32):
    if mode == wp.static(SupportScheme.Gather.value):
        return dhi
    elif mode == wp.static(SupportScheme.Scatter.value):
        return dhj
    elif mode == wp.static(SupportScheme.MeanSymmetric.value):
        return (dhi + dhj) / scalar_t(2.0)
    else:
        if hi >= hj:
            return dhi
        return dhj


# --------------------------------------------------------------------------
# New building block: the JVP of sphKernelLaplacian (kernels/laplacian.py's
# xi/xj/hi/hj wrapper) itself -- mirrors that function's two-branch
# SupportScheme dispatch (SuperSymmetric explicit, everything else via
# computePairwiseSupport's own dispatch -- see module docstring's finding).
# --------------------------------------------------------------------------

@wp.func
def _kernelLaplacianJVP(
    xij: vector(dtype=scalar_t, length=Any),  # type: ignore
    hi: scalar_t, hj: scalar_t,
    dxij: vector(dtype=scalar_t, length=Any),  # type: ignore
    dhi: scalar_t, dhj: scalar_t,
    mode: wp.uint32, kernel_id: wp.int32,
):
    if mode == wp.static(SupportScheme.SuperSymmetric.value):
        Li = sphKernelLaplacian_(xij, hi, kernel_id)
        Lj = sphKernelLaplacian_(xij, hj, kernel_id)
        val = (Li + Lj) * scalar_t(0.5)
        Gi = sphKernelLaplacianGradient_(xij, hi, kernel_id)
        Gj = sphKernelLaplacianGradient_(xij, hj, kernel_id)
        DhI = sphKernelLaplacianDkDh_(xij, hi, kernel_id)
        DhJ = sphKernelLaplacianDkDh_(xij, hj, kernel_id)
        dval = (wp.dot(Gi, dxij) + DhI * dhi + wp.dot(Gj, dxij) + DhJ * dhj) * scalar_t(0.5)
    else:
        hij = computePairwiseSupport(hi, hj, mode)
        dhij = _pairwiseSupportTangent(hi, hj, dhi, dhj, mode)
        val = sphKernelLaplacian_(xij, hij, kernel_id)
        G = sphKernelLaplacianGradient_(xij, hij, kernel_id)
        Dh = sphKernelLaplacianDkDh_(xij, hij, kernel_id)
        dval = wp.dot(G, dxij) + Dh * dhij
    return val, dval


@wp.kernel
def _pair_jvp_lap_1d(
    xi: wp.array(dtype=vec1_t), xj: wp.array(dtype=vec1_t),
    hi: wp.array(dtype=scalar_t), hj: wp.array(dtype=scalar_t),
    dxi: wp.array(dtype=vec1_t), dxj: wp.array(dtype=vec1_t),
    dhi: wp.array(dtype=scalar_t), dhj: wp.array(dtype=scalar_t),
    mode: wp.uint32, kernel_id: wp.int32,
    L_out: wp.array(dtype=scalar_t), dL_out: wp.array(dtype=scalar_t),
):
    p = wp.tid()
    xij = xi[p] - xj[p]
    dxij = dxi[p] - dxj[p]
    val, dval = _kernelLaplacianJVP(xij, hi[p], hj[p], dxij, dhi[p], dhj[p], mode, kernel_id)
    L_out[p] = val
    dL_out[p] = dval


@wp.kernel
def _pair_jvp_lap_2d(
    xi: wp.array(dtype=vec2_t), xj: wp.array(dtype=vec2_t),
    hi: wp.array(dtype=scalar_t), hj: wp.array(dtype=scalar_t),
    dxi: wp.array(dtype=vec2_t), dxj: wp.array(dtype=vec2_t),
    dhi: wp.array(dtype=scalar_t), dhj: wp.array(dtype=scalar_t),
    mode: wp.uint32, kernel_id: wp.int32,
    L_out: wp.array(dtype=scalar_t), dL_out: wp.array(dtype=scalar_t),
):
    p = wp.tid()
    xij = xi[p] - xj[p]
    dxij = dxi[p] - dxj[p]
    val, dval = _kernelLaplacianJVP(xij, hi[p], hj[p], dxij, dhi[p], dhj[p], mode, kernel_id)
    L_out[p] = val
    dL_out[p] = dval


_PAIR_JVP_LAP_BY_DIM = {1: _pair_jvp_lap_1d, 2: _pair_jvp_lap_2d}
_VEC_BY_DIM = {1: vec1_t, 2: vec2_t}


def _dense_kernelLaplacian_jvp(pos, sup, dpos, dsup, dim, mode, kernel_id):
    """All-pairs (i, j) including i==j, L_ij and dL_ij as (n, n) torch
    tensors. Safe for the same reason Tier 2.1/2.2's dense loops were:
    sphKernelLaplacian_/sphKernelLaplacianGradient_/sphKernelLaplacianDkDh_
    are all exactly zero for q=|x|/h>1 (and, per their own q<eps cutoff,
    for the self-pair too), so a pair outside the true support radius
    contributes nothing to either side regardless of whether the real
    neighbor search would have found it."""
    n = pos.shape[0]
    vec_t = _VEC_BY_DIM[dim]
    idx_i = torch.arange(n).repeat_interleave(n)
    idx_j = torch.arange(n).repeat(n)

    xi = wp.from_torch(pos[idx_i].contiguous(), dtype=vec_t)
    xj = wp.from_torch(pos[idx_j].contiguous(), dtype=vec_t)
    hi = wp.from_torch(sup[idx_i].contiguous(), dtype=scalar_t)
    hj = wp.from_torch(sup[idx_j].contiguous(), dtype=scalar_t)
    dxi = wp.from_torch(dpos[idx_i].contiguous(), dtype=vec_t)
    dxj = wp.from_torch(dpos[idx_j].contiguous(), dtype=vec_t)
    dhi = wp.from_torch(dsup[idx_i].contiguous(), dtype=scalar_t)
    dhj = wp.from_torch(dsup[idx_j].contiguous(), dtype=scalar_t)

    L_out = wp.zeros(n * n, dtype=scalar_t, device=DEVICE.type)
    dL_out = wp.zeros(n * n, dtype=scalar_t, device=DEVICE.type)
    wp.launch(_PAIR_JVP_LAP_BY_DIM[dim], dim=n * n,
              inputs=[xi, xj, hi, hj, dxi, dxj, dhi, dhj, mode, kernel_id],
              outputs=[L_out, dL_out], device=DEVICE.type)

    L = wp.to_torch(L_out).reshape(n, n)
    dL = wp.to_torch(dL_out).reshape(n, n)
    return L, dL


# --------------------------------------------------------------------------
# Ordinary calculus: Naive's q_ij == Tier 2.2's B_ij, reused verbatim (see
# module docstring -- q_ij doesn't depend on laplacianMode at all).
# --------------------------------------------------------------------------

def _gradient_weights_B(mass, density, dmass, ddensity, scheme: GradientScheme):
    """B_ij only (Tier 2.2's _gradient_weights, A_ij dropped -- Naive's q_ij
    never uses A_ij, it IS the B_ij term for every GradientScheme, exactly
    as Tier 2.2's docstring point 2 established for Brookshaw)."""
    density_i = density.unsqueeze(1)
    ddensity_i = ddensity.unsqueeze(1)
    mass_j = mass.unsqueeze(0)
    density_j = density.unsqueeze(0)
    dmass_j = dmass.unsqueeze(0)
    ddensity_j = ddensity.unsqueeze(0)

    Vj = mass_j / density_j
    dVj = dmass_j / density_j - mass_j * ddensity_j / density_j**2

    if scheme in (GradientScheme.Naive, GradientScheme.Difference, GradientScheme.Summation):
        B, dB = Vj, dVj
    elif scheme == GradientScheme.Symmetric:
        B = mass_j * density_i / density_j**2
        dB = (dmass_j * density_i / density_j**2
              + mass_j * ddensity_i / density_j**2
              - 2.0 * mass_j * density_i * ddensity_j / density_j**3)
    else:
        raise ValueError(scheme)
    return B, dB


def assembled_laplacian_naive_jvp(pos, sup, mass, density, fv_q, fv_r, dpos, dsup, dmass, ddensity, dim, mode, kernel_id, scheme):
    """Scalar field. q_ij = (fj-fi)*B_ij, L = Sum_j q_ij*L_ij (wp_laplacian.py's
    Naive branch: laplacian_contribution = q_ij * sphKernelLaplacian(...))."""
    L, dL = _dense_kernelLaplacian_jvp(pos, sup, dpos, dsup, dim, mode, kernel_id)
    B, dB = _gradient_weights_B(mass, density, dmass, ddensity, scheme)
    fi = fv_q.unsqueeze(1)
    fj = fv_r.unsqueeze(0)
    q = (fj - fi) * B
    dq = (fj - fi) * dB

    out = (q * L).sum(dim=1)
    d_out = (dq * L + q * dL).sum(dim=1)
    return out, d_out


# --------------------------------------------------------------------------
# Reference: reverse-mode Jacobian of the PRODUCTION operator (Tier 2.1/2.2's pattern).
# --------------------------------------------------------------------------

def _reference_jvp(f, primals, tangents):
    J = torch.autograd.functional.jacobian(f, primals, vectorize=False)
    out = f(*primals).detach()
    n_out = out.numel()
    acc = torch.zeros(n_out, dtype=DTYPE, device=DEVICE)
    for Jk, vk in zip(J, tangents):
        acc = acc + Jk.reshape(n_out, -1) @ vk.reshape(-1)
    return acc.reshape(out.shape)


def check(name, assembled, reference):
    assembled_flat, reference_flat = assembled.reshape(-1), reference.reshape(-1)
    assert assembled_flat.numel() == reference_flat.numel(), (
        f"{name}: shape mismatch assembled={tuple(assembled.shape)} reference={tuple(reference.shape)}"
    )
    scale = max(float(reference_flat.abs().max()), 1e-300)
    err = float((assembled_flat - reference_flat).abs().max()) / scale
    ok = err <= 1e-9
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:70s} rel_err={err:.3e}")
    return ok


def _compute_densities(pos, sup, mass, kinds, domain, adjacency):
    p = ParticleState(positions=pos.detach(), supports=sup.detach(), masses=mass.detach(), densities=None, kinds=kinds)
    rho = warpOperation(
        p, OperationProperties(kernel=KERNEL, operation=WarpOperation.Density,
                                supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll),
        domain, adjacency=adjacency,
    )
    return rho.detach().clone()


def _perturbed_case(n, dim, seed):
    torch.manual_seed(seed)
    if dim == 1:
        pos0, sup0, mass0 = line_case(n)
    else:
        pos0, sup0, mass0 = grid_case_2d(n)
    pos0, sup0, mass0 = pos0.detach(), sup0.detach(), mass0.detach()
    sup0 = sup0 * (1.0 + 0.15 * torch.linspace(-1, 1, sup0.shape[0], dtype=DTYPE))
    domain = make_domain(dim=dim)
    adjacency, kinds = build_adjacency(pos0, sup0, mass0, domain, mode=SupportScheme.KernelMeanSymmetric)
    density0 = _compute_densities(pos0, sup0, mass0, kinds, domain, adjacency)
    return pos0, sup0, mass0, density0, domain, adjacency, kinds


def run_laplacian_naive_case(n, dim, mode: SupportScheme, scheme: GradientScheme, kernel_id, seed=0):
    pos0, sup0, mass0, density0, domain, adjacency, kinds = _perturbed_case(n, dim, seed)

    def f(pos, sup, mass, density, qval, rval):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=density, kinds=kinds)
        return warpOperation(
            p, OperationProperties(kernel=KERNEL, operation=WarpOperation.Laplacian,
                                    supportMode=mode, operationMode=OperationDirection.AllToAll,
                                    gradientMode=scheme, laplacianMode=LaplacianScheme.Naive),
            domain, queryValues=qval, referenceValues=rval, adjacency=adjacency,
        )

    for t in (pos0, sup0, mass0, density0):
        t.requires_grad_(True)
    dpos, dsup, dmass, ddensity = (torch.randn_like(pos0), torch.randn_like(sup0) * 0.1,
                                    torch.randn_like(mass0), torch.randn_like(density0) * 0.1)
    fv_q = torch.randn(pos0.shape[0], dtype=DTYPE)
    fv_r = torch.randn(pos0.shape[0], dtype=DTYPE)

    reference = _reference_jvp(f, (pos0, sup0, mass0, density0, fv_q, fv_r), (dpos, dsup, dmass, ddensity, torch.zeros_like(fv_q), torch.zeros_like(fv_r)))
    _, d = assembled_laplacian_naive_jvp(pos0.detach(), sup0.detach(), mass0.detach(), density0.detach(), fv_q, fv_r,
                                          dpos, dsup, dmass, ddensity, dim, mode.value, kernel_id, scheme)
    return d, reference


def main():
    wp.init()
    print(__doc__.split("\n\n")[0])
    print()

    kernel_id = KERNEL.value
    ok = True

    print("Laplacian (Naive), 1D line of 7 particles, non-uniform supports, all GradientScheme x SupportScheme subset:")
    for scheme in GradientScheme:
        for mode in (SupportScheme.Gather, SupportScheme.Scatter, SupportScheme.MeanSymmetric,
                     SupportScheme.KernelMeanSymmetric, SupportScheme.SuperSymmetric, SupportScheme.PartialSymmetric):
            d, r = run_laplacian_naive_case(7, 1, mode, scheme, kernel_id)
            ok &= check(f"Laplacian/Naive JVP ({scheme.name}/{mode.name})", d, r)

    print("\nLaplacian (Naive), 2D 3x3 grid, non-uniform supports:")
    for scheme in GradientScheme:
        for mode in (SupportScheme.Gather, SupportScheme.MeanSymmetric, SupportScheme.KernelMeanSymmetric, SupportScheme.SuperSymmetric):
            d, r = run_laplacian_naive_case(3, 2, mode, scheme, kernel_id)
            ok &= check(f"Laplacian/Naive JVP ({scheme.name}/{mode.name})", d, r)

    print("\nKernelMeanSymmetric != SuperSymmetric for the Naive Laplacian JVP (unlike Tier 2.1/2.2's kernel")
    print("  value/gradient, where the two schemes coincide) -- sphKernelLaplacian only special-cases")
    print("  SuperSymmetric, KernelMeanSymmetric silently falls through to the max-h branch (see docstring):")
    d_kms, _ = run_laplacian_naive_case(7, 1, SupportScheme.KernelMeanSymmetric, GradientScheme.Naive, kernel_id, seed=1)
    d_ss, _ = run_laplacian_naive_case(7, 1, SupportScheme.SuperSymmetric, GradientScheme.Naive, kernel_id, seed=1)
    different = not bool(torch.allclose(d_kms, d_ss, atol=1e-12))
    print(f"  [{'PASS' if different else 'FAIL'}] assembled JVP genuinely differs between the two schemes: {different}")
    ok &= different

    print()
    if ok:
        print("ALL PASSED -- Tier 2.3's assembled JVP (sphKernelLaplacianGradient_/")
        print("  sphKernelLaplacianDkDh_, chain-ruled through sphKernelLaplacian's own")
        print("  SupportScheme dispatch and Naive's q_ij coefficient) matches the")
        print("  production LaplacianScheme.Naive operator's own reverse-mode derivative.")
    else:
        print("FAILED -- see the individual case(s) above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
