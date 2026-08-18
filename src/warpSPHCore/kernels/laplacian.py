from typing import Any
from ..type_config import *
import warp as wp
from warp.types import vector, matrix
from .properties import eval_C_d
from .eval_kernel import *
import numpy as np
from ..math import *
from ..type_config import scalar_t, dim_t
from .kernelFunctions import *
from ..util.support import computePairwiseSupport
from ..dataTypes.domain_t import domainData
from ..dataTypes.kernelState_t import kernelState

@wp.func
def sphKernelLaplacian_(x: vector(dtype=scalar_t, length=dim_t), h: scalar_t, kernel: wp.int32):
    dim = wp.int32(x.length)
    r = vectorNorm_warp(x)
    q = r / h
    eps = get_epsilon(r)
    r_eps = r + eps * h
    
    k1 = eval_dkdq(q, dim, kernel)   * eval_C_d(dim, kernel) / iPow(h, dim + 1)
    k2 = eval_d2kdq2(q, dim, kernel) * eval_C_d(dim, kernel) / iPow(h, dim + 2)
    
    s = wp.dot(x,x) / iPow(r_eps, 2)
    if q < eps:
        s = scalar_t(1.0)
    t = - wp.dot(x,x) / iPow(r_eps, 3)
    t += scalar_t(dim) / r_eps
    
    laplacian = s * k2 + t * k1
    if q < eps or q > scalar_t(1.0):
        laplacian = scalar_t(0.0)
    return laplacian

@wp.func
def sphKernelLaplacianGradient_(x: vector(dtype=scalar_t, length=dim_t), h: scalar_t, kernel: wp.int32):
    """d(sphKernelLaplacian_)/dx -- Tier 2.3 (warpier_adjoint.md): the position
    tangent of the analytic (LaplacianScheme.Naive) SPH Laplacian estimator.

    Not on any performance-relevant path -- Brookshaw, the estimator
    wp_laplacian.py's own comments treat as the consistent one, is covered by
    Tier 2.2's kernelGradient JVP instead -- but Naive IS a real, wired-in
    LaplacianScheme (wp_laplacian.py's Naive branch calls sphKernelLaplacian
    directly, and gradcheck_laplacian_native.py already reverse-mode
    validates it), so its forward-mode adjoint completes the same derivation
    as every other operator here rather than extending it to something
    hypothetical. Derived for methodological completeness of the adjoint SPH
    scheme, at the user's request, not because anything downstream consumes
    it yet.

    Derivation. sphKernelLaplacian_(x,h) = s*k2 + t*k1, with
        k1 = dW/dr  = eval_dkdq(q)  * C_d / h^(dim+1)   (== sphGradient_'s magnitude)
        k2 = d2W/dr2 = eval_d2kdq2(q) * C_d / h^(dim+2) (== sphKernelHessian_'s radial factor)
        k3 = d3W/dr3 = eval_d3kdq3(q) * C_d / h^(dim+3) (NEW: one more rung of the same
             eval_dkdq -> eval_d2kdq2 -> eval_d3kdq3 ladder Section C already validates;
             this is its first consumer)
        s = dot(x,x)/r_eps^2,  t = -dot(x,x)/r_eps^3 + dim/r_eps,  r_eps = r + eps*h
    (eps = get_epsilon(r) is a dtype-only constant, not itself a function of r or h --
    d(eps)/dx = d(eps)/dh = 0 throughout, so r_eps's only x/h-dependence is through r
    itself). Differentiating w.r.t. x by the product rule across s, t, k1, k2 -- using
    dr/dx = direction, dk1/dx = k2*direction, dk2/dx = k3*direction (the derivative
    shifts the whole k-ladder up one rung, the same identity sphGradientDkDh_'s
    docstring uses one rung lower for d/dh) -- collapses to:
        d(Laplacian)/dx = k2*ds/dx + k1*dt/dx + (s*k3 + t*k2)*direction
        ds/dx = 2*r*eps*h/r_eps^3 * direction          (O(eps): vanishes as r_eps -> r)
        dt/dx = direction*[(3*dot(x,x) - 2*r*r_eps)/r_eps^4 - dim/r_eps^2]
    Cross-checked (before ever touching wp.Tape) against the eps->0 textbook closed form
    Laplacian = k2 + (dim-1)/r*k1 differentiated directly: d/dr[...] = k3 -
    (dim-1)/r^2*k1 + (dim-1)/r*k2, algebraically identical to the bracketed term above
    once t -> (dim-1)/r. See warpier_adjoint.md Tier 2.3 for the full write-up.

    r=0 (self-pair) needs no special branch: `direction` comes from
    vectorNormalize_warp, which returns the zero vector (not NaN) for a zero-length
    input, so every term above -- each carrying an explicit factor of `direction` --
    evaluates to exactly zero there. This function does replicate
    sphKernelLaplacian_'s own `q<eps` cutoff (the value is identically 0 on that open
    region, so its derivative there is exactly 0 too, not merely untested) but does NOT
    attempt to differentiate across the discrete jump at q==eps itself -- a
    measure-zero point no validation sample here lands on, same convention as every
    other Section in kernel_sanity_native.py.
    """
    dim = wp.int32(x.length)
    r = vectorNorm_warp(x)
    q = r / h
    eps = get_epsilon(r)
    if q > scalar_t(1.0) or q < eps:
        return type(x)(scalar_t(0.0))

    r_eps = r + eps * h
    R = wp.dot(x, x)

    Cd = eval_C_d(dim, kernel)
    k1 = eval_dkdq(q, dim, kernel)   * Cd / iPow(h, dim + 1)
    k2 = eval_d2kdq2(q, dim, kernel) * Cd / iPow(h, dim + 2)
    k3 = eval_d3kdq3(q, dim, kernel) * Cd / iPow(h, dim + 3)

    s = R / iPow(r_eps, 2)
    t = - R / iPow(r_eps, 3) + scalar_t(dim) / r_eps

    direction = vectorNormalize_warp(input=x)
    ds_dx = scalar_t(2.0) * r * eps * h / iPow(r_eps, 3) * direction
    dt_dx = direction * ((scalar_t(3.0) * R - scalar_t(2.0) * r * r_eps) / iPow(r_eps, 4) - scalar_t(dim) / iPow(r_eps, 2))

    return k2 * ds_dx + k1 * dt_dx + (s * k3 + t * k2) * direction


@wp.func
def sphKernelLaplacianDkDh_(x: vector(dtype=scalar_t, length=dim_t), h: scalar_t, kernel: wp.int32):
    """d(sphKernelLaplacian_)/dh -- the /dh companion to
    sphKernelLaplacianGradient_ above; see that function's docstring for the
    full derivation context (Tier 2.3, warpier_adjoint.md) and why this is
    worth deriving despite Naive not being the performance-relevant scheme.

    Same s*k2+t*k1 closed form, differentiated w.r.t. h instead of x
    (dr/dh=0, dr_eps/dh=eps, dq/dh=-q/h). The mixed partials dk1/dh, dk2/dh
    follow the exact "k-ladder shifts up one rung, gains a -q*(next k) term"
    pattern sphGradientDkDh_'s docstring already established for dk1/dh;
    dk2/dh is that same pattern one rung higher, using k3 from
    sphKernelLaplacianGradient_ above:
        dk1/dh = -q*k2 - (dim+1)*k1/h    (== sphGradientDkDh_'s scalar magnitude)
        dk2/dh = -q*k3 - (dim+2)*k2/h    (NEW, one rung up)
        ds/dh  = -2*eps*dot(x,x)/r_eps^3
        dt/dh  = 3*eps*dot(x,x)/r_eps^4 - dim*eps/r_eps^2
        d(Laplacian)/dh = k2*ds/dh + k1*dt/dh - q*(s*k3+t*k2)
                          - (s*(dim+2)*k2 + t*(dim+1)*k1)/h
    Cross-checked against the same eps->0 textbook-form differentiation as the
    /dx function (d/dh[k2+(dim-1)/r*k1] expands to exactly this, term for
    term) before touching wp.Tape. No r=0 special-casing needed for the same
    reason as /dx: dot(x,x)=0 there kills every term that would otherwise
    blow up through r_eps=eps*h, and the q<eps cutoff below matches
    sphKernelLaplacian_'s own (the value is identically 0 on that region, so
    its h-derivative there is exactly 0 too).
    """
    dim = wp.int32(x.length)
    r = vectorNorm_warp(x)
    q = r / h
    eps = get_epsilon(r)
    if q > scalar_t(1.0) or q < eps:
        return scalar_t(0.0)

    r_eps = r + eps * h
    R = wp.dot(x, x)

    Cd = eval_C_d(dim, kernel)
    k1 = eval_dkdq(q, dim, kernel)   * Cd / iPow(h, dim + 1)
    k2 = eval_d2kdq2(q, dim, kernel) * Cd / iPow(h, dim + 2)
    k3 = eval_d3kdq3(q, dim, kernel) * Cd / iPow(h, dim + 3)

    s = R / iPow(r_eps, 2)
    t = - R / iPow(r_eps, 3) + scalar_t(dim) / r_eps

    ds_dh = - scalar_t(2.0) * eps * R / iPow(r_eps, 3)
    dt_dh = scalar_t(3.0) * eps * R / iPow(r_eps, 4) - scalar_t(dim) * eps / iPow(r_eps, 2)
    M = s * k3 + t * k2

    return k2 * ds_dh + k1 * dt_dh - q * M - (s * scalar_t(dim + 2) * k2 + t * scalar_t(dim + 1) * k1) / h


@wp.func
def sphKernelLaplacian(
    xi: vector(dtype=scalar_t, length=dim_t),
    xj: vector(dtype=scalar_t, length=dim_t),
    hi: scalar_t,
    hj: scalar_t,
    kernelProperties: kernelState,
    domainState: domainData,
):
    hij = computePairwiseSupport(hi, hj, kernelProperties.supportMode)
    xij = computeDistanceVec(xi, xj, domainState)
    if kernelProperties.supportMode == wp.static(SupportScheme.SuperSymmetric.value): # SuperSymmetric
        return (sphKernelLaplacian_(xij,hi,kernelProperties.kernelFunction) + sphKernelLaplacian_(xij,hj,kernelProperties.kernelFunction))/scalar_t(2.0)

    return sphKernelLaplacian_(xij, hij, kernelProperties.kernelFunction)
