"""Hessian-vector product of the Density operator w.r.t. positions
(`warpier_forward_mode_plan.md` Phase 4 Step 3, "`Hess C . v` is a JVP of
that JVP"): differentiate `computeSPHDensityGeometryJVP`'s own position
tangent once more, in the same direction. Concretely this reduces to
`HVP_i = sum_{j != i} m_j * H_ij @ (tangentQuery_i - tangentReference_j)`,
`H_ij` the pairwise kernel Hessian `kernels.hessian.sphKernelHessian` already
computes -- **no new kernel math**, the same callback Tier 2.1's own
derivation made for the first-order JVP (`warpier_adjoint.md` Tier 2.1
"Building blocks needed").

**Self-pairs (`j == i`) are always excluded, and this is an exact identity,
not a numerical-safety fallback -- correcting a claim that turned out to be
wrong (see below).** `C_i = sum_j m_j * W(x_i - x_j, h_ij)` depends on `x_i`
through *two* channels once `j = i`: the term's own `x_i` argument, and the
fact that its `x_j` argument is *also* `x_i` for that one term. Writing
`f(x) = W(x - x, h) = W(0, h)` for the self term as a function of the single
shared position variable makes this explicit: `f` is a constant (an SPH
particle's distance to itself is exactly zero everywhere in configuration
space, not just at the evaluation point), so `f'(x) = f''(x) = 0`
*identically*, for any kernel. Expanding via the chain rule on
`g(a, b) = W(a - b, h)` (treating the two occurrences of `x_i` as temporarily
independent, `a` and `b`, then setting `a = b = x_i`) makes *why* explicit:
translation invariance forces `d g/db = -dg/da` and `d^2g/(da db) = -d^2g/da^2
= -d^2g/db^2`, so `f''(x) = g_aa + 2 g_ab + g_bb = H(0,h) - 2 H(0,h) + H(0,h)
= 0` -- an exact cancellation for *any* finite `H(0,h)`, independent of its
value. Dropping the self term when assembling `d^2 C_i / dx_i^2` is therefore
exactly correct, not an approximation -- it is not there in the first place,
because the self term never truly varies with the single shared variable
`x_i`, only with the two independent arguments `sphKernelHessian(x_i, x_j,
...)` is evaluated against before they happen to coincide. (The same
identity is *also* why `d(grad_i C)/dx_k = -m_k H_ik` for `k != i` carries a
minus sign relative to the diagonal block, per
`wp_implicitShifting.py`'s docstring -- both are the same translation-
invariance fact, not two unrelated pitfalls.)

**This module's own earlier docstring, and `wp_implicitShifting.py`'s
(warpSPH), both previously claimed instead that `sphKernelHessian`'s value at
`r=0` is itself "numerically unstable" or "an arbitrary value" there, and
that *that* instability was why self-pairs get dropped. That claim is false**
-- `sphKernelHessian`'s near-origin regularization branch returns a
well-defined, finite, physically meaningful value at `r=0` (the kernel's own
curvature at its peak; confirmed directly, `wp_kernels.ipynb` in this repo's
root evaluates it at the exact midpoint of a 1023-particle line and finds a
smooth, continuous `-15.0`, matching its `-14.88` neighbors either side to
three digits -- not noise). The *value* was never the problem; the identity
above is why it still has to be excluded from this particular sum regardless
of how well-behaved it is. Verified two ways this doesn't leave a live gap:
(1) for the physically meaningful case `tangentQueryPositions ==
tangentReferencePositions` (the same particle moving in both roles, what
Phase 4 step 4's shifting matvec actually calls this with), the self term's
own `(tangentQuery_i - tangentReference_i) = 0` factor makes it contribute
exactly zero to the *pairwise-product* sum too, so keeping vs. dropping it
is a bitwise no-op there (checked directly, not just argued); (2) for the
deliberately asymmetric case Phase 4 step 4's `implicitShiftingAutomatic.py`
uses to isolate `implicitShifting._buildSystem`'s own `diagBlock`
(`tangentQueryPositions = e_d`, `tangentReferencePositions = 0`), dropping
is required to match that hand-built reference's own (correct) `diagBlock`
-- keeping the self term there produces a different, wrong answer. Both
checks live in `tests/operations/test_forward_mode_tier2_density_hvp_self_pair.py`.

**A third, independent confirmation, from the reverse-mode side.** A single
particle's own self-density contribution, differentiated by `warpOperation`'s
*production*, gradcheck-validated reverse-mode path (which routes the needed
`d(vectorNorm_warp)/dx` through `math/wp_normalize.py`'s manually-written,
eps-guarded `@wp.func_grad` adjoints, not a naive automatic one -- the
gradient of a normalized direction is exactly the kind of `x/|x|` expression
that would otherwise divide by zero at `r=0`), gives an exact `0.0` reverse
gradient at any position (matching the notebook's own "Kernel Gradients:
... -0." finding), and finite-differencing *that* gradient across a small
step gives exactly `0.0` for the self-Hessian too -- not `sphKernelHessian`'s
`-15.0`. This matches the chain-rule identity above (`0.0`, not `-15.0`, is
the true total second derivative) via a completely independent numerical
route: single reverse-mode backward passes at two nearby points, finite
differenced, never a genuine double-backward through the same graph.
**This also sharpens why composing an HVP via double-backward would be
dangerous, not just unavailable** (see below): `vectorNorm_warp`'s adjoint
itself calls `vectorNormalize_warp` (the guarded `x/|x|`), which has its
*own* manual adjoint built from `norm_hess_warp` specifically so that
differentiating a normalized direction a second time stays finite at `r=0`
instead of hitting the same `0/0` a naive automatic second derivative would.
Nothing in this module relies on that machinery -- `sphKernelHessian` is a
closed-form analytic formula, not a differentiated adjoint -- but it is the
concrete mechanism a from-scratch double-backward attempt would have to get
right, and silently corrupting it into NaN is an easy way to fail quietly.

Restricted to the position-only case (support/mass tangents frozen, i.e.
`dh=0` on both differentiation orders) to match `sphKernelHessian`'s own
scope, so this is directly comparable term-for-term to
`warpSPH/modules/shifting/wp_implicitShifting.py`'s hand-built `H` output --
the comparison baseline this exists to validate against
(`warpSPH/tests/test_implicitShiftingHessianJVP.py`).

**Composing this via generic torch machinery does not work, tried first.**
`torch.func.jvp` applied twice, or `torch.autograd.forward_ad.make_dual`/
`dual_level` nested, over `computeSPHDensityGeometryJVP` cannot propagate a
second tangent through a `wp.launch`-backed function in this codebase:
`computeSPHDensityGeometryJVP` is not wrapped in a `torch.autograd.Function`
at all (unlike `warpOperation`'s `StateAwareWarpFunction`, which has no
`jvp()` registered either -- see `scripts/spike_forward_mode_tier1.py`'s own
finding for the *first*-order case: `torch.autograd.forward_ad` silently
returns `tangent=None` there already). Empirically, `torch.func.jvp` errors
immediately (`RuntimeError: Cannot access data pointer of Tensor that
doesn't have storage` -- functorch's dual tensors have no real storage for
`wp.from_torch` to view), and `torch.autograd.forward_ad.make_dual` runs but
silently drops the tangent, same failure mode as the first-order case one
level down. This module is the "small explicit second-order helper" the plan
flagged as the fallback if composition didn't work -- it didn't.
"""

from typing import Any, Optional
import torch
import warp as wp
from warp.types import vector, matrix

from ..type_config import *
from ..dataTypes import *
from ..enumTypes import *
from ..util import castTorchToWarp, allocateTorchWarp
from ..kernels.hessian import sphKernelHessian
from ..util.stateUtil import getParticle
from ._jvpCommon import buildParticleSoA as _buildParticleSoA, buildDomainState as _buildDomainState, buildKernelState as _buildKernelState

__all__ = ['computeSPHDensityPositionHVP']


@wp.kernel
def _computeSPHDensityHVP_PairKernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,
    kernelProperties: kernelState,
    edgeI: wp.array(dtype=wp.int64),
    edgeJ: wp.array(dtype=wp.int64),
    outH: wp.array(dtype=Any),
):
    e = wp.tid()
    if e >= edgeI.shape[0]:
        return
    i = wp.int32(edgeI[e])
    j = wp.int32(edgeJ[e])

    xi, hi, _mi, _rhoi, _ki = getParticle(queryState, i)
    xj, hj, _mj, _rhoj, _kj = getParticle(referenceState, j)

    outH[e] = sphKernelHessian(xi, xj, hi, hj, kernelProperties, domainState)


def computeSPHDensityPositionHVP(
    queryParticles: ParticleState,
    domain: DomainDescription,
    kernel: KernelFunctions,
    supportMode: SupportScheme,
    adjacency: AdjacencyList,
    tangentQueryPositions: torch.Tensor,
    referenceParticles: Optional[ParticleState] = None,
    tangentReferencePositions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """`Hess(Density)_i @ v`, shape `[numQuery, dim]` -- the Hessian-vector
    product a Newton-Krylov matvec needs (Phase 4 Steps 3-4), with `v` =
    `tangentQueryPositions`/`tangentReferencePositions` (the same
    perturbation field in both roles, when query and reference are the same
    particle population, matching implicit shifting's own usage).

    Mirrors `wp_implicitShifting._multiplyLaplacianBlock`'s math exactly
    (`sum_{j != i} m_j * H_ij @ (v_i - v_j)`) but is assembled from warpSPHCore's
    own Tier-2.0 `sphKernelHessian` building block through this package's
    adjacency/domain/particle-state conventions, rather than warpSPH's
    hand-rolled per-pair kernel.
    """
    referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
    dim = domain.dim
    device, dtype = queryParticles.positions.device, queryParticles.positions.dtype
    nQuery = queryParticles.positions.shape[0]
    nRef = referenceParticles.positions.shape[0]

    tangentReferencePositions = (
        tangentReferencePositions if tangentReferencePositions is not None
        else torch.zeros((nRef, dim), device=device, dtype=dtype)
    )

    queryState = _buildParticleSoA(dim, queryParticles.positions, queryParticles.supports, queryParticles.masses)
    referenceState = _buildParticleSoA(dim, referenceParticles.positions, referenceParticles.supports, referenceParticles.masses)
    domainState = _buildDomainState(domain)
    kernelProperties = _buildKernelState(kernel, supportMode)

    edgeI = castTorchToWarp(adjacency.i)
    edgeJ = castTorchToWarp(adjacency.j)
    numPairs = adjacency.i.shape[0]

    H_t, H_w = allocateTorchWarp(numPairs, matrix(shape=(dim, dim), dtype=scalar_t), edgeI.device)

    wp.launch(
        _computeSPHDensityHVP_PairKernel,
        dim=numPairs,
        inputs=[queryState, referenceState, domainState, kernelProperties, edgeI, edgeJ, H_w],
        device=edgeI.device,
    )

    ii, jj = adjacency.i.long(), adjacency.j.long()
    # Self-pairs (i == j) are excluded exactly, not as a numerical-safety
    # measure -- see this module's docstring for the translation-invariance
    # identity (H(0,h) - 2*H(0,h) + H(0,h) = 0) showing the self term's true
    # contribution to d^2 C_i/dx_i^2 is zero for any finite H(0,h), including
    # sphKernelHessian's own well-defined value there.
    pairMask = ii != jj
    ii, jj, H_t = ii[pairMask], jj[pairMask], H_t[pairMask]

    massJ = referenceParticles.masses[jj]
    relTangent = tangentQueryPositions[ii] - tangentReferencePositions[jj]
    pairHVP = massJ[:, None] * torch.einsum('nab,nb->na', H_t, relTangent)

    HVP = torch.zeros((nQuery, dim), device=device, dtype=dtype)
    HVP.index_add_(0, ii, pairHVP)
    return HVP
