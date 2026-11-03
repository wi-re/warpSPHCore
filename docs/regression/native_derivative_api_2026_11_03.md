# Native core closure: derivative and public API parity

Stage five gives all seven uncorrected operations explicit combined JVPs and
ordinary Torch forward AD. Existing optimized primals and uncorrected reverse
kernels remain in place. Analytic forward products share the corrected pair
interface; density adds its mass-weighted kernel case. No finite differences or
per-edge Jacobian tape enter execution.

The public `warpOperationJVP` has independent query/reference roles and supports
fields, particle geometry/weights, explicit source volumes and CRK/renormalization
coefficient products. Explicit CRK volume, moments and factor helpers compose the
same native products with the existing Torch factor algebra. First-order limits
and fixed branch/rank contracts remain explicit; generic higher derivatives,
`torch.func` transforms and singular/cusp geometry AD are not added.

The density position block preserves the existing shifting helper's exact scope:
`sum(j != i) m_j H_ij (v_i-v_j)`. Equal global IDs are excluded, including separate
populations. Its regularized Hessian/origin convention and the legacy maximum
support for KernelMeanSymmetric are preserved; SuperSymmetric averages Hessians.
This is not a general density double backward. Geometry/support/domain stay fixed.
First-order vector/source-mass JVP/VJP uses owned CSR/CSC reductions without
floating atomics or per-edge Hessians. Eager row chunks carry the global row base.

`operations.py` selects the backend without importing Warp in a native process;
the old implementation lives in `wp_operations.py`. The flat adapter is shared.
Geometry-product compatibility names, root operator metadata, sampling and native
meshless raw passes are exported. Warp inline functions/containers remain
backend-specific, backed by native C++ headers rather than fake Python Warp types.
PartialSymmetric follows upstream's current maximum-support implementation.

| Qualification | Result |
|---|---|
| Four bounded native product contracts | CPU and CUDA pass |
| Combined field/geometry JVP | Finite differences, implicit dual propagation and reverse-product identity pass for all seven operations |
| Explicit CRK chain and corrected gradient | Factor/volume and composed output finite differences pass; helpers work inside an existing empty dual level |
| Density position block | Gradient finite differences, vector/mass gradcheck, JVP and self-ID checks pass in 1D/2D/3D |
| Public products against Warp | CPU and CUDA pass; all six support policies covered for the legacy position block |
| Existing affected Warp public derivative regressions | 20 passed on CPU |
| Native CPU suite | 3,625 passed, 470 existing skips; extended and other-device cases deselected |
| New/affected experimental HIP translation units | Differential, density position block and compact-index GPU code compile for gfx942 |

The CUDA dependent CRK/renormalization selection also passes 599 checks before the
final bounded CRK addition. No new timing or AMD runtime claim is made. Maintained
additions are four native contracts and one bounded Warp-parity contract.

Evidence, including failed/superseded probes and logs, is kept outside core at
`/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-core-derivative-api-20261103`.
