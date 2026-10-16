# Polynomial fits and interface reconstruction

For a checked runnable cubic fit and TENO/WENO jump example, see
[Runnable examples](examples.md).

For an illustrated comparison with standard SPH, CRK and gradient
renormalization, start with [How the schemes reconstruct a field](field-reconstruction-guide.md).

`warpSPHCore.polyfit` provides production `PolyFit`, MLS-TENO and MLS-WENO
reconstruction. These have their own constructors and methods; they are not
`warpOperation` variants. Fits assemble moments and field right-hand sides
with Warp, then solve dense local systems with Torch.

```python
from warpSPHCore.polyfit import PolyFit, Reconstructor

fit = PolyFit.build(particles, domain, kernel, order=3)
value = fit.values(field)
gradient = fit.gradient(field)
hessian = fit.hessian(field)        # scalar field, order >= 2
laplacian = fit.laplacian(field)
third_x = fit.derivative(field, (3, 0))  # 2D example

rec = Reconstructor.teno(particles, domain, kernel, order="O4")
left, right = rec.interfaceStates(field, *rec.pairs())
fallback = rec.last_fallback
```

## Fit definition and shapes

The basis contains monomials of total degree through `order`, in normalized
coordinates `(x_j-x_i)/h_i`. Kernel-weighted moments use Gather support and
reference volumes `m_j/rho_j`, or caller-supplied `referenceVolumes`.
`unitWeights=True` selects ordinary least-squares weights inside the radius.
Supports and the volume/density inputs must be physically usable and positive.

`constant=True` fits values directly (MLS/RKPM). It supports distinct
`referenceParticles`: scalar fields have shape `(N_reference,)`, bundles have
shape `(N_reference, components)`, and coefficients/read-outs have leading
dimension `N_query`. Empty reference/query populations preserve these shapes.
An empty reference population gives zero coefficients and deficiency flags.
Empty reference grids need explicit domain bounds.

`constant=False` fits differences `f_j-f_i` without the constant basis (LABFM),
and returns the original particle field for `values`. It requires `order>=1`
and one particle set: this API has no separate query field argument for LABFM
differences. Passing a distinct reference state raises before kernel launch.

`gradient` returns `(N_query, dim)` for a scalar or
`(N_query, components, dim)` for a bundle. Hessian/Laplacian read-outs currently
accept scalar fields of order at least two; otherwise they return None.
`evaluate` evaluates a particle's polynomial at physical offsets, and
`derivative` reads any represented partial derivative through the fit order.
Central fits support 1D–3D; 3D SPH qualification runs on CUDA.

These are **diffuse local-fit derivatives**: derivatives of the polynomial
at a fixed fit center. They do not differentiate an entire position-dependent
corrected RKPM kernel. They also do not guarantee pairwise conservative fluxes.

## Rank and conditioning

`M`, `Minv`, `num_nbrs`, `cond` and `deficient` remain available on the fit.
`cond` describes the Jacobi-equilibrated matrix. Zero diagonals receive a
neutral scale so absent basis directions remain finite zero directions.
Deficient coefficients are minimum norm in the equilibrated coordinates;
they do not guarantee polynomial reproduction on an unresolved stencil.

With `rtol=None`, the relative spectral cutoff is
`max(1e-12, n_basis * eps(dtype))`. An explicit `rtol` retains the caller's
policy. A fit is deficient if it has too few contributing neighbors, drops a
basis direction at that cutoff, exceeds `cond_max` (default `1e10`), or has
nonfinite condition. Neighbor count alone cannot establish rank.

The precision-aware default replaces the former `1e-12` default in float32.
It avoids interpreting roundoff-sized eigenvalues as resolved directions.
`cond_max` is a separate acceptance policy; selecting a small rank tolerance
does not certify accuracy. High-order/corner fits and their AD can amplify
float32 error well before complete rank loss. Choose conditioning limits for
the desired accuracy, and use float64 for tighter requirements.

## Nonlinear reconstruction and fallback

TENO supplies `"O4"`, `"O5"`, `"O6"` configurations; WENO accepts a polynomial
`degree`. Candidate sectors are defined in 1D/2D. A 3D sector reconstruction
raises before building candidate fits; 3D sector design remains separate work.

Candidates have their own radii, which can exceed the particle supports.
Omitted adjacency builds a list covering the largest radius. A supplied
adjacency must already cover every candidate; an ordinary smaller-radius
particle list is insufficient. Pair interface locations still use the
particle supports: the offset from `i` is `d*h_i/(h_i+h_j)`.

Unusable candidates receive zero weight. When all candidates for a component
are unusable, the output is the corresponding particle's **first-order state**.
It no longer uses a deficient central polynomial. This preserves constants
and has exact field derivatives on that fallback branch. It does not invent
a higher-order stencil or guarantee coupled hydrodynamic positivity.

`last_beta` and `last_omega` describe the most recent candidate indicators and
weights; `last_fallback` identifies the first-order fallback per particle or
per bundle component. On fallback rows, the central unit weight is only a
diagnostic placeholder: the returned state comes directly from the particle.
TENO's `last_central` records its actual central-stencil selection.
Inactive softmax rows use finite substitute logits before normalization, so
their zero adjoints do not become NaNs.

## Differentiation and lifecycle

Field reverse AD and Torch dual-tensor forward AD are supported. Field-only
fit tangents use the same linear RHS as the primal, including distinct MLS
references, explicit volumes and frozen kind/direction rules. Nonlinear
reconstruction differentiates its component-wise weights while the selected
branches remain fixed. Rank, radius, sector and TENO selection switches are
not smooth derivative points.

PolyFit geometry duals cover positions, supports, masses and densities of
one physical particle population with plain `m/rho` volumes. Its tangent moves
both query and reference roles. Combined field/geometry tangents use dedicated
Warp JVP kernels with the same frozen kinds and `operationMode` as the primal.
Build and use a geometry-dual fit inside the same `forward_ad.dual_level`.
The dense coefficient tangent assumes a full-rank selected system;
rank-deficient geometry derivatives are not qualified.
Geometry-dual qualification here is 2D; 3D central-fit reproduction does not
establish 3D geometry AD.

Separate-reference geometry duals and explicit-volume geometry duals raise.
Field duals do not have these exclusions. End-to-end geometry AD through the
entire nonlinear Reconstructor is not independently qualified here. Generic
`torch.func` transforms, nested geometry JVPs, characteristic reconstruction
and a common `warpOperation` dispatch are separate capabilities.

A fit/reconstructor holds geometry-dependent matrices and references to caller
tensors. Rebuild it after changes to positions, supports, mass/density volumes,
kinds, interaction rules or neighborhood coverage. Changing only field values
can reuse it. A supplied grid/list follows the search rebuild/capacity contract;
particle IDs stay in caller order and pair extraction consumes live entries.
Eager CPU/CUDA execution is qualified; a completely captured constructor,
nonlinear reconstruction or evolution is not a promised graph interface.

The production-path tests are `tests/operations/test_polyfit.py`,
`test_reconstruction.py` and `test_reconstruction_contracts.py`.
`scripts/diagnostics/qualify_reconstruction.py` runs a periodic frozen-particle
cubic MLS/LABFM advection ladder with analytic solution, conditioning, spatial
residuals and weighted mass drift. Its short smooth run is separate from
nonlinear discontinuity evolution or a conservative MFM/MFV scheme.
