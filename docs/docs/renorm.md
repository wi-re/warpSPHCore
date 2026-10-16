# Renormalization

The [De Courcy](https://doi.org/10.1016/j.jcp.2024.113192)
gradient-renormalization matrix $\mathbf{L}_i$ that the
[Gradient](operations/gradient) operator (and the
[Divergence](operations/divergence)/[Curl](operations/curl)/
[Laplacian](operations/laplacian) operators in their
CRK-corrected configurations) apply to the *uncorrected* gradient
result:

$$
\mathbf{G}_i \;\longleftarrow\; \mathbf{L}_i \, \mathbf{G}_i,
\qquad
\mathbf{L}_i = \mathrm{pinv}\!\left(\mathbf{C}_i\right),
$$

where $\mathbf{C}_i$ is the [Covariance](operations/covariance) of the
kernel gradient field at $i$. $\mathbf{C}_i$ measures the linear
reproduction error of the neighbor set; inverting it (pseudo-inverse,
when a stencil has unresolved directions) makes the
corrected Difference gradient reproduce linear fields on fully resolved
stencils. A free surface alone does not imply rank loss; a one-sided stencil
can still span every direction. Low-neighbor and truncated directions follow
the fallback policies below. The order of
application is fixed: CRK correction first, then
$\mathbf{L}_i\mathbf{G}_i$ (see the
[Gradient](operations/gradient) page).

Module: `src/warpSPHCore/renorm.py`.

## `computeRenormalizationMatrices`

```python
computeRenormalizationMatrices(
    queryParticles, operationProperties, domain,
    queryVolumes=None, referenceVolumes=None,
    adjacency=None,                 # AdjacencyList | CompactHashMap | None
    referenceParticles=None,
    crkState=None, gradHState=None, renormalizationState=None,
    returnEigVals=True,
)  -> (C, eigVals, RenormalizationState)      # returnEigVals=True
   -> RenormalizationState                    # returnEigVals=False
```

The pipeline (the `_`-suffixed internal function):

### 1. Covariance — on a *copy* of the properties

The operation is forced to `WarpOperation.Covariance` via
`dataclasses.replace(operationProperties, ...)` — never written back
into the caller's object. The old code did
`operationProperties.operation = WarpOperation.Covariance` in place; a
caller that reused its properties object afterwards would silently get
a covariance launch where it asked for a gradient, and the `(N, D, D)`
result is plausible enough to go unnoticed. The bug was found by a
Tier-1 spike that reused one properties object across the renorm call
and the gradient call that consumed its output; it is pinned by
`tests/operations/test_renorm_no_caller_mutation.py`.

The covariance call passes `covarianceReturnNumNeighbors=True` so the
low-neighbor fallback (below) works under **all three traversal
inputs** — explicit `AdjacencyList`, explicit `CompactHashMap`, or
`adjacency=None`: the count is a per-particle output of the
[Covariance kernel](operations/covariance), computed identically under
either traversal mode, rather than read from
`adjacency.numNeighbors`, which only exists for a list.

### 2. Low-neighbor-count fallback — identity, branch-free

A particle with too few neighbors to trust its covariance (free-surface
fingers, isolated particles) gets the **identity** instead of its own
untrustworthy matrix, so the pseudo-inverse does not amplify noise:

$$
\mathbf{C}_i \;\longleftarrow\;
\begin{cases} \mathbf{I} & n_i < d + 2 \\ \mathbf{C}_i & \text{otherwise} \end{cases}
\qquad
\text{implemented as } \mathrm{torch.where}(n_i < d+2,\; \mathbf{I},\; \mathbf{C}_i)
$$

This is deliberately a single elementwise `torch.where`, not
`if torch.any(mask): C[mask] = eye` — that guard was **two host
readbacks per step** in a real run (the `torch.any` in the Python
`if`, plus the mask count read by the boolean-masked assignment, which
is itself a synchronizing op): the fast path paid a stall to *maybe*
avoid a stall. `where` has neither, needs no clone, and gives the same
values and the same gradients (a low-neighbor row is replaced by a
constant either way, so nothing flows back through it). See
`docs/regression/real_workload_bottleneck_audit.md` (Step H sync
census) and the pinning test
`tests/operations/test_no_host_sync.py::test_renormalization_does_not_read_back`.

The 2D pseudo-inverse re-checks the same cutoff internally (see below);
applying it here as well covers 3D.

### 3. Pseudo-inverse — `pinv_warp`, dispatched on dimension

`pinv/wrapper.py::pinv_warp(C, numNbrs)`:

| dim | backend |
|---|---|
| 1 | `pinv1x1` (trivial reciprocal) |
| 2 | `pinv2x2_warpBackend` — a Warp kernel with the **closed-form symmetric eigendecomposition** (the general 2×2-SVD formula this used to use is unstable for near-isotropic covariances). Eigenvalue cutoff: an eigenvalue is inverted only if it is finite in absolute value *and* above `rcond = 1e-6` times the dominant one, so anisotropic/thin neighborhoods don't produce huge inverted eigenvalues. Re-checks `num_nbrs < 4` (i.e. `< dim + 2`) → identity. |
| 3 | Plain radial covariance uses Torch's symmetric spectral inverse at `rtol=1e-6`, with a selected-branch derivative shared by reverse AD and explicit JVP. Covariance with CRK/Grad-H/renorm corrections retains the general `torch.linalg.pinv` path. `eigVals` retain the existing diagnostics from `torch.linalg.eigvals`. |

**Outputs:** `C` `[N,D,D]` (the covariance, *after* the identity
fallback — the matrix the pseudo-inverse actually inverted), `eigVals`
`[N, D]`, and `RenormalizationState(renormalizationMatrices=L)`
`[N,D,D]`.

`computeRenormalizationMatrices` accepts `crkState`/`gradHState`/
`renormalizationState` and forwards them to its internal covariance
call — the renormalization matrix is then a correction *on top of* the
other corrections, which is the configuration the gradient JVP
scripts exercise.

On a fixed retained set, reverse and forward derivatives include rotation
between a retained eigenvalue `a` and a discarded eigenvalue `b`, with coefficient
`1/(a*(a-b))`. The usual Moore–Penrose backward at exact rank loss omits the
nonzero discarded `b` in that coefficient. The 3D plain-covariance path now
uses the selected spectral rule, including repeated full-rank eigenvalues,
reverse-over-JVP and second reverse derivatives. Thin 3D line/plane stencils
are qualified on CUDA. Cutoff/rank crossings remain nonsmooth; discarded
directions do not acquire a reproducing gradient.

## JVP

`computeRenormalizationMatricesJVP(queryParticles, operationProperties,
domain, queryTangentState, queryVolumes=None, referenceVolumes=None,
tangentReferenceVolumes=None, adjacency=None, referenceParticles=None,
referenceTangentState=None, returnEigVals=True)` →
`(C, eigVals, RenormalizationState, RenormalizationTangentState)` (or
the two states only if `returnEigVals=False`).

Given only *geometry* tangents (no pre-existing state), it returns
$d\mathbf{L}_i$ by the same two steps the primal applies to
$\mathbf{C}_i$:

1. **`dC`** from [`computeCovarianceGeometryJVP`](operations/covariance#jvp--hvp)
   (the raw, unmasked covariance tangent).
2. **The low-neighbor mask's exact-zero tangent**: the fallback
   replaces low-neighbor rows by the *constant* identity, so their
   tangent is exactly zero —
   `dC = torch.where(lowNbrMask, 0, dC_raw)`.
3. **The selected symmetric spectral-inverse derivative**. At full rank:

$$
d\mathbf{L}_i = -\,\mathbf{L}_i\, d\mathbf{C}_i\, \mathbf{L}_i
$$

For a truncated direction this also includes retained/discarded subspace
motion: between retained eigenvalue $a$ and discarded eigenvalue $b$, the
coefficient multiplying the corresponding entry of $d\mathbf C$ is
$1/(a(a-b))$. This includes small **nonzero** discarded eigenvalues; treating
them as exact zeros gives a different derivative. The implementation uses
projected small solves, avoiding eigenvector derivatives at repeated retained
eigenvalues. The retained set must stay fixed: crossing the cutoff is
nonsmooth. The primal cutoff and eigenvalue diagnostics are unchanged.

$\mathbf{L}_i$ is consumed from this function's own primal call
(same "consume an already-validated value" pattern the CRK JVP uses).
The neighbor count is recomputed via a second internal covariance call
(the primal function does not return it).

Both covariance calls and the covariance tangent honor the same particle
kinds and `operationProperties.operationMode`. A query row rejected by that
mode has the existing low-neighbor identity correction and zero tangent.

`crkState`/`gradHState` are **not** accepted here — the covariance JVP
kernel is deliberately CRK/grad-h-free, matching the "renorm alone
first" scoping. A downstream operator can consume independently constructed
CRK and renorm states/tangents together; this does not require constructing
the covariance with a CRK-corrected gradient. Full-rank evidence is in
`scripts/spikes/spike_forward_mode_tier2_renorm.py`; truncated-subspace and
composition regressions are listed below. All three traversal inputs,
including omitted adjacency, are accepted.

The lattice-normalization calibration ($1/L$ kernel scaling for a
uniform lattice's quadrature offset) is a separate, unrelated feature
that used to share this page — see
[Lattice normalization calibration](lattice-calibration).

## Tests

- `tests/operations/test_no_host_sync.py` — the branch-free low-neighbor
  fallback (readback counter on `renorm.py`) and
  `test_low_neighbour_fallback_still_replaces_those_rows` (identity,
  not zero, for `n_i < dim + 2`).
- `tests/operations/test_renorm_no_caller_mutation.py` — the
  properties-copy behavior.
- `scripts/gradcheck/gradcheck_renorm_native.py`,
  `scripts/gradcheck/gradcheck_renorm_uniform_grid_native.py` — forward-value
  parity + `torch.autograd.gradcheck` across all three traversal
  inputs (gated by `tests/operations/test_gradcheck_scripts.py`).
- `scripts/spikes/spike_forward_mode_tier2_renorm.py` — the JVP, float64.
- `tests/operations/test_renorm_rank_jvp.py` — truncated-subspace rotation,
  selected spectral finite differences, repeated retained eigenvalues and
  low-neighbor zero tangents.
- `tests/operations/test_capability_composition.py` — recomputed density,
  apparent-volume/CRK/renorm tangents through Gradient, and corrected SPH
  Dirichlet solves.

## See also

[Covariance](operations/covariance) · [Gradient — where $\mathbf{L}$
is applied](operations/gradient) ·
[Lattice normalization calibration](lattice-calibration)
