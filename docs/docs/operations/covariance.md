# Covariance

The per-particle kernel-weighted covariance matrix, the raw ingredient
of gradient renormalization. Kernel:
`coreOperations/wp_covariance.py` (`computeCovariance_Kernel`); JVP
`coreOperations/wp_covarianceJVP.py`.

## Equation

$$
C_i \;=\; \sum_{j \in \mathcal{N}(i)} V_j\,
          (x_j - x_i)\otimes \nabla W_{ij},
\qquad
n_i = \#\{\, j : W_{ij} > 0 \,\},
$$

with the same apparent volume $V_j = m_j/\rho_j$ (or
`referenceVolumes[j]`) and the same CRK-then-renormalized kernel
gradient as the [Gradient](gradient) family. The sign is
$(x_j - x_i)$, matching De Courcy et al. (2024) Eq. (34),

$$
L_i = \Big[-\textstyle\sum_j (x_i - x_j)\otimes \nabla_i W_{ij}\, V_j\Big]^{-1}
    = \Big[\textstyle\sum_j (x_j - x_i)\otimes \nabla_i W_{ij}\, V_j\Big]^{-1}.
$$

(The backend docstring records that an older comment had the opposite
sign; the code was always the De Courcy form — verified in
`DELTASPH_VALIDATION_PLAN.md` Part 8.13.)

**Output:** $C_i$ as a matrix-typed tensor `[N, D, D]` (stored as a
true matrix array, not a flattened vector — the covariance is always
$D \times D$ for $D \le 3$), optionally plus the integer neighbor count
`[N]` when `covarianceReturnNumNeighbors=True` (this is how
[Renormalization](../renorm) gets its low-neighbor-count mask under either
traversal mode).

## Public API

```python
C = sph.warpOperation(
    queryParticles, operationProperties, domain,
    adjacency=None,
    referenceParticles=None,
    queryVolumes=None, referenceVolumes=None,
    crkState=None, gradHState=None, renormalizationState=None,
    covarianceReturnNumNeighbors=False,
)
```

with `operation == WarpOperation.Covariance`. Covariance is special
cased in the dispatcher before the value-argument validation: it takes
**no field values at all** (only volumes).

| Option | Default | Effect |
|---|---|---|
| `supportMode` | `Gather$^1$` | pairwise support of $\nabla W_{ij}$ |
| `operationMode` | `AllToAll$^1$` | directional filtering |
| `referenceVolumes` / `crkState` / `renormalizationState` | `None` | as in Gradient (the kernel reads all of them) |
| `covarianceReturnNumNeighbors` | `False` | return `(C, numNeighbors)` instead of `C` |

$^1$ whatever the caller's `OperationProperties` carries — the
dispatcher forwards the object untouched (except `operation`).

## JVP / HVP

**Value JVP: none** — Covariance has no value input (a tangent standing
in for `queryValues` would be silently ignored, so
`warpOperationJVP` excludes it from the value-JVP set rather than let
that happen).

**Geometry JVP** — `warpOperationJVP(..., operation=Covariance,
queryTangentState=...)` computes $dC_i$ via
`computeCovarianceGeometryJVP`. That kernel is **deliberately
plain/uncorrected**: no CRK, no renormalization dispatch, no query-mass
term, no value input — matching its one internal consumer,
`computeRenormalizationMatricesJVP` (which therefore also rejects
`crkState`/`renormalizationState`). `warpOperationJVP` enforces the
same boundaries for the public path: any of `crkState`/`crkTangentState`/
`renormalizationState`/`renormalizationTangentState`/`gradHState`/value
tangents/`queryVolumes` raises `NotImplementedError`;
`referenceVolumes`/`tangentReferenceVolumes` are supported.

This is the main public consumer of the covariance JVP:
[Renormalization](../renorm#jvp) chains $dC_i$ through
$dL_i = -L_i\, dC_i\, L_i$.

Gated by `tests/operations/test_forward_mode_geometry_jvp_covariance.py`.

No HVP exists for Covariance.

## Tests

- `tests/operations/test_operations_core.py` — shape/finiteness of the
  matrix output and the neighbor count.
- `tests/operations/test_renorm_no_caller_mutation.py` — the renorm
  pipeline (built on this operator) never mutates the caller's
  `OperationProperties`.
- `tests/operations/test_forward_mode_geometry_jvp_covariance.py` —
  $dC_i$ vs reverse-mode Jacobian.
- `tests/operations/test_grid_modes.py` — adjacency vs grid parity,
  including `covarianceReturnNumNeighbors` under both.

## See also

[Renormalization](../renorm) · [Gradient](gradient)
