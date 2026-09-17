# warpSPHCore — operator wiki

`warpSPHCore` is the operator layer of the `warpSPH` family: a set of GPU
kernels, written in [NVIDIA Warp](https://github.com/NVlabs/warp) over
PyTorch tensors, that compute the pairwise sums at the heart of SPH —
density, gradient, divergence, curl, Laplacian, interpolation, and the
covariance matrix behind gradient renormalization — plus the CRK
(moment-based) correction machinery that lifts them to
linear-reproducing consistency.

The defining properties, all of which this wiki documents per operator:

- **Structured state, one dispatch.** Every operator is reached through one
  entry point, `warpOperation`, which takes semantic state objects
  (`ParticleState`, `OperationProperties`, `DomainDescription`, and
  optional correction states) and dispatches straight to the operator's
  backend. `sphOperation_warp` is the flat-tensor equivalent for callers
  without state objects on hand.
- **Two traversal modes, one kernel.** Each operator is a single Warp
  kernel pair that drives either an explicit neighbor list
  (`AdjacencyList`) or direct grid traversal over a `CompactHashMap`
  (Z-order-sorted cells + hash table). `adjacency=None` builds the hash map
  on the fly; see [Neighbor search](/docs/neighbor-search).
- **Differentiable end to end.** Every launch goes through a
  `torch.autograd.Function` bridge ([Autodiff machinery](/docs/autograd)):
  reverse-mode gradients come from Warp's tape; forward-mode JVPs
  (`warpOperationJVP`) are hand-derived per operator for value and
  geometry (position/support/mass/density) tangents, with the same
  scope restrictions enforced in one place. A density Hessian-vector
  product (`warpOperationHVP`) is the only second-order entry point.
- **Corrections are data, not code paths.** CRK correction
  (`CRKState`), gradient renormalization (`RenormalizationState`),
  apparent-volume substitution, and grad-h terms are passed as state
  objects; a kernel is "corrected" exactly when a state is supplied.
  The lattice-normalization calibration (the current branch's work) is
  the newest member of this family — see
  [Renormalization & lattice calibration](/docs/renorm).

## The operators

| Operator | Computes | Page |
|---|---|---|
| Density | $\rho_i = \sum_j m_j W_{ij}$ | [operations/density](/docs/operations/density) |
| Gradient | $\nabla f_i = \sum_j \mathrm{coeff}_{ij}\,\nabla W_{ij}$ (four schemes) | [operations/gradient](/docs/operations/gradient) |
| Divergence | $\nabla\!\cdot\!\mathbf{f}_i$ (incl. consistent / dot variants) | [operations/divergence](/docs/operations/divergence) |
| Curl | $\nabla\times\mathbf{f}_i$ (2D scalar, 3D vector) | [operations/curl](/docs/operations/curl) |
| Laplacian | $\nabla^2 f_i$ (four schemes: Naive, Brookshaw, Dot, Default) | [operations/laplacian](/docs/operations/laplacian) |
| Interpolate | $\tilde f_i = \sum_j f_j V_j W_{ij}$ | [operations/interpolate](/docs/operations/interpolate) |
| Covariance | $C_i = \sum_j V_j (x_j - x_i)\otimes \nabla W_{ij}$ | [operations/covariance](/docs/operations/covariance) |

## The correction machinery

| Page | Contents |
|---|---|
| [CRK overview](/docs/crk/crk-overview) | What CRK is, the corrected kernel/gradient, `computeCRKFactors` pipeline and its JVP |
| [CRK volume](/docs/crk/crk-volume) | Apparent volume $V_i = 1/\sum_j W_{ij}$ |
| [CRK moments](/docs/crk/crk-moments) | $m_0, m_1, m_2$ and their $\gamma$-derivatives |
| [CRK density](/docs/crk/crk-density) | The corrected consistency density (diagnostic) |
| [Kernels](/docs/kernels) | The 12 compact-form kernels, the derivative ladder, support schemes, kernel JVPs |
| [Renormalization](/docs/renorm) | Covariance $\to$ pseudo-inverse renormalization matrix, its JVP, and the lattice-normalization calibration ($L(n_h)$) |

## Infrastructure

| Page | Contents |
|---|---|
| [Neighbor search](/docs/neighbor-search) | Compact-hash grid (Z-order sort, cell table, hash table, count/collect), Verlet lists, other backends |
| [Autodiff machinery](/docs/autograd) | The `torch.autograd.Function` bridge, `OperatorSpec`/`SPHContext`, JVP dispatch, no-host-sync design |
| [Data types](/docs/data-types) | `ParticleState`, `OperationProperties`, `DomainDescription`, adjacency structures, correction states, all enums |
| [API reference](/docs/api) | Every public function in one table |

## How to read the operator pages

Each operator page is written against the code as it is on disk, and
follows a fixed structure:

1. **The equation** — the per-pair accumulation exactly as implemented in
   the Warp kernel, with the scheme-specific variants.
2. **Public API** — how to reach it through `warpOperation`
   (`WarpOperation.<Name>`) and the option table for
   `OperationProperties` / call arguments that apply to it.
3. **Corrections** — which of CRK / renormalization / apparent volume /
   grad-h the kernel reads, in which order.
4. **JVP / HVP** — the value- and geometry-tangent formulas, and which
   combinations `warpOperationJVP` accepts for this operator.
5. **Tests** — the files under `tests/operations/` that pin each claim,
   so a reader can verify a statement by running it.

Design decisions, derivations, and the history of each piece live in the
repo-root records — [`warpier_core.md`](https://github.com/wi-re/warpSPHCore/blob/main/warpier_core.md)
(structured-kernel ABI and call graph),
[`warpier_adjoint.md`](https://github.com/wi-re/warpSPHCore/blob/main/warpier_adjoint.md)
(JVP derivations, tier by tier),
[`warpier_fields.md`](https://github.com/wi-re/warpSPHCore/blob/main/warpier_fields.md)
(state objects, caching, the declared ABI), and
[`higher_order.md`](https://github.com/wi-re/warpSPHCore/blob/main/higher_order.md)
(higher-order convergence study plan). Warp-kernel authoring gotchas that
bit the implementations are logged in
[`docs/lessons_learned.md`](https://github.com/wi-re/warpSPHCore/blob/main/docs/lessons_learned.md);
performance measurement records are under `docs/regression/`.

## Requirements

- Python 3.10+, PyTorch, [warp](https://github.com/NVlabs/warp)
- Precision (`float32`/`float64`) is fixed at first import via the
  `warpSPHCore_PRECISION` environment variable (default `float32`) and
  cannot change mid-process — which is why the gradcheck test scripts
  run in subprocesses at float64. See [Quickstart](/docs/quickstart).
