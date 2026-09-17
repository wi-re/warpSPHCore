# Data types

The objects operators accept and return. Everything lives in
`src/warpSPHCore/dataTypes/` (the enums in
`src/warpSPHCore/enumTypes.py`).

Precision and dimension are global, set once at import by the
`warpSPHCore_PRECISION` / `warpSPHCore_DIM` environment variables (see
[Quickstart](quickstart#install)); `scalar_t` is the
resolved Warp dtype, and every dimension-dependent struct exists once
per dimension as `…_1`/`…_2`/`…_3` variants.

## Particle states

### `ParticleState`

```python
@dataclass
class ParticleState:
    positions: torch.Tensor        # [N, D]
    supports:  torch.Tensor        # [N]  per-particle smoothing length h_i
    masses:    torch.Tensor        # [N]
    kinds:     torch.Tensor        # [N]  int32, REQUIRED
    densities: Optional[torch.Tensor] = None   # [N]
```

- **`kinds` is required, not optional.** Every operator kernel indexes
  `kinds[i]` unconditionally (`getParticleData`);
  `OperationDirection.AllToAll` used to paper over a missing `kinds`
  tensor with an N-sized zero dummy — that path is gone. Pinned by
  `tests/operations/test_particle_state_kinds.py`.
- Positions may be **unwrapped** in a periodic domain; the kernels
  apply the minimum-image distance from the `DomainDescription`
  themselves.
- `densities` is optional so the [Density](operations/density) operator
  can run without pre-computed densities; a `None` slot is filled with
  a shared null field in the [flat layout](autograd#the-36-slot-flat-layout).

### `ParticleTangentState`

```python
@dataclass
class ParticleTangentState:
    positions: torch.Tensor        # [N, D]
    supports:  torch.Tensor        # [N]
    masses:    torch.Tensor        # [N]
    densities: Optional[torch.Tensor] = None
```

The JVP tangent counterpart, **minus `kinds`** (categorical — no
tangent). Supplied to `warpOperationJVP` as
`queryTangentState`/`referenceTangentState`.

(There is also a legacy `PointCloud(positions, supports)` for callers
that only carry geometry.)

## Domain

```python
@dataclass(slots=True)
class DomainDescription:
    min: torch.Tensor      # [D]
    max: torch.Tensor      # [D]
    periodic: torch.Tensor # [D] bool
    dim: int
```

The Warp-side mirror is `domainData` (`domainMin`, `domainMax`,
`periodicity: wp.array(wp.bool)`, `dim: wp.int32`).

## `OperationProperties`

A **frozen** dataclass — the operator configuration, hashable and safe
to share across calls:

| field | default | meaning |
|---|---|---|
| `kernel: KernelFunctions` | — (required) | which kernel |
| `operation: WarpOperation` | `Interpolate` | which operator |
| `gradientMode: GradientScheme` | `Naive` | the [Gradient](operations/gradient) scheme (and the Laplacian's $q_{ij}$) |
| `laplacianMode: LaplacianScheme` | `Brookshaw` | the [Laplacian](operations/laplacian) scheme |
| `positiveDivergence: bool` | `False` | the Laplacian's $+\max(\mathbf{G}\cdot\mathbf{n}, 0)$ term |
| `supportMode: SupportScheme` | `Gather` | the [pairwise support scheme](kernels#pairwise-evaluation-and-the-support-schemes) |
| `operationMode: OperationDirection` | `AllToAll` | which `kinds` participate (see [OperationDirection](#operationdirection)) |
| `divergenceDotMode: bool` | `False` | the [Divergence](operations/divergence) dot-mode contraction |
| `n_h: Optional[float]` | `None` | nominal support-to-spacing ratio, for the [lattice calibration](renorm#lattice-normalization-calibration) |
| `calibrateNormalization: bool` | `False` | enable the $1/L$ kernel scaling |

`__post_init__` raises unless `n_h` is positive whenever
`calibrateNormalization` is set — deliberately loud, so "calibration
requested, never computed" can never be silently confused with
"calibration off".

## Neighbor-search structures

### `AdjacencyList`

```python
@dataclass(slots=True)
class AdjacencyList:
    i: torch.Tensor            # int64, COO row (query) indices
    j: torch.Tensor            # int64, COO column (reference) indices
    numNeighbors: torch.Tensor # int32, per query particle
    edgeOffsets: torch.Tensor  # int32, CSR offsets
    numRows: int; numCols: int
    # Verlet-list bookkeeping (the positions the list was BUILT from,
    # not the current ones):
    queryPositions: torch.Tensor = None
    referencePositions: torch.Tensor = None
    querySupports: torch.Tensor = None
    referenceSupports: torch.Tensor = None
    hashMap: CompactHashMap = None
```

COO *and* CSR at once (neighbors are sorted by query particle, so it
cannot serve as CSC; `i` is redundant with `edgeOffsets` +
`numNeighbors` but kept to avoid reconstructing it). `int64` is
required: torch does not index with `int32` tensors. See [Neighbor
search](neighbor-search).

### `CompactHashMap`

```python
@dataclass
class CompactHashMap:
    sortedPositions: torch.Tensor   # reference particles in Z-order
    sortedSupports:  torch.Tensor
    sortIndex:       torch.Tensor   # original -> sorted permutation
    hashTable:       torch.Tensor   # [hashMapLength, 2] (start, count); empty = (-1, 0)
    sortedCellTable: torch.Tensor   # occupied cells sorted by hash: [cellIndex, start, count]
    qMin: torch.Tensor; qMax: torch.Tensor
    hCell: float               # cell size (>= global support)
    numCells: torch.Tensor
    mode_uint: int             # the support scheme, as its uint
    D: int
    searchRadius: int          # stencil radius in cells
    numOffsets: int
    cellOffsets: torch.Tensor  # [numOffsets, 3] stencil offsets
```

Built by [`buildCompactHashMap`](neighbor-search#the-compact-hash-pipeline-main-path);
usable both as the input to a neighbor *collection* (into an
`AdjacencyList`) and directly as an operator's `adjacency=` for
[grid traversal](neighbor-search).

## Correction states

| state | fields | produced by | consumed by |
|---|---|---|---|
| `CRKState` | `A [N]`, `B [N,D]`, `gradA [N,D]`, `gradB [N,D,D]` | [`computeCRKFactors`](crk/crk-overview) | Gradient/Divergence/Curl/Laplacian (Brookshaw/Dot/Default only) |
| `CRKTangentState` | same four, **no `Optional` fields** | [`computeCRKFactorsJVP`](crk/crk-overview) | `warpOperationJVP` (the JVP bridge substitutes primal-shaped zeros for any of the four that has no live tangent of its own) |
| `GradHState` | `queryOmegas [N]`, `referenceOmegas [N]` (optional — defaults to the query's) | caller | the gradient-h terms of Gradient/Divergence/Curl/Laplacian; **no JVP wiring — `warpOperationJVP` rejects `gradHState` outright** |
| `RenormalizationState` | `renormalizationMatrices [N,D,D]` | [`computeRenormalizationMatrices`](renorm) | Gradient (and the other corrected ops) |
| `RenormalizationTangentState` | `renormalizationMatrices [N,D,D]` | [`computeRenormalizationMatricesJVP`](renorm) | `warpOperationJVP` |

The Warp-side mirror is `correctionData_{dim}` — a canonical,
unconditionally-present struct with a `useVolume`/`useGradHTerms`/
`useGradientRenormalization`/`useCRK` boolean per section (the booleans
are what guard against Warp's zero-initialization of the array
fields), plus a field-for-field `correctionTangentData_{dim}` tangent
mirror (deliberately a *separate* struct: adding tangent fields to the
primal struct would grow every non-JVP kernel's ABI for no reason).
Per-query-particle forms are `ParticleCorrectionData_{dim}` /
`ParticleCorrectionTangentData_{dim}`.

## Kernel configuration

```python
@wp.struct
class kernelState:
    kernelFunction: wp.int32          # KernelFunctions value
    supportMode: wp.uint32            # SupportScheme value (11..16)
    gradientMode: wp.int32
    laplacianMode: wp.int32
    positiveDivergenceMode: wp.bool
    divergenceMode: wp.bool           # the Divergence dot-mode
    operationMode: wp.int32           # OperationDirection value
    calibrateNormalization: wp.bool   # guards the coefficient below
    normalizationCoefficient: scalar_t  # 1 / L, see lattice calibration
```

The boolean flag exists so the pair is safe under Warp's
zero-initialization of struct fields: several call sites build
`kernelState()` and set only the fields they know about — a bare
coefficient would default to `0.0` and **silently zero the kernel**.
`False` is "off"; the coefficient is read only when the flag is set,
and the host raises on `flag set + coefficient ≤ 0`.

The other per-dimension Warp structs: `particleDataSoA_{dim}`
(positions/supports/masses/densities/kinds as `wp.array`s),
`WarpParticle_{dim}` (the SoA struct's AoS per-particle form),
`domainData`, `adjacencyData` (`neighborList` int64,
`neighborOffsets`/`numNeighbors` int32), `gridData` (the
`CompactHashMap`'s arrays + `D`/`numOffsets` scalars).

## The `Field` abstraction

`util/fieldRegistry.py` + `dataTypes/field_t.py`: a cached dual
torch/Warp view of a tensor's storage, the identity layer the
[caching machinery](autograd#caching-layers) is built on.

- **Two ownership modes.** *Attached* fields (caller-owned tensors)
  live on the tensor as `t._wsc_field` (via `acquireView`); the
  `Field` holds no strong reference back, so the tensor dies by
  refcount alone — no cycle for the GC to break on its own schedule.
  *Standalone* fields (core-owned: null fields, adjacency) hold the
  tensor themselves, since the point is that the `Field` outlives any
  particular call.
- **Provenance check.** `Field.matches(t)` re-validates
  `data_ptr`/shape/strides/dtype before a cached view is reused —
  catching `resize_`, `.data` reassignment, and storage swaps.
- **Copy/pickle safety.** a `Field` wraps a `wp.array` with a live
  device pointer; `__copy__`/`__deepcopy__`/`__reduce__` all degrade to
  `None`, so a copied or pickled tensor arrives with a missing field
  and `acquireView` treats it as a miss and rebuilds cleanly — no
  cross-process device pointers.
- `Role` has a single member (`PRIMAL`): forward-mode AD landed via
  `StateAwareWarpFunction.jvp()`'s flat-tensor tangents, not via a
  second field role. `FieldKind` (SCALAR/VECTOR/MATRIX/INT32/INT64/
  VEC2I/VEC3I/VEC3L/BOOL) keys the **null-field registry** — the
  permanent placeholders the [flat layout](autograd#the-36-slot-flat-layout)
  fills disabled correction slots with.

`ExecutionMode` (NONE / REVERSE / FORWARD / AUTO) is the execution
mode carried on an execution context and folded into the StateBundle
cache key: **FORWARD is declared and deliberately unimplemented**
(rejected at the boundary), and AUTO is a caller-facing default that
resolves to the same struct rows as NONE/REVERSE — it is a placeholder,
not a fourth variant.

## Enums

`enumTypes.py`. Integer values are part of the ABI (they cross into
`wp.int32` fields and the `mode_uint` of the hash map).

### `SupportScheme`

| member | value | effective support |
|---|---|---|
| `Gather` | 11 | $h_i$ |
| `Scatter` | 12 | $h_j$ |
| `MeanSymmetric` | 13 | $(h_i + h_j)/2$ |
| `KernelMeanSymmetric` | 14 | $\tfrac12(k(q, h_i) + k(q, h_j))$ |
| `SuperSymmetric` | 15 | $\tfrac12(k(q, h_i) - k(q, h_j))$ — the CRK-SPH formulation |
| `PartialSymmetric` | 16 | $f_i$-weighted $h_i$ + $f_j$-weighted $h_j$ — the PESPH formulation |

### `KernelFunctions`

| member | value | | member | value |
|---|---|---|---|---|
| `Wendland2` | 0 | | `Poly6` | 20 |
| `Wendland4` | 1 | | `Spiky` | 21 |
| `Wendland6` | 2 | | `CubicSpline` | 30 |
| `QuarticSpline` | 31 | | `QuinticSpline` | 32 |
| `B7` | 33 | | `ViscosityKernel` | 40 |
| `CohesionKernel` | 41 | | `AdhesionKernel` | 42 |

(the formulas are on the [Kernel library](kernels) page)

### `GradientScheme` / `LaplacianScheme` / `WarpOperation`

| `GradientScheme` | value | | `LaplacianScheme` | value | | `WarpOperation` | value |
|---|---|---|---|---|---|---|---|
| `Naive` | 1 | | `Naive` | 1 | | `Interpolate` | 1 |
| `Symmetric` | 2 | | `Brookshaw` | 2 | | `Gradient` | 2 |
| `Difference` | 3 | | `Dot` | 3 | | `Divergence` | 3 |
| `Summation` | 4 | | `Default` | 4 | | `Curl` | 4 |
| | | | | | | `Laplacian` | 5 |
| | | | | | | `Density` | 6 |
| | | | | | | `Covariance` | 7 |
| | | | | | | `Custom` | 8 |

### `OperationDirection`

| member | value | | member | value |
|---|---|---|---|---|
| `TrueAllToToAll` | 0 | | `AllToAll` | 9 |
| `FluidToFluid` | 1 | | `AllToGhost` | 10 |
| `FluidToBoundary` | 2 | | `AllToFluid` | 11 |
| `BoundaryToFluid` | 3 | | `AllToBoundary` | 12 |
| `BoundaryToBoundary` | 4 | | `FluidToAll` | 13 |
| `FluidToGhost` | 5 | | `BoundaryToAll` | 14 |
| `GhostToFluid` | 6 | | | |
| `BoundaryToGhost` | 7 | | | |
| `GhostToBoundary` | 8 | | | |

with `ParticleType`: `Fluid=0`, `Boundary=1`, `Ghost=2`, `Other=3` —
the direction selects which `kinds` act as query vs. reference.

### `HashMapLengthMode`

`Fixed` (0 — a fixed table length, default 4096), `NumberOfParticles`
(1 — made odd), `NextPrime` (2 — the default; the table is the next
prime above the particle count).

### `ViscosityTerms`

`Default=0`, `MonaghanGingold1983=1`, `Cleary1998=2`,
`Monaghan1992=3`, `Monaghan1997a=4`, `Monaghan1997b=5`,
`Dukowicz=6`, `Price2012_98=7`, `Price2012=8`, `Price2008=9`,
`Wadsley2008=10`, `DeltaSPH=11` — viscosity-term selectors for the
integrator layer; the core operators here do not branch on them.

## See also

[API reference](api) (every public function) · [Autodiff machinery](autograd)
(the flat layout these objects flatten into) · `warpier_fields.md` (the
state-object design record: ownership, caching, the declared ABI)
