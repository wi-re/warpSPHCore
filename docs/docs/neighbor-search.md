# Neighbor search

How operators find $\mathcal{N}(i)$. Module:
`src/warpSPHCore/radiusSearch/`.

There are three ways an operator gets its neighbors:

1. **Explicit neighbor list** — an `AdjacencyList` (COO `i`/`j` pairs +
   CSR `edgeOffsets`/`numNeighbors`) built by a radius search, reused
   across many operator calls.
2. **Direct grid traversal** — pass a `CompactHashMap` (or `adjacency=None`,
   in which case one is built on the fly inside
   `extractStateInfo` with the SuperSymmetric support scheme). The
   operator kernel then iterates the cell stencil itself; this skips
   the neighbor-list build entirely, which is the right choice when the
   neighborhood is only accessed once per query set (the cost of
   building the list would dominate).
3. **Verlet list** — a neighbor list with a skin, updated
   incrementally between steps.

Both traversal modes (1 and 2) run the *same* operator kernel — every
operator is a dual-path kernel pair (`_Func_i` / `_Func_Adjacency` /
`_Kernel`) and `getIndexRange` abstracts the "where is particle $i$'s
neighbor block" question. Parity between the two is pinned by
`tests/operations/test_grid_modes.py`.

## The compact-hash pipeline (main path)

`radiusSearchCompactHashMap(queryParticles, domain, mode,
hashMapLengthMode=NextPrime, fixedHashMapLength=4096,
returnCompactHashMap=False, referenceParticles=None)` builds the hash
map and then collects neighbors into an `AdjacencyList`.
`buildCompactHashMap(...)` exposes the hash map alone;
`radiusSearchOnCompactHashMap(...)` the collection step alone.

Empty query, reference and both-empty populations are supported. The list
retains `numRows=N_query` and `numCols=N_reference`, with zero live edges,
zero row counts/offsets, and the selected integer dtypes. Exact allocation
(`capacity=None`) has zero pair storage; the default capacity policy retains
at least one padded slot filled with -1. Empty reference grids require explicit
domain `min` and `max`; bounds cannot be inferred from missing coordinates.
The grid remains valid with a positive cell size and no occupied reference cells.

Verlet rebuilds handle particle-count changes into and out of empty populations.
An unchanged empty pair domain needs no movement/support rebuild and reports
finite zero validity metrics. A count change invalidates the prior list.
In-place capacity reuse retains addresses only while the allocation fits;
captures must be rebuilt after a shape/address change. Fixed grid slots support
zero reference particles, including CUDA build replay at that fixed size.

All three entry points accept keyword-only `collection` and `queryOrder` options;
the same options are forwarded by `buildVerletList` and its raw-tensor variant.
`collection='auto'` (default) uses split collection on CUDA and fused on CPU.
Explicit `collection='split'` and `collection='fused'` override that choice.
Split collection writes `J` with the same traversal and ordering, then generates `I` from
CSR counts/offsets using 32 cooperating lanes per row. Both arrays retain their
usual format, padding and capacity/reuse behavior.

`queryOrder='auto'` (default) visits aliased self queries in the existing grid's
cell order. For sorted slot `s`, both count and collect write to original row
`sortIndex[s]`; fused collection writes that original I directly. Query positions
come from the already wrapped `sortedPositions`, and query supports are indexed
by original row. No extra sort, inverse permutation, particle-state reorder or
post-collection reindex pass is needed. J order within each original CSR row is
unchanged. Independent query/reference position layouts fall back to input
order, even when their populations are the same size. `queryOrder='input'`
forces input execution order; explicit `'spatial'` requires aliased positions.

The warpSPH frontend exposes these through `ComputeConfig.neighborCollection`
and `neighborQueryOrder`; `neighborGridBuild='scheme'|'eager'` separately selects
the scheme's existing pair-grid factory or bypasses it for an eager rebuild.
The query-order validation report and measurements are preserved in the
[external development archive](development-artifacts.md), under
`core/docs/development/query-order-validation.md`.

The stages (all in `radiusSearch/compactHash/`):

### 1. Cell size and support

- `computeGridSupport(supportsX, supportsY, scheme)` resolves the
  *global* support the grid must cover: Gather $\to \max h_i$,
  Scatter $\to \max h_j$, MeanSymmetric $\to \max (h_i+h_j)/2$, the
  three symmetric schemes $\to \max(\max h_i, \max h_j)$.
- `compute_h(qMin, qMax, support)` chooses the cell size as an integer
  division of the domain: $n_{\mathrm{cells}} = \mathrm{clamp}(\lfloor
  \text{extent}/h_{\max}\rfloor, 1)$, $h_{\mathrm{cell}} =
  \text{extent}/n_{\mathrm{cells}}$ (with a $10^{-4}$ inflation pass),
  so $h_{\mathrm{cell}} \ge h_{\max}$ and the domain holds an exact
  integer number of cells — no partial boundary cells.

### 2. Z-order sort

- On periodic axes the (query and reference) positions are wrapped
  into the domain with `torch.remainder` first.
- Each reference particle's cell index is
  $\lfloor (x - q_{\min})/h_{\mathrm{cell}}\rfloor$ clamped to
  $[0, n_{\mathrm{cells}}-1]$; periodic axes use a *floor* cell count
  (no trailing empty ghost cell, which would break wrap-around
  adjacency).
- The cell index is Morton-encoded (Z-order) by
  `computeZOrderIndex64` — `splitBy3Bits64` bit-interleaves the three
  coordinates into a 64-bit key (the 64-bit mask constants are
  assembled from 16-bit parts because Warp has no 64-bit literals —
  see `morton.py`'s comment).
- `argsort` on the linear indices gives `sortIndex` and
  `sortedPositions` — reference particles in Z-order, so each cell's
  particles are contiguous.

### 3. Cell table and hash table

- `torch.unique_consecutive` on the sorted linear indices gives the
  occupied-cell list with start offsets and counts:
  `cellTable = [cellIndex, cellStart, cellCount]`. Cell *grid* indices
  are de-linearized from the linear ids directly (no floating-point
  re-derivation, which could mismatch the build path at cell
  boundaries).
- Each occupied cell's grid index is hashed into
  `[0, hashMapLength)` by the Warp kernel `hashCells` (checked
  against the torch reference `hashGridIndicesTorch` at build time).
  `hashMapLength` comes from `HashMapLengthMode`: `Fixed`
  (default 4096), `NumberOfParticles` (made odd), or `NextPrime`
  (default mode; `getNextPrime(N)`).
- Occupied cells are re-sorted by hash value, and
  `hashTable[hash, 0:2] = (start, count)` is filled (empty slots are
  `(-1, 0)`). Lookup of a cell is: hash it, then walk the
  (usually length-1) chain of cells sharing its hash via
  `sortedCellTable`.

### 4. Stencil

- `searchRadius = max(1, ceil(hMax / hCell - 1e-6))` — the stencil
  must span every cell a support of $h_{\max}$ can reach; the tiny
  epsilon prevents floating-point noise from promoting an exact ratio
  of 1 to radius 2.
- `cellOffsets` is the $(2r+1)^D$-entry cartesian product of
  $[-r, r]^D$, padded to 3 columns for 1D/2D.

### 5. Count, then collect

- `radiusSearchCountNeighborsCompactHashMap` (one thread per query
  particle) walks the stencil — hashing the query's cell, walking
  chains, applying the per-pair support test
  (`isInSupport`: minimum-image distance $r \le h_{ij}$ under the
  scheme's pairwise support, with periodic wrap on cell components) —
  and writes `neighborCounts[i]`.
- Counts are summed and prefix-scanned in int64; the host-visible total and
  allocation capacity are checked before offsets are stored in their configured
  dtype. Count, collect and optional row-fill launches follow Torch's current
  stream. Then
  `radiusSearchCollectCompactHashMap` writes the `edge_i`/`edge_j`
  arrays, translating sorted indices back through `sortIndex`.

**Output:** an `AdjacencyList` with `paddedI`, `paddedJ` (configured particle-ID
dtype), `numNeighbors` (int32), `edgeOffsets` (configured CSR-offset dtype),
`numRows/numCols`, the
query/reference positions and supports, and a back-reference to the
`CompactHashMap` it was built from.
Particle IDs and offsets independently default to int32; configure
`warpSPHCore_INDEX_DTYPE` and `warpSPHCore_OFFSET_DTYPE` before import to select
int32 or int64. Warp's current flat-array dimension limit still applies.

The `CompactHashMap` itself holds: `sortedPositions`,
`sortedSupports`, `sortIndex`, `hashTable`, `sortedCellTable`,
`qMin/qMax`, `hCell`, `numCells`, `mode_uint`, `D`, `searchRadius`,
`numOffsets`, `cellOffsets`.

## Verlet lists

`buildVerletList(..., verletScale=1.0, priorNeighborhood=None)` — the
time-stepping path:

- builds from scratch via the compact-hash search with supports
  **scaled by `verletScale`** (the skin),
- reuses a `priorNeighborhood` when the particle counts match and the
  prior list is still valid (a validity check against the stored
  query/reference positions), otherwise rebuilds,
- `wp_updateVerlet`/`wp_countVerlet` then add newly-entered pairs
  incrementally between steps instead of rebuilding, and `filter`
  drops pairs that left the *true* support; `_verlet_validity_metrics`
  / `validCheck.py` quantify whether a list is still usable.

Use it when positions move a small fraction of the support per step
(the skin pays off); use a fresh `radiusSearchCompactHashMap` (or
`adjacency=None`) when they don't.

## Other backends

- `radiusSearch/naive/radius_naive.py` — the $O(N^2)$ all-pairs
  reference; used to validate the hashed search, never in production.
- `radiusSearch/small/wp_radius_small.py` — a small-system variant for
  cases where the hash-map overhead is not worth it.

## Test-data helpers

`generateNeighborTestData(nx, targetNumNeighbors, dim, periodic,
device)` (in `util/wp_util.py`) builds the standard test case used by
the notebooks and most of `tests/operations/`: an $n_x^d$ lattice in
$[-1, 1]^d$ with spacing $dx$, support $h$ chosen so the lattice has
~`targetNumNeighbors` neighbors via `volumeToSupport`:

$$
h = \begin{cases}
N_h\, v / 2 & d = 1 \\
\sqrt{N_h\, v/\pi} & d = 2 \\
\left(N_h\, v\,\cdot \tfrac{3}{4\pi}\right)^{1/3} & d = 3
\end{cases}
\qquad (v = dx^d)
$$

and returns `(positions, supports, n, domain, dx)`.
`n_h_to_nH(n_h, dim)` / `nH_to_nH` convert between the resolution knob
$n_h$ (particles per smoothing length, the dimension-comparable
quantity used by the [lattice
calibration](lattice-calibration)) and the
dimension-specific target neighbor count $N_h$.

## See also

[Data types — `CompactHashMap`, `AdjacencyList`](data-types) ·
[Autodiff machinery](autograd) (why the grid path has no host sync)
