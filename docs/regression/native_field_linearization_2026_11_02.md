# Production native field linearization and owned CSR/CSC pullbacks

Implemented on `codex/native-cluster-pairs`, based on `e2cc947a` (remaining
corrected primals). The ordinary public operation derivative contracts remain
unchanged. This milestone adds a separate production API for the field-only
products needed by fixed-geometry implicit solves.

## Scope and numerical contract

`warpSPHCore.native.linearizeFields(...)` binds geometry/corrections and exposes
`.value`, `.apply(query, reference)`, `.jvp(query=..., reference=...)`, and
`.vjp(cotangent, query=True, reference=True)`. VJP returns `FieldCotangents` with
independent query/reference roles. `apply` supports first-order Torch backwards.
Explicit products do not record derivative graphs. `CSRTopology` shares a lazy
transpose across operators and repeated solver products.

| Operator | Field derivative | Fixed coefficients and details |
| --- | --- | --- |
| Interpolation | Reference field only | Explicit/fallback volumes, CRK A/B; query VJP is `None` |
| Gradient | Query and reference fields | All gradient schemes; CRK spatial derivatives, reference grad-h, query renormalization |
| Divergence | Query and reference fields | Both tensor contraction axes; density-consistent weighting, both grad-h roles |
| Curl | Query and reference fields | 1D exact zero; existing 2D/3D vector/matrix contraction conventions |
| Laplacian | Query and reference fields | All four schemes; symmetric block transpose for Dot; Naive keeps the raw uncorrected kernel Laplacian |
| Density/covariance | Excluded | They do not consume the differentiated fields |

All 15 kernels, dimensions 1–3, FP32/FP64, supported scalar/vector/matrix shapes,
all five support policies and independent CSR ID/offset widths retain the primal
contracts. Gradients are partial derivatives with respect to field values:
positions, supports, domain, masses, densities, volumes, CRK coefficients and
spatial derivatives, grad-h and renormalization are fixed. There are no derivatives
through preparation, corrected geometry, neighbour topology or solver iterations.
Construct under `torch.no_grad()` to explicitly freeze differentiable coefficients.
Ordinary corrected `warpOperation` remains primal-only. Higher-order and dual/Torch
function-transform support is not added.

For an aliased physical field, explicit JVP requires its tangent in both roles;
explicit VJP returns role results to be added. If the factory's reference field
was omitted, `apply(field)` supplies both roles and Torch adds their backwards
results. Null tangents are real null pointers and exact zero inputs. Requested
VJP masks omit ownership launches; query-only VJP does not build CSC.

## Implementation and lifetime

JVP reuses the existing corrected primal policies with a nullable field loader.
Each operator adds `field_scales` and `field_contract` beside its numerical math.
The Laplacian primal also uses that contraction, avoiding duplicate Dot algebra.
The generic ownership executor uses existing `OwnedAdjointRow` traversal: CSR owns
query gradients, CSC owns reference gradients. It always recomputes original i,j
roles, including asymmetric supports, query corrections and masks. No floating
atomic reductions or edge Jacobian storage are needed.

The transpose builder lives in general native infrastructure. It counts/fills
using integer atomics, computes a device prefix sum, then sorts each CSC row by
original query ID. It handles prefixes, gaps, duplicates, tails and empty
populations. No runtime CSR value validation scan, host readback or debug parameter
was introduced. Construction is O(E) plus row sorting. CSC capacity equals CSR ID
capacity; its extra retained storage is eight bytes per capacity entry plus twelve
bytes per reference row, excluding temporary construction cursors/prefix-sum work.
CSC currently uses int64 IDs/offsets regardless of the primal widths.

`CSRTopology` borrows input arrays, caches CSC for that topology lifetime, and
checks tensor versions/identity. Captured geometry/coefficient changes invalidate
existing linearizations. Exposed CSC changes trigger reconstruction. Replacing an
external state attribute leaves an existing map bound to its old storage. Version
checks require ordinary tensors; inference-mode arrays must be cloned outside that
mode. Cross-stream CSC consumers wait for its construction event and record their
streams; callers still own normal input-stream ordering. This is not a concurrent
mutable-topology manager or graph-capture API.

The Torch backwards context saves fixed tensors, not primal fields or a solve
trajectory. Exemplar field tensors are borrowed for shape metadata. Sorted CSC
rows and fixed ownership reductions are repeatable within a backend/launch layout;
bitwise agreement across CPU/GPU or precisions is not promised. Ordinary native
reverse mode continues using its existing scatter atomics.

## Correctness and distribution qualification

Local environment: Torch `2.13.0+cu130`, CUDA 13 toolchain, x86_64 CPU, NVIDIA sm120.
Release build uses `-O2 -DNDEBUG`, native CPU target and LibTorch stable ABI 2.10.
This is capability/correctness qualification; no operation timing or throughput
claim is made under concurrent workloads.

| Check | Result |
| --- | --- |
| Complete source/prebuilt CPU+CUDA Torch suite | 20,642 passed; 1,580 skipped |
| New field-specific CPU+CUDA coverage within that suite | 2,273 passed; 129 unsupported-layout/device cases skipped |
| Python scaffolding suite | 23 passed |
| CPU editable/JIT complete suite | 10,068 passed; 517 skipped; 528 other-device cases deselected |
| CUDA editable/JIT complete suite | 10,574 passed; 536 skipped; 3 deselected |
| Installed CUDA wheel, CPU suite | 10,067 passed; 517 skipped; 529 deselected |
| Installed CUDA wheel, GPU suite | 10,574 passed; 536 skipped; 3 deselected |
| Wheel stable ABI audit | Passed; no direct ATen/c10/CPython symbols or absolute runtime paths |
| CUDA memcheck / synccheck | 687 passed each; 64 skipped; zero reported errors |
| Experimental HIP compile | Eight affected GPU translation units compiled for gfx942 |

Wheel checks run the production implicit example and complete suites with Warp
and compiler/JIT imports blocked. One nested import-blocking subprocess case is
omitted in wheel testing. Source and ELF `.text`/`.nv_fatbin` hashes match the tested
CMake binary. The local wheel targets x86_64/native CPU + sm120, pins this Torch
version, and is not a GH200 or manylinux qualification. HIP is offline compilation
only: no AMD runtime or ROCm wheel result is claimed. HIP emitted seven optimizer
loop-unrolling warnings, with no compilation failures.

Tests compare primal/JVP/VJP against independently differentiated Torch equations,
check dot-product identities, aliased-field gradcheck, optional roles, all kernels,
asymmetric supports and unequal populations. Additional cases cover CSR gaps and
duplicate multiplicity against primal basis matrices, isolated/empty rows, all
integer-width combinations, parallel transpose fill and CSC rows above the 64-entry
sort threshold, cross-stream reuse, mutation/replacement invalidation and unsupported
AD requests. Existing corrected primal and ordinary manual backwards matrices pass.

## Implicit-solve example

`scripts/examples/native_field_implicit.py` uses the production API for

`R(u) = u + dt * (u^3 - P u) - theta * source`,

with normalized interpolation `P` on fixed periodic 1D particles. Normalization is
an external diagonal operation, and its transpose divides the incoming cotangent
before the CSC gather. Newton/GMRES uses Jv products. The implicit backward solves
`R_u^T lambda = u` at the converged state for `L = ||u||^2 / 2`, then obtains the
parameter gradient from `lambda dot source`. No solve trajectory is recorded.
Independent dense equations are assembled only after the matrix-free solves.

Both CPU and CUDA converge in three Newton iterations, using 1/5/4 inner GMRES
iterations and six transposed GMRES iterations. Residual norms are below
`2.85e-13`, Jv/VJP dense errors below `5.2e-16`, adjoint-solve errors below `7e-14`,
and parameter-gradient errors below `9e-16`. The gradient is approximately
`6.42848358321845`. The independent verification initially used the 2D/3D Wendland
polynomial for its 1D problem; correcting it to the distinct 1D polynomial resolved
that oracle error without changing the native implementation.

## Integration and follow-up

The upstream sorter/CSR producer can provide its existing `AdjacencyList` directly.
Retain one `CSRTopology` while that graph is fixed and relinearize when geometry or
coefficients change. Reuse the topology object across multiple operator maps to
avoid redundant transposes. This does not require a frontend permutation change.

The next work is corrected geometry/coefficient derivatives and frontend-owned
operators on the shared policy interface, followed by clustered traversal with
its own ownership reversal. Compressed/two-bit production traversal still follows
that conversion. Field products currently select a general eight-lane CUDA owned
executor; dedicated derivative tuning, workspace reuse, compressed CSC, graph
capture and end-to-end frontend JFNK integration remain separate work.

Evidence archive: `core/experiments/native-field-ad-20261102` in
`/home/lu26029/dev/warpSPHDevelopment`. It contains logs, wheel, HIP objects,
source snapshot, build configuration and hash provenance. Generated binaries are
not committed to the source worktree.
