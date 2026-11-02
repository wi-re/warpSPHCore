# Native neighborhood template prototype

This is an opt-in experiment. Production Python dispatch and the existing Warp
and native operators are unchanged. Common traversal/execution types live in
`src/warpSPHCore/native/csrc/execution`; operation policies live alongside each
function in its module's `native/csrc_<function>` directory.

## Contract

`Row<Operation, Traversal, Loader>` executes the following contract:

1. `operation.prepare(i)` makes query-local state (geometry, support/kernel
   coefficients and any query values). `operation.allowed(i)` applies the
   query-direction predicate.
2. `operation.identity(query)` initializes the accumulator.
3. `traversal.each(i, lane, lanes, visit)` supplies `{reference, edge}`. An edge
   ordinal is traversal-local and can be -1 without stored edges. A future
   transpose reader using cached original-edge values must map that ordinal
   back to the forward edge; recomputing pair derivatives needs no such map.
4. `loader.allowed(j)` and `loader.load(j)` provide the reference record required
   by this operation. `operation.interact(query, reference, neighbor, state)`
   adds a directed interaction. It never writes another particle's output.
5. `state.reduce(reduction)` declares each sum, count, maximum or selection
   reduction. This declaration supports mixed accumulator types without a
   fixed-size array of floats masquerading as every possible reduction.
6. Lane zero calls `operation.finish(query, state)`, which owns nonlinear
   postprocessing and stores. Output, saved state and diagnostic pointers can
   independently be null.

CPU uses the existing Torch stable-ABI `parallel_for` launcher. GPU execution
uses 8/16/32-lane groups. The shared-cell executor stages source records and
feeds the same operation directly: it is an execution policy, not a fake CSR
iterator. Padding groups participate in every block barrier.

For a concrete short example, read
`coreOperations/native/csrc_interpolate/nt_interpolate_operator.h`. It declares
its accumulator, adds weighted fields, optionally normalizes by an accumulated
weight, and optionally stores the pre-finalization state. The loader can change
plain interpolation from separate volume/field values to preweighted values
without changing traversal. All requirements are compile-time types; no virtual
calls or per-pair Python callbacks are involved.

Implemented primal policies: density; scalar/vector interpolation; scalar
Difference gradient and vector Jacobian; covariance and count; normalized
interpolation; all six CRK moment tensors and count; maximum-with-winner as a
selection-contract example. The moment policy has 52 floating sums plus a count
in 3D. The selector breaks ties by lowest reference storage index; this is an
explicit prototype convention, not a promise to match traversal-dependent Warp
tie breaking.

CSR uses int32 or int64 neighbor IDs, int64 offsets and int32 counts. Contiguous
source ranges and shared query/source cell tiles are implemented. Compressed
neighbors can later implement the traversal contract; two-bit decoding is not
implemented here. The private experiment ABI uses int32 particle counts and
range bounds, so counts must fit int32.

## Derivatives and implicit solves

Manual first derivatives are implemented for Wendland2/Gather density,
interpolation, Difference gradient/Jacobian and normalized interpolation. JVP
uses the forward CSR. VJP splits query-owned CSR and reference-owned transposed
CSR gathers. The reference-owned pass evaluates the **original directed pair**;
it does not replace `h_i` with `h_j` or reverse a nonsymmetric operator. No
original-edge map is required while pair coefficients are recomputed.

Normalized finalization has a module-owned `finalize_vjp`: a pointwise pass
pulls the output cotangent back to numerator/denominator cotangents once per
query. The pair pullback consumes those values. The matched atomic control uses
the same pair arithmetic, with output clearing included in timings.

Null tangent/gradient pointers express absent inputs/outputs. Field-only
requests select a narrower compile-time specialization and skip ownership passes
with no requested output. A first-order `torch.autograd.Function` bridge saves
inputs, geometry metadata, topology and required finalizer state, checks tensor
versions through Torch, and accumulates correctly when query and reference
arguments alias. Explicit JVP/VJP are also callable without recording a Torch
solve graph. `implicit_demo.py` demonstrates Newton-GMRES and a transpose solve
for an implicit parameter derivative; dense matrices are built only afterward
for verification.

This initial AD implementation assumes fixed topology and domain bounds. It
does not implement corrected-CRK derivatives, moments/selection AD, apparent-
volume AD, cell/cluster VJP, higher derivatives, or automatic differentiation of
new C++ operation policies. Those require explicit mathematical policies; the
template does not generate derivatives from arbitrary C++.

## Build and run

Run with a Torch development environment and CUDA/HIP compiler as appropriate.
The experiment has no JIT on its execution path. Example from this worktree:

```sh
python experiments/template_interface/build.py --core "$PWD" --backend cpu --output /tmp/template-cpu.so
python experiments/template_interface/build.py --core "$PWD" --backend cuda --arch sm_120 --output /tmp/template-cuda.so
HIP_PLATFORM=amd python experiments/template_interface/build.py --core "$PWD" --backend hip --arch gfx942 --output /tmp/template-hip.so
python experiments/template_interface/verify.py --cpu /tmp/template-cpu.so --gpu /tmp/template-cuda.so --output /tmp/verification.json
python experiments/template_interface/implicit_demo.py --cpu /tmp/template-cpu.so --gpu /tmp/template-cuda.so --output /tmp/implicit.json
```

CPU links Torch's stable scheduler API; CUDA/HIP arithmetic is a borrowed-pointer
C ABI with no Torch headers. This experiment does not add wheel/CMake targets or
register public operators. HIP has compile qualification only on this machine.

`template_runtime.py` is a private test wrapper, not a general tensor-validation
API. It requires correctly typed, contiguous, same-device buffers and valid,
nonoverlapping CSR/range/tile plans. Execution does not scan CSR, synchronize for
validation or copy inputs to the CPU. Build/transposition/verification are
explicit separate steps. A resident packed-reference buffer must be rebuilt when
reference fields change. The caller must establish cross-stream dependencies;
the wrapper records borrowed tensor lifetimes on the current Torch stream.

`run_study.py --help` describes the scaling study. It requires immutable existing
native/planner binaries supplied with `--baseline`, plus `--cpu` and `--gpu`.
The historical measurements in the report used V2; final compact packing and
once-per-query finalization are measured separately by `packing_study.py` against
those frozen V2 binaries. `analyze.py` generates graphs from the saved JSONs.
Sorting, radius search, transpose construction and correctness checks are
outside operation timings. Packing and output allocation are included in normal
forward timings; resident direct/template controls exclude them.
