---
title: Experimental native backend
---

# Experimental native backend

The native backend builds neighbourhoods, evaluates all 15 radial kernel families,
and runs the seven standard SPH operators without installing or importing Warp.
It includes manual first-order adjoints for smooth kernels. CPU, CUDA and HIP
share the same numerical headers; Warp remains the default backend. Native CRK
preparation and all six applicable corrected core operations support first-order
Torch backwards and forward AD, including derivatives through the prepared CRK
coefficients. Fixed-geometry solver products remain available through
`native.linearizeFields`. Nonsmooth/singular geometry contracts, higher-order native
derivatives and a completely native frontend remain unported. The 0.7.1 frontend
can run complete eager simulations with native core operators and Warp frontend
kernels using the explicit compatibility selection below.

## Install and select

```bash
pip install -e ".[native]"
export WARPSPHCORE_BACKEND=native
python scripts/examples/native_core.py --device cpu
python scripts/examples/native_core.py --device cuda
```

Use a separate Python environment for an editable installation when other work
is using the main checkout. Local qualification used this worktree through
`PYTHONPATH`, leaving the existing editable installation unchanged.

Select the backend before the first package import. Changing it in a running
process is unsupported. For the existing backend, install `.[warp]` and leave
`WARPSPHCORE_BACKEND` unset, or set it to `warp`. The development extra includes
both Warp and Ninja so existing development/CI installation commands still work.

The loader first looks for a compatible packaged library. It checks the exact
qualified Torch build, backend runtime, machine architecture, requested GPU
coverage, source/header revision and (for host-tuned builds) CPU features before
loading it. This path imports no compiler helpers. Otherwise it uses a cached
Torch JIT extension, requiring a C++17 compiler and Ninja; GPU compilation also
requires the matching CUDA/ROCm toolkit.

`WARPSPHCORE_NATIVE_LOAD=auto|prebuilt|jit` chooses packaged-first selection,
strict compiler-free execution, or forced source compilation.
`WARPSPHCORE_NATIVE_BUILD=cpu|gpu|auto` controls backend capability; `auto`
requests GPU support when a GPU is visible. For a CPU-only wheel on a GPU host,
select `WARPSPHCORE_NATIVE_BUILD=cpu`. A loaded backend cannot be replaced in the
same process. A binary load/registration error is reported directly rather than
attempting to register a second copy through JIT.

The C++ boundary now uses stable LibTorch tensors, registration, dispatcher,
parallel execution, device guards and stream APIs, targeting Torch 2.10. It has
no direct ATen/c10 C++ or Python C-API dependencies. Autograd capability checks stay
at the Python entry points; the stable Tensor API does not expose a
`requires_grad` query. Native source builds require Torch >= 2.10; binary wheels
are deliberately pinned and qualified against only the exact build used to
produce them. Stable ABI use does not claim a tested multi-Torch release matrix.

### Editable/source installation

Plain `pip install -e .` installs source wrappers and headers without compiling
native code or importing Torch during packaging. Install `.[native]` to include
Ninja. The first native call builds all float32/float64, 1D/2D/3D variants once.
JIT host code defaults to `-march=native` on x86 or `-mcpu=native` on aarch64;
set `WARPSPHCORE_NATIVE_CPU_ARCH=portable` for a baseline build, or a compiler
CPU target name. CUDA JIT targets visible devices unless
`TORCH_CUDA_ARCH_LIST` is supplied. `TORCH_EXTENSIONS_DIR` selects a persistent
cache; use node-local storage or prebuilt wheels for large cluster jobs.
`CUDA_HOME`/`ROCM_HOME` configure toolkit discovery when needed. HIP preprocessing
uses private build copies and never writes generated headers into the package.
Importing the package or calling `nativeCapabilities()` compiles nothing.

### Using the 0.7.1 frontend

Keep the frontend's Warp runtime and select the native core before import:

```bash
export WARPSPHCORE_BACKEND=warp
export WARPSPHCORE_OPERATORS=native
# Run the usual frontend entry point in eager mode.
```

`warpSPHCore.backend` identifies the process runtime; `operator_backend`
identifies this core execution selection.

This redirects the public `warpOperation`/flat operation calls and CRK
volume/moment/consistency-density accumulations to native code. Existing Torch
particle states, int32/int64 CSR buffers and Morton ordering pass through without
copies or hidden permutations. Calibrated summation density uses the upstream
cached lattice factor and a Torch output scale in this compatibility adapter.
The pure native API's calibration restriction is unchanged. No CSR content scans,
CPU readbacks or extra debug options are introduced in these operator calls.

Frontend force/work kernels, fused CRK coefficient construction, shifting,
pressure iterations, communication and explicit Verlet-list construction remain
their existing implementations. Missing adjacency uses the native operator's
own compact-hash search. This is a mixed execution path, not a Warp-free fluid
solver. Unsupported inputs raise; there is no implicit operator fallback.
In particular, native operators do not yet consume upstream `ChunkedAdjacency`:
owned-row multi-GPU/multi-node execution still needs Warp operators. The explicit
Warp JVP/HVP and CRK JVP entry points reject this selection; native first-order
Torch AD and `native.linearizeFields` retain their separately qualified scope.
Full native frontend CUDA graph execution is not qualified.

For a complete warmed-step comparison, put the paired 0.7.1 source roots on
`PYTHONPATH` and run each operator backend in its own process:

```bash
python scripts/benchmarks/bench_simulations.py --operators warp \
  --case tgv-wc --scheme deltaSPH --nx 128 --device cpu \
  --order morton --n-h 4 --sound-speed 10 --out /tmp/simulation-warp
python scripts/benchmarks/bench_simulations.py --operators native \
  --case tgv-wc --scheme deltaSPH --nx 128 --device cpu \
  --order morton --n-h 4 --sound-speed 10 --out /tmp/simulation-native
```

The primary timer includes integrator stages, search and layout preparation;
the accepted-boundary interval also includes runner diagnostics and outputs.
Compilation/setup/warmup are excluded. `--snapshots` saves every accepted state
for numerical comparisons outside the primary timer. `--profile` records two
warmed steps to show which native kernels the solver actually calls; profiled
timings are not benchmark results. Physical projection limits may shorten a
requested fixed step; compare the recorded clocks as well as the fields.

### Building native wheels

Use the target Torch installation and its matching toolkit. Build dependencies
are `build`, `scikit-build-core>=0.10,<1`, `wheel`, CMake and Ninja. The repository
script uses the active environment without isolated dependency replacement and
pins the resulting wheel's Torch requirement to that exact installation.
CPU, CUDA and HIP wheels are separate artifacts with backend build tags.

```bash
python scripts/packaging/build_native_wheel.py --backend cpu --output dist/cpu

# Explicit GPU targets, portable host CPU code within the build's CPU platform:
python scripts/packaging/build_native_wheel.py --backend cuda \
  --gpu-arch '80-real;90-real;120-real' --output dist/cuda

# Run separately in a ROCm Torch environment:
python scripts/packaging/build_native_wheel.py --backend hip \
  --gpu-arch 'gfx942;gfx1100' --output dist/rocm

python scripts/packaging/check_native_wheel.py dist/cuda/*.whl --device cuda --tests
python scripts/packaging/audit_native_abi.py dist/cuda/*.whl
```

Both GPU variants include parallel CPU execution. All six precision/dimension
variants are compiled together. `--torch-version` can require an exact installed
version before building; `--jobs`, `CXX`, `CC`, `CUDACXX` and repeatable
`--cmake-arg KEY=VALUE` support site toolchains. CUDA architecture suffixes follow
CMake conventions: `-real` includes machine code, `-virtual` includes PTX, and a
bare target includes both. PTX can still involve driver compilation at runtime;
choose real targets for a fixed cluster deployment.

### GH200 / site-specific HPC deployment

Build on an aarch64 build node with the cluster's Torch/CUDA installation. H100
uses target `90`; the same build command also compiles CPU kernels for Grace.
A native CPU target assumes the build host represents the compute nodes.

```bash
# On the target GH200/Grace build node:
python scripts/packaging/build_native_wheel.py --backend cuda \
  --gpu-arch 90-real --cpu-arch native --output dist/gh200

# Install that aarch64 wheel in the matching runtime environment:
pip install --no-deps dist/gh200/*.whl
WARPSPHCORE_BACKEND=native WARPSPHCORE_NATIVE_LOAD=prebuilt python simulation.py
```

A wheel built on x86_64 is not an aarch64 wheel. This work qualifies native
builds on the current x86_64 host; GH200 execution and cross-compilation have
not been qualified. The build script uses the running machine's platform,
Torch libraries and CPU compiler. `portable` avoids host-specific CPU tuning
but does not certify a manylinux/glibc baseline. Public manylinux distribution
still needs a controlled build image and dependency audit; site wheels target
the cluster's existing environment. Runtime Torch and accelerator libraries
remain required; compilers and Ninja are unnecessary for a compatible wheel.

## Standalone C++ execution

The optional `warpsph_core` library has no Torch, Python or Warp build/runtime
dependency. It exports a C++17 API from `native/csrc/api.h`, installed as `api.h`
beneath the CMake target's include directory. The exported target is
`warpSPHCore::core`. The Torch adapter and standalone library compile the same
module-owned primal and adjoint policies; they do not maintain separate formulas.

```bash
# CPU: requires a C++17 compiler, CMake and OpenMP (no Python discovery).
cmake -S . -B build/standalone \
  -DWARPSPHCORE_BUILD_STANDALONE=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build build/standalone -j2
ctest --test-dir build/standalone --output-on-failure
cmake --install build/standalone

# A separate C++ application consumes the installed package:
cmake -S scripts/examples/standalone -B build/consumer \
  -DCMAKE_PREFIX_PATH="$PWD/install"
cmake --build build/consumer -j2
build/consumer/sph_example
# Optional: explicit mass derivatives of sum(density).
build/consumer/sph_example --mass-gradient

# Standalone CUDA build (includes CPU execution):
cmake -S . -B build/standalone-cuda \
  -DWARPSPHCORE_BUILD_STANDALONE=ON -DWARPSPHCORE_NATIVE_BACKEND=cuda \
  -DCMAKE_CUDA_ARCHITECTURES='90-real;120-real' -DCMAKE_BUILD_TYPE=Release

# Standalone ROCm build, without an AMD GPU on the build host:
cmake -S . -B build/standalone-hip \
  -DWARPSPHCORE_BUILD_STANDALONE=ON -DWARPSPHCORE_NATIVE_BACKEND=hip \
  -DCMAKE_HIP_ARCHITECTURES='gfx942;gfx1100' -DCMAKE_BUILD_TYPE=Release
```

The commented example in `scripts/examples/standalone/main.cpp` shows input
ownership and layouts, query/reference roles, CSR construction, density, and a
scalar-field gradient that uses the computed density. The optional mass-gradient
example is a separate function. Numerical smoke assertions live under
`tests/native/cpp/standalone_smoke.cpp`.

Build each GPU directory with `cmake --build <directory> -j2`. Sites can supply
`CMAKE_CUDA_COMPILER`, `CMAKE_HIP_COMPILER`, `CMAKE_HIP_COMPILER_ROCM_ROOT` and
`CMAKE_PREFIX_PATH` for their toolkit. GPU runtimes must be available to the
application's dynamic loader, for example through cluster environment modules.
`WARPSPHCORE_CPU_ARCH=portable|native|<target>` applies to the standalone CPU
implementation too. `WARPSPHCORE_STANDALONE_OPENMP=OFF` selects serial CPU code;
`WARPSPHCORE_STANDALONE_TESTS=OFF` omits the test programs and test-only bridge.

`Problem` describes particle/domain/field/CSR views. Each buffer is contiguous,
non-owning storage with a capacity measured in elements. Positions have shape
`[N,D]`, supports/masses/densities/kinds `[N]`, and scalar/vector/matrix fields
`[N]`, `[N,D]`, `[N,D,D]`. Floating-point buffers use `Problem.precision`, kinds
and neighbour counts use int32, and periodicity uses bool. CSR offsets and IDs
independently accept int32 or int64. An output's logical size is given by
`output_elements(problem)`. The caller allocates it and retains all buffers
until the work on its stream has completed.

```cpp
#include <api.h>
using namespace sph_native;
Context context;                       // parallel CPU, or set device/stream
Problem problem;                       // fill borrowed particle/domain/CSR views
problem.properties.operation = Operation::Gradient;
Buffer result{output_pointer, output_capacity};
forward(context, problem, result);
```

All seven standard primals, 15 kernels, precisions, dimensions and field contracts
listed below are available. `Context` selects CPU, CUDA or HIP, device index,
caller stream and CPU thread count. Each GPU library includes
CPU execution and its selected GPU backend. `supports(device)` reports build
capability; it does not probe installed hardware. CPU buffers must be host
accessible; GPU buffers must be accessible on the selected device.

Operator entry points check argument metadata (sizes, dtypes, options and storage
compatibility). They do not scan numerical contents or CSR data. There is no
validation mode, debug flag, or verification parameter in the execution API.
Callers own input validity and topology updates. Optional, synchronizing input
verifiers live separately under `tests/verify`; the runtime imports none of them.
Numerical outputs must not overlap their input storage. Primal/VJP launches on
GPU remain stream-ordered without host content-check synchronization.

Radius search uses a compact hash of occupied reference cells. Cells are grouped
by the existing uint32 prime hash, retain full int64 Cartesian keys to resolve
collisions, and contain stable original reference IDs. CPU and GPU share cell
indexing, periodic wrapping, unique stencil traversal and the exact minimum-image
distance predicate. Cell width conservatively bounds the selected support policy,
with padding for floating-point minimum-image arithmetic. Extremely small grids
are coarsened to preserve the existing scalar squared-distance underflow convention. Periodic axes use floor
cell counts, and one/two-cell axes visit each cell once. Nonperiodic out-of-domain
positions clamp only for indexing; distance checks still use their actual positions.
The native cutoff remains strict (`distance < support`), including zero-radius
rows. Warp's compact search uses an inclusive cutoff; exact boundary membership
therefore retains the established backend distinction.

Build a reusable index, count, scan, allocate and collect:

```cpp
auto index = build_compact_hash(context, problem, SearchOptions{4096});
radius_count(context, problem, index, counts);
const auto edges = radius_scan(context, problem.query.count, counts_view, offsets);
// Allocate edges int64 IDs and set problem.adjacency.counts/offsets.
radius_collect(context, problem, index, indices);
```

`SearchOptions.hash_buckets=0` selects approximately one bucket per reference;
one bucket is legal and resolves all collisions using full cell keys. The overloads
without an index build a temporary one. The standalone example retains one index
across count and collect. The Torch adapter builds the same native index directly
from Torch memory and shares the count/collect rows; it no longer scans all pairs.
The final CSR scan also uses the common native CPU/CUDA/HIP scan, rather than
Torch tensor scans or scalar reductions. CSR rows preserve ascending original
reference IDs using an in-place heap sort.

An index owns its buffers, is movable, and stays bound to its build device,
precision, dimensionality, reference count and domain. Queries may move or be
replaced within `statistics().radius_bound`. Reference motion within the same cells and
support changes within that bound are valid. The caller rebuilds the index when
references cross cells, the domain changes, or the support exceeds its build
bound. Indexed calls check device/shape metadata, without rescanning geometry.
`statistics()` reports occupied-cell counts,
conceptual grid dimensions, cell width and the reusable radius bound. The floating-
point padding is reserved for traversal correctness and cannot be consumed by
support growth. Periodic coordinates whose image count cannot be represented
reliably in the selected precision are rejected explicitly. Only occupied-cell storage is allocated, including
sparse 3D domains; axis counts are capped conservatively so int64 keys cannot overflow.

For native frontends, `index.view()` borrows read-only device pointers and grid
metadata. Include `execution/compactTraversal.h` and use
`compact::CellTraversal<T,D>` to enumerate candidate cell ranges; each stored
particle carries its original reference ID. Keep the index alive until work using
its view has completed. The frontend applies its own pair predicate and accumulation.

CPU construction uses `std::sort`. CUDA/HIP construction uses shared portable
bottom-up merge sorting, marks cell boundaries, scans them on device and emits
compact bucket/cell tables. Construction completes on the caller's stream before
publishing the index. This first GPU sorter performs a binary rank search per
element per merge pass; it is a correctness implementation rather than a tuned
radix-sort replacement. Allocation, validation and exact edge-capacity boundaries
still synchronize; graph capture and Verlet maintenance remain future work.

CPU CSR scanning uses checked int64 accumulation. CUDA and HIP share a hierarchical
device prefix scan: counts, tile sums and offsets remain on the device. Only the
small total/status result returns to the host so the caller can allocate exact
edge capacity. Negative counts and int64 overflow are rejected. Collection also
synchronizes, rejects inconsistent counts and bounds its writes before reporting
a mismatch. Dense neighborhoods still require visiting every relevant pair.

`adjoint(context, problem, cotangent, gradients)` exposes an explicit VJP for an
individual operation. The caller retains the unmodified primal inputs and CSR,
provides a cotangent matching the logical output, selects floating-point roles
with `gradient_bit(QX)` etc., and supplies corresponding `Adjoints.slots` buffers.
By default these buffers are zeroed before accumulation; `accumulate=true` adds
to existing values. Exact aliases between gradient buffers combine role
contributions; partial overlap is rejected. The same smooth-kernel restrictions
apply as for Torch reverse mode. This interface does not record an AD graph,
manage saved tensors, or implement a second graph engine.

The standalone library supplies operations, not a complete solver or file-format
runner. Initial conditions can be generated in Python and stored in HDF5 with a
YAML configuration. Loading that state into owned C++ CPU/GPU buffers, solver
steps and snapshot output belong to the application/frontend. HDF5 and YAML
libraries are not required by the numerical core.

The CPU/CUDA test bridge can be checked against independent Python oracles:

```bash
pytest --suite standalone --device cuda \
  --native-library=build/standalone-cuda/libnative_standalone_test.so -q
```

Local qualification built and ran the standalone CPU and CUDA executables and an
external installed-package consumer. Link dependencies contained no Torch,
CPython or Warp libraries. The standalone ROCm library also compiled and linked
for both gfx942 and gfx1100; actual AMD execution remains unqualified. CPU/GPU
primal and explicit-adjoint tests cover all operator schemes, field ranks,
precisions, kernels, support policies, CSR widths, empty populations, explicit
accumulation, aliasing, bounds errors and caller streams. Parallel CPU tests use
both one and four workers with shared reference-gradient scatters. The standalone
matrix qualifies both CPU and CUDA. CPU thread-count checks run only in the CPU
suite. The native Torch adapter and compiler-free CPU wheel have separate checks,
including a stable-ABI audit. The ROCm
consumer executed the CPU path successfully and skipped its unavailable AMD
GPU test. No performance measurements were made.

## Supported contracts

| Capability | Scope |
|---|---|
| Execution | Parallel CPU; CUDA; experimental HIP (ROCm compilation gate) |
| Spatial dimensions | 1, 2, 3 |
| Floating-point types | float32 and float64, selected from tensors at runtime |
| Neighbours | Compact-hash radius search, minimum images, canonical CSR, ascending reference IDs |
| Kernels | Wendland2/4/6, cubic/quartic/quintic spline, B7/B8, HOCT4, Gaussian, Poly6, Spiky, Viscosity, Cohesion, Adhesion |
| Operators | Density, Interpolate, Gradient, Divergence, Curl, Laplacian, Covariance |
| Fields | Scalar, vector and matrix fields; each field axis has spatial extent D |
| Schemes | All four gradient schemes; all four Laplacian schemes; both divergence contractions |
| Reverse AD | Manual first-order VJPs for the 11 smooth-origin kernels; fixed CSR topology |
| Supports | Gather, Scatter, MeanSymmetric, KernelMeanSymmetric, SuperSymmetric |
| Categories | All existing operation-direction enums, preserving query/reference roles |
| Geometry | Independent query/reference populations; mixed periodic axes; unwrapped positions |
| CSR input | int32/int64 particle IDs and offsets independently; int32 counts; padded tails ignored |
| Empty sets | Zero query/reference populations and finite zero sums |
| Tensor access | Direct Torch storage; inputs must be contiguous and on the same device |

Use `radiusSearch(particles, domain, referenceParticles=None, mode=...)` to build
an unpadded CSR list. Use `warpOperation(...)` or its native alias
`sphOperation(...)` with the established semantic state and operation properties.
When adjacency is omitted, the native compact-hash search builds it. Common
`ParticleState`, `DomainDescription`, `OperationProperties`, `CapacityPolicy`
and `AdjacencyList` definitions are shared with the Warp backend; they are not
separate native copies. Their established dataTypes imports remain available.

Search visits nearby occupied cells after index construction. Dense or very
large-radius queries can still visit every reference. Native search emits int64
IDs/offsets regardless of Warp import-time type settings.
CSR tensors supplied by another producer may use either integer width.

Corrected interpolation supports CRK A/B and explicit volumes. Gradient,
divergence, curl and covariance support CRK spatial derivatives and query gradient
renormalization; the three gradient-based Laplacian schemes do as well. Grad-h
scales the reference field for gradient, and both query and reference fields for
divergence, curl and Laplacian. Covariance does not consume grad-h. Naive Laplacian
uses the raw kernel Laplacian, ignoring CRK/renormalization while retaining volume
and grad-h weighting. These are the existing Warp equations.

`consistentDivergence=True` selects density-consistent divergence weighting.
`covarianceReturnNumNeighbors=True` returns `(covariance, counts)`, counting
positive corrected-kernel values rather than CSR degree. These flags route to
corrected native policies, including when no other correction is supplied. Counts
are discrete and have no derivative.
Density ignores common correction bundles, matching its unchanged Warp equation.
The corrected `warpOperation` paths and CRK preparation support first-order
backwards and Torch forward AD (`torch.autograd.forward_ad`). Correction inputs
and field-only products support all kernels; geometry derivatives retain the
11 smooth-kernel restriction. For repeated field-only products on fixed geometry,
use the explicit linearization below. Calibration, positive divergence,
partial support policies, capacity/rebuild reuse and general graph capture remain
unsupported. Dot Laplacian requires a vector or matrix field outside 1D.
Unsupported requests raise explicitly.

`warpSPHCore.kernels.evaluateKernel(displacements, supports, kernel)` returns value,
gradient, Hessian, support derivative, gradient-support derivative, regularized
Laplacian and radial metadata. This primitive is forward-only as a Torch operation;
its explicit derivative outputs are available to callers and shared operator adjoints.
Smooth kernels use analytic origin limits and a near-origin derivative expansion.
Cusp/singular force-kernel Hessians may be undefined at the origin. The legacy
Laplacian estimator retains its regularization and zero result below its epsilon.

C++ checks dispatch metadata without scanning numerical values or CSR contents.
No operation carries a debug mode or verification parameter. Optional diagnostics
live separately under `tests/verify`, and callers explicitly invoke them when
investigating input data. Exact-sized search still returns counts needed for
allocation to the host; that is part of search construction rather than operator
verification. Graph capture remains unsupported by the Torch wrapper.

## Source ownership and common wrappers

Shared numerical primitives mirror the core's existing structure under
`native/csrc`. Operator and algorithm implementations belong to their owning
module's `native` subpackage. The latter contains both Warp and native sources,
behind a Python wrapper exported by its `__init__.py`.

```text
warpSPHCore/
  native/
    _build.py                    # JIT build and prebuilt-first entry point
    _sources.py                  # manifest inventory, source hashes, private HIP staging
    _loader.py                   # packaged binary compatibility checks
    api.py                       # explicit-native compatibility entry points
    csrc/
      config.h
      dataTypes/                 # array and particle/domain/CSR views
      math/                      # distance and power primitives
      util/                      # support and interaction-direction rules
      kernels/
        kernel.h                 # support-aware pair kernel evaluation
        kernelFunctions/
          *.h                    # all 15 radial families
        radialJet.h              # analytic radial derivatives through third order
        evaluation.h             # spatial and support derivatives
        adjoint.h                # shared kernel VJPs
      execution/                 # CPU/GPU scheduling, type dispatch and CSR traversal
      torch/                     # reusable native tensor validation
      extension.cpp              # shared library capability registration
    _inputs.py                   # shared Python geometry/CSR/tensor preparation
  coreOperations/
    native/
      __init__.py                # common computeDensity/computeGradient exports
      wrapper.py                 # select Warp or native implementation
      wp_density.py              # Warp implementation, including tape support
      wp_gradient.py             # remaining five wp_ implementations also colocated
      nt_density.py              # native density Python/AD boundary
      nt_gradient.py             # nt_ entry points exist for all seven operators
      nt_density_backward.py     # module-owned reverse entry point
      sources.json               # this module's CPU/GPU translation units
      csrc_density/
        nt_density_forward.h     # shared CPU/CUDA/HIP primal row implementation
        nt_density_forward.cpp   # checks, CPU launch and Torch registration
        nt_density_forward.cu    # thin GPU launch adapter
        nt_density_adjoint.h     # shared numerical VJP
        nt_density_adjoint.cpp
        nt_density_adjoint.cu
      csrc_gradient/             # same forward/adjoint layout for each operator
        nt_gradient_forward.h
        nt_gradient_forward.cpp
        nt_gradient_forward.cu
    wp_density.py                # old import path aliases the colocated module
    wp_gradient.py
  kernels/native/                # explicit kernel evaluator and csrc_kernel/
  radiusSearch/
    native/
      __init__.py
      wrapper.py                 # selected Warp or native compact-hash backend
      nt_radius.py
      sources.json
      csrc_radius/
        nt_radius_forward.h      # shared indexed count/collect rows
        nt_radius_forward.cpp
        nt_radius_forward.cu
        nt_radius_dispatch.h
      csrc_compact/              # framework-free index API and CPU/GPU builders
        nt_compact.cpp
        nt_compact_cpu.cpp
        nt_compact_gpu.cu
```

Shared index views and traversal live in `native/csrc/dataTypes/compactHash.h`,
`math/grid.h` and `execution/compactTraversal.h`. The device prefix scan and
runtime guards live under `native/csrc/execution/`; both the standalone and
Torch builds compile the common execution sources.

Higher-level code imports the common wrapper from its local `native` package.
The Warp entry point in `operations.py` now reaches the same density/gradient
wrapper boundary as native execution. Warp implementation files load only on
that selected path. Old direct kernel/spec imports remain aliases of the
colocated module, including private specs and monkeypatches.

CPU and GPU adapters schedule the same header-defined row code. Module-local
C++ files own their Torch registration. The central extension does not contain
an operation switch or a list of module source filenames. It discovers each
`native/sources.json` without importing Python modules. Manifest paths are relative
to that `native` directory, such as `csrc_density/nt_density_forward.cpp`.
Each function keeps its compiled sources together under `csrc_<function>/`;
reusable headers belong in the core's shared `native/csrc` tree. CSR traversal
lives in `native/csrc/execution/traversal.h` and accepts a pair-operation policy,
so frontend code can include it as `"execution/traversal.h"` using the shared
include root. `warpSPHCore.native.get_include()` returns that installed header
root without triggering a build, for external CMake or extension builds.
Only helpers specific to a module stay local. Module sources can
therefore be added without modifying the central builder. Shared validation
remains in C++; Python owns argument preparation and capability/AD policy.
Shared wrapper preparation lives in `native/_inputs.py`: geometry is reused by
radius search and pair operators, and the common operator tensor layout/CSR
handling is available to frontend wrappers. Operator-specific field requirements
and supported schemes stay in their `nt_<operator>.py` entry points.

`discover_sources(package_root)` supports nested frontend layouts such as
`warpSPH/modules/crk/native`. `load_extension(name, cpu_sources, gpu_sources,
include_paths=...)` can build a frontend's own source group and registration
namespace, automatically adding the shared core header root. A frontend need not move its solver kernels into the core package.
Parent package initialisers must also avoid eager Warp imports on a native path.

CRK preparation exports live in `crk.native`, with `nt_volume.py`, `nt_moments.py`,
`nt_density.py` and `nt_factors.py` wrapping module-owned `csrc_<function>` policies.
The small batched factor solve is shared Torch algebra in `crk/crk_terms.py`.
`crk` selects its native exports without importing Warp when the native backend is
selected. The historical Warp implementation remains the numerical reference.

Torch JIT derives object names from source basenames. The shared loader stages
small include wrappers for colliding names across modules, preserving original
relative-header lookup and compiler dependencies. Frontends can therefore retain
names such as `nt_density_forward.cpp` in their own module directories.

Tensor ownership stays with Torch. GPU execution uses Torch's current stream
and device guard. `scripts/diagnostics/native_kernel_smoke.cpp` includes the
module-local primal headers directly and builds them without Torch.

## Native CRK preparation and frontend integration

```python
# Select WARPSPHCORE_BACKEND=native before importing warpSPHCore.
prepared = sph.prepareCRK(
    query, domain, kernel,
    operationMode=sph.OperationDirection.TrueAllToToAll,
    adjacency=query_to_reference_csr,
    referenceState=reference,
    referenceAdjacency=reference_to_reference_csr,
)
# Existing corrected operators consume these arrays directly:
options = dict(referenceParticles=reference,
               adjacency=query_to_reference_csr,
               referenceVolumes=prepared.referenceVolumes,
               crkState=prepared.crkState)
```

`prepared` contains `queryVolumes`, `referenceVolumes`, `consistencyDensity` and
`crkState`. `computeCRKFactors(...)` preserves the existing three-value tuple
(query volumes, diagnostic density, CRKState); use `prepareCRK` when distinct
reference volumes are also needed. The standalone pass exports are
`computeCRKVolume`, `computeCRKMoments` and `computeCRKDensity`. The latter is the
CRK consistency ratio, not the ordinary SPH density operator.

The frontend owns particle ordering and topology lifetime. All outputs retain
input order; sorting requires applying the corresponding permutation to particle
fields and producing CSR in that same order. Supplied lists accept independent
int32/int64 IDs and offsets. No additional search runs when both required lists
are supplied. If lists are omitted, native compact hashing constructs them.
A distinct reference cloud needs a reference-to-reference list for its volumes;
a query-to-reference list cannot substitute for it. `referenceVolumeMode` selects
that volume interaction rule independently, including for aliased populations.

The query/reference list must cover both Gather volume and Scatter moment pairs.
A union/SuperSymmetric search is suitable for unequal supports. Coefficients and
volumes depend on current positions, supports and interaction categories; reuse
of CSR alone does not justify reusing old correction values. The existing
identity limiter is retained for degenerate systems. Moment counts follow Warp's
admitted-edge convention, including zero-weight edges in a supplied broad list.

All particle-pair passes use the shared prepare/accumulate/finalize interface;
volume reciprocals and diagnostic ratios run once per query. CPU scheduling and
GPU stream/device ownership use Torch. No CSR scans, host readbacks or primal
scatter atomics are added. The factor algebra retains its numerical limiter and
Torch derivatives. The native volume/moment/diagnostic passes now provide manual
first-order JVP/VJP, so the complete preparation pipeline is differentiable.
The determinant/count/Schur limiter masks are fixed decisions for AD. Inactive
identity-correction branches have zero coefficient derivative; crossing a limiter
threshold or changing rank is not a smooth derivative contract.

Run `scripts/examples/native_corrected.py --device cpu` (or `cuda`) with the
native backend selected for preparation and all six corrected core primals.
The example checks linear interpolation/gradient/divergence/curl, identity
covariance and zero constant-field Laplacian.
See [preparation qualification](../regression/native_crk_preparation_2026_11_02.md)
and [remaining corrected primals](../regression/native_corrected_core_2026_11_02.md).

## Forward, backward and adjoint file boundaries

The uncorrected implementation uses one primal calculation for inference and a
grad-enabled forward. A Torch `autograd.Function` bridge saves inputs with version checks,
prunes unrequested input gradients, and calls module-owned numerical adjoints.
Backward orchestration and the adjoint are two layers of the same reverse pass;
there is no second derivative formula in Python.

| File family | Responsibility |
|---|---|
| `nt_density.py` | Python entry, capability checks and argument preparation |
| `csrc_density/nt_density_forward.*` | Shared primal and CPU/GPU adapters |
| `nt_density_backward.py` | Module-owned reverse entry into the common Torch bridge |
| `native/_autograd.py` | Saved tensor/version handling, requested gradients and Torch backward protocol |
| `csrc_density/nt_density_adjoint.*` | Analytic VJP contractions and CPU/GPU adapters |
| `native/csrc/execution/adjointTraversal.h` | Shared CSR reverse traversal and scatter accumulation |
| `native/csrc/kernels/adjoint.h` | Spatial/support kernel VJPs |

All seven uncorrected operators follow this layout. Gradients cover positions, supports,
reference masses, relevant query/reference densities and fields, and periodic
domain bounds. Aliased query/reference roles return separate contributions for
Torch to accumulate. Radius search runs outside the differentiation graph;
neighbour membership, periodic winding and cutoff transitions are discrete.

Reverse mode supports Wendland2/4/6, cubic/quartic/quintic, B7/B8, Gaussian,
Poly6 and Cohesion. HOCT4, Spiky, Viscosity and Adhesion reverse mode remains
explicitly unsupported pending nonsmooth/singular contracts. On this uncorrected
bridge, JVP, HVP and higher orders are unsupported; the backward bridge is once-differentiable. Reference
scatter uses CPU atomic compare/exchange or GPU atomic addition. Strict Torch
deterministic mode rejects this path; warn-only mode warns and executes.

## Correctness qualification

Tests are separated by layer, then selected by execution device before native
parametrization. Each suite runs in a fresh process because the package backend
and Warp precision are import-time choices. A bare `pytest` runs the fast Python
scaffolding suite and compiles no numerical code.

| Suite | Location | Responsibility |
|---|---|---|
| `python` | `tests/python` | Models, source ownership, build/loading contracts, CI selection |
| `warp` | `tests/warp` | Warp kernels, operators, search and frontend contracts |
| `torch` | `tests/native/torch` | Native Torch adapter, autograd, kernel/primal integration |
| `standalone` | `tests/native/standalone` | Framework-free library against independent numerical oracles |
| `parity` | `tests/parity` | Explicit native/Warp comparisons |

C++ tests live in `tests/native/cpp`; reusable oracles live in `tests/_support`.
The standalone numerical matrix owns the full shared-core checks. Expensive
Torch integration matrices carry `extended`; adapter changes run them, while
shared C++ changes can run the smaller adapter contract suite. CI uses
`scripts/testing/select_suites.py` to select owners from changed paths; docs-only
changes do not launch numerical jobs. Packaging checks qualify wheel loading
and ABI without repeating the full numerical matrix.

```bash
pytest --suite python -q
pytest --suite torch --device cpu -m "not extended" -q
pytest --suite torch --device cuda -q
pytest --suite standalone --device cpu \
  --native-library=build/standalone/libnative_standalone_test.so -q
pytest --suite standalone --device cuda \
  --native-library=build/standalone-cuda/libnative_standalone_test.so -q
pytest --suite warp --device cpu -q
pytest --suite parity --device cuda -q
warpSPHCore_PRECISION=float64 pytest --suite parity --device cpu -q
# C++ checks do not require Python or Torch:
ctest --test-dir build/standalone-cuda -L cuda --output-on-failure
ctest --test-dir build/standalone-cuda -L cpu --output-on-failure
```

A specific test path infers its suite, e.g. `pytest tests/native/torch/test_native.py`.
`--device all` deliberately selects every available device; the default is CPU.
An explicitly requested unavailable device fails early. `--device hip` requires
ROCm Torch for Python-driven tests; compile/link qualification and standalone
C++ HIP tests require no ROCm Torch. HIP remains experimental until an AMD GPU
is available for execution. Use the equals form of `--native-library=...` to
prevent pytest's initial root discovery treating the library as a test path.

Native tests use an independent Torch all-pairs oracle and check malformed
inputs, empty sets, directions, mixed index widths, padding and stream use. A
fresh subprocess installs an import blocker for `warp` and runs the complete
native example. Direct Warp comparisons use the same CSR, converting IDs and
offsets to the Warp process's configured integer widths. No timing is measured.

Local qualification uses Python 3.13.14, Torch 2.13.0+cu130, Warp 1.17.0,
CUDA 13.0 and HIP 7.2.4. Kernel tests compare analytic derivatives through third
order with independent Torch differentiation, including support knots, origin
limits and normalization. Primal tests cover field ranks, dimensions, support
policies and schemes against a dense Torch oracle. Adjoint tests compare with
independent Torch AD and finite differences, including periodic images, aliases,
chained density calculations and mutation checks. CPU and CUDA are exercised.

Separate Warp comparisons cover both precisions. Tensor parity also exposed two
existing shared Warp helper errors: 3D vector gradients flattened outer products
with the axes swapped, and matrix divergence selected the opposite documented
contraction axis. The shared helpers now follow the documented field-axis order;
this changes those affected Warp outputs as well as aligning both backends.

Wheel qualification blocks compiler and Warp imports and runs the numerical
suite against the extracted binary. Symbol auditing rejects direct ATen/c10 and
CPython dependencies. A plain editable install was checked in a disposable
environment. No operator or timestep performance measurements are made.

Standalone CPU qualification:

```bash
c++ -std=c++17 -Isrc/warpSPHCore/native/csrc -Isrc/warpSPHCore \
  scripts/diagnostics/native_kernel_smoke.cpp -o /tmp/native-cpu-smoke
/tmp/native-cpu-smoke
```

## HIP toolchains and NVIDIA

HIP can compile for NVIDIA through nvcc. This checks HIP source compatibility;
it does not execute the AMD runtime or qualify an AMD Torch extension.

The local `warpsph-native-hip` Conda environment was installed with
`hipcc=7.2.4`, `hip-devel=7.2.4` and Python 3.13 from conda-forge. That package
omits the NVIDIA `nvidia_detail` headers. The isolated environment is supplemented
with the official `ROCm/hipother` tag `rocm-7.2.4`, commit
`e56867449451fa4e12d068fedf502423a8d29191`; its source record and license are
under that environment's `share/hipother/`. Its `bin/hipcc-nvidia` wrapper selects
`HIP_PLATFORM=nvidia`, supplies the HIP include directory and uses the existing
CUDA toolkit. The pre-existing `warp` environment was not modified.

```bash
# With HIP/NVIDIA and CUDA toolkit paths configured:
hipcc-nvidia -x cu -std=c++17 -arch=sm_120 -DSPH_SMOKE_HIP \
  -Isrc/warpSPHCore/native/csrc -Isrc/warpSPHCore scripts/diagnostics/native_kernel_smoke.cpp \
  -o /tmp/native-hip-smoke
# Ensure the CUDA runtime library directory is available to the dynamic loader.
/tmp/native-hip-smoke

# Compile for AMD without requiring an AMD GPU on the build host:
HIP_PLATFORM=amd hipcc -x hip --offload-arch=gfx1100 -std=c++17 \
  -DSPH_SMOKE_HIP -Isrc/warpSPHCore/native/csrc -Isrc/warpSPHCore \
  -c scripts/diagnostics/native_kernel_smoke.cpp -o /tmp/native-amd-smoke.o
```

The shared row smoke checks float32/float64 and 1D/2D/3D. Local qualification
includes CPU execution, CUDA Torch execution, HIP-on-NVIDIA execution and an
offline AMD gfx1100/gfx942 compilation. After stable-ABI migration, all 16
GPU adapter translation units (primals, adjoints, kernel evaluation and search)
compiled offline for both AMD targets,
using the installed Torch stable headers and official matching rocThrust/rocPRIM
headers (with their release version headers generated). This is compilation
qualification, not a linked ROCm Torch wheel. A ROCm Torch build and real AMD hardware are
still required to qualify the complete AMD extension and its runtime behavior.
AMD support remains experimental; real-device access is not a release gate for
this backend. Repeat the offline ROCm compile check when shared kernels change.

The expanded port passed 2,536 numerical/layout/loader checks against the CUDA
wheel and 1,288 applicable checks against the CPU wheel, with compiler and Warp
imports blocked. Both wheels passed the stable-ABI symbol audit. Float64 primal
parity and module-layout checks passed 398 cases (41 unsupported/native-only
cases skipped). Float32 primal and kernel parity passed 482 cases (40 skipped);
separate float64 kernel parity passed 90 cases. The Warp
axis-helper and existing geometry-JVP regressions passed 64 cases; focused field,
stream, neighbor-lane and dual-wrapper regressions passed another 56. The final
source/JIT suite passed 2,537 checks; empty-population VJPs and a nondefault CUDA
stream are covered.

## Fixed-geometry field products for implicit solves

```python
from warpSPHCore.native import CSRTopology, linearizeFields

# Share this object across operators on one topology. The native CSR transpose
# is constructed lazily on the first reference pullback, then reused.
topology = CSRTopology(adjacency)
linear = linearizeFields(
    query, properties, domain, query_values, reference_values,
    referenceParticles=reference, adjacency=topology,
    referenceVolumes=volumes, crkState=coefficients)
value = linear.value
jv = linear.jvp(query=query_direction, reference=reference_direction)
query_bar, reference_bar = linear.vjp(output_cotangent)
# Evaluate other fields and use ordinary first-order Torch backwards:
value = linear.apply(new_query_values, new_reference_values)
```

This API supports interpolation, gradient, divergence, curl and Laplacian,
including their supported volume, CRK, grad-h, renormalization and consistent
divergence options. All 15 kernels, dimensions 1–3, both precisions, field ranks
and mixed CSR integer widths follow the primal contracts. Density and covariance
have no differentiated field input and are excluded. `jvp` and `vjp` are numerical
products without a derivative graph. `apply` supports first-order Torch backwards;
higher-order derivatives and `torch.func`/dual-tensor integration are not supplied.

The map is linear in the supplied fields. Geometry, masses, densities, volumes,
CRK coefficients and their spatial derivatives, grad-h and renormalization are
fixed coefficients. This is a partial derivative, not differentiation through
CRK preparation or a moving-particle residual. Construction rejects differentiable
fixed coefficients unless called inside `torch.no_grad()` to freeze them explicitly.
Changes to a borrowed fixed tensor or CSR array invalidate existing maps. Replacing
an external state attribute does not update an existing map; construct a new map
for the new state. Version tracking requires ordinary versioned tensors, including
when constructing under `no_grad`; inference-mode tensors must be cloned outside
`torch.inference_mode()` first.

Query and reference roles are independent. An omitted `jvp` role is exactly zero
and uses a native null pointer. If one physical field occupies both roles, pass
its tangent to both roles and sum the two VJP results. Interpolation has no query
field dependency, so its query cotangent is `None`. If `referenceValues` was omitted
at construction, `apply(field)` uses that field in both roles; Torch autograd then
adds their gradients for the shared tensor. `vjp(..., query=False)` or
`reference=False` omits that output and its ownership pass. A query-only VJP does
not construct CSC. No primal field values, edge Jacobians or solver trajectory
are needed for these pullbacks; borrowed exemplar tensors retain shape metadata.

The shared traversal owns query cotangents through CSR and reference cotangents
through CSC. Both evaluate the original directed i,j interaction, preserving
query corrections, asymmetric supports, masks and density weights. No floating
atomics are used. Construction uses integer count/fill atomics, a device prefix
sum and sorted CSC rows; no CSR value validation scan or GPU-to-CPU readback is
introduced. Prefixes, gaps, duplicate edges and unused capacity are supported.
Transpose capacity equals CSR ID capacity. Reference VJPs therefore retain an
additional int64 ID buffer plus int64 offsets/int32 counts until the topology is
released. Cached CSC tensors are version guarded; cross-stream reuse waits on its
construction event and records the consuming stream. Usual Torch stream ordering
for the caller's input arrays is still required. A `CSRTopology` is not a mutable
multi-threaded topology manager; finish changing the graph before linearizing.

Sorted CSC rows and owned reductions give repeatable execution on a fixed backend
and launch layout; CPU/GPU or precision changes need not be bitwise identical.
Uncorrected `warpOperation` reverse mode retains its prior scatter-atomic behavior.
These products do not add graph-capture or compilation support. There is no
runtime CSR validation mode or debug flag.

Run `scripts/examples/native_field_implicit.py --device cpu` (or `cuda`) with the
native backend selected. It solves a nonlinear residual with Newton/GMRES Jv
products, then solves the transposed implicit adjoint at the converged state.
Independent dense equations are used only afterwards for verification. See the
[field-linearization qualification](../regression/native_field_linearization_2026_11_02.md)
for test coverage, distribution checks and remaining integration work.

## Complete corrected and preparation derivatives

Ordinary corrected calls now support first-order Torch backwards and forward AD:

```python
query.positions.requires_grad_()
preparation = prepareCRK(query, domain, kernel, referenceState=reference)
value = warpOperation(
    query, properties, domain, query_values, reference_values,
    referenceParticles=reference, referenceVolumes=preparation.referenceVolumes,
    crkState=preparation.crkState)
position_gradient, = torch.autograd.grad(value.sum(), query.positions)

# Other input tensors can be made dual in the same way, including supports,
# volumes, CRK A/B/gradA/gradB, omegas and renormalization matrices.
with torch.autograd.forward_ad.dual_level():
    dual_positions = torch.autograd.forward_ad.make_dual(positions, direction)
    # Build a ParticleState with dual_positions and call preparation/operators.
```

These derivatives include query/reference fields, explicit or fallback volumes,
masses/densities used by the chosen weighting, CRK A/B/gradA/gradB, both applicable
grad-h roles, query renormalization, positions/supports and periodic domain bounds.
Unused inputs retain their primal meaning: interpolation ignores query fields and
spatial CRK coefficients, covariance ignores fields/grad-h, and Naive Laplacian
ignores CRK/renormalization. Field-only requests reuse the existing optimized field
policies. Geometry requests support the same 11 smooth-origin kernels as ordinary
native reverse mode; coefficient and field requests support all 15 kernels.
Density continues using its existing uncorrected derivative path.

The common bridge is `native/_differential.py`; native pair preparation, directional
products and analytic pullbacks live under `coreOperations/native/csrc_differential`.
JVP uses a first-order scalar direction inside each pair; there is no tape or
custom C++ graph across operations. Manual VJP uses CSR query ownership and CSC
reference ownership. Per-query domain-bound contributions are reduced with Torch.
No floating scatter atomics or per-edge Jacobians are used by these pair passes.
The corrected primal operator policies remain the reference for their equations.

CRK volume saves its scalar kernel sum per query for reciprocal finalization;
consistency density saves numerator/denominator for ratio finalization. Moment
VJPs consume cotangents of the six existing moment tensors. The factor solve uses
its existing Torch algebra and active limiter branch. Preparation shares one query
`CSRTopology` across volume, moments and diagnostic passes, with a separate reference
self-topology when required. The backward bridge saves borrowed inputs with Torch
version checks; mutating geometry, coefficients or CSR before backwards is rejected.

Topology, categories, positive-weight counts, periodic winding, support-selection
branches and limiter masks are discrete. At a max-support tie the native query
branch is retained. Gradient-based Laplacians retain the exact-self zero branch
and its zero branch derivative, including with nonzero CRK gradients. Geometry at
force-kernel cusps, branch transitions or limiter/rank changes is not covered by a
smooth derivative claim. Inactive CRK identity-limiter outputs have zero derivative.
Higher-order differentiation, HVP and `torch.func` transforms are unsupported;
backwards is once-differentiable, and JVPs are first-order numerical products.
Reverse-over-JVP and dual inputs to backwards are rejected explicitly, so the
Torch factor algebra cannot hide missing native higher-order terms.

This extends the corrected dispatch and CRK APIs. Uncorrected calls retain the
legacy reverse bridge and its forward-AD limitations. The fixed-geometry
`linearizeFields` API deliberately still freezes geometry/corrections for repeated
solver products. Use `torch.no_grad()` when only a primal result is required.

Run `scripts/examples/native_crk_autograd.py --device cpu` (or `cuda`). It computes
a nonlinear-field loss through complete preparation and a corrected gradient,
then checks the JVP/VJP identity and finite differences. See the
[corrected AD qualification](../regression/native_corrected_ad_2026_11_02.md).

## Core completion order

Frontend ports stay in a separate worktree until these core stages are closed.
The research bundles have moved to their independent source repository; builds
and tests do not require that checkout or the external evidence archive.

| Order | Core work | State |
|---|---|---|
| 1 | Verlet lifecycle/capacity reuse; standard operator options | Implemented and qualified on CPU/CUDA; new GPU paths compile for experimental HIP |
| 2 | Renormalization, small matrices, solver dependencies | Implemented and qualified on CPU/CUDA; new GPU paths compile for experimental HIP |
| 3 | Polynomial reconstruction and MFM/MFV | Next |
| 4 | Chunked/owned-row search and native operator traversal | Pending |
| 5 | Remaining derivative and public API parity | Pending; includes uncorrected JVP and existing density-HVP scope |
| 6 | Final distribution/merge qualification | Pending; no AMD hardware gate |

Higher-order and nonsmooth derivative claims need explicit numerical contracts;
a successful primal port does not imply those derivatives. Reusable capture/
compiler integration and public release automation retain their separate scope.

### Native Verlet and operator options

The native backend exports `buildVerletList`, `filterVerletList`,
`verletStaleFlag`, `verletUsage` and `deferVerletChecks`. Motion/support reductions
and candidate filtering share CPU/CUDA/HIP numerical headers. Validity flags stay
on the current Torch stream; eager rebuilding reads one decision scalar and
variable-size collection reads its edge count. No CSR-content verification or
particle readback is introduced. Capacity headroom, CSR-only storage, optional
COO rows, snapshot ownership and opt-in in-place buffers follow the core policy.
Support-policy changes and versioned domain changes conservatively rebuild.
Unversioned inference-mode domain tensors conservatively rebuild as well.

Native filtering preserves Warp's inclusive cutoff (`distance <= support`);
fresh radius search retains its strict cutoff. Tensor-supported Verlet scales
must satisfy the caller's usual scale >= 1 contract. Variable-sized native
build/filter remains outside CUDA graph capture; deferred flags do not add a
claim of a capture-compatible complete search lifecycle. Warp grid factories
and Warp-specific collection/query-order tuning are not native execution modes.

`calibrateNormalization` now applies to every standard native operator, including
corrected operations, first-order derivatives and fixed-geometry field maps.
The cached coefficient uses shipped native kernel equations in a Warp-free
process. Positive-divergence Laplacians support primals and active-branch
first-order AD. They retain Warp's matrix behavior (only the first `D` flattened
channels are written); scalar fields are admitted only in 1D. The clipped field
map is nonlinear, so use ordinary JVP/VJP rather than `linearizeFields`.

See the [lifecycle/options qualification](../regression/native_core_lifecycle_2026_11_03.md).

## Performance comparison

The [corrected 2026-10-30 comparison](../regression/native_backend_production_performance_2026_10_30.md)
uses the same normal execution policy for Warp/Torch, native/Torch and standalone
C++: no particle-content or CSR verification inside operation calls. It covers
fresh search and all seven primals on CPU/CUDA in 1D/2D/3D through one million particles.
Standalone allocation and output reuse are measured separately. The earlier
comparison included inappropriate per-call verification and is superseded.

Build and run instructions are in
[`scripts/benchmarks/native/README.md`](../../scripts/benchmarks/native/README.md).
Raw results and scaling plots live in the external development archive at
`core/experiments/native-backend-20261030-million/`.

The [CPU worker-count follow-up](../regression/native_backend_cpu_threads_2026_10_30.md)
compares 4, 8, 16 and 24 workers on the 24-core local machine, with controlled
CPU placements. The earlier four-worker setting is a benchmark choice, not a
runtime limit. Its graph, raw samples and topology manifest live in
`core/experiments/native-backend-20261030-cpu-threads/`.

The paired FP32/FP64 interpolation follow-up uses a common million-particle 2D
CSR and CPU counts of 16/24 workers. Its graph and measurements are in
`core/experiments/native-backend-20261030-float32/`; reproduction uses
`scripts/benchmarks/bench_interpolation_precision.py`. The timing-only bridge
supports both precisions and checks its private descriptor ABI before execution.


The [2026-10-31 optimization pass](../regression/native_backend_optimization_2026_10_31.md)
keeps Warp as the readable reference and optimizes native/Torch scheduling,
prepared kernel math, periodic geometry and compact-hash sorting. In the measured
million-particle 2D case, FP64 GPU primals improve by 2.36–2.96× over the saved
checkpoint and GPU fresh search by 2.83×. FP32 GPU interpolation is essentially
unchanged, and CPU fresh search needs further work. The report records numerical
qualification, including FP32 cutoff rounding, and retains all timing ranges.
Graphs and raw evidence live in `core/experiments/native-performance-20261031/`.


The [FP32 GPU exploration](../regression/native_backend_fp32_2026_10_31.md)
profiles and reduces repeated reference-side gathers with aligned packed records,
and uses four-lane, cell-ordered radius search. At one million particles in 2D,
FP32 native/Torch primals improve 3.14–5.24× over `0560de5`; interpolation takes
0.332 ms versus 1.487 ms before and 1.493 ms for Warp. Fresh search improves
6.15 → 2.78 ms. Packing applies to the common Wendland2/Gather path from 32K
queries upward, adding a preparation launch and typically 16–32 MB of scratch
per million references. Smaller and other supported cases retain the existing
path. FP64 SPH and CPU implementations remain unchanged. The report covers
limits, independent primal/VJP tests, offline HIP compilation and raw evidence.


The [cell-to-cell interpolation study](../regression/native_backend_cell_tiles_2026_10_31.md)
implements a separate Morton-sorted CUDA toy with bounded tasks and shared-memory
reference tiles. At 1M particles and n_h=4, FP32 interpolation improves 1.58× in
2D and 1.47× in 3D over already-sorted native CSR. Fresh 3D planning plus
interpolation improves much more by avoiding particle CSR construction; the
report separates that result from operator throughput. Sparse neighborhoods,
1D and FP64 often favor CSR. Production kernels are unchanged. The primary
benchmark now defaults to n_h=4, with n_h=2.57 as comparison and --neighbors 32
for reproducing historical runs.

### Full simulation stage profiling

The benchmark-only `--stage-profile` option on
`scripts/benchmarks/bench_simulations.py` uses the frontend's existing numbered
gates and actual Warp/native dispatch to partition complete step time. It keeps
mixed CRK stages explicit and adds no instrumentation to production simulation
code. `plot_simulation_stages.py` creates stacked comparisons: colors identify
simulation aspects, diagonal hatching identifies remaining Warp work, and dots
identify scaffolding/Torch/dispatch. Black diamonds show matched ordinary runs
because instrumentation itself adds cost. See the
[stage attribution report](../regression/native_frontend_stages_2026_11_03.md)
for scope, raw evidence and reproduction.

### Native dense helpers and solvers

The native root exports the core `linalg` API and
`computeRenormalizationMatrices` / `computeRenormalizationMatricesJVP` without
importing Warp. Dense inverse/application, small-block preconditioning and
CG/BiCGStab workspace updates share compiled CPU/CUDA/HIP numerical headers.
Solver control flow remains shared with Warp; reductions use Torch in a native
process. Captured workspace chunks avoid device-wide synchronization, while
outer convergence checkpoints and true-residual verification retain host reads.
The workspace path remains inference-only. Ordinary solvers and dense spectral
AD retain their Torch implementations. See the
[dense and solver qualification](../regression/native_dense_solvers_2026_11_03.md)
for numerical policies, derivative scope and verification.
