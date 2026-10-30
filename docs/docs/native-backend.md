---
title: Experimental native backend
---

# Experimental native backend

The native backend builds neighbourhoods, evaluates all 15 radial kernel families,
and runs the seven standard SPH operators without installing or importing Warp.
It includes manual first-order adjoints for smooth kernels. CPU, CUDA and HIP
share the same numerical headers; Warp remains the default backend. Corrections,
higher-order derivatives and complete frontend simulations remain unported.

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

# Standalone CUDA build (includes CPU execution):
cmake -S . -B build/standalone-cuda \
  -DWARPSPHCORE_BUILD_STANDALONE=ON -DWARPSPHCORE_NATIVE_BACKEND=cuda \
  -DCMAKE_CUDA_ARCHITECTURES='90-real;120-real' -DCMAKE_BUILD_TYPE=Release

# Standalone ROCm build, without an AMD GPU on the build host:
cmake -S . -B build/standalone-hip \
  -DWARPSPHCORE_BUILD_STANDALONE=ON -DWARPSPHCORE_NATIVE_BACKEND=hip \
  -DCMAKE_HIP_ARCHITECTURES='gfx942;gfx1100' -DCMAKE_BUILD_TYPE=Release
```

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
caller stream, CPU thread count and validation mode. Each GPU library includes
CPU execution and its selected GPU backend. `supports(device)` reports build
capability; it does not probe installed hardware. CPU buffers must be host
accessible; GPU buffers must be accessible on the selected device.

`Validation::Contents` is the default and checks finite/valid geometry, positive
densities, fields and live CSR indices. These checks can copy GPU data to the
host and synchronize the supplied stream. `Validation::Metadata` checks sizes,
shapes, options and output/input overlap, assuming the contents have already
been validated. It allows primal/VJP work to be enqueued without a final host
synchronize. Callers must revalidate after changing geometry, densities or CSR
contents. Numerical outputs must not overlap their input storage.

For the existing all-pairs search, call `radius_count`, then `radius_scan` to
produce int64 offsets and return the required edge capacity, allocate int64 IDs,
and call `radius_collect`. Counts and collection run on the selected device.
The initial scan uses a host prefix scan (including a counts download and offsets
upload on GPU). Scan and collection synchronize; collection rejects stale counts
and bounds its writes before reporting a mismatch. This provides correctness and
standalone capability; scalable device search/scan remains future work.

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
WARPSPHCORE_BACKEND=native \
WARPSPHCORE_STANDALONE_LIBRARY=build/standalone-cuda/libnative_standalone_test.so \
  pytest tests/native/test_standalone.py -q
```

Local qualification built and ran the standalone CPU and CUDA executables and an
external installed-package consumer. Link dependencies contained no Torch,
CPython or Warp libraries. The standalone ROCm library also compiled and linked
for both gfx942 and gfx1100; actual AMD execution remains unqualified. CPU/GPU
primal and explicit-adjoint tests cover all operator schemes, field ranks,
precisions, kernels, support policies, CSR widths, empty populations, explicit
accumulation, aliasing, bounds errors and caller streams. Parallel CPU tests use
both one and four workers with shared reference-gradient scatters. The final
standalone matrix passed 1,512 CPU/CUDA cases (756 applicable to the CPU-only
build). The existing native Torch suite still passed 2,537 cases; its rebuilt
CPU wheel passed 1,288 compiler-free checks and the stable-ABI audit. The ROCm
consumer executed the CPU path successfully and skipped its unavailable AMD
GPU test. No performance measurements were made.

## Supported contracts

| Capability | Scope |
|---|---|
| Execution | Parallel CPU; CUDA; experimental HIP (ROCm compilation gate) |
| Spatial dimensions | 1, 2, 3 |
| Floating-point types | float32 and float64, selected from tensors at runtime |
| Neighbours | Exact all-pairs radius search, canonical CSR, ascending reference IDs |
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
When adjacency is omitted, the native all-pairs search builds it. Common
`ParticleState`, `DomainDescription`, `OperationProperties`, `CapacityPolicy`
and `AdjacencyList` definitions are shared with the Warp backend; they are not
separate native copies. Their established dataTypes imports remain available.

Search is **O(Nquery * Nreference)** and intended for small correctness fixtures.
Scalable spatial indexing and Verlet maintenance remain future work. Native
search emits int64 IDs/offsets regardless of Warp import-time type settings.
CSR tensors supplied by another producer may use either integer width.

Corrections (including volume overrides, CRK, grad-h, renormalization, calibration,
positive divergence and consistent divergence), covariance counts, partial support
policies, capacity/rebuild reuse and graph capture are unsupported. Dot Laplacian
requires a vector or matrix field outside 1D. Unsupported requests raise explicitly.

`warpSPHCore.kernels.evaluateKernel(displacements, supports, kernel)` returns value,
gradient, Hessian, support derivative, gradient-support derivative, regularized
Laplacian and radial metadata. This primitive is forward-only as a Torch operation;
its explicit derivative outputs are available to callers and shared operator adjoints.
Smooth kernels use analytic origin limits and a near-origin derivative expansion.
Cusp/singular force-kernel Hessians may be undefined at the origin. The legacy
Laplacian estimator retains its regularization and zero result below its epsilon.

C++ checks metadata, finite geometry, support/density validity, CSR offsets,
capacity and live indices. Conservative content checks and exact-sized search
allocation read device results on the host. Therefore native execution does not yet
provide the existing Warp launch path's no-host-sync or capture contract.
Future prepared execution must amortize validation without overlooking tensor
mutation or storage replacement. No performance claim is made for this pass.

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
      wrapper.py                 # existing Warp compact hash or native all-pairs
      nt_radius.py
      sources.json
      csrc_radius/
        nt_radius_forward.h      # shared count/collect row implementations
        nt_radius_forward.cpp
        nt_radius_forward.cu
        nt_radius_dispatch.h
```

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

For CRK, the intended ownership is `crk/native/{wrapper.py, wp_*.py, nt_*.py,
sources.json}`, with compiled sources under `crk/native/csrc_<function>/`. Its existing
high-level correction pipeline should call exports from `crk.native`. CRK
implementations have not been moved or ported.

Tensor ownership stays with Torch. GPU execution uses Torch's current stream
and device guard. `scripts/diagnostics/native_kernel_smoke.cpp` includes the
module-local primal headers directly and builds them without Torch.

## Forward, backward and adjoint file boundaries

The implementation uses one primal calculation for inference and a grad-enabled
forward. A Torch `autograd.Function` bridge saves inputs with version checks,
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

All seven operators follow this layout. Gradients cover positions, supports,
reference masses, relevant query/reference densities and fields, and periodic
domain bounds. Aliased query/reference roles return separate contributions for
Torch to accumulate. Radius search runs outside the differentiation graph;
neighbour membership, periodic winding and cutoff transitions are discrete.

Reverse mode supports Wendland2/4/6, cubic/quartic/quintic, B7/B8, Gaussian,
Poly6 and Cohesion. HOCT4, Spiky, Viscosity and Adhesion reverse mode remains
explicitly unsupported pending nonsmooth/singular contracts. JVP, HVP and higher
orders are unsupported; the backward bridge is once-differentiable. Reference
scatter uses CPU atomic compare/exchange or GPU atomic addition. Strict Torch
deterministic mode rejects this path; warn-only mode warns and executes.

## Correctness qualification

Run native tests and Warp reference tests in separate processes. Warp precision
must be selected before importing its types.

```bash
WARPSPHCORE_BACKEND=native pytest tests/native/ -q
WARPSPHCORE_NATIVE_PARITY=1 pytest tests/native/test_warp_parity.py tests/native/test_kernel_warp_parity.py tests/native/test_primal_warp_parity.py -q
warpSPHCore_PRECISION=float64 WARPSPHCORE_NATIVE_PARITY=1 \
  pytest tests/native/test_warp_parity.py tests/native/test_kernel_warp_parity.py tests/native/test_primal_warp_parity.py -q
```

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

## Remaining milestones

1. Scalable native spatial search and Verlet/capacity lifecycle.
2. Corrections, remaining field contracts and nonsmooth kernel adjoints.
3. JVP, reverse-over-JVP and explicit supported HVP contracts.
4. Prepared validation/workspaces, capture and compiler integration.
5. Public manylinux wheel release automation; AMD runtime qualification when hardware is available.
6. Frontend-owned Warp kernels needed by a selected complete simulation.

Performance work follows those capability and correctness gates. Concurrent
simulation use makes isolated operator timing unsuitable for this pass.
