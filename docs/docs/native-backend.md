---
title: Experimental native backend
---

# Experimental native backend

The native backend is a correctness-first vertical slice of the core. It can
build a neighbourhood and evaluate density and a scalar-field gradient without
installing or importing Warp. The existing Warp backend remains the default.
This is not yet full operator, derivative or frontend simulation parity.

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
no direct ATen/c10 C++ or Python C-API dependencies. Forward-only AD checks stay
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

## Supported contracts

| Capability | Scope |
|---|---|
| Execution | Parallel CPU; CUDA; experimental HIP (ROCm compilation gate) |
| Spatial dimensions | 1, 2, 3 |
| Floating-point types | float32 and float64, selected from tensors at runtime |
| Neighbours | Exact all-pairs radius search, canonical CSR, ascending reference IDs |
| Density | Wendland2 mass-weighted kernel sum |
| Gradient | Wendland2 Difference scheme, scalar values, output `[N,D]` |
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

Autograd, dual tensors, other kernel functions, vector/tensor-valued gradients,
corrections, other operations, capacity/rebuild reuse and graph capture are not
implemented. Unsupported requests raise rather than importing/falling back to
Warp. `torch.no_grad()` may be used for inference with grad-requiring inputs.

C++ checks metadata, finite geometry, support/density validity, CSR offsets,
capacity and live indices. Conservative content checks and exact-sized search
allocation read device results on the host. Therefore this pilot does not yet
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
          wendland2.h
      execution/                 # CPU/GPU scheduling, type dispatch and CSR traversal
      torch/                     # reusable native tensor validation
      extension.cpp              # shared library capability registration
    _inputs.py                   # shared Python geometry/CSR/tensor preparation
  coreOperations/
    native/
      __init__.py                # common computeDensity/computeGradient exports
      wrapper.py                 # select Warp or native implementation
      wp_density.py              # Warp implementation, including tape support
      wp_gradient.py
      nt_density.py              # native density Python/AD boundary
      nt_gradient.py
      sources.json               # this module's CPU/GPU translation units
      csrc_density/
        nt_density_forward.h     # shared CPU/CUDA/HIP primal row implementation
        nt_density_forward.cpp   # checks, CPU launch and Torch registration
        nt_density_forward.cu    # thin GPU launch adapter
      csrc_gradient/
        nt_gradient_forward.h
        nt_gradient_forward.cpp
        nt_gradient_forward.cu
    wp_density.py                # old import path aliases the colocated module
    wp_gradient.py
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
implementations have not been moved or ported in this pilot.

Tensor ownership stays with Torch. GPU execution uses Torch's current stream
and device guard. `scripts/diagnostics/native_kernel_smoke.cpp` includes the
module-local primal headers directly and builds them without Torch.

## Forward, backward and adjoint file boundaries

`nt_density.py` is the native Python entry point. It owns dispatch, capability
checks, input preparation and the eventual autograd interface. Its primal
calculation lives in `csrc_density/nt_density_forward.*`, with numerical code written once
in the header and CPU/GPU launch adapters in `.cpp`/`.cu`.

The same primal calculation should serve inference and a grad-enabled forward
call. The latter adds the bridge/context required to save inputs; it does not
need a second copy of the forward mathematics. The current pilot still rejects
AD, rather than implying that the manual derivative implementation exists.

Future reverse-mode files have separate responsibilities:

| File family | Intended responsibility |
|---|---|
| `nt_density.py` | Python entry, capability checks, selecting inference or an AD bridge |
| `csrc_density/nt_density_forward.*` | Primal density calculation used by either forward route |
| `nt_density_backward.py` | Torch-facing reverse orchestration: saved tensors, requested input gradients, calling adjoint operators and returning results |
| `csrc_density/nt_density_adjoint.h/.cpp/.cu` | Numerical VJP: accumulate derivatives of the primal with an incoming output cotangent |
| `csrc_density/nt_density_jvp.*` | Forward-mode tangent evaluation, when implemented |

Backward and adjoint here describe two layers of the same reverse-mode pass;
there should not be two independent implementations of its derivative formula.
The Torch integration mechanism (registered custom operators versus an autograd
Function bridge) still needs deliberation. Small orchestration may stay in the
entry file until it warrants a separate backward file. No empty derivative
files are added just to reserve these names.

The manual adjoint design must handle aliased query/reference roles, saved
input version checks, requested-gradient pruning, self-pair limits and the
currently supported higher-order contracts. Forward row ownership alone is
insufficient: reference gradients can receive contributions from many query
rows, requiring a deliberate CPU reduction/GPU accumulation policy. Adjoint
pair mathematics should remain shared across targets, with accumulation policy
in the execution layer. Kernel derivatives belong beside the common kernel
functions. Neighbour topology and cutoff/rank transitions retain their explicit
nondifferentiable or nonsmooth contracts.

## Correctness qualification

Run native tests and Warp reference tests in separate processes. Warp precision
must be selected before importing its types.

```bash
WARPSPHCORE_BACKEND=native pytest tests/native/test_native.py tests/native/test_layout.py tests/native/test_build.py -q
WARPSPHCORE_NATIVE_PARITY=1 pytest tests/native/test_warp_parity.py -q
warpSPHCore_PRECISION=float64 WARPSPHCORE_NATIVE_PARITY=1 \
  pytest tests/native/test_warp_parity.py -q
```

Native tests use an independent Torch all-pairs oracle and check malformed
inputs, empty sets, directions, mixed index widths, padding and stream use. A
fresh subprocess installs an import blocker for `warp` and runs the complete
native example. Direct Warp comparisons use the same CSR, converting IDs and
offsets to the Warp process's configured integer widths. No timing is measured.

Local qualification on 30 October 2026 used Python 3.13.14, Torch
2.13.0+cu130, Warp 1.17.0 for reference comparisons, CUDA 13.0 and HIP 7.2.4.
The numerical gates passed 232 native CPU/CUDA checks, 90 cross-backend cases
per precision, and 42 focused existing package/type/state/operator regressions.
The built wheel's native example also passed in an isolated interpreter with
Warp imports blocked. The module-layout pass also checks lazy wrapper imports, old Warp module
aliases, nested frontend manifest discovery and packaged source ownership.
The stable-ABI migration passed 241 native numerical/layout/loader checks,
plus a private HIP-staging check. The packaged CUDA suite passed with JIT and
Warp imports blocked; the CPU wheel also passed its applicable numerical suite.
Both wheels passed the symbol audit for direct ATen/c10/CPython dependencies.
A plain editable installation was checked in a disposable environment.
No operator or timestep performance measurements were made.

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
offline AMD gfx1100/gfx942 compilation. After stable-ABI migration, all three
full GPU adapter translation units also compiled offline for both AMD targets,
using the installed Torch stable headers and official matching rocThrust/rocPRIM
headers (with their release version headers generated). This is compilation
qualification, not a linked ROCm Torch wheel. A ROCm Torch build and real AMD hardware are
still required to qualify the complete AMD extension and its runtime behavior.
AMD support remains experimental; real-device access is not a release gate for
this pilot. Repeat the offline ROCm compile check when shared kernels change.

## Remaining milestones

1. Scalable native spatial search and Verlet/capacity lifecycle.
2. Remaining operators, field ranks, kernels and corrections.
3. VJP, JVP, reverse-over-JVP and explicit supported HVP contracts.
4. Prepared validation/workspaces, capture and compiler integration.
5. Public manylinux wheel release automation; AMD runtime qualification when hardware is available.
6. Frontend-owned Warp kernels needed by a selected complete simulation.

Performance work follows those capability and correctness gates. Concurrent
simulation use makes isolated operator timing unsuitable for this pass.
