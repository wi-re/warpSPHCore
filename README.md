# warpSPHCore

warpSPHCore is a Warp- and PyTorch-based backend for core Smoothed Particle Hydrodynamics (SPH) operations: GPU-accelerated compact-hash neighbor search plus differentiable density, gradient, divergence, curl, Laplacian, and interpolation operators, with CRK (corrected reproducing kernel) and gradient-renormalization corrections.

**Full reference documentation, with the equations, options, and JVP scope for every operator, lives at [fluids.dev/warpSPHCore](https://fluids.dev/warpSPHCore/).** This README covers install and a minimal working example; the wiki is the source of truth for everything else.

## Highlights

- Compact-hash radius search for particle neighborhoods (adjacency-list or on-the-fly grid traversal)
- One entry point, `warpOperation`, dispatching to all seven operators via `OperationProperties`
- Multiple support, gradient, and Laplacian schemes
- CRK correction and gradient-renormalization utilities for linear-reproducing consistency
- Reverse-mode gradients and forward-mode JVPs (value and geometry tangents) through the same `torch.autograd.Function` bridge
- PyTorch tensor inputs and outputs, with Warp kernels under the hood
- Local polynomial derivatives, TENO/WENO interface reconstruction and ideal-gas MFM/MFV rates
- Matrix-free solvers with constraints, nullspaces and diagonal/block preconditioning

## Installation

Install the package in editable mode while developing:

```bash
pip install -e .
```

If you want to run the example notebooks in `notebooks/`, install the notebook extras as well:

```bash
pip install -e ".[notebooks]"
```

For packaging and publishing, install the development extras or at least `build` and `twine`:

```bash
pip install -e ".[dev]"
```

## Requirements

Core package requirements:

- Python 3.10+
- PyTorch
- warp-lang
- NumPy

Python 3.10+ is the package metadata range. CPU CI is configured for Python
3.12; the current local CPU/CUDA qualification uses Python 3.13.14, Torch
2.13.0+cu130, Warp 1.17.0 and NumPy 2.5.2. Older dependency combinations and
every Python version in that metadata range are not individually qualified.
See [qualification and development records](docs/docs/development-artifacts.md).

Notebook and plotting extras used by `notebooks/` (all installed by
`pip install -e ".[notebooks]"`):

- matplotlib, ipywidgets, ipympl — used across the notebook set
- pandas, seaborn, tqdm — used by `warp_profile.ipynb` specifically

GPU notes:

- For GPU execution, use a CUDA-capable PyTorch build and a compatible NVIDIA driver.
- A separate CUDA toolkit installation is not required for this package.
- Warp still needs to be initialized once per process with `wp.init()`.

## Precision and index configuration

Set configuration before importing `warpSPHCore`; types are fixed when Warp
structs and kernels are defined:

```python
import os
os.environ["warpSPHCore_PRECISION"] = "float32"     # or float64
os.environ["warpSPHCore_INDEX_DTYPE"] = "int32"     # particle IDs; or int64
os.environ["warpSPHCore_OFFSET_DTYPE"] = "int32"    # CSR addresses; or int64
import warpSPHCore as sph
```

All three settings are independent. Both integer settings default to int32;
per-particle neighbor counts remain int32, with totals and prefix scans computed
in int64 and checked before narrowing. An optional `warpSPHCore_config` module
can instead define `precision`, `dim`, `index_dtype` and `offset_dtype`, taking
precedence over environment settings. Inspect the active types with
`sph.get_type_config()`.

The current flat-array representation retains Warp's signed-int32 dimension
limit, including neighbor capacity headroom; selecting int64 offsets does not
remove that backend limit. Particle IDs and edge offsets have separate meanings:
total neighbor entries can exceed int32 well before the particle population does.
Torch consumers that require long indices can convert the views returned by
`adjacency.livePairs()`; production edge traversal uses Warp CSR kernels.

## Quickstart

```python
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore import (
    ParticleState, OperationProperties, DomainDescription,
    WarpOperation, KernelFunctions, GradientScheme, SupportScheme,
    OperationDirection,
)
from warpSPHCore.util import generateNeighborTestData

wp.init()
device = "cuda"  # or "cpu"

# An nx^d lattice in [-1, 1]^d with ~55 neighbors per particle.
nx, target_neighbors, dim, periodic = 32, 55, 2, True
positions, supports, n, domain, dx = generateNeighborTestData(
    nx, target_neighbors, dim, periodic, device
)
masses = torch.full_like(supports, dx**dim)
kinds = torch.zeros(n, dtype=torch.int32, device=device)
particles = ParticleState(positions, supports, masses, kinds)

# One entry point for every operator: OperationProperties.operation selects it.
properties = OperationProperties(
    kernel=KernelFunctions.Wendland2,
    operation=WarpOperation.Density,
    supportMode=SupportScheme.Gather,
    operationMode=OperationDirection.AllToAll,
)
rho = sph.warpOperation(particles, properties, domain)
particles.densities = rho

# A gradient of a vector field, Difference scheme.
vec = torch.sin(torch.pi * positions)  # smooth on the periodic domain
properties = OperationProperties(
    kernel=KernelFunctions.Wendland2,
    operation=WarpOperation.Gradient,
    supportMode=SupportScheme.Gather,
    operationMode=OperationDirection.AllToAll,
    gradientMode=GradientScheme.Difference,
)
grad = sph.warpOperation(
    particles, properties, domain,
    queryValues=vec, referenceValues=vec,
)
```

`warpOperation` covers CRK correction (`crkState=`) and gradient renormalization (`renormalizationState=`) the same way — see [Quickstart](https://fluids.dev/warpSPHCore/docs/quickstart) and the [per-operator pages](https://fluids.dev/warpSPHCore/docs/operations/density) on the wiki for corrections, forward/reverse-mode differentiation, and every option.

## Runnable composition examples

After installing the package, run the checked examples in fresh precision processes:

```bash
python scripts/examples/run_all.py --device cpu --precision float32
python scripts/examples/run_all.py --device cpu --precision float64
# With a real CUDA device:
python scripts/examples/run_all.py --device cuda --precision float64
```

The examples cover constrained diffusion, singular graph solves, a vector
GMRES problem with block preconditioning, MINRES/Richardson, corrected
operator derivatives, polynomial/nonlinear reconstruction and MFM/MFV rates
with a short periodic update. Each checks its numerical result and raises
on failure. See [examples and their assumptions](docs/docs/examples.md).
The files live in the repository and can also use an installed wheel.

## Development records

Detailed development reports, raw results and generated plot collections live in
`../warpSPHDevelopment/core/`, outside this checkout. The shared
`warpSPHDevelopment` folder can hold other projects and is backed up manually.
Reproduction code, public docs and curated guide figures remain in Git. See the
[artifact layout and regeneration instructions](docs/docs/development-artifacts.md).

## Common entry points

- `warpSPHCore.warpOperation` / `warpSPHCore.warpOperationJVP`: the state-object entry points (above)
- `warpSPHCore.sphOperation_warp`: the flat-tensor entry point for callers without state objects
- `warpSPHCore.radiusSearchCompactHashMap` / `buildCompactHashMap`: compact-hash neighbor search
- `warpSPHCore.crk.computeCRKFactors`: CRK apparent-volume and correction tensors
- `warpSPHCore.renorm.computeRenormalizationMatrices`: gradient-renormalization matrices
- `warpSPHCore.polyfit.PolyFit` / `Reconstructor`: local polynomial derivatives and nonlinear interface states with rank diagnostics and first-order fallback; see [Reconstruction](docs/docs/reconstruction.md)
- [How the schemes reconstruct a field](docs/docs/field-reconstruction-guide.md): illustrated SPH, CRK, MLS/LABFM and TENO/WENO comparison with reproducible production examples.
- `warpSPHCore.mfm.MeshlessGeometry` / `MeshlessWarp`: conservative MFM/MFV rates, geometric guards and explicit face closure; see [MFM/MFV](docs/docs/mfm.md).
- `warpSPHCore.linalg`: verified CG/BiCG/BiCGStab/GMRES/MINRES and bounded Richardson, restricted systems with Dirichlet lifting, explicit nullspace/compatibility/gauge helpers, diagonal/block preconditioners, optional native CG/BiCGStab workspaces, stationary iteration, and small symmetric pseudoinverses; see [Linear algebra](docs/docs/linalg.md)
- `warpSPHCore.util.generateNeighborTestData`: a regular test lattice with a target neighbor count

See the [API reference](https://fluids.dev/warpSPHCore/docs/api) for the full function list and the [Data types](https://fluids.dev/warpSPHCore/docs/data-types) page for every enum (`WarpOperation`, `KernelFunctions`, `SupportScheme`, `GradientScheme`, `OperationDirection`, ...).

## Repository layout

```
src/warpSPHCore/     the package
tests/               pytest suite (unit tests + the gradcheck/spike script gate)
docs/                the Docusaurus wiki source (fluids.dev/warpSPHCore)
notebooks/           example notebooks + their shared helper modules (demo_util.py, profile_util.py)
scripts/
  gradcheck/         torch.autograd.gradcheck canary scripts, one per operator (run by tests/operations/test_gradcheck_scripts.py)
  spikes/            forward-mode JVP tier spikes (exploratory, gated the same way as gradcheck)
  diagnostics/       operation_matrix.py (forward-value sweep) and other ad hoc measurement tools
  benchmarks/        call-overhead and real-workload performance benchmarks
  repro/             standalone bug-reproduction scripts (illustrative, not gated)
  packaging/         PyPI publish helpers
```

The `gradcheck` and `operation-matrix` project skills (see `.claude/skills/`) document how and when to run the `scripts/gradcheck/` and `scripts/diagnostics/operation_matrix.py` gates.

## Publishing to PyPI

- `scripts/packaging/setup_pypi_token.sh`: stores a PyPI API token in `~/.pypirc`
- `scripts/packaging/publish_pypi.sh`: builds, validates, and uploads the package

Typical release flow:

```bash
bash scripts/packaging/setup_pypi_token.sh pypi
bash scripts/packaging/publish_pypi.sh
```

For a TestPyPI dry run:

```bash
bash scripts/packaging/setup_pypi_token.sh testpypi
bash scripts/packaging/publish_pypi.sh --testpypi
```

Before publishing, bump the package version in both `pyproject.toml` and `src/warpSPHCore/__init__.py`. The publish script checks that these two versions match and stops if they do not.

## Testing

```bash
pytest tests/
```

On a CUDA host, `python scripts/diagnostics/run_cuda_gate.py` checks linalg,
native pinv streams/graphs, neighbor rebuilds and tiled operators in separate
float32 and float64 processes. It fails when CUDA is unavailable.

This includes `tests/operations/test_gradcheck_scripts.py`, which runs every script under `scripts/gradcheck/` (plus the Tier-1/Tier-2 forward-mode spikes under `scripts/spikes/`) as a subprocess, since `warpSPHCore_PRECISION` is baked into every compiled kernel at first import and cannot change mid-process. For the forward-value diagnostic matrix and a quick smoke sweep, see `scripts/diagnostics/run_operation_matrix_sweep.sh --quick`.
