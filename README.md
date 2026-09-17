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

Notebook and plotting extras used by `notebooks/` (all installed by
`pip install -e ".[notebooks]"`):

- matplotlib, ipywidgets, ipympl — used across the notebook set
- pandas, seaborn, tqdm — used by `warp_profile.ipynb` specifically

GPU notes:

- For GPU execution, use a CUDA-capable PyTorch build and a compatible NVIDIA driver.
- A separate CUDA toolkit installation is not required for this package.
- Warp still needs to be initialized once per process with `wp.init()`.

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
masses = torch.full((n,), 1.0, device=device)
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

# A gradient of a vector field, Difference scheme.
vec = torch.stack([positions[:, 0], positions[:, 1]], dim=1)
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

## Common entry points

- `warpSPHCore.warpOperation` / `warpSPHCore.warpOperationJVP`: the state-object entry points (above)
- `warpSPHCore.sphOperation_warp`: the flat-tensor entry point for callers without state objects
- `warpSPHCore.radiusSearchCompactHashMap` / `buildCompactHashMap`: compact-hash neighbor search
- `warpSPHCore.crk.computeCRKFactors`: CRK apparent-volume and correction tensors
- `warpSPHCore.renorm.computeRenormalizationMatrices`: gradient-renormalization matrices
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

This includes `tests/operations/test_gradcheck_scripts.py`, which runs every script under `scripts/gradcheck/` (plus the Tier-1/Tier-2 forward-mode spikes under `scripts/spikes/`) as a subprocess, since `warpSPHCore_PRECISION` is baked into every compiled kernel at first import and cannot change mid-process. For the forward-value diagnostic matrix and a quick smoke sweep, see `scripts/diagnostics/run_operation_matrix_sweep.sh --quick`.
