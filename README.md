# warpSPHCore

warpSPHCore is a Warp- and PyTorch-based backend for core Smoothed Particle Hydrodynamics (SPH) operations. The package focuses on GPU-accelerated neighborhood construction and particle operators such as density estimation, interpolation, gradients, divergence, curl, Laplacians, and CRK-corrected variants.

The repository also includes notebooks that compare the Warp implementation against diffSPH reference workflows and demonstrate common operator setups.

## Highlights

- Compact-hash radius search for particle neighborhoods
- Unified SPH operator entry point via `sphOperation_warp`
- Multiple support and gradient schemes
- CRK correction utilities for corrected interpolation and gradients
- PyTorch tensor inputs and outputs, with Warp kernels under the hood

## Installation

Install the package in editable mode while developing:

```bash
pip install -e .
```

If you want to run the example notebooks in this repository, install the notebook extras as well:

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

Notebook and plotting extras used in this repository:

- matplotlib
- ipywidgets
- ipympl

GPU notes:

- For GPU execution, use a CUDA-capable PyTorch build and a compatible NVIDIA driver.
- A separate CUDA toolkit installation is not required anymore for this package setup.
- Warp still needs to be initialized once per process with `wp.init()`.

## Package Overview

Common entry points:

- `warpSPHCore.radiusSearchCompactHashMap`: builds adjacency lists with compact hashing
- `warpSPHCore.sphOperation_warp`: dispatches SPH operators from one high-level function
- `warpSPHCore.crk.computeCRKFactors`: computes CRK apparent area and correction tensors
- `warpSPHCore.util.generateNeighborTestData`: helper for generating regular particle test sets
- `warpSPHCore.util.getNextPrime`: helper for choosing compact-hash table sizes

Important enums:

- `WarpOperation`: selects the SPH operator
- `KernelFunctions`: selects the smoothing kernel
- `SupportScheme`: selects gather/scatter/symmetric support handling
- `GradientScheme`: selects the gradient formulation
- `OperationDirection`: selects all-to-all or filtered interaction directionality

## Example: Radius Search with Compact Hashing

This follows the same setup pattern used by `prepData` in the demo utilities.

```python
import torch
import warp as wp

from warpSPHCore import radiusSearchCompactHashMap
from warpSPHCore.util import generateNeighborTestData, getNextPrime

wp.init()

device = torch.device("cuda")
nx = 128
dim = 2
target_num_neighbors = 50
periodic = True

positions, supports, num_particles, domain, dx = generateNeighborTestData(
    nx, target_num_neighbors, dim, periodic, device
)

query_positions = positions.contiguous()
reference_positions = positions.contiguous()
query_supports = supports
reference_supports = supports

hash_map_length = getNextPrime(num_particles)

adjacency = radiusSearchCompactHashMap(
    query_positions,
    reference_positions,
    query_supports,
    reference_supports,
    domain.periodic,
    domain,
    "gather",
    hash_map_length,
)

print(adjacency.numNeighbors.shape)
print(adjacency.edgeOffsets[:5])
print(adjacency.j[:10])
```

The returned adjacency list stores edge-level connectivity in COO/CSR-like form:

- `adjacency.i`: query particle index per edge
- `adjacency.j`: neighbor particle index per edge
- `adjacency.numNeighbors`: neighbor count per query particle
- `adjacency.edgeOffsets`: CSR-style start offset per query particle

## Example: SPH Interpolation

This mirrors the interpolation workflow used in `warp_interpolate.ipynb`.

```python
import torch
import warp as wp

from warpSPHCore import radiusSearchCompactHashMap, sphOperation_warp
from warpSPHCore.enumTypes import KernelFunctions, OperationDirection, SupportScheme, WarpOperation
from warpSPHCore.util import generateNeighborTestData, getNextPrime

wp.init()

device = torch.device("cuda")
nx = 128
dim = 2
target_num_neighbors = 50
periodic = True

positions, supports, num_particles, domain, dx = generateNeighborTestData(
    nx, target_num_neighbors, dim, periodic, device
)

query_positions = positions.contiguous()
reference_positions = positions.contiguous()
query_supports = supports
reference_supports = supports

particle_mass = dx ** dim
query_masses = torch.full((num_particles,), particle_mass, device=device)
reference_masses = torch.full((num_particles,), particle_mass, device=device)

adjacency = radiusSearchCompactHashMap(
    query_positions,
    reference_positions,
    query_supports,
    reference_supports,
    domain.periodic,
    domain,
    "gather",
    getNextPrime(num_particles),
)

densities = sphOperation_warp(
    query_positions,
    reference_positions,
    query_supports,
    reference_supports,
    query_masses,
    reference_masses,
    None,
    None,
    None,
    None,
    domain,
    adjacency,
    operation=WarpOperation.Density,
    kernel=KernelFunctions.Wendland2,
    supportMode=SupportScheme.Gather,
)

field = torch.sin(query_positions[:, 0])

interpolated = sphOperation_warp(
    query_positions,
    reference_positions,
    query_supports,
    reference_supports,
    query_masses,
    reference_masses,
    densities,
    densities,
    field,
    field,
    domain=domain,
    adjacency=adjacency,
    operation=WarpOperation.Interpolate,
    operationMode=OperationDirection.AllToAll,
    kernel=KernelFunctions.Wendland2,
    supportMode=SupportScheme.Gather,
)

print(interpolated.shape)
```

## Example: CRK-Corrected Gradient

This matches the corrected linear-field gradient workflow in `warp_gradient.ipynb`.

```python
import torch
import warp as wp

from warpSPHCore import radiusSearchCompactHashMap, sphOperation_warp
from warpSPHCore.crk import computeCRKFactors
from warpSPHCore.enumTypes import (
    GradientScheme,
    KernelFunctions,
    OperationDirection,
    SupportScheme,
    WarpOperation,
)
from warpSPHCore.util import generateNeighborTestData, getNextPrime

wp.init()

device = torch.device("cuda")
nx = 128
dim = 2
target_num_neighbors = 50
periodic = True

positions, supports, num_particles, domain, dx = generateNeighborTestData(
    nx, target_num_neighbors, dim, periodic, device
)

query_positions = positions.contiguous()
reference_positions = positions.contiguous()
query_supports = supports
reference_supports = supports

particle_mass = dx ** dim
query_masses = torch.full((num_particles,), particle_mass, device=device)
reference_masses = torch.full((num_particles,), particle_mass, device=device)

adjacency = radiusSearchCompactHashMap(
    query_positions,
    reference_positions,
    query_supports,
    reference_supports,
    domain.periodic,
    domain,
    "gather",
    getNextPrime(num_particles),
)

densities = sphOperation_warp(
    query_positions,
    reference_positions,
    query_supports,
    reference_supports,
    query_masses,
    reference_masses,
    None,
    None,
    None,
    None,
    domain,
    adjacency,
    operation=WarpOperation.Density,
    kernel=KernelFunctions.Wendland2,
    supportMode=SupportScheme.Gather,
)

apparent_area, crk_density, A, B, gradA, gradB = computeCRKFactors(
    query_positions,
    reference_positions,
    query_supports,
    reference_supports,
    query_masses,
    reference_masses,
    domain=domain,
    adjacency=adjacency,
    operationMode=OperationDirection.AllToAll,
    kernel=KernelFunctions.Wendland2,
    supportMode=SupportScheme.Gather,
)

field = query_positions[:, 0] * 5.0 + 10.0

gradient = sphOperation_warp(
    query_positions,
    reference_positions,
    query_supports,
    reference_supports,
    query_masses,
    reference_masses,
    densities,
    densities,
    field,
    field,
    domain=domain,
    adjacency=adjacency,
    operation=WarpOperation.Gradient,
    operationMode=OperationDirection.AllToAll,
    kernel=KernelFunctions.Wendland2,
    supportMode=SupportScheme.SuperSymmetric,
    gradientMode=GradientScheme.Naive,
    useCRK=True,
    crk_A=A,
    crk_B=B,
    crk_gradA=gradA,
    crk_gradB=gradB,
    useVolume=True,
    queryVolumes=apparent_area,
    referenceVolumes=apparent_area,
)

print(gradient[:5])
```

`computeCRKFactors` also returns `crk_density`, which can be useful in other corrected workflows, but the gradient notebook example uses the standard density estimate together with CRK volume and correction tensors.

## Repository Notes

- `demo_util.py` contains helper setup code used across the notebooks.
- `warp_interpolate.ipynb` demonstrates interpolation calls and visualization.
- `warp_gradient.ipynb` demonstrates both standard and CRK-corrected gradients.
- `packages.md` is no longer authoritative for CUDA toolkit setup; the separate CUDA install noted there is not required anymore.

## Publishing To PyPI

This repository now includes two helper scripts:

- `scripts/setup_pypi_token.sh`: stores a PyPI API token in `~/.pypirc`
- `scripts/publish_pypi.sh`: builds, validates, and uploads the package

Typical release flow:

```bash
bash scripts/setup_pypi_token.sh pypi
bash scripts/publish_pypi.sh
```

For a TestPyPI dry run:

```bash
bash scripts/setup_pypi_token.sh testpypi
bash scripts/publish_pypi.sh --testpypi
```

Before publishing, bump the package version in both `pyproject.toml` and `src/warpSPHCore/__init__.py`. The publish script checks that these two versions match and stops if they do not.