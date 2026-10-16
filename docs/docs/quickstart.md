# Quickstart

## Install

```bash
pip install -e .          # core (torch, warp-lang, numpy)
# or with test tooling:
pip install -e ".[dev]"
```

Precision is chosen **before first import** through an environment
variable (it is baked into every Warp kernel at import time and cannot
change mid-process):

```bash
export warpSPHCore_PRECISION=float32   # default; float64 for high-accuracy runs
# export warpSPHCore_DIM=3             # optional: pin the spatial dimension
```

## A minimal run: density and gradient

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
from warpSPHCore.renorm import computeRenormalizationMatrices

wp.init()
device = "cuda"

# 1. Build a state: an nx^d lattice in [-1, 1]^d with ~55 neighbors per particle.
nx, target_neighbors, dim, periodic = 32, 55, 2, True
positions, supports, n, domain, dx = generateNeighborTestData(
    nx, target_neighbors, dim, periodic, device
)
masses   = torch.full_like(supports, dx**dim)      # unit nominal density
kinds    = torch.zeros(n, dtype=torch.int32, device=device)
particles = ParticleState(positions, supports, masses, kinds)

# 2. The one entry point. `warpOperation` dispatches on
#    OperationProperties.operation; `adjacency=None` builds a
#    CompactHashMap on the fly (pass an AdjacencyList or a
#    CompactHashMap to reuse one across many calls).
properties = OperationProperties(
    kernel=KernelFunctions.Wendland2,
    operation=WarpOperation.Density,
    supportMode=SupportScheme.Gather,
    operationMode=OperationDirection.AllToAll,
)
rho = sph.warpOperation(particles, properties, domain)
particles.densities = rho
density_properties = properties
print("density:", rho.shape)          # [N]

# 3. A gradient of a vector field, Difference scheme.
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
print("gradient:", grad.shape)        # [N, 2] -> [N, 2, 2]
```

The same call pattern covers every operator — see
[API reference](api) and the per-operator pages for which
`OperationProperties` fields and which value/volume/correction arguments
each one consumes.

## Adding corrections

```python
from warpSPHCore.crk import computeCRKFactors

# CRK correction (moment-based linear reproduction):
apparent_area, crk_density, crk_state = computeCRKFactors(
    particles, domain, KernelFunctions.Wendland2,
)

# Gradient renormalization (De Courcy et al. 2024 style):
renorm = computeRenormalizationMatrices(
    particles, properties, domain,
    referenceVolumes=apparent_area,
)   # -> (C, eigVals, RenormalizationState)

grad_crk = sph.warpOperation(
    particles, properties, domain,
    queryValues=vec, referenceValues=vec,
    referenceVolumes=apparent_area,
    crkState=crk_state,                 # CRK-corrected kernel gradient
    renormalizationState=renorm[2],     # ... then renormalized
)
```

A correction is "on" exactly when its state object is passed — there are
no boolean flags in the state API (the flat `sphOperation_warp` keeps
`useCRK`/`useVolume`/... flags for manual callers, validated against the
matching tensors).

## Differentiating an operator

Reverse mode — gradients w.r.t. positions, supports, masses, densities
and the field values:

```python
positions.requires_grad_(True)
rho = sph.warpOperation(particles, density_properties, domain)
rho.sum().backward()
print(positions.grad.shape)
```

Forward mode — a JVP in any combination of value and geometry
directions, through the same public entry point:

```python
from warpSPHCore import ParticleTangentState

dpositions = torch.randn_like(positions)
tangent = ParticleTangentState(positions=dpositions, supports=None, masses=None)
drho = sph.warpOperationJVP(
    particles, density_properties, domain,
    queryTangentState=tangent,
    referenceTangentState=tangent,  # move both roles of this physical cloud
)
```

Every operator's JVP scope — which tangents it accepts, which
combinations raise `NotImplementedError`, and why — is documented on its
operator page under **JVP / HVP** and summarized in
[Autodiff machinery](autograd).

## What the notebooks show

The `notebooks/` directory ships Jupyter notebooks that exercise each
operator on a lattice: `warp_density.ipynb`, `warp_gradient.ipynb`,
`warp_divergence.ipynb`, `warp_curl.ipynb`, `warp_laplacian.ipynb`,
`warp_interpolate.ipynb`, `warp_renorm.ipynb`, `warp_profile.ipynb`.
They use the same `warpOperation` / `warpOperationJVP` entry points as
this quickstart.

## Running the tests

```bash
pytest tests/operations            # CPU + CUDA (CUDA skipped if absent)
```

The gradcheck suites run as subprocesses at float64
(`tests/operations/test_gradcheck_scripts.py`) because precision is
baked in at import time.

For complete checked compositions, including the density/correction/gradient
derivative chain, start with [Runnable examples](examples.md). The density
derivative snippets above differentiate the density operator alone; cached
correction tensors must be rebuilt inside a differentiated composition.
