# Meshless finite mass and finite volume

[Runnable examples](examples.md) includes both backends/modes, raw conserved
state checks and a short periodic update with an explicit rebuild policy.

`warpSPHCore.mfm` supplies ideal-gas MFM/MFV geometry, reconstruction,
Riemann fluxes and conservative rates. `MeshlessGeometry` plus `mfmRates`
is the Torch reference; `MeshlessWarp` implements the same formulation over
CSR neighborhoods. A solver supplies particle drift, support updates, physical
boundaries and time integration.

## MFM versus MFV

Both modes evolve particle-integrated conserved quantities

$$
Q_i=(m_i,\;m_iv_i,\;m_ie_{\mathrm{tot},i}),\qquad
e_{\mathrm{tot}}=\frac{P}{(\gamma-1)\rho}+\frac{|v|^2}{2}.
$$

| Mode | Face treatment | Mass update |
|---|---|---|
| `MFM` | The face moves with the Riemann contact speed | Exactly zero mass rate per particle |
| `MFV` | The face stays at rest in the selected moving frame | Mass exchanges along with momentum and energy |

Fluxes are transformed back to lab-frame conserved quantities. Each unordered
pair contributes equal and opposite changes, so total mass, momentum and
energy cancel to accumulation roundoff. This differs from local uniform-state
preservation and angular-momentum conservation; noncentral faces do not
automatically conserve angular momentum. Mode names are case insensitive.
Unknown modes raise in both public backends.

## Geometry, gradients and faces

Effective volumes come from geometry, not the input density:

$$
\omega_i=\sum_j W(|x_j-x_i|,h_i),\qquad V_i=1/\omega_i.
$$

The self contribution is included. With minimum-image periodic offsets
$d_{ij}=x_j-x_i$,

$$
E_i=\sum_j W_{ij}(h_i)d_{ij}d_{ij}^T,\qquad
\nabla f_i=E_i^{-1}\sum_jW_{ij}(h_i)(f_j-f_i)d_{ij}.
$$

Resolved affine data give the affine gradient to numerical precision.
Deficient rows return zero gradient, giving first-order reconstruction there.
This relates to the [local polynomial difference fit](field-reconstruction-guide.md),
but MFM geometry and volumes have their own definition. Before guards and
projection, the face oriented from $i$ to $j$ is

$$
A^0_{ij}=V_iW_{ij}(h_i)E_i^{-1}d_{ij}
         +V_jW_{ij}(h_j)E_j^{-1}d_{ij}.
$$

Torch stores one copy per unordered pair; Warp reconstructs directed
contributions from symmetric CSR. An explicit supplied list must cover both
support radii and contain the reverse of each nonself interaction. The
moment kernels add the self term explicitly and skip listed self entries.
An arbitrary directed/kind-filtered SPH list is not
a conservative MFM graph. Omitted adjacency builds a SuperSymmetric list.
The Warp wrapper constructs kind-zero particles; it is not an
`OperationDirection` dispatcher.

### Rank and guards

`cond` is the moment diagnostic $\|E\|_F\|E^{-1}\|_F/d$ after scaling by
$\omega h^2$. Difficult rows use a pseudoinverse. When it drops a spatial
direction, `cond` is infinite, rather than the deceptively small condition
of the retained subspace. This triggers gradient deficiency and, with guards
enabled, face fallback. A pseudoinverse cannot create a missing direction.

`cond_max=1e3` controls gradient fallback. `face_cond_max` separately controls
geometric fallback: default `1e6` in float64 and approximately `2.1e4` in
float32. Backward-pointing faces also fall back to a radial SPH-style face.
Centered volume weights are on by default; the optional area cap is off.
Guards precede projection. Inspect `deficient`, `cond`, and Torch `fallback`
or Warp `condBad`; the latter two describe different guard stages.

## Closure and uniform-state preservation

Antisymmetry gives global conservation even if local faces leak. A static
uniform-pressure state nevertheless experiences

$$
\dot p_i=-P\sum_jA_{ij}.
$$

| `closure` | Formulation |
|---|---|
| `"none"` | Guarded unprojected faces |
| `"project"` (default) | Antisymmetric graph correction reducing each face sum |

Projection adds $\kappa_{ij}(\lambda_j-\lambda_i)$, with default
$\kappa_{ij}=|A_{ij}|$, by solving a graph-Laplacian system. The solver has
best-iterate, finite-value and potential-cap guards. Tolerance is relative to
the initial leakage: Warp defaults are `1e-6` in float64 and `1e-4` in
float32. Failed or capped solves need not reach that tolerance. A small wave
can expose leakage negligible for a much larger signal.

Torch exposes `A0` before guards, `A` after guards/projection, and `closure()`
for the final face sum. Warp `residual()` returns the **unprojected** residual
and face norm; `closureResidualRatio` reports the solve's relative residual.

Closure enforces a zeroth-moment condition. It does not enforce

$$
\sum_j A_{ij}\otimes(x_{*,ij}-x_i)=V_iI,
$$

the first-moment identity for exact local linear-flux divergence. Projection
can change this moment. First-moment-consistent faces are a separate numerical
project. On an open domain, missing physical boundary faces still need a
boundary model; graph projection is not a wall or inflow condition.

## Reconstruction and flux

MFM uses matrix gradients, a particle slope limiter and a pair limiter. It
does not automatically substitute TENO/WENO. The pair location is

$$
x_* = x_i+\frac{h_i}{h_i+h_j}(x_j-x_i).
$$

`order=1` uses piecewise-constant states. `order=2` adds limited spatial
reconstruction and, for positive `dt`, half-step primitive prediction.
`dt=0` skips prediction. Current defaults combine lab-frame prediction,
a time-centered moving frame and Gaburov-first HLLC wave speeds, followed
by Roe, PVRS and Rusanov speed fallbacks. Changing only one part is not a
qualified replacement for this post-fix combination.

Invalid star states can retry first-order states, then zero-jump states.
Torch diagnostics include `stage`, `flux`, `Sstar`, `Pstar` and `alpha`.
For MFV with positive `dt`, `massFluxLimit=0.1` caps each pair's transfer at
ten percent of the donor's current mass, limiting mass flux only. Momentum
and energy keep their full flux. A per-pair cap does not prove positive mass
after summing all outgoing transfers, or positive internal energy after an
arbitrary timestep. Check the actual update. `primitives` floors pressure,
so validation must also inspect raw internal energy.

The gas law is ideal gas with scalar $\gamma>1$. Torch `faceFlux` accepts a
custom MFM `starFn`, forwarded by `mfmRates`. This replaces neither the
ideal-gas predictor nor the CFL estimate. Warp rates have no custom-star hook.

## Public use

```python
from warpSPHCore.mfm import (
    MeshlessGeometry, MeshlessWarp, conserved, mfmRates, signalTimestep,
)

geometry = MeshlessGeometry.build(particles, domain, kernel, closure="project")
Q = conserved(rho * geometry.volume, rho, velocity, pressure, gamma)
dt = float(signalTimestep(geometry, rho, velocity, pressure, gamma).min())
rate, diagnostics = mfmRates(
    geometry, rho, velocity, pressure, gamma, dt=dt, mode="MFM", order=2,
)
Q_next = Q + dt * rate

geometry_warp = MeshlessWarp.build(
    particles.positions, particles.supports, domain, kernel, closure="project",
)
rate_warp = geometry_warp.rates(
    rho, velocity, pressure, gamma, dt=dt, mode="MFM", order=2,
)
```

Rates have shape `(N, dim+2)`, scalar primitives `(N,)`, velocity `(N, dim)`.
Positions/supports and Warp primitive inputs must use the process precision
selected before importing the core. `SUPPORTED_KERNELS` lists the five Torch
kernel choices used in the backend comparisons; Torch rejects other kernels.
Warp uses the core kernel dispatcher, but additional kernel choices are not
qualified by these MFM tests.
Inputs need finite positions, positive finite supports, and physically valid
positive density/pressure; geometric guards are not an invalid-state repair.

This example shows a rate/update calculation. A complete step also drifts
positions, wraps periodic coordinates, rebuilds geometry and updates supports.
`higherOrderSPH/harness/mfm_driver.py` demonstrates this composition. Its
`H_ITERS` option separates moved-particle support/volume iteration from the
default one-step-lag policy; neither policy is silently substituted here.

Rebuild geometry after positions, supports, domain or graph membership change.
Torch stores geometry-dependent arrays; Warp retains prepared tensors and
optional captured closure buffers. Particle-count, capacity and graph-storage
changes can require rebuild/recapture. A fully captured public evolution step
is not promised.

## Differentiation and qualification

The Torch reference supports reverse AD and forward duals on fixed topology
and smooth selected branches. Tests include geometry-gradient derivatives and
fixed-geometry MFM/MFV field adjoint/dual agreement, with finite differences
along a primitive-scaling direction. Rank changes, limiter ties, neighbor
changes, retry switches and stopping-iteration changes are not smooth
derivative points. This is not a qualification of every derivative through
long moving-particle evolution.
Geometry derivative qualification is for resolved stencils; derivatives of
truncated moments and active geometric fallbacks have not been established
by these tests.

**`MeshlessWarp` is forward-only.** Build, gradients, rates, timestep and
closure reject live gradient/dual inputs explicitly. An ordinary forward in
`torch.no_grad()` can consume `requires_grad` metadata. Low-level exported
Warp kernels do not establish a complete flux/closure AD path; that path is
separately scoped.

The [development archive](development-artifacts.md), under
`core/docs/development/mfm-validation.md`, records production tests, sound-wave/Sod
ladders, post-fix jittered Gresho repeats and the frozen geometry probe.
Reproduction scripts are `scripts/diagnostics/qualify_mfm.py` and
`qualify_mfm_geometry.py`. Performance is not an acceptance criterion.
