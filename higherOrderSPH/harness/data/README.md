# Saved particle distributions (harness test data)

Real, *evolved* SPH particle distributions captured from the tgv-wc PDE
benchmark case (`warpSPH.cases.tgvWeaklyCompressible`, nx=128, 2D periodic
Taylor-Green vortex, k=2, nu=0.01, uMag=1, L=2 pi, t=2.0, dt=1e-3, RK2,
float64), one file per particle-shifting (PST) setting:

* `tgv2d_noshift_nx128.npz` -- delta-SPH, PST **off** (`shifting=False`).
  The uncorrected distribution: whatever disorder the flow itself leaves
  behind.
* `tgv2d_fullshift_nx128.npz` -- delta+-SPH, **full-strength** Sun 2017
  Eq. (7) shifting (scheme `sun2017DeltaSPH`). The redistributed
  distribution the shifting is supposed to maintain.

Purpose: test data for the static higher-order probes. Instead of only
lattice + random jitter, the operator-consistency probes can now run on a
*physically disordered* distribution and quantify how the distribution's
anisotropy degrades the recorded field (Vacondio et al. 2021, SPH grand
challenges, GC1). The per-snapshot anisotropy summaries below were computed
with `harness/distribution.py` at capture time and are stored in the file.
First usage: the disorder probe of `pde/TGV_NOTES.md` section 7.5 (static
operator probes on these distributions vs a purely-jittered baseline).

## npz layout

Common metadata keys (0-d arrays): `L`, `dx`, `nx`, `dim`, `scheme`,
`shifting`, `kernel`, `n_h`, `targetDt`, `t_limit`, `integration`,
`precision`, `mass_min`, `mass_max`, `times`.

Per snapshot t (keys `t0`, `t0.5`, `t1`, `t1.5`, `t2`):

* `t*_positions` -- (N, 2) float64, raw simulation coordinates. The code
  does not re-wrap them into the box, so the net drift grows with t (up to
  ~0.4 L at t=2 in these files); the periodic minimum-image handling takes
  that into account
* `t*_masses` -- (N,) float64
* `t*_h` -- uniform support (float)
* `t*_first_anisotropy_rms` / `_max` -- first-moment residual measure
* `t*_second_anisotropy_rms` / `_max` -- second-moment (shape) measure, in
  [0, 1]
* `t*_kernel_sum_min` / `_max` -- kernel-sum spread (C_i)

## Loading into the harness

```python
import numpy as np, torch
from harness.particle_sets import build_case_from_positions
from harness.distribution import distribution_moments, anisotropy_summary
from warpSPHCore.enumTypes import KernelFunctions

d = np.load("tgv2d_noshift_nx128.npz")
t = "t2"
case = build_case_from_positions(
    positions=torch.tensor(d[f"{t}_positions"], dtype=torch.float64, device="cuda"),
    masses=torch.tensor(d[f"{t}_masses"], dtype=torch.float64, device="cuda"),
    h=float(d[f"{t}_h"]), box=np.array([float(d["L"])] * 2),
    dx=float(d["dx"]), target_neighbors=32,   # nominal neighbour count, metadata only
    periodic=True, device="cuda",
    kernel=KernelFunctions.Wendland4,
)   # -> Case with densities + adjacency, ready for the probes

mom = distribution_moments(case.positions.cpu(), case.particles.masses.cpu(),
                           h=case.h,
                           L=torch.tensor([float(d["L"])] * 2,
                                          dtype=torch.float64),
                           dim=2, kernel="wendland4", periodic=True)
anisotropy_summary(mom)   # must reproduce the stored t*_..._rms/_max keys
```

Note: the probes require `warpSPHCore_PRECISION=float64` (as the rest of
the harness). The anisotropy module is pure torch and precision-agnostic.
