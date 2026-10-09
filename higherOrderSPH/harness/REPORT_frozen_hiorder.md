# Frozen-particle PDE leg -- Phase 4/5 operators

Generated 2026-10-09 11:33 by `run_frozen_hi.py` (float64, 2-D periodic, jitter 0.3, RK4, cfl 0.05 advection / 0.02 diffusion, t_end 0.5, ladder [288, 572, 1152, 2304]). Stock leg (standard / crk / renorm): `REPORT_frozen.md`.

| equation | mode | N neighbours | order (L2 vs dx) | L2 error @ coarsest -> finest | diverged |
|---|---|---|---|---|---|
| advection | rkpm1 | 40 | 1.94 | 3.52e-01 -> 4.84e-02 | no |
| advection | rkpm2 | 40 | 1.94 | 3.53e-01 -> 4.86e-02 | no |
| advection | rkpm3 | 60 | 3.91 | 3.11e-02 -> 5.73e-04 | no |
| advection | labfm2 | 25 | 1.95 | 2.30e-01 -> 3.12e-02 | no |
| advection | labfm4 | 40 | 3.93 | 1.49e-02 -> 2.69e-04 | no |
| advection | labfm6 | 60 | 5.90 | 1.04e-03 -> 2.51e-06 | no |
| advection | labfm8 | 80 | 7.87 | 5.62e-05 -> 1.83e-08 | no |
| diffusion | rkpm2 | 40 | 2.05 | 2.03e-02 -> 2.55e-03 | no |
| diffusion | rkpm3 | 60 | 2.06 | 3.03e-02 -> 3.75e-03 | no |
| diffusion | labfm2 | 25 | 2.02 | 9.33e-03 -> 1.21e-03 | no |
| diffusion | labfm4 | 40 | 3.95 | 4.73e-04 -> 8.58e-06 | no |
| diffusion | labfm6 | 60 | 5.91 | 2.81e-05 -> 6.88e-08 | no |
| diffusion | labfm8 | 80 | 7.88 | 1.47e-06 -> 4.80e-10 | no |
