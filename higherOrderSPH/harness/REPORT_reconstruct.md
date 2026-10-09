# Phase 6/7 static evidence -- TENO vs WENO vs plain MLS interface states

Generated 2026-10-09 11:25 by `run_reconstruct.py` (float64, 2-D hex lattice, jitter 0.3, periodic, particle counts (288, 1152, 4608)). Matched formal order per block of three rows. See the module docstring of `run_reconstruct.py` for the metric definitions; **no Riemann solver is involved** -- these are reconstruction states only.

| scheme | smooth order | smooth err @ finest | step overshoot | jump capture | false jump (same side) | smooth-background error >= 2 dx from the jump | same, no jump present |
|---|---|---|---|---|---|---|---|
| mls3 | 4.08 | 7.69e-06 | 4.04e-01 | 0.175 | 3.82e-01 | 7.66e-03 | 5.96e-06 |
| teno-O4 | 4.08 | 7.69e-06 | 1.49e-10 | 1.000 | 2.18e-10 | 7.66e-03 | 5.96e-06 |
| weno-M3 | 4.01 | 6.19e-05 | 2.22e-16 | 1.000 | 2.22e-16 | 2.63e-02 | 5.73e-05 |
| mls4 | 4.94 | 6.45e-07 | 2.59e-01 | 0.151 | 1.85e-01 | 2.27e-02 | 4.42e-07 |
| teno-O5 | 4.94 | 6.45e-07 | 2.58e-08 | 1.000 | 3.48e-08 | 6.63e-03 | 4.42e-07 |
| weno-M4 | 4.85 | 7.74e-06 | 2.22e-16 | 1.000 | 2.22e-16 | 7.48e-03 | 5.21e-06 |
| mls5 | 5.94 | 4.73e-08 | 2.07e-01 | 0.040 | 1.03e-01 | 3.35e-02 | 3.14e-08 |
| teno-O6 | 5.94 | 4.73e-08 | 2.41e-08 | 1.000 | 6.58e-08 | 8.56e-03 | 3.14e-08 |
| weno-M5 | 5.93 | 2.25e-07 | 2.22e-16 | 1.000 | 4.44e-16 | 1.38e-04 | 2.07e-07 |
