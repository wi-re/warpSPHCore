# Phase 5 report -- LABFM order vs stencil size

Generated 2026-10-09 11:33 by `run_labfm.py` (float64, 2-D periodic, kernel Wendland2, jitter 0.3, field `sin_cos`, ladder N = (288, 572, 1152, 2304, 4608)). Operator: `rkpm.py` `constant=False` (LABFM: kernel-weighted polynomial fit of f_j - f_i). Orders are the log-log slope of the interior L-inf error vs dx at fixed N (neighbours); `sat` = flat/floor-limited series.

## Observed order, gradient (G) and Laplacian (L), by scheme order k and stencil size

| k | N (nbrs) | gradient | Laplacian | grad err @ finest | cond (median) | rank-deficient |
|---|---|---|---|---|---|---|
| 2 | 25 | 1.95 | 1.54 | 3.00e-02 | 2.1 | 0 |
| 2 | 40 | 1.95 | 1.77 | 4.57e-02 | 2 | 0 |
| 2 | 60 | 1.94 | 1.90 | 6.63e-02 | 2 | 0 |
| 2 | 80 | 1.93 | 1.94 | 8.67e-02 | 2 | 0 |
| 2 | 100 | 1.92 | 1.94 | 1.08e-01 | 2 | 0 |
| 2 | 150 | 1.88 | 1.91 | 1.60e-01 | 2 | 0 |
| 4 | 25 | 3.94 | 2.81 | 5.63e-05 | 45 | 0 |
| 4 | 40 | 3.91 | 3.19 | 1.29e-04 | 42 | 0 |
| 4 | 60 | 3.92 | 3.43 | 2.51e-04 | 43 | 0 |
| 4 | 80 | 3.93 | 3.62 | 4.43e-04 | 42 | 0 |
| 4 | 100 | 3.91 | 3.79 | 6.80e-04 | 42 | 0 |
| 4 | 150 | 3.86 | 3.86 | 1.49e-03 | 42 | 0 |
| 6 | 25 | sat | -1.12 | 9.83e-01 | 8.1e+16 | 1 |
| 6 | 40 | 5.90 | 4.72 | 1.82e-07 | 8.4e+02 | 0 |
| 6 | 60 | 5.91 | 5.11 | 5.65e-07 | 7.6e+02 | 0 |
| 6 | 80 | 5.92 | 5.06 | 1.11e-06 | 8.3e+02 | 0 |
| 6 | 100 | 5.89 | 5.45 | 2.23e-06 | 7.7e+02 | 0 |
| 6 | 150 | 5.86 | 5.65 | 7.27e-06 | 7.8e+02 | 0 |
| 8 | 25 | sat | -1.01 | 1.49e+00 | 4e+17 | 1 |
| 8 | 40 | sat | -0.75 | 1.69e-01 | 1.8e+17 | 1 |
| 8 | 60 | 7.88 | 6.76 | 6.42e-10 | 1.8e+04 | 0 |
| 8 | 80 | 7.88 | 6.79 | 1.94e-09 | 1.7e+04 | 0 |
| 8 | 100 | 7.88 | 6.89 | 4.15e-09 | 1.7e+04 | 0 |
| 8 | 150 | 7.88 | 7.31 | 2.03e-08 | 1.6e+04 | 0 |

## Smallest stencil reaching gradient order >= k - 0.5

| k | measured N_min | paper N | paper N_crit (ABF) | N_poly |
|---|---|---|---|---|
| 2 | 25 | — | 8 | 6 |
| 4 | 25 | ~25 | 21 | 15 |
| 6 | 40 | ~50 (h/dr=2) | 37 | 28 |
| 8 | 60 | ~60-78 | 57 | 45 |
