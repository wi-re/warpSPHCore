# Interface-state reconstruction -- smooth-region order

Generated 2026-10-09 11:15 by `run_interface.py` (float64, 2-D periodic jitter 0.3, ladder (288, 572, 1152, 2304)). Error = max over all pairs of |f_L - f(mid)|, |f_R - f(mid)|; jump = max |f_L - f_R|.

| scheme | N nbrs | order(err) | order(jump) | err @ finest | jump @ finest |
|---|---|---|---|---|---|
| mls1 | 40 | 1.81 | 2.78 | 2.85e-02 | 1.27e-02 |
| mls2 | 40 | 2.89 | 2.89 | 1.83e-03 | 3.48e-03 |
| mls3 | 60 | 3.87 | 4.74 | 3.40e-04 | 1.81e-04 |
| labfm2 | 25 | 2.92 | 2.92 | 9.85e-04 | 1.90e-03 |
| labfm4 | 40 | 4.80 | 4.51 | 1.15e-05 | 2.23e-05 |
