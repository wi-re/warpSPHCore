# Frozen-particle linear PDE leg

Periodic 2D densest-packing lattice, jitter 0.3, target neighbors 40, Wendland2, float64, RK4; u0 = sin(kappa . x), k = (1, 1), a = (1.0, 0.5), nu = 0.05, t_end = 0.5. Spatial operator = the only error source (`run_frozen.py`).

## Rows

| equation | mode | N | dx | dt | steps | diverged | err L2 | err Linf | max|u| / exact |
|---|---|---|---|---|---|---|---|---|---|
| advection | standard | 572 | 0.04545 | 0.00806 | 62 | False | 2.340e-01 | 4.054e-01 | 1.1118 |
| advection | standard | 1152 | 0.03125 | 0.00556 | 90 | False | 1.443e-01 | 3.235e-01 | 1.1472 |
| advection | standard | 2200 | 0.02273 | 0.00407 | 123 | False | 9.544e-02 | 2.351e-01 | 1.1638 |
| advection | standard | 4736 | 0.01562 | 0.00279 | 179 | False | 6.735e-02 | 2.083e-01 | 1.1623 |
| advection | standard | 9360 | 0.01111 | 0.00198 | 252 | False | 5.408e-02 | 2.972e-01 | 1.1460 |
| advection | crk | 572 | 0.04545 | 0.00806 | 62 | False | 1.967e-01 | 3.201e-01 | 1.0482 |
| advection | crk | 1152 | 0.03125 | 0.00556 | 90 | False | 1.010e-01 | 1.870e-01 | 1.0432 |
| advection | crk | 2200 | 0.02273 | 0.00407 | 123 | False | 5.354e-02 | 1.082e-01 | 1.0320 |
| advection | crk | 4736 | 0.01562 | 0.00279 | 179 | False | 2.530e-02 | 5.829e-02 | 1.0159 |
| advection | crk | 9360 | 0.01111 | 0.00198 | 252 | False | 1.320e-02 | 3.361e-02 | 1.0123 |
| advection | renorm | 572 | 0.04545 | 0.00806 | 62 | False | 1.972e-01 | 3.018e-01 | 1.0128 |
| advection | renorm | 1152 | 0.03125 | 0.00556 | 90 | False | 1.005e-01 | 1.635e-01 | 1.0259 |
| advection | renorm | 2200 | 0.02273 | 0.00407 | 123 | False | 5.250e-02 | 8.876e-02 | 1.0176 |
| advection | renorm | 4736 | 0.01562 | 0.00279 | 179 | False | 2.430e-02 | 4.901e-02 | 1.0077 |
| advection | renorm | 9360 | 0.01111 | 0.00198 | 252 | False | 1.243e-02 | 2.807e-02 | 1.0082 |
| advection | renormVal | 572 | 0.04545 | 0.00806 | 62 | False | 1.972e-01 | 3.018e-01 | 1.0128 |
| advection | renormVal | 1152 | 0.03125 | 0.00556 | 90 | False | 1.005e-01 | 1.635e-01 | 1.0259 |
| advection | renormVal | 2200 | 0.02273 | 0.00407 | 123 | False | 5.250e-02 | 8.876e-02 | 1.0176 |
| advection | renormVal | 4736 | 0.01562 | 0.00279 | 179 | False | 2.430e-02 | 4.901e-02 | 1.0077 |
| advection | renormVal | 9360 | 0.01111 | 0.00198 | 252 | False | 1.243e-02 | 2.807e-02 | 1.0082 |
| diffusion | standard | 572 | 0.04545 | 0.00413 | 121 | False | 8.931e-03 | 1.516e-02 | 1.1029 |
| diffusion | standard | 1152 | 0.03125 | 0.00195 | 256 | False | 5.364e-03 | 8.454e-03 | 1.0641 |
| diffusion | standard | 2200 | 0.02273 | 0.00103 | 484 | False | 3.806e-03 | 6.020e-03 | 1.0420 |
| diffusion | standard | 4736 | 0.01562 | 0.000488 | 1024 | False | 2.996e-03 | 4.686e-03 | 1.0320 |
| diffusion | standard | 9360 | 0.01111 | 0.000247 | 2025 | False | 2.554e-03 | 3.867e-03 | 1.0270 |
| diffusion | crk | 572 | 0.04545 | 0.00413 | 121 | False | 6.202e-03 | 9.297e-03 | 1.0630 |
| diffusion | crk | 1152 | 0.03125 | 0.00195 | 256 | False | 2.944e-03 | 4.397e-03 | 1.0330 |
| diffusion | crk | 2200 | 0.02273 | 0.00103 | 484 | False | 1.540e-03 | 2.413e-03 | 1.0175 |
| diffusion | crk | 4736 | 0.01562 | 0.000488 | 1024 | False | 7.307e-04 | 1.157e-03 | 1.0081 |
| diffusion | crk | 9360 | 0.01111 | 0.000247 | 2025 | False | 3.736e-04 | 7.132e-04 | 1.0050 |
| diffusion | renorm | 572 | 0.04545 | 0.00413 | 121 | False | 6.834e-03 | 1.156e-02 | 1.0780 |
| diffusion | renorm | 1152 | 0.03125 | 0.00195 | 256 | False | 3.397e-03 | 5.855e-03 | 1.0414 |
| diffusion | renorm | 2200 | 0.02273 | 0.00103 | 484 | False | 1.869e-03 | 3.541e-03 | 1.0221 |
| diffusion | renorm | 4736 | 0.01562 | 0.000488 | 1024 | False | 1.077e-03 | 2.309e-03 | 1.0132 |
| diffusion | renorm | 9360 | 0.01111 | 0.000247 | 2025 | False | 5.920e-04 | 1.473e-03 | 1.0068 |
| diffusion | renormVal | 572 | 0.04545 | 0.00413 | 121 | False | 6.834e-03 | 1.156e-02 | 1.0780 |
| diffusion | renormVal | 1152 | 0.03125 | 0.00195 | 256 | False | 3.397e-03 | 5.855e-03 | 1.0414 |
| diffusion | renormVal | 2200 | 0.02273 | 0.00103 | 484 | False | 1.869e-03 | 3.541e-03 | 1.0221 |
| diffusion | renormVal | 4736 | 0.01562 | 0.000488 | 1024 | False | 1.077e-03 | 2.309e-03 | 1.0132 |
| diffusion | renormVal | 9360 | 0.01111 | 0.000247 | 2025 | False | 5.920e-04 | 1.473e-03 | 1.0068 |

## Observed orders (err L2 vs dx)

| equation | mode | slope | r^2 | pairwise | note |
|---|---|---|---|---|---|
| advection | standard | 1.05 | 0.982 | 1.29, 1.30, 0.93, 0.64 |  |
| advection | crk | 1.93 | 1.000 | 1.78, 1.99, 2.00, 1.91 |  |
| advection | renorm | 1.98 | 0.999 | 1.80, 2.04, 2.06, 1.97 |  |
| advection | renormVal | 1.98 | 0.999 | 1.80, 2.04, 2.06, 1.97 |  |
| diffusion | standard | 0.88 | 0.954 | 1.36, 1.08, 0.64, 0.47 |  |
| diffusion | crk | 2.00 | 1.000 | 1.99, 2.04, 1.99, 1.97 |  |
| diffusion | renorm | 1.72 | 0.998 | 1.87, 1.88, 1.47, 1.76 |  |
| diffusion | renormVal | 1.72 | 0.998 | 1.87, 1.88, 1.47, 1.76 |  |
