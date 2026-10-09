# Phase 4 report -- order-p MLS/RKPM reproducing-kernel operator

Generated 2026-10-09 11:33 by `run_rkpm.py` (float64, 2-D, kernel Wendland2, jitter 0.3, 40 target neighbours). Operator: `rkpm.py` (pure-torch local polynomial fit, diffuse derivatives). Reference columns `crk` / `renorm` from the unchanged Phase 0 suites.

## Monomial patch tests -- open domain, interior

Max error; ~1e-15..1e-12 = exact reproduction. `rkpmP` should be exact through degree P for value / gradient and, for P >= 2, Laplacian / Hessian.

| field | field_degree | probe | crk | renorm | rkpm1 | rkpm2 | rkpm3 |
|---|---|---|---|---|---|---|---|
| 1 | 0 | gradient | 0.000e+00 | 0.000e+00 | 3.121e-13 | 2.391e-13 | 2.034e-12 |
| 1 | 0 | interpolate | 4.441e-16 | 5.548e-02 | 1.066e-14 | 4.441e-15 | 4.263e-14 |
| Mx | 1 | gradient | 1.709e-15 | 1.874e-15 | 3.235e-13 | 1.921e-13 | 2.460e-12 |
| Mx | 1 | interpolate | 4.965e-16 | 5.764e-02 | 9.416e-15 | 5.796e-15 | 3.902e-14 |
| x | 1 | gradient | 4.500e-16 | 4.450e-16 | 1.954e-13 | 6.852e-14 | 1.721e-12 |
| x | 1 | interpolate | 3.331e-16 | 4.091e-02 | 6.883e-15 | 3.220e-15 | 2.842e-14 |
| x^2 | 2 | gradient | 6.683e-03 | 3.656e-03 | 4.940e-03 | 5.145e-14 | 1.448e-12 |
| x^2 | 2 | hessian | 1.014e-01 | 7.077e-02 | n/a | 4.910e-12 | 2.307e-11 |
| x^2 | 2 | interpolate | 8.212e-04 | 3.492e-02 | 8.212e-04 | 3.109e-15 | 2.398e-14 |
| x^2 | 2 | laplacian | 7.515e-01 | 7.416e+00 | n/a | 4.096e-12 | 3.022e-11 |
| x^2y | 3 | gradient | 7.319e-03 | 4.886e-03 | 5.384e-03 | 7.886e-04 | 8.647e-13 |
| x^2y | 3 | hessian | 9.008e-02 | 5.371e-02 | n/a | 9.783e-03 | 1.039e-11 |
| x^2y | 3 | interpolate | 6.958e-04 | 2.349e-02 | 6.958e-04 | 4.333e-06 | 1.094e-14 |
| x^2y | 3 | laplacian | 6.619e-01 | 6.702e+00 | n/a | 9.978e-03 | 1.396e-11 |
| x^2y^2 | 4 | gradient | 1.172e-02 | 6.251e-03 | 7.327e-03 | 1.778e-03 | 9.747e-06 |
| x^2y^2 | 4 | hessian | 9.647e-02 | 1.643e-01 | n/a | 1.719e-02 | 1.907e-03 |
| x^2y^2 | 4 | interpolate | 1.132e-03 | 1.760e-02 | 1.132e-03 | 7.711e-06 | 4.822e-07 |
| x^2y^2 | 4 | laplacian | 6.730e-01 | 6.255e+00 | n/a | 1.953e-02 | 2.686e-03 |
| x^3 | 3 | gradient | 1.909e-02 | 1.185e-02 | 1.449e-02 | 2.241e-03 | 1.223e-12 |
| x^3 | 3 | hessian | 2.932e-01 | 2.001e-01 | n/a | 1.371e-02 | 1.913e-11 |
| x^3 | 3 | interpolate | 2.146e-03 | 3.158e-02 | 2.146e-03 | 9.337e-06 | 1.932e-14 |
| x^3 | 3 | laplacian | 1.039e+00 | 1.005e+01 | n/a | 1.690e-02 | 2.555e-11 |
| x^3y | 4 | gradient | 1.468e-02 | 7.374e-03 | 7.609e-03 | 2.633e-03 | 9.233e-06 |
| x^3y | 4 | hessian | 2.313e-01 | 1.201e-01 | n/a | 2.724e-02 | 2.897e-03 |
| x^3y | 4 | interpolate | 1.716e-03 | 2.123e-02 | 1.716e-03 | 1.228e-05 | 1.131e-07 |
| x^3y | 4 | laplacian | 7.438e-01 | 8.485e+00 | n/a | 2.410e-02 | 4.172e-04 |
| x^4 | 4 | gradient | 3.590e-02 | 2.529e-02 | 2.783e-02 | 7.910e-03 | 2.354e-05 |
| x^4 | 4 | hessian | 5.472e-01 | 3.617e-01 | n/a | 5.138e-02 | 7.961e-03 |
| x^4 | 4 | interpolate | 3.839e-03 | 3.055e-02 | 3.839e-03 | 3.397e-05 | 1.504e-06 |
| x^4 | 4 | laplacian | 1.288e+00 | 1.212e+01 | n/a | 6.042e-02 | 7.897e-03 |
| xy | 2 | gradient | 3.057e-03 | 2.872e-03 | 2.927e-03 | 3.339e-14 | 1.017e-12 |
| xy | 2 | hessian | 2.672e-02 | 2.450e-02 | n/a | 2.869e-12 | 1.316e-11 |
| xy | 2 | interpolate | 5.236e-05 | 2.650e-02 | 5.236e-05 | 2.442e-15 | 1.338e-14 |
| xy | 2 | laplacian | 5.514e-01 | 5.076e+00 | n/a | 2.969e-12 | 1.765e-11 |
| xy^2 | 3 | gradient | 6.785e-03 | 4.197e-03 | 4.696e-03 | 7.739e-04 | 6.089e-13 |
| xy^2 | 3 | hessian | 1.175e-01 | 1.450e-01 | n/a | 8.338e-03 | 5.972e-12 |
| xy^2 | 3 | interpolate | 7.178e-04 | 1.916e-02 | 7.178e-04 | 3.544e-06 | 6.245e-15 |
| xy^2 | 3 | laplacian | 6.261e-01 | 5.514e+00 | n/a | 9.380e-03 | 8.007e-12 |
| xy^3 | 4 | gradient | 1.498e-02 | 1.076e-02 | 1.072e-02 | 2.712e-03 | 1.275e-05 |
| xy^3 | 4 | hessian | 3.266e-01 | 3.867e-01 | n/a | 2.471e-02 | 2.905e-03 |
| xy^3 | 4 | interpolate | 1.679e-03 | 1.800e-02 | 1.679e-03 | 9.644e-06 | 1.522e-07 |
| xy^3 | 4 | laplacian | 7.307e-01 | 6.434e+00 | n/a | 2.353e-02 | 3.427e-04 |
| y | 1 | gradient | 4.490e-16 | 4.450e-16 | 2.071e-13 | 1.509e-13 | 1.221e-12 |
| y | 1 | interpolate | 3.331e-16 | 3.802e-02 | 5.884e-15 | 2.998e-15 | 1.860e-14 |
| y^2 | 2 | gradient | 6.669e-03 | 4.233e-03 | 4.080e-03 | 1.115e-13 | 7.204e-13 |
| y^2 | 2 | hessian | 1.583e-01 | 1.583e-01 | n/a | 6.601e-12 | 8.003e-12 |
| y^2 | 2 | interpolate | 8.292e-04 | 3.399e-02 | 8.292e-04 | 2.665e-15 | 8.160e-15 |
| y^2 | 2 | laplacian | 8.300e-01 | 6.536e+00 | n/a | 3.765e-12 | 1.112e-11 |
| y^3 | 3 | gradient | 1.686e-02 | 1.329e-02 | 1.211e-02 | 2.252e-03 | 4.938e-13 |
| y^3 | 3 | hessian | 4.430e-01 | 4.372e-01 | n/a | 1.856e-02 | 3.661e-12 |
| y^3 | 3 | interpolate | 2.142e-03 | 3.100e-02 | 2.142e-03 | 8.808e-06 | 3.678e-15 |
| y^3 | 3 | laplacian | 1.124e+00 | 8.177e+00 | n/a | 1.901e-02 | 5.164e-12 |
| y^4 | 4 | gradient | 3.193e-02 | 2.705e-02 | 2.456e-02 | 7.850e-03 | 2.287e-05 |
| y^4 | 4 | hessian | 8.081e-01 | 7.888e-01 | n/a | 7.215e-02 | 7.962e-03 |
| y^4 | 4 | interpolate | 3.710e-03 | 2.876e-02 | 3.710e-03 | 3.181e-05 | 1.472e-06 |
| y^4 | 4 | laplacian | 1.354e+00 | 9.164e+00 | n/a | 7.354e-02 | 7.933e-03 |

## Monomial patch tests -- open domain, boundary band

Max error; ~1e-15..1e-12 = exact reproduction. `rkpmP` should be exact through degree P for value / gradient and, for P >= 2, Laplacian / Hessian.

| field | field_degree | probe | crk | renorm | rkpm1 | rkpm2 | rkpm3 |
|---|---|---|---|---|---|---|---|
| 1 | 0 | gradient | 0.000e+00 | 0.000e+00 | 3.413e-13 | 1.064e-12 | 1.534e-11 |
| 1 | 0 | interpolate | 4.441e-16 | 2.268e-01 | 1.132e-14 | 2.176e-14 | 3.442e-14 |
| Mx | 1 | gradient | 7.504e-15 | 5.466e-15 | 4.209e-13 | 1.335e-12 | 2.304e-11 |
| Mx | 1 | interpolate | 4.710e-16 | 3.549e-01 | 1.507e-14 | 2.710e-14 | 2.868e-14 |
| x | 1 | gradient | 1.110e-15 | 4.444e-16 | 3.230e-13 | 1.032e-12 | 2.315e-11 |
| x | 1 | interpolate | 3.331e-16 | 2.252e-01 | 1.121e-14 | 2.010e-14 | 1.932e-14 |
| x^2 | 2 | gradient | 4.166e-02 | 4.715e-02 | 4.906e-02 | 9.917e-13 | 1.972e-11 |
| x^2 | 2 | hessian | 8.924e-01 | 9.998e-01 | n/a | 3.862e-11 | 1.916e-09 |
| x^2 | 2 | interpolate | 9.591e-04 | 2.320e-01 | 9.591e-04 | 1.987e-14 | 1.554e-14 |
| x^2 | 2 | laplacian | 1.393e+02 | 1.268e+02 | n/a | 3.445e-11 | 1.437e-09 |
| x^2y | 3 | gradient | 4.925e-02 | 6.010e-02 | 6.574e-02 | 1.337e-03 | 1.655e-11 |
| x^2y | 3 | hessian | 1.073e+00 | 1.220e+00 | n/a | 8.698e-02 | 1.619e-09 |
| x^2y | 3 | interpolate | 1.006e-03 | 2.252e-01 | 1.006e-03 | 9.540e-06 | 8.771e-15 |
| x^2y | 3 | laplacian | 1.695e+02 | 1.551e+02 | n/a | 3.966e-02 | 1.272e-09 |
| x^2y^2 | 4 | gradient | 9.780e-02 | 1.143e-01 | 1.203e-01 | 3.064e-03 | 8.104e-05 |
| x^2y^2 | 4 | hessian | 2.487e+00 | 2.073e+00 | n/a | 2.765e-01 | 7.443e-03 |
| x^2y^2 | 4 | interpolate | 1.308e-03 | 2.187e-01 | 1.308e-03 | 2.081e-05 | 5.545e-07 |
| x^2y^2 | 4 | laplacian | 1.963e+02 | 1.803e+02 | n/a | 1.059e-01 | 6.077e-03 |
| x^3 | 3 | gradient | 1.214e-01 | 1.319e-01 | 1.426e-01 | 2.760e-03 | 1.645e-11 |
| x^3 | 3 | hessian | 2.632e+00 | 2.819e+00 | n/a | 1.899e-01 | 1.812e-09 |
| x^3 | 3 | interpolate | 2.653e-03 | 2.402e-01 | 2.653e-03 | 1.605e-05 | 1.366e-14 |
| x^3 | 3 | laplacian | 2.055e+02 | 1.870e+02 | n/a | 1.901e-01 | 1.380e-09 |
| x^3y | 4 | gradient | 1.175e-01 | 1.416e-01 | 1.523e-01 | 4.578e-03 | 1.063e-04 |
| x^3y | 4 | hessian | 2.469e+00 | 2.624e+00 | n/a | 2.859e-01 | 9.971e-03 |
| x^3y | 4 | interpolate | 2.644e-03 | 2.305e-01 | 2.644e-03 | 2.809e-05 | 3.634e-07 |
| x^3y | 4 | laplacian | 2.311e+02 | 2.111e+02 | n/a | 2.452e-01 | 9.583e-03 |
| x^4 | 4 | gradient | 2.359e-01 | 2.558e-01 | 2.765e-01 | 9.352e-03 | 4.176e-04 |
| x^4 | 4 | hessian | 5.458e+00 | 5.772e+00 | n/a | 7.167e-01 | 3.586e-02 |
| x^4 | 4 | interpolate | 4.893e-03 | 2.479e-01 | 4.893e-03 | 6.005e-05 | 1.458e-06 |
| x^4 | 4 | laplacian | 2.697e+02 | 2.453e+02 | n/a | 7.367e-01 | 4.333e-02 |
| xy | 2 | gradient | 1.320e-02 | 2.120e-02 | 2.093e-02 | 6.070e-13 | 1.641e-11 |
| xy | 2 | hessian | 4.095e-01 | 5.869e-01 | n/a | 3.171e-11 | 1.826e-09 |
| xy | 2 | interpolate | 2.155e-04 | 2.195e-01 | 2.155e-04 | 5.551e-15 | 8.438e-15 |
| xy | 2 | laplacian | 1.058e+02 | 9.731e+01 | n/a | 3.200e-11 | 1.182e-09 |
| xy^2 | 3 | gradient | 6.892e-02 | 7.691e-02 | 7.689e-02 | 1.344e-03 | 1.629e-11 |
| xy^2 | 3 | hessian | 1.660e+00 | 1.430e+00 | n/a | 8.470e-02 | 1.692e-09 |
| xy^2 | 3 | interpolate | 9.070e-04 | 2.135e-01 | 9.070e-04 | 1.092e-05 | 5.773e-15 |
| xy^2 | 3 | laplacian | 1.371e+02 | 1.266e+02 | n/a | 5.363e-02 | 1.298e-09 |
| xy^3 | 4 | gradient | 1.621e-01 | 1.763e-01 | 1.734e-01 | 5.243e-03 | 7.881e-05 |
| xy^3 | 4 | hessian | 3.883e+00 | 3.412e+00 | n/a | 3.528e-01 | 6.554e-03 |
| xy^3 | 4 | interpolate | 2.050e-03 | 2.072e-01 | 2.050e-03 | 4.131e-05 | 2.120e-07 |
| xy^3 | 4 | laplacian | 1.649e+02 | 1.529e+02 | n/a | 2.563e-01 | 5.222e-03 |
| y | 1 | gradient | 8.882e-16 | 4.450e-16 | 1.772e-13 | 6.189e-13 | 1.459e-11 |
| y | 1 | interpolate | 3.331e-16 | 2.132e-01 | 5.884e-15 | 5.884e-15 | 2.165e-14 |
| y^2 | 2 | gradient | 4.875e-02 | 5.139e-02 | 4.987e-02 | 6.016e-13 | 1.792e-11 |
| y^2 | 2 | hessian | 1.034e+00 | 1.063e+00 | n/a | 3.038e-11 | 1.968e-09 |
| y^2 | 2 | interpolate | 8.915e-04 | 2.077e-01 | 8.915e-04 | 3.886e-15 | 1.343e-14 |
| y^2 | 2 | laplacian | 1.147e+02 | 1.028e+02 | n/a | 3.049e-11 | 1.416e-09 |
| y^3 | 3 | gradient | 1.358e-01 | 1.431e-01 | 1.394e-01 | 2.954e-03 | 1.904e-11 |
| y^3 | 3 | hessian | 3.059e+00 | 2.967e+00 | n/a | 1.958e-01 | 1.805e-09 |
| y^3 | 3 | interpolate | 2.320e-03 | 2.020e-01 | 2.320e-03 | 1.864e-05 | 8.715e-15 |
| y^3 | 3 | laplacian | 1.630e+02 | 1.461e+02 | n/a | 2.030e-01 | 1.218e-09 |
| y^4 | 4 | gradient | 2.523e-01 | 2.655e-01 | 2.598e-01 | 1.002e-02 | 2.034e-04 |
| y^4 | 4 | hessian | 6.012e+00 | 5.845e+00 | n/a | 7.264e-01 | 2.048e-02 |
| y^4 | 4 | interpolate | 4.145e-03 | 1.962e-01 | 4.145e-03 | 6.202e-05 | 1.637e-06 |
| y^4 | 4 | laplacian | 2.060e+02 | 1.846e+02 | n/a | 7.526e-01 | 2.085e-02 |

## Observed orders -- resolve-open (error vs dx)

| probe | field | mode | region | slope | r2 | saturated |
|---|---|---|---|---|---|---|
| gradient | 1 | crk | boundary | exact | exact | no |
| gradient | 1 | crk | interior | exact | exact | no |
| gradient | 1 | renorm | boundary | exact | exact | no |
| gradient | 1 | renorm | interior | exact | exact | no |
| gradient | 1 | rkpm1 | boundary | -1.43 | 0.728 | no |
| gradient | 1 | rkpm1 | interior | -1.21 | 0.859 | no |
| gradient | 1 | rkpm2 | boundary | -0.76 | 0.559 | no |
| gradient | 1 | rkpm2 | interior | -1.54 | 0.946 | no |
| gradient | 1 | rkpm3 | boundary | -1.47 | 0.946 | no |
| gradient | 1 | rkpm3 | interior | -1.36 | 0.827 | no |
| gradient | gauss | crk | boundary | 1.63 | 0.980 | no |
| gradient | gauss | crk | interior | 1.60 | 0.999 | no |
| gradient | gauss | renorm | boundary | 1.69 | 0.975 | no |
| gradient | gauss | renorm | interior | 1.76 | 0.998 | no |
| gradient | gauss | rkpm1 | boundary | 1.68 | 0.979 | no |
| gradient | gauss | rkpm1 | interior | 1.69 | 1.000 | no |
| gradient | gauss | rkpm2 | boundary | 2.42 | 0.996 | no |
| gradient | gauss | rkpm2 | interior | 1.86 | 0.998 | no |
| gradient | gauss | rkpm3 | boundary | 3.28 | 0.997 | no |
| gradient | gauss | rkpm3 | interior | 3.31 | 1.000 | no |
| gradient | sin_cos | crk | boundary | 0.76 | 0.971 | no |
| gradient | sin_cos | crk | interior | 1.70 | 0.996 | no |
| gradient | sin_cos | renorm | boundary | 0.86 | 0.986 | no |
| gradient | sin_cos | renorm | interior | 1.86 | 0.997 | no |
| gradient | sin_cos | rkpm1 | boundary | 0.77 | 0.976 | no |
| gradient | sin_cos | rkpm1 | interior | 1.77 | 0.997 | no |
| gradient | sin_cos | rkpm2 | boundary | 1.92 | 0.995 | no |
| gradient | sin_cos | rkpm2 | interior | 1.94 | 0.999 | no |
| gradient | sin_cos | rkpm3 | boundary | 2.75 | 0.992 | no |
| gradient | sin_cos | rkpm3 | interior | 2.98 | 0.994 | no |
| gradient | x | crk | boundary | n/a | n/a | yes |
| gradient | x | crk | interior | n/a | n/a | yes |
| gradient | x | renorm | boundary | n/a | n/a | yes |
| gradient | x | renorm | interior | n/a | n/a | yes |
| gradient | x | rkpm1 | boundary | -2.67 | 0.976 | no |
| gradient | x | rkpm1 | interior | -1.70 | 0.924 | no |
| gradient | x | rkpm2 | boundary | -1.04 | 0.351 | no |
| gradient | x | rkpm2 | interior | -1.56 | 0.934 | no |
| gradient | x | rkpm3 | boundary | -2.01 | 0.922 | no |
| gradient | x | rkpm3 | interior | -1.77 | 0.726 | no |
| gradient | x^2 | crk | boundary | 1.00 | 1.000 | no |
| gradient | x^2 | crk | interior | n/a | n/a | yes |
| gradient | x^2 | renorm | boundary | 1.01 | 0.999 | no |
| gradient | x^2 | renorm | interior | 1.03 | 0.998 | no |
| gradient | x^2 | rkpm1 | boundary | 1.00 | 1.000 | no |
| gradient | x^2 | rkpm1 | interior | 0.84 | 0.948 | no |
| gradient | x^2 | rkpm2 | boundary | -0.81 | 0.216 | no |
| gradient | x^2 | rkpm2 | interior | -1.92 | 0.953 | no |
| gradient | x^2 | rkpm3 | boundary | -2.32 | 0.926 | no |
| gradient | x^2 | rkpm3 | interior | -2.15 | 0.666 | no |
| gradient | xy | crk | boundary | 0.80 | 0.996 | no |
| gradient | xy | crk | interior | 0.98 | 0.968 | no |
| gradient | xy | renorm | boundary | 0.95 | 0.979 | no |
| gradient | xy | renorm | interior | n/a | n/a | yes |
| gradient | xy | rkpm1 | boundary | 0.94 | 0.996 | no |
| gradient | xy | rkpm1 | interior | n/a | n/a | yes |
| gradient | xy | rkpm2 | boundary | -0.95 | 0.368 | no |
| gradient | xy | rkpm2 | interior | -2.29 | 0.901 | no |
| gradient | xy | rkpm3 | boundary | -1.67 | 0.825 | no |
| gradient | xy | rkpm3 | interior | -2.19 | 0.769 | no |
| gradient | y | crk | boundary | n/a | n/a | yes |
| gradient | y | crk | interior | n/a | n/a | yes |
| gradient | y | renorm | boundary | n/a | n/a | yes |
| gradient | y | renorm | interior | n/a | n/a | yes |
| gradient | y | rkpm1 | boundary | -1.41 | 0.698 | no |
| gradient | y | rkpm1 | interior | -1.28 | 0.801 | no |
| gradient | y | rkpm2 | boundary | -0.84 | 0.542 | no |
| gradient | y | rkpm2 | interior | -1.94 | 0.901 | no |
| gradient | y | rkpm3 | boundary | -1.81 | 0.903 | no |
| gradient | y | rkpm3 | interior | -1.89 | 0.741 | no |
| gradient | y^2 | crk | boundary | 1.06 | 0.994 | no |
| gradient | y^2 | crk | interior | 0.78 | 0.960 | no |
| gradient | y^2 | renorm | boundary | 1.05 | 0.996 | no |
| gradient | y^2 | renorm | interior | 0.94 | 0.995 | no |
| gradient | y^2 | rkpm1 | boundary | 1.01 | 0.998 | no |
| gradient | y^2 | rkpm1 | interior | 0.72 | 0.951 | no |
| gradient | y^2 | rkpm2 | boundary | -1.03 | 0.634 | no |
| gradient | y^2 | rkpm2 | interior | -2.04 | 0.942 | no |
| gradient | y^2 | rkpm3 | boundary | -1.65 | 0.806 | no |
| gradient | y^2 | rkpm3 | interior | -2.46 | 0.897 | no |
| hessian | gauss | crk | boundary | 2.48 | 0.932 | no |
| hessian | gauss | crk | interior | 1.54 | 0.999 | no |
| hessian | gauss | renorm | boundary | 2.35 | 0.870 | no |
| hessian | gauss | renorm | interior | 1.56 | 0.998 | no |
| hessian | gauss | rkpm2 | boundary | 1.46 | 0.983 | no |
| hessian | gauss | rkpm2 | interior | 1.74 | 0.997 | no |
| hessian | gauss | rkpm3 | boundary | 2.15 | 0.995 | no |
| hessian | gauss | rkpm3 | interior | 1.82 | 0.997 | no |
| hessian | sin_cos | crk | boundary | n/a | n/a | yes |
| hessian | sin_cos | crk | interior | 1.07 | 0.935 | no |
| hessian | sin_cos | renorm | boundary | n/a | n/a | yes |
| hessian | sin_cos | renorm | interior | 1.03 | 0.897 | no |
| hessian | sin_cos | rkpm2 | boundary | 0.92 | 0.995 | no |
| hessian | sin_cos | rkpm2 | interior | 1.59 | 0.987 | no |
| hessian | sin_cos | rkpm3 | boundary | 2.12 | 0.980 | no |
| hessian | sin_cos | rkpm3 | interior | 1.93 | 0.997 | no |
| hessian | x^2 | crk | boundary | n/a | n/a | yes |
| hessian | x^2 | crk | interior | n/a | n/a | yes |
| hessian | x^2 | renorm | boundary | n/a | n/a | yes |
| hessian | x^2 | renorm | interior | n/a | n/a | yes |
| hessian | x^2 | rkpm2 | boundary | -1.95 | 0.630 | no |
| hessian | x^2 | rkpm2 | interior | -3.15 | 0.973 | no |
| hessian | x^2 | rkpm3 | boundary | -3.10 | 0.981 | no |
| hessian | x^2 | rkpm3 | interior | -3.01 | 0.978 | no |
| hessian | xy | crk | boundary | n/a | n/a | yes |
| hessian | xy | crk | interior | n/a | n/a | yes |
| hessian | xy | renorm | boundary | n/a | n/a | yes |
| hessian | xy | renorm | interior | n/a | n/a | yes |
| hessian | xy | rkpm2 | boundary | -1.95 | 0.683 | no |
| hessian | xy | rkpm2 | interior | -3.02 | 0.957 | no |
| hessian | xy | rkpm3 | boundary | -3.09 | 0.907 | no |
| hessian | xy | rkpm3 | interior | -3.33 | 0.946 | no |
| hessian | y^2 | crk | boundary | n/a | n/a | yes |
| hessian | y^2 | crk | interior | n/a | n/a | yes |
| hessian | y^2 | renorm | boundary | n/a | n/a | yes |
| hessian | y^2 | renorm | interior | n/a | n/a | yes |
| hessian | y^2 | rkpm2 | boundary | -2.38 | 0.891 | no |
| hessian | y^2 | rkpm2 | interior | -2.96 | 0.954 | no |
| hessian | y^2 | rkpm3 | boundary | -3.09 | 0.975 | no |
| hessian | y^2 | rkpm3 | interior | -2.50 | 0.709 | no |
| interpolate | 1 | crk | boundary | n/a | n/a | yes |
| interpolate | 1 | crk | interior | n/a | n/a | yes |
| interpolate | 1 | renorm | boundary | n/a | n/a | yes |
| interpolate | 1 | renorm | interior | n/a | n/a | yes |
| interpolate | 1 | rkpm1 | boundary | -0.44 | 0.185 | no |
| interpolate | 1 | rkpm1 | interior | -1.36 | 0.914 | no |
| interpolate | 1 | rkpm2 | boundary | n/a | n/a | yes |
| interpolate | 1 | rkpm2 | interior | -1.88 | 0.798 | no |
| interpolate | 1 | rkpm3 | boundary | -1.86 | 0.926 | no |
| interpolate | 1 | rkpm3 | interior | -1.19 | 0.822 | no |
| interpolate | gauss | crk | boundary | 2.95 | 0.999 | no |
| interpolate | gauss | crk | interior | 1.84 | 1.000 | no |
| interpolate | gauss | renorm | boundary | 2.68 | 0.994 | no |
| interpolate | gauss | renorm | interior | 0.98 | 0.892 | no |
| interpolate | gauss | rkpm1 | boundary | 2.95 | 0.999 | no |
| interpolate | gauss | rkpm1 | interior | 1.84 | 1.000 | no |
| interpolate | gauss | rkpm2 | boundary | 3.26 | 0.999 | no |
| interpolate | gauss | rkpm2 | interior | 3.08 | 0.983 | no |
| interpolate | gauss | rkpm3 | boundary | 5.19 | 0.965 | no |
| interpolate | gauss | rkpm3 | interior | 3.80 | 1.000 | no |
| interpolate | sin_cos | crk | boundary | 1.80 | 0.999 | no |
| interpolate | sin_cos | crk | interior | 1.95 | 1.000 | no |
| interpolate | sin_cos | renorm | boundary | n/a | n/a | yes |
| interpolate | sin_cos | renorm | interior | 0.95 | 0.974 | no |
| interpolate | sin_cos | rkpm1 | boundary | 1.80 | 0.999 | no |
| interpolate | sin_cos | rkpm1 | interior | 1.95 | 1.000 | no |
| interpolate | sin_cos | rkpm2 | boundary | 2.82 | 1.000 | no |
| interpolate | sin_cos | rkpm2 | interior | 2.91 | 0.998 | no |
| interpolate | sin_cos | rkpm3 | boundary | 3.88 | 0.999 | no |
| interpolate | sin_cos | rkpm3 | interior | 3.91 | 1.000 | no |
| interpolate | x | crk | boundary | n/a | n/a | yes |
| interpolate | x | crk | interior | n/a | n/a | yes |
| interpolate | x | renorm | boundary | n/a | n/a | yes |
| interpolate | x | renorm | interior | n/a | n/a | yes |
| interpolate | x | rkpm1 | boundary | -0.97 | 0.590 | no |
| interpolate | x | rkpm1 | interior | -2.06 | 0.938 | no |
| interpolate | x | rkpm2 | boundary | -1.60 | 0.432 | no |
| interpolate | x | rkpm2 | interior | -0.94 | 0.920 | no |
| interpolate | x | rkpm3 | boundary | -1.57 | 0.942 | no |
| interpolate | x | rkpm3 | interior | -1.43 | 0.850 | no |
| interpolate | x^2 | crk | boundary | 1.97 | 0.998 | no |
| interpolate | x^2 | crk | interior | 1.97 | 1.000 | no |
| interpolate | x^2 | renorm | boundary | n/a | n/a | yes |
| interpolate | x^2 | renorm | interior | n/a | n/a | yes |
| interpolate | x^2 | rkpm1 | boundary | 1.97 | 0.998 | no |
| interpolate | x^2 | rkpm1 | interior | 1.97 | 1.000 | no |
| interpolate | x^2 | rkpm2 | boundary | -1.82 | 0.469 | no |
| interpolate | x^2 | rkpm2 | interior | -0.87 | 0.987 | no |
| interpolate | x^2 | rkpm3 | boundary | -1.50 | 0.804 | no |
| interpolate | x^2 | rkpm3 | interior | -1.41 | 0.777 | no |
| interpolate | xy | crk | boundary | 2.11 | 0.991 | no |
| interpolate | xy | crk | interior | 1.87 | 0.998 | no |
| interpolate | xy | renorm | boundary | n/a | n/a | yes |
| interpolate | xy | renorm | interior | n/a | n/a | yes |
| interpolate | xy | rkpm1 | boundary | 2.11 | 0.991 | no |
| interpolate | xy | rkpm1 | interior | 1.87 | 0.998 | no |
| interpolate | xy | rkpm2 | boundary | -0.37 | 0.024 | no |
| interpolate | xy | rkpm2 | interior | -0.79 | 0.874 | no |
| interpolate | xy | rkpm3 | boundary | -1.74 | 0.569 | no |
| interpolate | xy | rkpm3 | interior | -0.81 | 0.605 | no |
| interpolate | y | crk | boundary | n/a | n/a | yes |
| interpolate | y | crk | interior | n/a | n/a | yes |
| interpolate | y | renorm | boundary | n/a | n/a | yes |
| interpolate | y | renorm | interior | n/a | n/a | yes |
| interpolate | y | rkpm1 | boundary | -0.36 | 0.115 | no |
| interpolate | y | rkpm1 | interior | -1.34 | 0.998 | no |
| interpolate | y | rkpm2 | boundary | -0.79 | 0.570 | no |
| interpolate | y | rkpm2 | interior | -1.27 | 0.471 | no |
| interpolate | y | rkpm3 | boundary | -2.16 | 0.930 | no |
| interpolate | y | rkpm3 | interior | n/a | n/a | yes |
| interpolate | y^2 | crk | boundary | 1.93 | 0.999 | no |
| interpolate | y^2 | crk | interior | 1.99 | 1.000 | no |
| interpolate | y^2 | renorm | boundary | n/a | n/a | yes |
| interpolate | y^2 | renorm | interior | n/a | n/a | yes |
| interpolate | y^2 | rkpm1 | boundary | 1.93 | 0.999 | no |
| interpolate | y^2 | rkpm1 | interior | 1.99 | 1.000 | no |
| interpolate | y^2 | rkpm2 | boundary | -0.97 | 0.950 | no |
| interpolate | y^2 | rkpm2 | interior | -1.36 | 0.510 | no |
| interpolate | y^2 | rkpm3 | boundary | -2.40 | 0.950 | no |
| interpolate | y^2 | rkpm3 | interior | -0.80 | 0.977 | no |
| laplacian | gauss | crk | boundary | n/a | n/a | yes |
| laplacian | gauss | crk | interior | 1.17 | 0.888 | no |
| laplacian | gauss | renorm | boundary | n/a | n/a | yes |
| laplacian | gauss | renorm | interior | n/a | n/a | yes |
| laplacian | gauss | rkpm2 | boundary | 1.44 | 0.987 | no |
| laplacian | gauss | rkpm2 | interior | 1.74 | 0.997 | no |
| laplacian | gauss | rkpm3 | boundary | 2.12 | 0.994 | no |
| laplacian | gauss | rkpm3 | interior | 1.82 | 0.997 | no |
| laplacian | sin_cos | crk | boundary | -1.23 | 0.982 | no |
| laplacian | sin_cos | crk | interior | n/a | n/a | yes |
| laplacian | sin_cos | renorm | boundary | -1.20 | 0.988 | no |
| laplacian | sin_cos | renorm | interior | -1.06 | 0.858 | no |
| laplacian | sin_cos | rkpm2 | boundary | 0.69 | 0.963 | no |
| laplacian | sin_cos | rkpm2 | interior | 1.62 | 0.986 | no |
| laplacian | sin_cos | rkpm3 | boundary | 2.01 | 0.980 | no |
| laplacian | sin_cos | rkpm3 | interior | 1.95 | 0.998 | no |
| laplacian | x^2 | crk | boundary | -1.09 | 0.974 | no |
| laplacian | x^2 | crk | interior | -1.68 | 0.999 | no |
| laplacian | x^2 | renorm | boundary | -1.06 | 0.984 | no |
| laplacian | x^2 | renorm | interior | -1.67 | 0.999 | no |
| laplacian | x^2 | rkpm2 | boundary | -2.93 | 0.718 | no |
| laplacian | x^2 | rkpm2 | interior | -2.68 | 0.990 | no |
| laplacian | x^2 | rkpm3 | boundary | -3.03 | 0.872 | no |
| laplacian | x^2 | rkpm3 | interior | -2.83 | 0.964 | no |
| laplacian | xy | crk | boundary | -1.13 | 0.999 | no |
| laplacian | xy | crk | interior | -1.21 | 0.936 | no |
| laplacian | xy | renorm | boundary | -1.12 | 1.000 | no |
| laplacian | xy | renorm | interior | -1.51 | 0.994 | no |
| laplacian | xy | rkpm2 | boundary | -2.50 | 0.749 | no |
| laplacian | xy | rkpm2 | interior | -2.69 | 0.988 | no |
| laplacian | xy | rkpm3 | boundary | -3.34 | 0.948 | no |
| laplacian | xy | rkpm3 | interior | -2.73 | 0.949 | no |
| laplacian | y^2 | crk | boundary | -1.08 | 0.991 | no |
| laplacian | y^2 | crk | interior | -1.23 | 0.903 | no |
| laplacian | y^2 | renorm | boundary | -1.08 | 0.990 | no |
| laplacian | y^2 | renorm | interior | -1.32 | 0.957 | no |
| laplacian | y^2 | rkpm2 | boundary | -2.55 | 0.937 | no |
| laplacian | y^2 | rkpm2 | interior | -3.09 | 0.902 | no |
| laplacian | y^2 | rkpm3 | boundary | -3.40 | 0.989 | no |
| laplacian | y^2 | rkpm3 | interior | -2.96 | 0.992 | no |

## Observed orders -- resolve-periodic (error vs dx)

| probe | field | mode | region | slope | r2 | saturated |
|---|---|---|---|---|---|---|
| gradient | gauss_periodic | crk | interior | 1.60 | 0.999 | no |
| gradient | gauss_periodic | renorm | interior | 1.76 | 0.998 | no |
| gradient | gauss_periodic | rkpm1 | interior | 1.69 | 1.000 | no |
| gradient | gauss_periodic | rkpm2 | interior | 1.86 | 0.998 | no |
| gradient | gauss_periodic | rkpm3 | interior | 3.31 | 1.000 | no |
| gradient | sin_a | crk | interior | 1.39 | 0.994 | no |
| gradient | sin_a | renorm | interior | 1.72 | 1.000 | no |
| gradient | sin_a | rkpm1 | interior | 1.65 | 1.000 | no |
| gradient | sin_a | rkpm2 | interior | 1.97 | 1.000 | no |
| gradient | sin_a | rkpm3 | interior | 2.91 | 0.997 | no |
| gradient | sin_b | crk | interior | 1.39 | 0.983 | no |
| gradient | sin_b | renorm | interior | 1.70 | 0.984 | no |
| gradient | sin_b | rkpm1 | interior | 1.50 | 0.973 | no |
| gradient | sin_b | rkpm2 | interior | 1.96 | 0.994 | no |
| gradient | sin_b | rkpm3 | interior | 2.98 | 0.975 | no |
| gradient | sin_cos | crk | interior | 1.69 | 0.996 | no |
| gradient | sin_cos | renorm | interior | 1.83 | 0.998 | no |
| gradient | sin_cos | rkpm1 | interior | 1.76 | 0.997 | no |
| gradient | sin_cos | rkpm2 | interior | 1.93 | 0.999 | no |
| gradient | sin_cos | rkpm3 | interior | 3.09 | 0.998 | no |
| hessian | gauss_periodic | crk | interior | 1.54 | 0.999 | no |
| hessian | gauss_periodic | renorm | interior | 1.56 | 0.998 | no |
| hessian | gauss_periodic | rkpm2 | interior | 1.74 | 0.997 | no |
| hessian | gauss_periodic | rkpm3 | interior | 1.82 | 0.997 | no |
| hessian | sin_a | crk | interior | 1.27 | 0.993 | no |
| hessian | sin_a | renorm | interior | 1.33 | 0.989 | no |
| hessian | sin_a | rkpm2 | interior | 1.39 | 0.981 | no |
| hessian | sin_a | rkpm3 | interior | 1.96 | 1.000 | no |
| hessian | sin_b | crk | interior | 1.22 | 0.942 | no |
| hessian | sin_b | renorm | interior | 1.40 | 0.974 | no |
| hessian | sin_b | rkpm2 | interior | 1.29 | 0.953 | no |
| hessian | sin_b | rkpm3 | interior | 1.93 | 0.989 | no |
| hessian | sin_cos | crk | interior | 1.56 | 0.995 | no |
| hessian | sin_cos | renorm | interior | 1.52 | 0.985 | no |
| hessian | sin_cos | rkpm2 | interior | 1.59 | 0.990 | no |
| hessian | sin_cos | rkpm3 | interior | 1.92 | 0.998 | no |
| interpolate | gauss_periodic | crk | interior | 1.84 | 1.000 | no |
| interpolate | gauss_periodic | renorm | interior | 0.98 | 0.892 | no |
| interpolate | gauss_periodic | rkpm1 | interior | 1.84 | 1.000 | no |
| interpolate | gauss_periodic | rkpm2 | interior | 3.08 | 0.983 | no |
| interpolate | gauss_periodic | rkpm3 | interior | 3.80 | 1.000 | no |
| interpolate | sin_a | crk | interior | 1.97 | 1.000 | no |
| interpolate | sin_a | renorm | interior | n/a | n/a | yes |
| interpolate | sin_a | rkpm1 | interior | 1.97 | 1.000 | no |
| interpolate | sin_a | rkpm2 | interior | 2.73 | 0.995 | no |
| interpolate | sin_a | rkpm3 | interior | 3.96 | 1.000 | no |
| interpolate | sin_b | crk | interior | 1.94 | 0.998 | no |
| interpolate | sin_b | renorm | interior | n/a | n/a | yes |
| interpolate | sin_b | rkpm1 | interior | 1.94 | 0.998 | no |
| interpolate | sin_b | rkpm2 | interior | 2.82 | 0.979 | no |
| interpolate | sin_b | rkpm3 | interior | 3.91 | 0.997 | no |
| interpolate | sin_cos | crk | interior | 1.94 | 1.000 | no |
| interpolate | sin_cos | renorm | interior | 0.85 | 0.934 | no |
| interpolate | sin_cos | rkpm1 | interior | 1.94 | 1.000 | no |
| interpolate | sin_cos | rkpm2 | interior | 3.05 | 0.995 | no |
| interpolate | sin_cos | rkpm3 | interior | 3.88 | 1.000 | no |
| laplacian | gauss_periodic | crk | interior | 1.17 | 0.888 | no |
| laplacian | gauss_periodic | renorm | interior | n/a | n/a | yes |
| laplacian | gauss_periodic | rkpm2 | interior | 1.74 | 0.997 | no |
| laplacian | gauss_periodic | rkpm3 | interior | 1.82 | 0.997 | no |
| laplacian | sin_a | crk | interior | -0.83 | 0.817 | no |
| laplacian | sin_a | renorm | interior | -1.36 | 0.985 | no |
| laplacian | sin_a | rkpm2 | interior | 1.29 | 0.976 | no |
| laplacian | sin_a | rkpm3 | interior | 1.96 | 1.000 | no |
| laplacian | sin_b | crk | interior | -0.78 | 0.838 | no |
| laplacian | sin_b | renorm | interior | -1.19 | 0.999 | no |
| laplacian | sin_b | rkpm2 | interior | 1.18 | 0.937 | no |
| laplacian | sin_b | rkpm3 | interior | 1.92 | 0.990 | no |
| laplacian | sin_cos | crk | interior | n/a | n/a | yes |
| laplacian | sin_cos | renorm | interior | -1.14 | 0.863 | no |
| laplacian | sin_cos | rkpm2 | interior | 1.59 | 0.993 | no |
| laplacian | sin_cos | rkpm3 | interior | 1.95 | 0.998 | no |

## Observed orders -- smoothing (error vs h)

| probe | field | mode | region | slope | r2 | saturated |
|---|---|---|---|---|---|---|
| gradient | 1 | crk | boundary | exact | exact | no |
| gradient | 1 | crk | interior | exact | exact | no |
| gradient | 1 | renorm | boundary | exact | exact | no |
| gradient | 1 | renorm | interior | exact | exact | no |
| gradient | 1 | rkpm1 | boundary | 0.14 | 0.011 | no |
| gradient | 1 | rkpm1 | interior | -0.90 | 0.880 | no |
| gradient | 1 | rkpm2 | boundary | -18.85 | 0.638 | no |
| gradient | 1 | rkpm2 | interior | -1.07 | 0.854 | no |
| gradient | 1 | rkpm3 | boundary | -21.91 | 0.893 | no |
| gradient | 1 | rkpm3 | interior | -22.43 | 0.645 | no |
| gradient | gauss | crk | boundary | n/a | n/a | yes |
| gradient | gauss | crk | interior | n/a | n/a | yes |
| gradient | gauss | renorm | boundary | 1.01 | 0.995 | no |
| gradient | gauss | renorm | interior | 1.02 | 0.906 | no |
| gradient | gauss | rkpm1 | boundary | 1.07 | 0.982 | no |
| gradient | gauss | rkpm1 | interior | 0.90 | 0.962 | no |
| gradient | gauss | rkpm2 | boundary | -0.69 | 0.114 | no |
| gradient | gauss | rkpm2 | interior | 1.57 | 0.991 | no |
| gradient | gauss | rkpm3 | boundary | -3.17 | 0.810 | no |
| gradient | gauss | rkpm3 | interior | -3.33 | 0.343 | no |
| gradient | sin_cos | crk | boundary | n/a | n/a | yes |
| gradient | sin_cos | crk | interior | n/a | n/a | yes |
| gradient | sin_cos | renorm | boundary | n/a | n/a | yes |
| gradient | sin_cos | renorm | interior | n/a | n/a | yes |
| gradient | sin_cos | rkpm1 | boundary | n/a | n/a | yes |
| gradient | sin_cos | rkpm1 | interior | n/a | n/a | yes |
| gradient | sin_cos | rkpm2 | boundary | -2.65 | 0.374 | no |
| gradient | sin_cos | rkpm2 | interior | 1.57 | 0.986 | no |
| gradient | sin_cos | rkpm3 | boundary | -6.81 | 0.879 | no |
| gradient | sin_cos | rkpm3 | interior | -6.75 | 0.539 | no |
| gradient | x | crk | boundary | 0.43 | 0.247 | no |
| gradient | x | crk | interior | n/a | n/a | yes |
| gradient | x | renorm | boundary | n/a | n/a | yes |
| gradient | x | renorm | interior | n/a | n/a | yes |
| gradient | x | rkpm1 | boundary | -0.76 | 0.451 | no |
| gradient | x | rkpm1 | interior | -0.94 | 0.901 | no |
| gradient | x | rkpm2 | boundary | -30.88 | 0.633 | no |
| gradient | x | rkpm2 | interior | -2.27 | 0.923 | no |
| gradient | x | rkpm3 | boundary | -37.09 | 0.838 | no |
| gradient | x | rkpm3 | interior | -30.77 | 0.624 | no |
| gradient | y | crk | boundary | n/a | n/a | yes |
| gradient | y | crk | interior | n/a | n/a | yes |
| gradient | y | renorm | boundary | n/a | n/a | yes |
| gradient | y | renorm | interior | n/a | n/a | yes |
| gradient | y | rkpm1 | boundary | 0.25 | 0.014 | no |
| gradient | y | rkpm1 | interior | -1.23 | 0.920 | no |
| gradient | y | rkpm2 | boundary | -29.31 | 0.606 | no |
| gradient | y | rkpm2 | interior | -0.92 | 0.508 | no |
| gradient | y | rkpm3 | boundary | -37.49 | 0.839 | no |
| gradient | y | rkpm3 | interior | -30.13 | 0.591 | no |
| interpolate | 1 | crk | boundary | n/a | n/a | yes |
| interpolate | 1 | crk | interior | n/a | n/a | yes |
| interpolate | 1 | renorm | boundary | 1.47 | 0.986 | no |
| interpolate | 1 | renorm | interior | -1.18 | 0.938 | no |
| interpolate | 1 | rkpm1 | boundary | 0.71 | 0.241 | no |
| interpolate | 1 | rkpm1 | interior | n/a | n/a | yes |
| interpolate | 1 | rkpm2 | boundary | n/a | n/a | yes |
| interpolate | 1 | rkpm2 | interior | -2.19 | 0.956 | no |
| interpolate | 1 | rkpm3 | boundary | n/a | n/a | yes |
| interpolate | 1 | rkpm3 | interior | -2.15 | 0.641 | no |
| interpolate | gauss | crk | boundary | 3.02 | 0.998 | no |
| interpolate | gauss | crk | interior | 2.05 | 1.000 | no |
| interpolate | gauss | renorm | boundary | 1.78 | 0.998 | no |
| interpolate | gauss | renorm | interior | n/a | n/a | yes |
| interpolate | gauss | rkpm1 | boundary | 3.02 | 0.998 | no |
| interpolate | gauss | rkpm1 | interior | 2.05 | 1.000 | no |
| interpolate | gauss | rkpm2 | boundary | 4.58 | 0.923 | no |
| interpolate | gauss | rkpm2 | interior | 3.79 | 0.931 | no |
| interpolate | gauss | rkpm3 | boundary | 7.31 | 0.840 | no |
| interpolate | gauss | rkpm3 | interior | 7.07 | 0.886 | no |
| interpolate | sin_cos | crk | boundary | 1.93 | 0.999 | no |
| interpolate | sin_cos | crk | interior | 2.03 | 0.999 | no |
| interpolate | sin_cos | renorm | boundary | 1.53 | 0.936 | no |
| interpolate | sin_cos | renorm | interior | n/a | n/a | yes |
| interpolate | sin_cos | rkpm1 | boundary | 1.93 | 0.999 | no |
| interpolate | sin_cos | rkpm1 | interior | 2.03 | 0.999 | no |
| interpolate | sin_cos | rkpm2 | boundary | 4.15 | 0.974 | no |
| interpolate | sin_cos | rkpm2 | interior | 2.43 | 0.907 | no |
| interpolate | sin_cos | rkpm3 | boundary | 7.29 | 0.870 | no |
| interpolate | sin_cos | rkpm3 | interior | 7.61 | 0.866 | no |
| interpolate | x | crk | boundary | n/a | n/a | yes |
| interpolate | x | crk | interior | n/a | n/a | yes |
| interpolate | x | renorm | boundary | 1.78 | 0.935 | no |
| interpolate | x | renorm | interior | -1.33 | 0.960 | no |
| interpolate | x | rkpm1 | boundary | n/a | n/a | yes |
| interpolate | x | rkpm1 | interior | n/a | n/a | yes |
| interpolate | x | rkpm2 | boundary | -0.07 | 0.002 | no |
| interpolate | x | rkpm2 | interior | -1.52 | 0.621 | no |
| interpolate | x | rkpm3 | boundary | n/a | n/a | yes |
| interpolate | x | rkpm3 | interior | -5.45 | 0.645 | no |
| interpolate | y | crk | boundary | n/a | n/a | yes |
| interpolate | y | crk | interior | n/a | n/a | yes |
| interpolate | y | renorm | boundary | 1.52 | 0.997 | no |
| interpolate | y | renorm | interior | -1.51 | 0.997 | no |
| interpolate | y | rkpm1 | boundary | 0.63 | 0.106 | no |
| interpolate | y | rkpm1 | interior | n/a | n/a | yes |
| interpolate | y | rkpm2 | boundary | n/a | n/a | yes |
| interpolate | y | rkpm2 | interior | -2.69 | 0.966 | no |
| interpolate | y | rkpm3 | boundary | -0.62 | 0.372 | no |
| interpolate | y | rkpm3 | interior | -5.23 | 0.696 | no |

## p = 1 cross-check against CRK

- `interpolate`: max |rkpm1 - crk| interior 7.55e-15, boundary 9.33e-15 (field scale 9.99e-01)
- `gradient`: max |rkpm1 - crk| interior 6.75e-03, boundary 3.91e-02 (field scale 2.00e+00)

## Moment-matrix conditioning

| order | n_basis | jitter | target_neighbors | region | cond_median | cond_p95 | cond_max | deficient_frac |
|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 0.1 | 12 | interior | 1.11 | 1.18 | 1.25 | 0 |
| 1 | 3 | 0.1 | 12 | boundary | 1.26 | 2.83 | 3.36 | 0 |
| 1 | 3 | 0.1 | 20 | interior | 1.05 | 1.09 | 1.12 | 0 |
| 1 | 3 | 0.1 | 20 | boundary | 1.18 | 3.86 | 4.28 | 0 |
| 1 | 3 | 0.1 | 35 | interior | 1.03 | 1.05 | 1.07 | 0 |
| 1 | 3 | 0.1 | 35 | boundary | 1.11 | 4.58 | 5.36 | 0 |
| 1 | 3 | 0.1 | 60 | interior | 1.02 | 1.03 | 1.04 | 0 |
| 1 | 3 | 0.1 | 60 | boundary | 1.13 | 5.19 | 6.43 | 0 |
| 1 | 3 | 0.3 | 12 | interior | 1.35 | 1.66 | 1.82 | 0 |
| 1 | 3 | 0.3 | 12 | boundary | 1.67 | 3.15 | 4.1 | 0 |
| 1 | 3 | 0.3 | 20 | interior | 1.17 | 1.29 | 1.36 | 0 |
| 1 | 3 | 0.3 | 20 | boundary | 1.25 | 3.86 | 4.59 | 0 |
| 1 | 3 | 0.3 | 35 | interior | 1.09 | 1.15 | 1.23 | 0 |
| 1 | 3 | 0.3 | 35 | boundary | 1.14 | 4.53 | 5.49 | 0 |
| 1 | 3 | 0.3 | 60 | interior | 1.05 | 1.09 | 1.12 | 0 |
| 1 | 3 | 0.3 | 60 | boundary | 1.16 | 5.13 | 6.77 | 0 |
| 1 | 3 | 0.5 | 12 | interior | 1.61 | 2.34 | 3.07 | 0 |
| 1 | 3 | 0.5 | 12 | boundary | 2.07 | 3.88 | 5.82 | 0 |
| 1 | 3 | 0.5 | 20 | interior | 1.28 | 1.54 | 1.72 | 0 |
| 1 | 3 | 0.5 | 20 | boundary | 1.4 | 3.96 | 4.86 | 0 |
| 1 | 3 | 0.5 | 35 | interior | 1.15 | 1.27 | 1.39 | 0 |
| 1 | 3 | 0.5 | 35 | boundary | 1.2 | 4.45 | 5.57 | 0 |
| 1 | 3 | 0.5 | 60 | interior | 1.09 | 1.15 | 1.2 | 0 |
| 1 | 3 | 0.5 | 60 | boundary | 1.18 | 5.07 | 7.04 | 0 |
| 2 | 6 | 0.1 | 12 | interior | 5.11 | 5.54 | 5.89 | 0 |
| 2 | 6 | 0.1 | 12 | boundary | 8.27 | 8.68e+04 | 1.07e+17 | 0.011 |
| 2 | 6 | 0.1 | 20 | interior | 6.67 | 6.79 | 6.89 | 0 |
| 2 | 6 | 0.1 | 20 | boundary | 7.71 | 80.1 | 921 | 0 |
| 2 | 6 | 0.1 | 35 | interior | 6.83 | 7.06 | 7.14 | 0 |
| 2 | 6 | 0.1 | 35 | boundary | 7.82 | 55.1 | 161 | 0 |
| 2 | 6 | 0.1 | 60 | interior | 6.83 | 7 | 7.07 | 0 |
| 2 | 6 | 0.1 | 60 | boundary | 7.71 | 58.7 | 102 | 0 |
| 2 | 6 | 0.3 | 12 | interior | 5.24 | 6.61 | 7.69 | 0 |
| 2 | 6 | 0.3 | 12 | boundary | 15.1 | 8.08e+05 | 2.07e+17 | 0.0258 |
| 2 | 6 | 0.3 | 20 | interior | 6.56 | 7.05 | 7.52 | 0 |
| 2 | 6 | 0.3 | 20 | boundary | 7.87 | 116 | 3.92e+03 | 0 |
| 2 | 6 | 0.3 | 35 | interior | 6.79 | 7.45 | 7.74 | 0 |
| 2 | 6 | 0.3 | 35 | boundary | 7.82 | 67.3 | 218 | 0 |
| 2 | 6 | 0.3 | 60 | interior | 6.79 | 7.3 | 7.56 | 0 |
| 2 | 6 | 0.3 | 60 | boundary | 7.88 | 62.5 | 118 | 0 |
| 2 | 6 | 0.5 | 12 | interior | 5.76 | 8.59 | 12 | 0 |
| 2 | 6 | 0.5 | 12 | boundary | 45.2 | 8.53e+06 | 1.1e+17 | 0.0282 |
| 2 | 6 | 0.5 | 20 | interior | 6.43 | 7.34 | 8.15 | 0 |
| 2 | 6 | 0.5 | 20 | boundary | 8.21 | 161 | 4.39e+04 | 0 |
| 2 | 6 | 0.5 | 35 | interior | 6.68 | 7.77 | 8.32 | 0 |
| 2 | 6 | 0.5 | 35 | boundary | 7.82 | 77.7 | 310 | 0 |
| 2 | 6 | 0.5 | 60 | interior | 6.75 | 7.6 | 8.05 | 0 |
| 2 | 6 | 0.5 | 60 | boundary | 7.85 | 69.7 | 135 | 0 |
| 3 | 10 | 0.1 | 12 | interior | 2.62e+03 | 1.14e+04 | 2.99e+04 | 0 |
| 3 | 10 | 0.1 | 12 | boundary | 8.25e+04 | 2.42e+17 | 1e+18 | 0.385 |
| 3 | 10 | 0.1 | 20 | interior | 15.1 | 16.4 | 17.2 | 0 |
| 3 | 10 | 0.1 | 20 | boundary | 28.5 | 8.88e+05 | 1.16e+18 | 0.0294 |
| 3 | 10 | 0.1 | 35 | interior | 15 | 15.6 | 15.9 | 0 |
| 3 | 10 | 0.1 | 35 | boundary | 30.1 | 3.05e+03 | 6.08e+04 | 0 |
| 3 | 10 | 0.1 | 60 | interior | 14.4 | 14.7 | 15 | 0 |
| 3 | 10 | 0.1 | 60 | boundary | 23.5 | 1.31e+03 | 4.91e+03 | 0 |
| 3 | 10 | 0.3 | 12 | interior | 2.07e+03 | 2.4e+06 | 9.05e+16 | 0.012 |
| 3 | 10 | 0.3 | 12 | boundary | 2.72e+16 | 5.04e+17 | 3.61e+18 | 0.587 |
| 3 | 10 | 0.3 | 20 | interior | 19.4 | 24.6 | 28.8 | 0 |
| 3 | 10 | 0.3 | 20 | boundary | 39.4 | 1.66e+05 | 1.44e+17 | 0.0294 |
| 3 | 10 | 0.3 | 35 | interior | 16.3 | 18.5 | 19.7 | 0 |
| 3 | 10 | 0.3 | 35 | boundary | 33.9 | 3.45e+03 | 1.81e+05 | 0 |
| 3 | 10 | 0.3 | 60 | interior | 15 | 16 | 17.1 | 0 |
| 3 | 10 | 0.3 | 60 | boundary | 24 | 1.49e+03 | 7.57e+03 | 0 |
| 3 | 10 | 0.5 | 12 | interior | 1.21e+03 | 6.4e+06 | 1.85e+18 | 0.0302 |
| 3 | 10 | 0.5 | 12 | boundary | 3.37e+16 | 4.53e+17 | 5.8e+18 | 0.641 |
| 3 | 10 | 0.5 | 20 | interior | 27.9 | 41.7 | 49.1 | 0 |
| 3 | 10 | 0.5 | 20 | boundary | 61.8 | 2.36e+05 | 1.39e+17 | 0.0302 |
| 3 | 10 | 0.5 | 35 | interior | 18.2 | 23 | 26.1 | 0 |
| 3 | 10 | 0.5 | 35 | boundary | 35.5 | 3.07e+03 | 6.06e+05 | 0 |
| 3 | 10 | 0.5 | 60 | interior | 16 | 17.7 | 19.4 | 0 |
| 3 | 10 | 0.5 | 60 | boundary | 25.7 | 1.59e+03 | 1.23e+04 | 0 |
