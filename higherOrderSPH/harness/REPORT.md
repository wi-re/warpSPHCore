# Convergence harness -- baseline report (before column)

Generated 2026-09-22 15:48 by `run_baseline.py` (float64, kernels: Wendland2, jitter 0.3, default target neighbors 40, patch N target 1152).

Frozen reference for the higher-order phases (parent plan `../../higher_order.md`): later phases diff against these numbers and do not re-baseline silently. A harness change is a versioning event that re-runs every phase.

## 1. Monomial patch tests -- open domain, interior

Max error per monomial / probe / mode. ~1e-15 (float64) = exact reproduction.

| kernel | field | field_degree | probe | standard | crk | renorm |
|---|---|---|---|---|---|---|
| Wendland2 | 1 | 0 | gradient | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| Wendland2 | 1 | 0 | interpolate | 5.548e-02 | 6.661e-16 | 5.548e-02 |
| Wendland2 | Mx | 1 | gradient | 1.599e-01 | 1.450e-03 | 2.215e-15 |
| Wendland2 | Mx | 1 | interpolate | 5.764e-02 | 8.899e-16 | 5.764e-02 |
| Wendland2 | x | 1 | gradient | 9.980e-02 | 7.482e-04 | 7.777e-16 |
| Wendland2 | x | 1 | interpolate | 4.091e-02 | 5.551e-16 | 4.091e-02 |
| Wendland2 | x^2 | 2 | gradient | 1.639e-01 | 6.703e-03 | 3.656e-03 |
| Wendland2 | x^2 | 2 | interpolate | 3.492e-02 | 8.212e-04 | 3.492e-02 |
| Wendland2 | x^2 | 2 | laplacian | 7.253e+00 | 7.515e-01 | 7.416e+00 |
| Wendland2 | x^2y | 3 | gradient | 1.268e-01 | 7.311e-03 | 4.886e-03 |
| Wendland2 | x^2y | 3 | interpolate | 2.349e-02 | 6.958e-04 | 2.349e-02 |
| Wendland2 | x^2y | 3 | laplacian | 7.053e+00 | 6.619e-01 | 6.702e+00 |
| Wendland2 | x^2y^2 | 4 | gradient | 1.168e-01 | 1.176e-02 | 6.251e-03 |
| Wendland2 | x^2y^2 | 4 | interpolate | 1.760e-02 | 1.132e-03 | 1.760e-02 |
| Wendland2 | x^2y^2 | 4 | laplacian | 6.590e+00 | 6.730e-01 | 6.255e+00 |
| Wendland2 | x^3 | 3 | gradient | 1.998e-01 | 1.911e-02 | 1.185e-02 |
| Wendland2 | x^3 | 3 | interpolate | 3.158e-02 | 2.146e-03 | 3.158e-02 |
| Wendland2 | x^3 | 3 | laplacian | 9.871e+00 | 1.039e+00 | 1.005e+01 |
| Wendland2 | x^3y | 4 | gradient | 1.460e-01 | 1.463e-02 | 7.374e-03 |
| Wendland2 | x^3y | 4 | interpolate | 2.123e-02 | 1.716e-03 | 2.123e-02 |
| Wendland2 | x^3y | 4 | laplacian | 9.004e+00 | 7.438e-01 | 8.485e+00 |
| Wendland2 | x^4 | 4 | gradient | 2.141e-01 | 3.592e-02 | 2.529e-02 |
| Wendland2 | x^4 | 4 | interpolate | 3.055e-02 | 3.839e-03 | 3.055e-02 |
| Wendland2 | x^4 | 4 | laplacian | 1.195e+01 | 1.288e+00 | 1.212e+01 |
| Wendland2 | xy | 2 | gradient | 9.272e-02 | 3.068e-03 | 2.872e-03 |
| Wendland2 | xy | 2 | interpolate | 2.650e-02 | 5.236e-05 | 2.650e-02 |
| Wendland2 | xy | 2 | laplacian | 5.058e+00 | 5.514e-01 | 5.076e+00 |
| Wendland2 | xy^2 | 3 | gradient | 1.248e-01 | 6.786e-03 | 4.197e-03 |
| Wendland2 | xy^2 | 3 | interpolate | 1.916e-02 | 7.178e-04 | 1.916e-02 |
| Wendland2 | xy^2 | 3 | laplacian | 5.404e+00 | 6.261e-01 | 5.514e+00 |
| Wendland2 | xy^3 | 4 | gradient | 1.585e-01 | 1.499e-02 | 1.076e-02 |
| Wendland2 | xy^3 | 4 | interpolate | 1.800e-02 | 1.679e-03 | 1.800e-02 |
| Wendland2 | xy^3 | 4 | laplacian | 6.351e+00 | 7.307e-01 | 6.434e+00 |
| Wendland2 | y | 1 | gradient | 1.015e-01 | 8.165e-04 | 6.673e-16 |
| Wendland2 | y | 1 | interpolate | 3.802e-02 | 6.661e-16 | 3.802e-02 |
| Wendland2 | y^2 | 2 | gradient | 1.678e-01 | 6.638e-03 | 4.233e-03 |
| Wendland2 | y^2 | 2 | interpolate | 3.399e-02 | 8.292e-04 | 3.399e-02 |
| Wendland2 | y^2 | 2 | laplacian | 6.412e+00 | 8.300e-01 | 6.536e+00 |
| Wendland2 | y^3 | 3 | gradient | 2.233e-01 | 1.687e-02 | 1.329e-02 |
| Wendland2 | y^3 | 3 | interpolate | 3.100e-02 | 2.142e-03 | 3.100e-02 |
| Wendland2 | y^3 | 3 | laplacian | 8.064e+00 | 1.124e+00 | 8.177e+00 |
| Wendland2 | y^4 | 4 | gradient | 2.668e-01 | 3.194e-02 | 2.705e-02 |
| Wendland2 | y^4 | 4 | interpolate | 2.876e-02 | 3.710e-03 | 2.876e-02 |
| Wendland2 | y^4 | 4 | laplacian | 9.511e+00 | 1.354e+00 | 9.164e+00 |

## 2. Monomial patch tests -- open domain, boundary band

Same tests on particles within one support of an open wall (truncated kernel support).

| kernel | field | field_degree | probe | standard | crk | renorm |
|---|---|---|---|---|---|---|
| Wendland2 | 1 | 0 | gradient | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| Wendland2 | 1 | 0 | interpolate | 2.268e-01 | 8.882e-16 | 2.268e-01 |
| Wendland2 | Mx | 1 | gradient | 1.236e+00 | 2.454e-01 | 6.077e-15 |
| Wendland2 | Mx | 1 | interpolate | 3.549e-01 | 9.155e-16 | 3.549e-01 |
| Wendland2 | x | 1 | gradient | 5.782e-01 | 1.473e-01 | 6.685e-16 |
| Wendland2 | x | 1 | interpolate | 2.252e-01 | 6.661e-16 | 2.252e-01 |
| Wendland2 | x^2 | 2 | gradient | 1.154e+00 | 2.872e-01 | 4.715e-02 |
| Wendland2 | x^2 | 2 | interpolate | 2.320e-01 | 9.591e-04 | 2.320e-01 |
| Wendland2 | x^2 | 2 | laplacian | 6.649e+01 | 1.393e+02 | 1.268e+02 |
| Wendland2 | x^2y | 3 | gradient | 9.424e-01 | 3.278e-01 | 6.010e-02 |
| Wendland2 | x^2y | 3 | interpolate | 2.252e-01 | 1.006e-03 | 2.252e-01 |
| Wendland2 | x^2y | 3 | laplacian | 6.607e+01 | 1.695e+02 | 1.551e+02 |
| Wendland2 | x^2y^2 | 4 | gradient | 1.137e+00 | 3.588e-01 | 1.143e-01 |
| Wendland2 | x^2y^2 | 4 | interpolate | 2.187e-01 | 1.308e-03 | 2.187e-01 |
| Wendland2 | x^2y^2 | 4 | laplacian | 7.859e+01 | 1.963e+02 | 1.803e+02 |
| Wendland2 | x^3 | 3 | gradient | 1.726e+00 | 4.212e-01 | 1.319e-01 |
| Wendland2 | x^3 | 3 | interpolate | 2.402e-01 | 2.653e-03 | 2.402e-01 |
| Wendland2 | x^3 | 3 | laplacian | 9.986e+01 | 2.055e+02 | 1.870e+02 |
| Wendland2 | x^3y | 4 | gradient | 1.431e+00 | 4.522e-01 | 1.416e-01 |
| Wendland2 | x^3y | 4 | interpolate | 2.305e-01 | 2.644e-03 | 2.305e-01 |
| Wendland2 | x^3y | 4 | laplacian | 9.111e+01 | 2.311e+02 | 2.111e+02 |
| Wendland2 | x^4 | 4 | gradient | 2.294e+00 | 5.500e-01 | 2.558e-01 |
| Wendland2 | x^4 | 4 | interpolate | 2.479e-01 | 4.893e-03 | 2.479e-01 |
| Wendland2 | x^4 | 4 | laplacian | 1.334e+02 | 2.697e+02 | 2.453e+02 |
| Wendland2 | xy | 2 | gradient | 5.847e-01 | 2.062e-01 | 2.120e-02 |
| Wendland2 | xy | 2 | interpolate | 2.195e-01 | 2.155e-04 | 2.195e-01 |
| Wendland2 | xy | 2 | laplacian | 4.130e+01 | 1.058e+02 | 9.731e+01 |
| Wendland2 | xy^2 | 3 | gradient | 1.062e+00 | 2.817e-01 | 7.691e-02 |
| Wendland2 | xy^2 | 3 | interpolate | 2.135e-01 | 9.070e-04 | 2.135e-01 |
| Wendland2 | xy^2 | 3 | laplacian | 5.570e+01 | 1.371e+02 | 1.266e+02 |
| Wendland2 | xy^3 | 4 | gradient | 1.582e+00 | 4.085e-01 | 1.763e-01 |
| Wendland2 | xy^3 | 4 | interpolate | 2.072e-01 | 2.050e-03 | 2.072e-01 |
| Wendland2 | xy^3 | 4 | laplacian | 7.598e+01 | 1.649e+02 | 1.529e+02 |
| Wendland2 | y | 1 | gradient | 6.501e-01 | 1.209e-01 | 6.668e-16 |
| Wendland2 | y | 1 | interpolate | 2.132e-01 | 4.441e-16 | 2.132e-01 |
| Wendland2 | y^2 | 2 | gradient | 1.247e+00 | 2.191e-01 | 5.139e-02 |
| Wendland2 | y^2 | 2 | interpolate | 2.077e-01 | 8.915e-04 | 2.077e-01 |
| Wendland2 | y^2 | 2 | laplacian | 5.681e+01 | 1.147e+02 | 1.028e+02 |
| Wendland2 | y^3 | 3 | gradient | 1.794e+00 | 3.199e-01 | 1.431e-01 |
| Wendland2 | y^3 | 3 | interpolate | 2.020e-01 | 2.320e-03 | 2.020e-01 |
| Wendland2 | y^3 | 3 | laplacian | 8.205e+01 | 1.630e+02 | 1.461e+02 |
| Wendland2 | y^4 | 4 | gradient | 2.292e+00 | 4.644e-01 | 2.655e-01 |
| Wendland2 | y^4 | 4 | interpolate | 1.962e-01 | 4.145e-03 | 1.962e-01 |
| Wendland2 | y^4 | 4 | laplacian | 1.054e+02 | 2.060e+02 | 1.846e+02 |

## 3. Observed orders -- resolve-open/wendland2 (error vs dx)

`slope` = d log E / d log dx; `saturated` = flat error series (floor/plateau, no order to read); `exact` = errors at machine precision (nothing to fit).

| probe | field | mode | region | slope | r2 | saturated |
|---|---|---|---|---|---|---|
| gradient | 1 | crk | boundary | exact | exact | no |
| gradient | 1 | crk | interior | exact | exact | no |
| gradient | 1 | renorm | boundary | exact | exact | no |
| gradient | 1 | renorm | interior | exact | exact | no |
| gradient | 1 | standard | boundary | exact | exact | no |
| gradient | 1 | standard | interior | exact | exact | no |
| gradient | gauss | crk | boundary | 1.62 | 0.980 | no |
| gradient | gauss | crk | interior | 1.60 | 0.999 | no |
| gradient | gauss | renorm | boundary | 1.69 | 0.975 | no |
| gradient | gauss | renorm | interior | 1.76 | 0.998 | no |
| gradient | gauss | standard | boundary | 1.41 | 0.941 | no |
| gradient | gauss | standard | interior | 0.95 | 0.982 | no |
| gradient | sin_cos | crk | boundary | 0.70 | 0.983 | no |
| gradient | sin_cos | crk | interior | 1.70 | 0.996 | no |
| gradient | sin_cos | renorm | boundary | 0.86 | 0.986 | no |
| gradient | sin_cos | renorm | interior | 1.86 | 0.997 | no |
| gradient | sin_cos | standard | boundary | n/a | n/a | yes |
| gradient | sin_cos | standard | interior | n/a | n/a | yes |
| gradient | x | crk | boundary | -0.15 | 0.025 | no |
| gradient | x | crk | interior | n/a | n/a | yes |
| gradient | x | renorm | boundary | n/a | n/a | yes |
| gradient | x | renorm | interior | n/a | n/a | yes |
| gradient | x | standard | boundary | n/a | n/a | yes |
| gradient | x | standard | interior | n/a | n/a | yes |
| gradient | x^2 | crk | boundary | -0.09 | 0.008 | no |
| gradient | x^2 | crk | interior | n/a | n/a | yes |
| gradient | x^2 | renorm | boundary | 1.01 | 0.999 | no |
| gradient | x^2 | renorm | interior | 1.03 | 0.998 | no |
| gradient | x^2 | standard | boundary | n/a | n/a | yes |
| gradient | x^2 | standard | interior | n/a | n/a | yes |
| gradient | xy | crk | boundary | -0.56 | 0.241 | no |
| gradient | xy | crk | interior | 0.97 | 0.970 | no |
| gradient | xy | renorm | boundary | 0.95 | 0.979 | no |
| gradient | xy | renorm | interior | n/a | n/a | yes |
| gradient | xy | standard | boundary | n/a | n/a | yes |
| gradient | xy | standard | interior | n/a | n/a | yes |
| gradient | y | crk | boundary | n/a | n/a | yes |
| gradient | y | crk | interior | n/a | n/a | yes |
| gradient | y | renorm | boundary | n/a | n/a | yes |
| gradient | y | renorm | interior | n/a | n/a | yes |
| gradient | y | standard | boundary | n/a | n/a | yes |
| gradient | y | standard | interior | n/a | n/a | yes |
| gradient | y^2 | crk | boundary | n/a | n/a | yes |
| gradient | y^2 | crk | interior | 0.79 | 0.962 | no |
| gradient | y^2 | renorm | boundary | 1.05 | 0.996 | no |
| gradient | y^2 | renorm | interior | 0.94 | 0.995 | no |
| gradient | y^2 | standard | boundary | n/a | n/a | yes |
| gradient | y^2 | standard | interior | -0.65 | 0.766 | no |
| interpolate | 1 | crk | boundary | n/a | n/a | yes |
| interpolate | 1 | crk | interior | n/a | n/a | yes |
| interpolate | 1 | renorm | boundary | n/a | n/a | yes |
| interpolate | 1 | renorm | interior | n/a | n/a | yes |
| interpolate | 1 | standard | boundary | n/a | n/a | yes |
| interpolate | 1 | standard | interior | n/a | n/a | yes |
| interpolate | gauss | crk | boundary | 2.95 | 0.999 | no |
| interpolate | gauss | crk | interior | 1.84 | 1.000 | no |
| interpolate | gauss | renorm | boundary | 2.68 | 0.994 | no |
| interpolate | gauss | renorm | interior | 0.98 | 0.892 | no |
| interpolate | gauss | standard | boundary | 2.68 | 0.994 | no |
| interpolate | gauss | standard | interior | 0.98 | 0.892 | no |
| interpolate | sin_cos | crk | boundary | 1.80 | 0.999 | no |
| interpolate | sin_cos | crk | interior | 1.95 | 1.000 | no |
| interpolate | sin_cos | renorm | boundary | n/a | n/a | yes |
| interpolate | sin_cos | renorm | interior | 0.95 | 0.974 | no |
| interpolate | sin_cos | standard | boundary | n/a | n/a | yes |
| interpolate | sin_cos | standard | interior | 0.95 | 0.974 | no |
| interpolate | x | crk | boundary | n/a | n/a | yes |
| interpolate | x | crk | interior | n/a | n/a | yes |
| interpolate | x | renorm | boundary | n/a | n/a | yes |
| interpolate | x | renorm | interior | n/a | n/a | yes |
| interpolate | x | standard | boundary | n/a | n/a | yes |
| interpolate | x | standard | interior | n/a | n/a | yes |
| interpolate | x^2 | crk | boundary | 1.97 | 0.998 | no |
| interpolate | x^2 | crk | interior | 1.97 | 1.000 | no |
| interpolate | x^2 | renorm | boundary | n/a | n/a | yes |
| interpolate | x^2 | renorm | interior | n/a | n/a | yes |
| interpolate | x^2 | standard | boundary | n/a | n/a | yes |
| interpolate | x^2 | standard | interior | n/a | n/a | yes |
| interpolate | xy | crk | boundary | 2.11 | 0.991 | no |
| interpolate | xy | crk | interior | 1.87 | 0.998 | no |
| interpolate | xy | renorm | boundary | n/a | n/a | yes |
| interpolate | xy | renorm | interior | n/a | n/a | yes |
| interpolate | xy | standard | boundary | n/a | n/a | yes |
| interpolate | xy | standard | interior | n/a | n/a | yes |
| interpolate | y | crk | boundary | -0.42 | 0.259 | no |
| interpolate | y | crk | interior | n/a | n/a | yes |
| interpolate | y | renorm | boundary | n/a | n/a | yes |
| interpolate | y | renorm | interior | n/a | n/a | yes |
| interpolate | y | standard | boundary | n/a | n/a | yes |
| interpolate | y | standard | interior | n/a | n/a | yes |
| interpolate | y^2 | crk | boundary | 1.93 | 0.999 | no |
| interpolate | y^2 | crk | interior | 1.99 | 1.000 | no |
| interpolate | y^2 | renorm | boundary | n/a | n/a | yes |
| interpolate | y^2 | renorm | interior | n/a | n/a | yes |
| interpolate | y^2 | standard | boundary | n/a | n/a | yes |
| interpolate | y^2 | standard | interior | n/a | n/a | yes |
| laplacian | gauss | crk | boundary | n/a | n/a | yes |
| laplacian | gauss | crk | interior | 1.17 | 0.888 | no |
| laplacian | gauss | renorm | boundary | n/a | n/a | yes |
| laplacian | gauss | renorm | interior | n/a | n/a | yes |
| laplacian | gauss | standard | boundary | n/a | n/a | yes |
| laplacian | gauss | standard | interior | n/a | n/a | yes |
| laplacian | sin_cos | crk | boundary | -1.23 | 0.982 | no |
| laplacian | sin_cos | crk | interior | n/a | n/a | yes |
| laplacian | sin_cos | renorm | boundary | -1.20 | 0.988 | no |
| laplacian | sin_cos | renorm | interior | -1.06 | 0.858 | no |
| laplacian | sin_cos | standard | boundary | -1.07 | 0.999 | no |
| laplacian | sin_cos | standard | interior | -1.08 | 0.858 | no |
| laplacian | x^2 | crk | boundary | -1.09 | 0.974 | no |
| laplacian | x^2 | crk | interior | -1.68 | 0.999 | no |
| laplacian | x^2 | renorm | boundary | -1.06 | 0.984 | no |
| laplacian | x^2 | renorm | interior | -1.67 | 0.999 | no |
| laplacian | x^2 | standard | boundary | -0.99 | 0.998 | no |
| laplacian | x^2 | standard | interior | -1.64 | 0.999 | no |
| laplacian | xy | crk | boundary | -1.13 | 0.999 | no |
| laplacian | xy | crk | interior | -1.21 | 0.936 | no |
| laplacian | xy | renorm | boundary | -1.12 | 1.000 | no |
| laplacian | xy | renorm | interior | -1.51 | 0.994 | no |
| laplacian | xy | standard | boundary | -1.10 | 0.997 | no |
| laplacian | xy | standard | interior | -1.54 | 0.996 | no |
| laplacian | y^2 | crk | boundary | -1.08 | 0.991 | no |
| laplacian | y^2 | crk | interior | -1.23 | 0.903 | no |
| laplacian | y^2 | renorm | boundary | -1.08 | 0.990 | no |
| laplacian | y^2 | renorm | interior | -1.32 | 0.957 | no |
| laplacian | y^2 | standard | boundary | -1.04 | 0.989 | no |
| laplacian | y^2 | standard | interior | -1.32 | 0.947 | no |

Saturated series: gradient/sin_cos/standard (boundary), gradient/sin_cos/standard (interior), gradient/x/crk (interior), gradient/x/renorm (boundary), gradient/x/renorm (interior), gradient/x/standard (boundary), gradient/x/standard (interior), gradient/x^2/crk (interior), gradient/x^2/standard (boundary), gradient/x^2/standard (interior), gradient/xy/renorm (interior), gradient/xy/standard (boundary), gradient/xy/standard (interior), gradient/y/crk (boundary), gradient/y/crk (interior), gradient/y/renorm (boundary), gradient/y/renorm (interior), gradient/y/standard (boundary), gradient/y/standard (interior), gradient/y^2/crk (boundary), gradient/y^2/standard (boundary), interpolate/1/crk (boundary), interpolate/1/crk (interior), interpolate/1/renorm (boundary), interpolate/1/renorm (interior), interpolate/1/standard (boundary), interpolate/1/standard (interior), interpolate/sin_cos/renorm (boundary), interpolate/sin_cos/standard (boundary), interpolate/x/crk (boundary), interpolate/x/crk (interior), interpolate/x/renorm (boundary), interpolate/x/renorm (interior), interpolate/x/standard (boundary), interpolate/x/standard (interior), interpolate/x^2/renorm (boundary), interpolate/x^2/renorm (interior), interpolate/x^2/standard (boundary), interpolate/x^2/standard (interior), interpolate/xy/renorm (boundary), interpolate/xy/renorm (interior), interpolate/xy/standard (boundary), interpolate/xy/standard (interior), interpolate/y/crk (interior), interpolate/y/renorm (boundary), interpolate/y/renorm (interior), interpolate/y/standard (boundary), interpolate/y/standard (interior), interpolate/y^2/renorm (boundary), interpolate/y^2/renorm (interior), interpolate/y^2/standard (boundary), interpolate/y^2/standard (interior), laplacian/gauss/crk (boundary), laplacian/gauss/renorm (boundary), laplacian/gauss/renorm (interior), laplacian/gauss/standard (boundary), laplacian/gauss/standard (interior), laplacian/sin_cos/crk (interior)

## 4. Observed orders -- resolve-periodic/wendland2 (error vs dx)

`slope` = d log E / d log dx; `saturated` = flat error series (floor/plateau, no order to read); `exact` = errors at machine precision (nothing to fit).

| probe | field | mode | region | slope | r2 | saturated |
|---|---|---|---|---|---|---|
| gradient | gauss_periodic | crk | interior | 1.60 | 0.999 | no |
| gradient | gauss_periodic | renorm | interior | 1.76 | 0.998 | no |
| gradient | gauss_periodic | standard | interior | 0.95 | 0.982 | no |
| gradient | sin_a | crk | interior | 1.39 | 0.994 | no |
| gradient | sin_a | renorm | interior | 1.72 | 1.000 | no |
| gradient | sin_a | standard | interior | n/a | n/a | yes |
| gradient | sin_b | crk | interior | 1.39 | 0.983 | no |
| gradient | sin_b | renorm | interior | 1.70 | 0.984 | no |
| gradient | sin_b | standard | interior | n/a | n/a | yes |
| gradient | sin_cos | crk | interior | 1.69 | 0.996 | no |
| gradient | sin_cos | renorm | interior | 1.83 | 0.998 | no |
| gradient | sin_cos | standard | interior | n/a | n/a | yes |
| interpolate | gauss_periodic | crk | interior | 1.84 | 1.000 | no |
| interpolate | gauss_periodic | renorm | interior | 0.98 | 0.892 | no |
| interpolate | gauss_periodic | standard | interior | 0.98 | 0.892 | no |
| interpolate | sin_a | crk | interior | 1.97 | 1.000 | no |
| interpolate | sin_a | renorm | interior | n/a | n/a | yes |
| interpolate | sin_a | standard | interior | n/a | n/a | yes |
| interpolate | sin_b | crk | interior | 1.94 | 0.998 | no |
| interpolate | sin_b | renorm | interior | n/a | n/a | yes |
| interpolate | sin_b | standard | interior | n/a | n/a | yes |
| interpolate | sin_cos | crk | interior | 1.94 | 1.000 | no |
| interpolate | sin_cos | renorm | interior | 0.85 | 0.934 | no |
| interpolate | sin_cos | standard | interior | 0.85 | 0.934 | no |
| laplacian | gauss_periodic | crk | interior | 1.17 | 0.888 | no |
| laplacian | gauss_periodic | renorm | interior | n/a | n/a | yes |
| laplacian | gauss_periodic | standard | interior | n/a | n/a | yes |
| laplacian | sin_a | crk | interior | -0.83 | 0.817 | no |
| laplacian | sin_a | renorm | interior | -1.36 | 0.985 | no |
| laplacian | sin_a | standard | interior | -1.34 | 0.978 | no |
| laplacian | sin_b | crk | interior | -0.78 | 0.838 | no |
| laplacian | sin_b | renorm | interior | -1.19 | 0.999 | no |
| laplacian | sin_b | standard | interior | -1.18 | 0.999 | no |
| laplacian | sin_cos | crk | interior | n/a | n/a | yes |
| laplacian | sin_cos | renorm | interior | -1.14 | 0.863 | no |
| laplacian | sin_cos | standard | interior | -1.13 | 0.872 | no |

Saturated series: gradient/sin_a/standard (interior), gradient/sin_b/standard (interior), gradient/sin_cos/standard (interior), interpolate/sin_a/renorm (interior), interpolate/sin_a/standard (interior), interpolate/sin_b/renorm (interior), interpolate/sin_b/standard (interior), laplacian/gauss_periodic/renorm (interior), laplacian/gauss_periodic/standard (interior), laplacian/sin_cos/crk (interior)

## 5. Observed orders -- smoothing/wendland2 (error vs h)

`slope` = d log E / d log h; `saturated` = flat error series (floor/plateau, no order to read); `exact` = errors at machine precision (nothing to fit).

| probe | field | mode | region | slope | r2 | saturated |
|---|---|---|---|---|---|---|
| gradient | 1 | crk | boundary | exact | exact | no |
| gradient | 1 | crk | interior | exact | exact | no |
| gradient | 1 | renorm | boundary | exact | exact | no |
| gradient | 1 | renorm | interior | exact | exact | no |
| gradient | 1 | standard | boundary | exact | exact | no |
| gradient | 1 | standard | interior | exact | exact | no |
| gradient | gauss | crk | boundary | n/a | n/a | yes |
| gradient | gauss | crk | interior | n/a | n/a | yes |
| gradient | gauss | renorm | boundary | 1.01 | 0.995 | no |
| gradient | gauss | renorm | interior | 1.02 | 0.906 | no |
| gradient | gauss | standard | boundary | n/a | n/a | yes |
| gradient | gauss | standard | interior | -1.41 | 0.914 | no |
| gradient | sin_cos | crk | boundary | n/a | n/a | yes |
| gradient | sin_cos | crk | interior | n/a | n/a | yes |
| gradient | sin_cos | renorm | boundary | n/a | n/a | yes |
| gradient | sin_cos | renorm | interior | n/a | n/a | yes |
| gradient | sin_cos | standard | boundary | n/a | n/a | yes |
| gradient | sin_cos | standard | interior | -1.29 | 0.928 | no |
| gradient | x | crk | boundary | n/a | n/a | yes |
| gradient | x | crk | interior | -4.99 | 0.965 | no |
| gradient | x | renorm | boundary | n/a | n/a | yes |
| gradient | x | renorm | interior | n/a | n/a | yes |
| gradient | x | standard | boundary | n/a | n/a | yes |
| gradient | x | standard | interior | -2.01 | 0.992 | no |
| gradient | y | crk | boundary | n/a | n/a | yes |
| gradient | y | crk | interior | -4.61 | 0.985 | no |
| gradient | y | renorm | boundary | n/a | n/a | yes |
| gradient | y | renorm | interior | n/a | n/a | yes |
| gradient | y | standard | boundary | n/a | n/a | yes |
| gradient | y | standard | interior | -2.11 | 0.999 | no |
| interpolate | 1 | crk | boundary | n/a | n/a | yes |
| interpolate | 1 | crk | interior | n/a | n/a | yes |
| interpolate | 1 | renorm | boundary | 1.47 | 0.986 | no |
| interpolate | 1 | renorm | interior | -1.18 | 0.938 | no |
| interpolate | 1 | standard | boundary | 1.47 | 0.986 | no |
| interpolate | 1 | standard | interior | -1.18 | 0.938 | no |
| interpolate | gauss | crk | boundary | 3.02 | 0.998 | no |
| interpolate | gauss | crk | interior | 2.05 | 1.000 | no |
| interpolate | gauss | renorm | boundary | 1.78 | 0.998 | no |
| interpolate | gauss | renorm | interior | n/a | n/a | yes |
| interpolate | gauss | standard | boundary | 1.78 | 0.998 | no |
| interpolate | gauss | standard | interior | n/a | n/a | yes |
| interpolate | sin_cos | crk | boundary | 1.93 | 0.999 | no |
| interpolate | sin_cos | crk | interior | 2.03 | 0.999 | no |
| interpolate | sin_cos | renorm | boundary | 1.53 | 0.936 | no |
| interpolate | sin_cos | renorm | interior | n/a | n/a | yes |
| interpolate | sin_cos | standard | boundary | 1.53 | 0.936 | no |
| interpolate | sin_cos | standard | interior | n/a | n/a | yes |
| interpolate | x | crk | boundary | 0.92 | 0.899 | no |
| interpolate | x | crk | interior | n/a | n/a | yes |
| interpolate | x | renorm | boundary | 1.78 | 0.935 | no |
| interpolate | x | renorm | interior | -1.33 | 0.960 | no |
| interpolate | x | standard | boundary | 1.78 | 0.935 | no |
| interpolate | x | standard | interior | -1.33 | 0.960 | no |
| interpolate | y | crk | boundary | 0.63 | 0.539 | no |
| interpolate | y | crk | interior | 0.92 | 0.899 | no |
| interpolate | y | renorm | boundary | 1.52 | 0.997 | no |
| interpolate | y | renorm | interior | -1.51 | 0.997 | no |
| interpolate | y | standard | boundary | 1.52 | 0.997 | no |
| interpolate | y | standard | interior | -1.51 | 0.997 | no |

Saturated series: gradient/gauss/crk (boundary), gradient/gauss/crk (interior), gradient/gauss/standard (boundary), gradient/sin_cos/crk (boundary), gradient/sin_cos/crk (interior), gradient/sin_cos/renorm (boundary), gradient/sin_cos/renorm (interior), gradient/sin_cos/standard (boundary), gradient/x/crk (boundary), gradient/x/renorm (boundary), gradient/x/renorm (interior), gradient/x/standard (boundary), gradient/y/crk (boundary), gradient/y/renorm (boundary), gradient/y/renorm (interior), gradient/y/standard (boundary), interpolate/1/crk (boundary), interpolate/1/crk (interior), interpolate/gauss/renorm (interior), interpolate/gauss/standard (interior), interpolate/sin_cos/renorm (interior), interpolate/sin_cos/standard (interior), interpolate/x/crk (interior)

## 6. Renorm condition numbers

| N | dim | dx | fallback_frac | h | h_over_dx | jitter | kernel | max | mean | n | p50 | p95 | region | target_neighbors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 572 | 2 | 0.04545 | 0 | 0.08267 | 1.819 | 0.1 | Wendland2 | 1.135 | 1.054 | 390 | 1.05 | 1.109 | interior | 12 |
| 572 | 2 | 0.04545 | 0 | 0.08267 | 1.819 | 0.1 | Wendland2 | 3.47 | 1.361 | 182 | 1.112 | 2.03 | boundary | 12 |
| 572 | 2 | 0.04545 | 0 | 0.1067 | 2.348 | 0.1 | Wendland2 | 1.079 | 1.032 | 368 | 1.03 | 1.063 | interior | 20 |
| 572 | 2 | 0.04545 | 0 | 0.1067 | 2.348 | 0.1 | Wendland2 | 2.694 | 1.32 | 204 | 1.073 | 1.859 | boundary | 20 |
| 572 | 2 | 0.04545 | 0 | 0.1412 | 3.106 | 0.1 | Wendland2 | 1.055 | 1.021 | 294 | 1.02 | 1.04 | interior | 35 |
| 572 | 2 | 0.04545 | 0 | 0.1412 | 3.106 | 0.1 | Wendland2 | 2.381 | 1.286 | 278 | 1.226 | 1.748 | boundary | 35 |
| 572 | 2 | 0.04545 | 0 | 0.1849 | 4.067 | 0.1 | Wendland2 | 1.042 | 1.014 | 236 | 1.012 | 1.027 | interior | 60 |
| 572 | 2 | 0.04545 | 0 | 0.1849 | 4.067 | 0.1 | Wendland2 | 2.562 | 1.282 | 336 | 1.187 | 1.692 | boundary | 60 |
| 572 | 2 | 0.04545 | 0 | 0.08267 | 1.819 | 0.3 | Wendland2 | 1.544 | 1.178 | 417 | 1.158 | 1.378 | interior | 12 |
| 572 | 2 | 0.04545 | 0.006452 | 0.08267 | 1.819 | 0.3 | Wendland2 | 3.141 | 1.517 | 155 | 1.375 | 2.332 | boundary | 12 |
| 572 | 2 | 0.04545 | 0 | 0.1067 | 2.348 | 0.3 | Wendland2 | 1.256 | 1.101 | 368 | 1.092 | 1.203 | interior | 20 |
| 572 | 2 | 0.04545 | 0 | 0.1067 | 2.348 | 0.3 | Wendland2 | 3.317 | 1.369 | 204 | 1.198 | 1.939 | boundary | 20 |
| 572 | 2 | 0.04545 | 0 | 0.1412 | 3.106 | 0.3 | Wendland2 | 1.155 | 1.064 | 303 | 1.059 | 1.125 | interior | 35 |
| 572 | 2 | 0.04545 | 0 | 0.1412 | 3.106 | 0.3 | Wendland2 | 2.533 | 1.304 | 269 | 1.193 | 1.798 | boundary | 35 |
| 572 | 2 | 0.04545 | 0 | 0.1849 | 4.067 | 0.3 | Wendland2 | 1.109 | 1.038 | 243 | 1.036 | 1.074 | interior | 60 |
| 572 | 2 | 0.04545 | 0 | 0.1849 | 4.067 | 0.3 | Wendland2 | 2.696 | 1.29 | 329 | 1.183 | 1.727 | boundary | 60 |
| 572 | 2 | 0.04545 | 0 | 0.08267 | 1.819 | 0.5 | Wendland2 | 2.256 | 1.337 | 430 | 1.292 | 1.758 | interior | 12 |
| 572 | 2 | 0.04545 | 0.007042 | 0.08267 | 1.819 | 0.5 | Wendland2 | 4.706 | 1.753 | 142 | 1.594 | 3.019 | boundary | 12 |
| 572 | 2 | 0.04545 | 0 | 0.1067 | 2.348 | 0.5 | Wendland2 | 1.519 | 1.179 | 373 | 1.159 | 1.353 | interior | 20 |
| 572 | 2 | 0.04545 | 0 | 0.1067 | 2.348 | 0.5 | Wendland2 | 4.291 | 1.449 | 199 | 1.32 | 2.077 | boundary | 20 |
| 572 | 2 | 0.04545 | 0 | 0.1412 | 3.106 | 0.5 | Wendland2 | 1.274 | 1.109 | 307 | 1.097 | 1.212 | interior | 35 |
| 572 | 2 | 0.04545 | 0 | 0.1412 | 3.106 | 0.5 | Wendland2 | 2.641 | 1.323 | 265 | 1.19 | 1.823 | boundary | 35 |
| 572 | 2 | 0.04545 | 0 | 0.1849 | 4.067 | 0.5 | Wendland2 | 1.176 | 1.063 | 244 | 1.062 | 1.122 | interior | 60 |
| 572 | 2 | 0.04545 | 0 | 0.1849 | 4.067 | 0.5 | Wendland2 | 2.757 | 1.295 | 328 | 1.196 | 1.76 | boundary | 60 |

## 7. Findings

Curated interpretation of the tables above (regenerated reports do not carry the curation): see [`FINDINGS.md`](FINDINGS.md).
