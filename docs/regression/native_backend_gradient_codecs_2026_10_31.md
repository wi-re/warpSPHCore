# Heavier GPU operators: sorting, CRK and neighbor compression

Primal scalar gradients, vector Jacobians, divergence and covariance, with and without CRK kernel-gradient correction. All timings below are allocating operator calls, including packing and wrapper dispatch. Search, sorting, conversion, encoding, CRK-factor construction and qualification are excluded.

Artifact archive: `/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-gradient-codecs-20261031`. JSON/CSV files retain individual samples, accuracy checks, configurations and planning costs. Throughput PNGs use zero-based axes.

## Findings

**float32, 16M, 3D, n_h=4:** scalar-gradient sorting changes Warp from 1397.55 to 144.59 ms (9.67×). The CRK Jacobian takes 189.97 ms in sorted Warp, 34.55 ms in the matching-CSR64 native prototype (5.50× throughput), and 30.10 ms with native CSR32 (6.31×). Thus narrower IDs account for only part of the native gain.

The same CRK Jacobian takes 34.27 ms with two-bit gaps (+13.8% time versus native CSR32) and 39.19 ms with nibble lists (+30.2%). Two-bit topology is 2.276 GB versus 17.414 GB for CSR32. Small timing differences should be treated as comparable performance, not reliable speedups.

**float64, 16M, 3D, n_h=4:** scalar-gradient sorting changes Warp from 1465.51 to 1062.51 ms (1.38×). The CRK Jacobian takes 1503.18 ms in sorted Warp, 483.13 ms in the matching-CSR64 native prototype (3.11× throughput), and 490.45 ms with native CSR32 (3.06×). Thus narrower IDs account for only part of the native gain.

The same CRK Jacobian takes 493.37 ms with two-bit gaps (+0.6% time versus native CSR32) and 486.49 ms with nibble lists (-0.8%). Two-bit topology is 2.276 GB versus 17.414 GB for CSR32. Small timing differences should be treated as comparable performance, not reliable speedups.

Field rank and CRK also change accumulator size, correction loads and selected schedules. Decode overhead therefore does not decline uniformly with more mathematical work. See the per-operation ratios below; some FP32 vector cases are nearly throughput-neutral while others retain a measurable penalty.

The experimental raw consumers also improve some uncorrected operations over existing production native/Torch. They pack reference volume/fields in both precisions, prepare query/kernel coefficients once and stage independent gathers. These results are not a production dispatch or AD change, and are specific to the stated Wendland2/Gather/Difference scope.

## Scope and controls

* 28 completed held-out fixtures, 1792 operator/path comparisons.
* 1M and 4M particles in 1D/2D/3D, plus 16M in 3D; FP32/FP64; n_h=4 and 2.57.
* RTX PRO 6000 Blackwell Workstation, sm_120; CUDA 13 native build, Warp 1.17.0, PyTorch 2.13.0+cu130.
* Wendland2, Gather, Difference fields and TrueAllToAll; random uniform periodic particle sets, uniform supports/mass/density, smooth periodic scalar/vector fields.
* Independent 262,144-particle screens freeze segment, adjacent-entry and batching choices before held-out sizes. Blocks contain 128 threads. Warp retains its existing 32-lane kernel.
* Warp uses CSR64. Both native CSR64 and CSR32 prototype controls are included so narrower IDs are not mistaken for a backend advantage.
* Production native/Torch rejects CRK corrections. Corrected native results are explicitly experimental primals; no Warp fallback is timed as native.
* Sorted-backend timings are randomized paired batches. Original/sorted phases run separately to avoid retaining two multi-billion-edge CSR64 graphs simultaneously.
* Particles, fields, volumes and all CRK coefficients follow the same permutation. The same correction coefficients are reused after sorting. Neighbor counts are checked exactly; operator permutation parity is sampled at up to 16,384 deterministic particles.

## Correctness

* 28,032 standalone configurations: independent dense FP64 oracles, nonsymmetric gradB, variable/zero supports and mixed periodicity; exact raw64/raw32/byte/two-bit/nibble results at matching schedules.
* A nondefault stream is included. Targeted sanitizer checks cover 160 configurations across all four operators, both correction modes and every format; memcheck zero errors, racecheck zero errors/warnings.
* An initial unfiltered sanitizer run was stopped during expensive instrumentation of thousands of unused specializations. It is not counted as a passed check. Targeted kernel filters and a representative schedule were then used.
* Query-batched Warp is bit-identical to full calls for factors and all eight operator cases in both precisions, checked separately before the oversized runs.
* Real apparent-volume CRK factors are constructed from Warp moments. Small fixtures verify equivalence of the dimensionless and physical coefficient solves where the physical limiter is inactive.

## Actual CRK factors and scale

The existing solver applies an absolute det(m2)<1e-14 cutoff. In a fixed unit box, physical m2 scales as h² and its determinant as h^(2D), so shrinking support can turn nearly every sufficiently large 3D fixture into an identity correction. This experiment normalizes moments and their derivatives by h, invokes the unchanged coefficient solver in batches, then transforms B, gradA and gradB back by 1/h, 1/h and 1/h². Support is held fixed during the coordinate derivative.

The shared CRK setup runs on the original-order graph before permutation. Its recorded cost must not be interpreted as the cost of a sorted CRK setup or as a full-step benchmark.

This is an experimental preparation step, not a production solver change. It keeps the corrected workload physically nontrivial across scales. The measured CRK-factor setup omits the convenience wrapper’s unrelated diagnostic-density pass. Both setup wall cost and reproduction residuals are retained in each JSON.

| Precision | D | n_h | N | Physical cutoff fraction | Actual identity fraction | CRK setup ms |
|---|---:|---:|---:|---:|---:|---:|
| float32 | 1 | 2.57 | 1000000 | 0.012812 | 0.006156 | 389.482 |
| float32 | 1 | 2.57 | 4000000 | 0.0749185 | 0.00798 | 414.481 |
| float32 | 1 | 4 | 1000000 | 0.000836 | 0.000347 | 390.969 |
| float32 | 1 | 4 | 4000000 | 0.00482625 | 0.00044275 | 413.205 |
| float32 | 2 | 2.57 | 1000000 | 0.001919 | 0 | 367.602 |
| float32 | 2 | 2.57 | 4000000 | 0.805963 | 0 | 625.445 |
| float32 | 2 | 4 | 1000000 | 0 | 0 | 363.625 |
| float32 | 2 | 4 | 4000000 | 4.25e-06 | 0 | 624.770 |
| float32 | 3 | 2.57 | 1000000 | 0.000858 | 0 | 444.512 |
| float32 | 3 | 2.57 | 16000000 | 1 | 0 | 3202.236 |
| float32 | 3 | 2.57 | 4000000 | 1 | 0 | 941.665 |
| float32 | 3 | 4 | 1000000 | 0 | 0 | 445.260 |
| float32 | 3 | 4 | 16000000 | 1 | 0 | 3764.979 |
| float32 | 3 | 4 | 4000000 | 0 | 0 | 971.654 |
| float64 | 1 | 2.57 | 1000000 | 0.012693 | 0.005949 | 435.340 |
| float64 | 1 | 2.57 | 4000000 | 0.0744267 | 0.0057745 | 649.063 |
| float64 | 1 | 4 | 1000000 | 0.00083 | 0.000343 | 448.075 |
| float64 | 1 | 4 | 4000000 | 0.00480225 | 0.000322 | 659.705 |
| float64 | 2 | 2.57 | 1000000 | 0.001919 | 0 | 620.150 |
| float64 | 2 | 2.57 | 4000000 | 0.80597 | 0 | 1679.016 |
| float64 | 2 | 4 | 1000000 | 0 | 0 | 622.048 |
| float64 | 2 | 4 | 4000000 | 4.25e-06 | 0 | 1721.062 |
| float64 | 3 | 2.57 | 1000000 | 0.000858 | 0 | 1245.905 |
| float64 | 3 | 2.57 | 16000000 | 1 | 0 | 16189.471 |
| float64 | 3 | 2.57 | 4000000 | 1 | 0 | 4218.551 |
| float64 | 3 | 4 | 1000000 | 0 | 0 | 1361.348 |
| float64 | 3 | 4 | 16000000 | 1 | 0 | 17947.791 |
| float64 | 3 | 4 | 4000000 | 0 | 0 | 4652.336 |

## Large 3D results

At 16M particles, n_h=4 in 3D, Warp rejects the roughly 4.3-billion-entry array because each array dimension must fit signed int32. Those Warp results therefore use four query batches with shared full reference arrays; the timed operation includes concatenation. Planning/rebasing stays outside. Native uses one full graph call. Smaller graphs, including 16M at n_h=2.57, use ordinary single-call Warp. The original failure log and the adapter verification are retained.

Times are CUDA-event medians in milliseconds. A missing production-native corrected entry means unsupported, not zero time. Neighbor memory is retained topology only; CSR construction and simultaneous comparison layouts still determine peak device allocation.

### float32, n_h=4, 16M particles

| Operation | CRK | Warp original CSR64 | Warp Morton CSR64 | Production native CSR32 | Prototype native CSR64 | Prototype native CSR32 | Byte delta | Two-bit gaps | Unity/nibble |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Scalar gradient | no | 1397.554 | 144.588 | 35.562 | 26.119 | 20.040 | 24.170 | 25.495 | 26.299 |
| Scalar gradient | yes | 1727.856 | 171.616 | — | 29.321 | 27.165 | 30.107 | 31.145 | 34.093 |
| Vector Jacobian | no | 1431.117 | 162.366 | 33.966 | 29.110 | 27.443 | 30.616 | 27.929 | 31.467 |
| Vector Jacobian | yes | 1759.518 | 189.969 | — | 34.552 | 30.105 | 35.963 | 34.273 | 39.189 |
| Vector divergence | no | 1467.356 | 164.960 | 35.106 | 25.845 | 22.576 | 24.862 | 23.836 | 29.712 |
| Vector divergence | yes | 1795.932 | 193.564 | — | 29.356 | 26.673 | 34.526 | 32.175 | 38.869 |
| Covariance | no | 1044.032 | 252.636 | 34.253 | 25.221 | 20.782 | 23.394 | 22.690 | 28.482 |
| Covariance | yes | 740.702 | 308.379 | — | 30.051 | 27.563 | 33.634 | 30.343 | 35.973 |

Compression relative to the same experimental native CSR32 consumer:

| Operation | CRK | Byte speed ratio | Two-bit speed ratio | Nibble speed ratio | Byte GB | Two-bit GB | Nibble GB |
|---|---|---:|---:|---:|---:|---:|---:|
| Scalar gradient | no | 0.829× | 0.786× | 0.762× | 6.140 | 2.276 | 2.327 |
| Scalar gradient | yes | 0.902× | 0.872× | 0.797× | 6.140 | 2.276 | 3.457 |
| Vector Jacobian | no | 0.896× | 0.983× | 0.872× | 7.078 | 2.276 | 2.327 |
| Vector Jacobian | yes | 0.837× | 0.878× | 0.768× | 6.140 | 2.276 | 2.327 |
| Vector divergence | no | 0.908× | 0.947× | 0.760× | 6.140 | 2.276 | 2.327 |
| Vector divergence | yes | 0.773× | 0.829× | 0.686× | 6.140 | 2.276 | 2.327 |
| Covariance | no | 0.888× | 0.916× | 0.730× | 7.078 | 2.276 | 2.327 |
| Covariance | yes | 0.819× | 0.908× | 0.766× | 6.140 | 2.276 | 2.327 |

### float32, n_h=2.57, 16M particles

| Operation | CRK | Warp original CSR64 | Warp Morton CSR64 | Production native CSR32 | Prototype native CSR64 | Prototype native CSR32 | Byte delta | Two-bit gaps | Unity/nibble |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Scalar gradient | no | 358.001 | 37.740 | 7.076 | 8.258 | 6.707 | 7.294 | 7.909 | 8.313 |
| Scalar gradient | yes | 446.819 | 43.894 | — | 9.747 | 8.236 | 9.075 | 10.265 | 11.550 |
| Vector Jacobian | no | 367.818 | 45.233 | 8.019 | 9.678 | 9.337 | 9.590 | 9.621 | 11.108 |
| Vector Jacobian | yes | 454.148 | 52.735 | — | 10.680 | 10.132 | 11.299 | 12.506 | 13.621 |
| Vector divergence | no | 374.119 | 38.797 | 7.349 | 8.621 | 8.223 | 7.563 | 7.936 | 9.292 |
| Vector divergence | yes | 463.797 | 45.732 | — | 9.495 | 9.465 | 11.070 | 10.279 | 12.157 |
| Covariance | no | 268.380 | 71.472 | 6.968 | 7.933 | 7.081 | 7.092 | 7.420 | 9.762 |
| Covariance | yes | 190.031 | 90.532 | — | 10.307 | 10.105 | 9.728 | 9.499 | 12.276 |

Compression relative to the same experimental native CSR32 consumer:

| Operation | CRK | Byte speed ratio | Two-bit speed ratio | Nibble speed ratio | Byte GB | Two-bit GB | Nibble GB |
|---|---|---:|---:|---:|---:|---:|---:|
| Scalar gradient | no | 0.920× | 0.848× | 0.807× | 2.735 | 0.967 | 1.287 |
| Scalar gradient | yes | 0.908× | 0.802× | 0.713× | 2.735 | 0.967 | 1.287 |
| Vector Jacobian | no | 0.974× | 0.970× | 0.841× | 2.735 | 0.967 | 1.287 |
| Vector Jacobian | yes | 0.897× | 0.810× | 0.744× | 2.735 | 0.967 | 1.287 |
| Vector divergence | no | 1.087× | 1.036× | 0.885× | 2.735 | 0.967 | 1.287 |
| Vector divergence | yes | 0.855× | 0.921× | 0.779× | 2.735 | 0.967 | 1.287 |
| Covariance | no | 0.998× | 0.954× | 0.725× | 2.735 | 0.967 | 1.287 |
| Covariance | yes | 1.039× | 1.064× | 0.823× | 2.735 | 0.967 | 1.287 |

### float64, n_h=4, 16M particles

| Operation | CRK | Warp original CSR64 | Warp Morton CSR64 | Production native CSR32 | Prototype native CSR64 | Prototype native CSR32 | Byte delta | Two-bit gaps | Unity/nibble |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Scalar gradient | no | 1465.507 | 1062.514 | 308.723 | 259.924 | 259.331 | 262.565 | 261.086 | 262.187 |
| Scalar gradient | yes | 1777.247 | 1386.193 | — | 432.006 | 425.585 | 432.314 | 426.584 | 433.181 |
| Vector Jacobian | no | 1471.088 | 1183.189 | 378.808 | 311.202 | 312.492 | 313.513 | 313.510 | 358.629 |
| Vector Jacobian | yes | 1785.542 | 1503.180 | — | 483.134 | 490.447 | 494.145 | 493.374 | 486.489 |
| Vector divergence | no | 1467.007 | 1081.172 | 341.164 | 285.180 | 284.576 | 281.576 | 282.018 | 281.783 |
| Vector divergence | yes | 1782.729 | 1397.504 | — | 446.129 | 449.960 | 451.090 | 451.006 | 446.816 |
| Covariance | no | 2458.282 | 2434.190 | 332.863 | 299.300 | 295.741 | 297.590 | 297.807 | 311.425 |
| Covariance | yes | 3370.457 | 3342.546 | — | 473.705 | 461.778 | 475.576 | 497.367 | 463.242 |

Compression relative to the same experimental native CSR32 consumer:

| Operation | CRK | Byte speed ratio | Two-bit speed ratio | Nibble speed ratio | Byte GB | Two-bit GB | Nibble GB |
|---|---|---:|---:|---:|---:|---:|---:|
| Scalar gradient | no | 0.988× | 0.993× | 0.989× | 7.078 | 2.276 | 3.457 |
| Scalar gradient | yes | 0.984× | 0.998× | 0.982× | 7.078 | 2.276 | 2.327 |
| Vector Jacobian | no | 0.997× | 0.997× | 0.871× | 9.146 | 2.276 | 2.327 |
| Vector Jacobian | yes | 0.993× | 0.994× | 1.008× | 7.078 | 2.276 | 3.457 |
| Vector divergence | no | 1.011× | 1.009× | 1.010× | 9.146 | 2.276 | 3.457 |
| Vector divergence | yes | 0.997× | 0.998× | 1.007× | 7.078 | 2.276 | 3.457 |
| Covariance | no | 0.994× | 0.993× | 0.950× | 9.146 | 2.276 | 2.327 |
| Covariance | yes | 0.971× | 0.928× | 0.997× | 7.078 | 2.276 | 3.457 |

### float64, n_h=2.57, 16M particles

| Operation | CRK | Warp original CSR64 | Warp Morton CSR64 | Production native CSR32 | Prototype native CSR64 | Prototype native CSR32 | Byte delta | Two-bit gaps | Unity/nibble |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Scalar gradient | no | 381.532 | 370.205 | 93.851 | 89.340 | 80.900 | 85.825 | 80.420 | 81.254 |
| Scalar gradient | yes | 483.470 | 479.986 | — | 129.058 | 128.526 | 129.298 | 129.836 | 128.829 |
| Vector Jacobian | no | 443.699 | 432.099 | 116.287 | 96.910 | 96.387 | 101.046 | 100.569 | 105.939 |
| Vector Jacobian | yes | 549.266 | 540.550 | — | 148.766 | 158.983 | 150.383 | 149.137 | 151.130 |
| Vector divergence | no | 404.726 | 378.858 | 103.378 | 85.978 | 86.972 | 86.496 | 86.951 | 86.249 |
| Vector divergence | yes | 500.237 | 487.881 | — | 134.183 | 134.183 | 141.463 | 134.569 | 134.377 |
| Covariance | no | 829.624 | 827.279 | 103.065 | 90.718 | 95.708 | 103.026 | 94.484 | 95.247 |
| Covariance | yes | 1125.888 | 1122.117 | — | 141.887 | 141.656 | 141.646 | 141.676 | 142.068 |

Compression relative to the same experimental native CSR32 consumer:

| Operation | CRK | Byte speed ratio | Two-bit speed ratio | Nibble speed ratio | Byte GB | Two-bit GB | Nibble GB |
|---|---|---:|---:|---:|---:|---:|---:|
| Scalar gradient | no | 0.943× | 1.006× | 0.996× | 2.235 | 0.967 | 1.287 |
| Scalar gradient | yes | 0.994× | 0.990× | 0.998× | 2.735 | 0.967 | 1.287 |
| Vector Jacobian | no | 0.954× | 0.958× | 0.910× | 2.735 | 0.967 | 1.287 |
| Vector Jacobian | yes | 1.057× | 1.066× | 1.052× | 2.735 | 0.967 | 1.287 |
| Vector divergence | no | 1.005× | 1.000× | 1.008× | 2.735 | 0.967 | 1.287 |
| Vector divergence | yes | 0.949× | 0.997× | 0.999× | 2.235 | 0.967 | 1.287 |
| Covariance | no | 0.929× | 1.013× | 1.005× | 2.735 | 0.967 | 1.287 |
| Covariance | yes | 1.000× | 1.000× | 0.997× | 2.735 | 0.967 | 1.287 |

## Numerical limits

192 recorded comparisons did not meet the standard cross-backend relative-L2 tolerance. See `unqualified-comparisons.json` for exact paths and errors. The benchmark permits only the inherited FP32 periodic 1D geometry discrepancy; unexpected disagreements in other cases stop the run.

Warp’s periodic geometry performs (dx+L/2)-L/2 even for a very small in-range displacement. Native geometry preserves that displacement directly. At million-particle 1D scales, FP32 rounding can therefore be a material fraction of support. Timings for such configurations are retained but are not claimed to compare numerically equivalent outputs. Native raw/compressed equivalence is separately established by the independent dense suite.

## Boundaries

These are fixed-topology primals. There is no direct compressed radius-search builder, CRK backward/adjoint implementation, or production CRK dispatch added here. CRK-factor construction remains Warp/Torch preparation shared by every corrected consumer. Covariance’s Warp wrapper also produces a neighbor-count output even when the caller requests only the matrix; native primals produce the requested matrix directly.

Source, binaries, checksums, GPU activity and the final commit are retained in the artifact audit. The implementation lives in `experiments/gradient_codecs/`; earlier codec/planning modules are reused rather than copied into production.

## Selected plots

[FP32 3D gradients](/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-gradient-codecs-20261031/throughput-float32-d3-h4-gradients.png) · [FP64 3D gradients](/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-gradient-codecs-20261031/throughput-float64-d3-h4-gradients.png)

[FP32 divergence/covariance](/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-gradient-codecs-20261031/throughput-float32-d3-h4-other.png) · [FP64 divergence/covariance](/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-gradient-codecs-20261031/throughput-float64-d3-h4-other.png)

[FP32 relative throughput](/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-gradient-codecs-20261031/relative-float32-h4.png) · [FP64 relative throughput](/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-gradient-codecs-20261031/relative-float64-h4.png)

The archive also contains corresponding n_h=2.57 and 1D/2D plots.
