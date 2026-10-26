# Core SPH operator inventory and FP32 scaling

21 October 2026

This inventory covers the seven core SPH operators, renormalization, CRK pair stages and core factor assembly, and local pseudo-inverse APIs. It measures public-call time and useful physical-pair throughput across particle counts, neighbor targets, traversal and lane options, with separate compiler and live launch diagnostics. **FP32 is the analysis target.** Completed FP64 runs are retained as secondary raw artifacts and are not used to rank bottlenecks on this machine.

The results show substantial variation between public wrappers, kernel families, traversal and problem size. Kernel register use spans 30–128 registers/thread in the FP32 inventory; 253 Warp kernel captures supply achieved occupancy, scheduler and memory diagnostics. CSR neighbor consumers reach tens of billions of directed interactions/s at large N, while small launches carry appreciable dispatch and allocation costs. The 3D core renormalization inverse policy has a distinct launch-heavy scaling problem. These observations are an inventory for subsequent investigation; no production kernels, defaults or algorithms were changed.

## Simulation relevance

The distinction between a supported library path and a path used by a simulation is essential. **CRK moments on grid traversal are unused in actual simulations**, as clarified by the user. They are excluded from the main CRK figures and bottleneck discussion. Their already-collected records remain in the original raw datasets as additional API coverage.

The current frontend's [CRK step](/home/lu26029/dev/warpSPH/src/warpSPH/schemes/compressible/crkSPH.py:114) obtains a CSR pair list via [stagePairList](/home/lu26029/dev/warpSPH/src/warpSPH/utils/pairList.py:24) even when the support solve runs on a grid. Its [crkFactors implementation](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/terms.py:117) calls the core volume/moment/density kernels but assembles the local coefficients in an existing fused frontend Warp kernel. It can reuse supplied volumes or skip consistency density. The core Torch `crk_terms` and full `crk_factors` timings therefore describe those core APIs and the differentiation fallback, not an ordinary frontend CRK timestep. No frontend end-to-end timing is claimed.

Renormalization is called by the frontend's [density-gradient correction](/home/lu26029/dev/warpSPH/src/warpSPH/modules/density/gradRhoL.py:35), surface detection and an optional shifting gate. Its cost matters when that feature and dimension are enabled; it is not a cost paid by every SPH scheme.

| Inventory family | Timed forms and scope | Default or usage notes |
| --- | --- | --- |
| Density, interpolation | Scalar field; CSR/grid; 1/32 lanes | 32 lanes by default |
| Gradient, divergence | Difference baseline; frozen renorm/CRK; CSR/grid plain | 32 lanes by default; scalar gradient/vector divergence |
| Curl, Laplacian | Vector curl/scalar Brookshaw; frozen corrections; formulation sweep | Serial by default; 32-lane variants explicitly selected |
| Covariance | Matrix output; CSR/grid; 1/32 lanes | 32 lanes by default; builder input |
| CRK volume, moments, density | Separated pair stages; frozen inputs where applicable | Main inventory uses CSR; volume has no tiled kernel |
| Renormalization | Covariance, fallback, inverse and eigenvalue diagnostics together | Dimension-dependent inverse policy; actual frontend callers exist |
| Core CRK factors and local terms | Complete core builder / frozen Torch coefficient assembly | Additional core API and AD-fallback coverage; ordinary frontend assembly differs |
| Native symmetric pinv, applyPinv | Matched-cutoff synthetic symmetric matrices, including rank-deficient rows | Separate APIs; not automatic substitutes for correction wrappers |
| pinv policy, Torch pinv | Dimension-dependent wrapper / general Torch baseline | Policy includes eigenvalue diagnostics; contracts differ from standalone APIs |

## Workloads and measurement contract

Hardware: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0, 97,887 MiB memory, driver 610.43.02; Torch 2.13.0+cu130; Warp reports 1.17.0. The GPU also drives the desktop. Clocks were not locked or modified, and retained compute contexts and intermittent desktop activity were recorded. Timing jobs ran sequentially. These are workstation measurements with scatter, not exclusive-device peak specifications.

The baseline is a deterministic periodic equal-support all-fluid lattice. 2D sizes are 1,024, 4,096, 16,384, 65,536, 262,144 and 1,048,576 particles; 3D sizes are 512, 4,096, 32,768, 262,144 and 2,097,152. Neighbor targets are 25/55/110/220. At the largest lattice, actual CSR entries per particle, including self, are 21/57/109/221 in 2D and 27/57/93/251 in 3D. Rates use the exact measured edge count at each size, not the requested target.

Every timed case first checks finite output and parity with the same formulation's serial CSR call. Native inverse/application also use a float64 reference with matched `rtol=1e-6`. The FP32 elementwise gate is `rtol=atol=2e-4`. This checks variant parity, not an independent physical accuracy oracle. Failed cases retain their errors and have no timing point; tolerances were not relaxed. Cancellation near zero can trigger the gate, so these flags alone do not establish production correctness bugs.

Four warmups precede nine synchronized public calls on a separate stream. Raw wall and CUDA-event intervals are retained. Wall time includes Python dispatch, allocation, event recording and final synchronization. CUDA-event intervals can include launch gaps and are not isolated kernel timings. Topology construction and frozen correction preparation are excluded; full correction builders include their own factor construction. The grid comparisons use a prebuilt grid, and do not measure rebuild amortization.

Directed non-self physical pairs define interactions/s; reverse pairs count separately. Fused CRK moments have one pair pass and full core CRK factors have three. Grid candidate visits are not the throughput denominator. Standalone matrix stages use matrices/s. Independent stage timings are not an additive attribution of a complete call.

A one-second projected per-call budget bounds expensive larger cases: a measured smaller point in the same series is scaled by the particle-count ratio solely to decide whether to skip. These estimates are scheduling metadata, never plotted measurements. Local matrix cases are deduplicated across irrelevant topology/neighbor options.

| FP32 dataset | Timed and qualified | Numerical flags | Budget skips |
| --- | --- | --- | --- |
| [benchmark-2d-float32.json](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/benchmark-2d-float32.json) | 1111 | 69 | 2 |
| [benchmark-3d-float32.json](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/benchmark-3d-float32.json) | 904 | 30 | 51 |
| [inventory-corrected-2d-float32.json](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-corrected-2d-float32.json) | 45 | 3 | 0 |
| [inventory-crk-consumers-3d-float32.json](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-crk-consumers-3d-float32.json) | 24 | 0 | 0 |
| [inventory-formulations-2d-float32.json](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-formulations-2d-float32.json) | 73 | 19 | 4 |
| [inventory-options-2d-float32.json](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-options-2d-float32.json) | 240 | 12 | 0 |

**2,397 timed FP32 cases** across these datasets; 746 have IQR greater than 10% of the median. IQR bands accompany every graph, so small differences between settings should not be treated as stable wins. The two baseline raw datasets include the secondary unused grid CRK API records; the main plotting views omit correction builders/stages on grid.

The additional FP32 sweeps cover 2D frozen renormalization/CRK gradient, divergence, curl and Laplacian at three sizes; 3D frozen CRK consumers at 512/32,768/262,144; all four gradient and Laplacian schemes in 2D; and Wendland2/Wendland6/CubicSpline with Gather/KernelMeanSymmetric on CSR at three sizes. Equal supports make some support formulas algebraically identical, so this is overhead/dispatch coverage rather than a heterogeneous-support study. Dot Laplacian uses a vector field and is labeled separately.

Not timed here: linear solves, AD/JVP/HVP execution, prepared execution APIs, arbitrary tensor-valued fields, irregular/free-surface/boundary populations, heterogeneous supports, index-width changes, and neighbor-search rebuild frequency. The 3D frozen renormalization consumer sweep was not expanded because preparing those factors invokes the expensive 3D inverse policy; its builder and frozen CRK consumers are measured separately. The core enums and source contain further options; the measured matrix above is the bounded inventory, not every Cartesian combination.

## Scaling graphs

Each PNG has synchronized public-call time on the left and interactions/s or matrices/s on the right, with median and IQR. SVG and raw-point/source manifests sit beside it. Links labeled 55/220 select neighbor targets; local stages have one graph because neighbor count is irrelevant. CRK and renormalization plots use CSR only. Core seven plots retain CSR/grid and serial/32-lane comparisons.

| Operator | 2D FP32 figures | 3D FP32 figures |
| --- | --- | --- |
| density | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-55/density-2d-float32-wall.png) / [220](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-220/density-2d-float32-wall.png) | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-55/density-3d-float32-wall.png) / [220](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-220/density-3d-float32-wall.png) |
| interpolate | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-55/interpolate-2d-float32-wall.png) / [220](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-220/interpolate-2d-float32-wall.png) | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-55/interpolate-3d-float32-wall.png) / [220](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-220/interpolate-3d-float32-wall.png) |
| gradient | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-55/gradient-2d-float32-wall.png) / [220](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-220/gradient-2d-float32-wall.png) | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-55/gradient-3d-float32-wall.png) / [220](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-220/gradient-3d-float32-wall.png) |
| divergence | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-55/divergence-2d-float32-wall.png) / [220](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-220/divergence-2d-float32-wall.png) | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-55/divergence-3d-float32-wall.png) / [220](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-220/divergence-3d-float32-wall.png) |
| curl | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-55/curl-2d-float32-wall.png) / [220](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-220/curl-2d-float32-wall.png) | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-55/curl-3d-float32-wall.png) / [220](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-220/curl-3d-float32-wall.png) |
| laplacian | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-55/laplacian-2d-float32-wall.png) / [220](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-220/laplacian-2d-float32-wall.png) | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-55/laplacian-3d-float32-wall.png) / [220](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-220/laplacian-3d-float32-wall.png) |
| covariance | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-55/covariance-2d-float32-wall.png) / [220](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-220/covariance-2d-float32-wall.png) | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-55/covariance-3d-float32-wall.png) / [220](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-220/covariance-3d-float32-wall.png) |
| crk_volume | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-55/crk_volume-2d-float32-wall.png) / [220](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-220/crk_volume-2d-float32-wall.png) | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-55/crk_volume-3d-float32-wall.png) / [220](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-220/crk_volume-3d-float32-wall.png) |
| crk_moments | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-55/crk_moments-2d-float32-wall.png) / [220](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-220/crk_moments-2d-float32-wall.png) | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-55/crk_moments-3d-float32-wall.png) / [220](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-220/crk_moments-3d-float32-wall.png) |
| crk_density | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-55/crk_density-2d-float32-wall.png) / [220](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-220/crk_density-2d-float32-wall.png) | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-55/crk_density-3d-float32-wall.png) / [220](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-220/crk_density-3d-float32-wall.png) |
| renorm | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-55/renorm-2d-float32-wall.png) / [220](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-220/renorm-2d-float32-wall.png) | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-55/renorm-3d-float32-wall.png) / [220](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-220/renorm-3d-float32-wall.png) |
| crk_factors | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-55/crk_factors-2d-float32-wall.png) / [220](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-220/crk_factors-2d-float32-wall.png) | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-55/crk_factors-3d-float32-wall.png) / [220](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-220/crk_factors-3d-float32-wall.png) |
| pinv_symmetric | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-55/pinv_symmetric-2d-float32-wall.png) | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-55/pinv_symmetric-3d-float32-wall.png) |
| apply_pinv | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-55/apply_pinv-2d-float32-wall.png) | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-55/apply_pinv-3d-float32-wall.png) |
| pinv_policy | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-55/pinv_policy-2d-float32-wall.png) | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-55/pinv_policy-3d-float32-wall.png) |
| torch_pinv | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-55/torch_pinv-2d-float32-wall.png) | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-55/torch_pinv-3d-float32-wall.png) |
| crk_terms | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-2d-float32-55/crk_terms-2d-float32-wall.png) | [55](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-figures-3d-float32-55/crk_terms-3d-float32-wall.png) |

| Corrected consumer | 2D FP32 | 3D FP32 |
| --- | --- | --- |
| gradient | [2D renorm and CRK](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-corrected-2d-float32-figures/gradient-2d-float32-wall.png) | [3D CRK](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-crk-consumers-3d-float32-figures/gradient-3d-float32-wall.png) |
| divergence | [2D renorm and CRK](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-corrected-2d-float32-figures/divergence-2d-float32-wall.png) | [3D CRK](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-crk-consumers-3d-float32-figures/divergence-3d-float32-wall.png) |
| curl | [2D renorm and CRK](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-corrected-2d-float32-figures/curl-2d-float32-wall.png) | [3D CRK](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-crk-consumers-3d-float32-figures/curl-3d-float32-wall.png) |
| laplacian | [2D renorm and CRK](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-corrected-2d-float32-figures/laplacian-2d-float32-wall.png) | [3D CRK](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-crk-consumers-3d-float32-figures/laplacian-3d-float32-wall.png) |

Formulation graphs: [gradient](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-formulations-2d-float32-figures/gradient-2d-float32-wall.png), [laplacian](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-formulations-2d-float32-figures/laplacian-2d-float32-wall.png). Kernel/support graphs: [density](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-options-2d-float32-figures/density-2d-float32-wall.png), [interpolate](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-options-2d-float32-figures/interpolate-2d-float32-wall.png), [gradient](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-options-2d-float32-figures/gradient-2d-float32-wall.png), [divergence](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-options-2d-float32-figures/divergence-2d-float32-wall.png), [curl](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-options-2d-float32-figures/curl-2d-float32-wall.png), [laplacian](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-options-2d-float32-figures/laplacian-2d-float32-wall.png), [covariance](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-options-2d-float32-figures/covariance-2d-float32-wall.png).

## Representative FP32 public-call results

The following CSR serial points use target 55: N=1,048,576 in 2D and N=2,097,152 in 3D. G interactions/s means billions of directed non-self physical interactions per second. Explicit serial settings make comparison consistent; they do not imply that serial is each operator's default.

| Operator | 2D ms | 2D G interactions/s | 3D ms | 3D G interactions/s |
| --- | --- | --- | --- | --- |
| density | 1.238 | 47.42 | 2.275 | 51.61 |
| interpolate | 1.269 | 46.28 | 2.288 | 51.33 |
| gradient | 1.606 | 36.56 | 3.566 | 32.94 |
| divergence | 1.576 | 37.25 | 3.672 | 31.98 |
| curl | 1.626 | 36.11 | 3.758 | 31.25 |
| laplacian | 1.908 | 30.77 | 4.304 | 27.29 |
| covariance | 2.137 | 27.48 | 5.379 | 21.83 |
| crk_volume | 1.076 | 54.57 | 2.083 | 56.39 |
| crk_moments | 2.347 | 25.02 | 9.526 | 12.33 |
| crk_density | 1.654 | 35.51 | 3.882 | 30.26 |

| Core API | Rate unit | 2D N | 2D ms | 2D rate | 3D N | 3D ms | 3D rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| renorm | G interactions/s | 1,048,576 | 2.311 | 25.4 | 4,096 | 668.916 | 0.000343 |
| crk_factors | G interactions/s | 1,048,576 | 49.863 | 3.53 | 2,097,152 | 82.196 | 4.29 |
| pinv_symmetric | M matrices/s | 1,048,576 | 0.096 | 1.09e+04 | 2,097,152 | 0.305 | 6.88e+03 |
| apply_pinv | M matrices/s | 1,048,576 | 0.112 | 9.36e+03 | 2,097,152 | 0.305 | 6.87e+03 |
| pinv_policy | M matrices/s | 1,048,576 | 0.105 | 9.99e+03 | 4,096 | 684.423 | 0.00598 |
| torch_pinv | M matrices/s | 1,048,576 | 72.153 | 14.5 | 2,097,152 | 311.067 | 6.74 |
| crk_terms | M matrices/s | 1,048,576 | 45.718 | 22.9 | 2,097,152 | 65.866 | 31.8 |

Core CRK full/local values describe the Torch-based core API, not the ordinary fused frontend path. The native local APIs use synthetic symmetric batches; the matrix policy also computes diagnostics, so their timings are not an interchangeable implementation comparison. The missing large 3D renormalization/policy points are budget skips, not zero time.

## Traversal and launch options

Large-N examples below use target 55 and show time in ms. These are the basic simulation-relevant operator families; unused grid CRK moments are intentionally absent.

| Operator | CSR serial | CSR 32 lanes | Grid serial | Grid 32 lanes |
| --- | --- | --- | --- | --- |
| 2D density | 1.238 | 1.428 | 2.728 | 9.833 |
| 2D gradient | 1.606 | 1.890 | 3.587 | 13.407 |
| 2D curl | 1.626 | 1.876 | 3.387 | 13.493 |
| 2D covariance | 2.137 | 2.770 | 5.812 | 19.593 |
| 3D density | 2.275 | 3.569 | 25.829 | 50.024 |
| 3D gradient | 3.566 | 5.496 | 32.635 | 60.165 |
| 3D curl | 3.758 | 5.774 | 34.148 | 67.674 |
| 3D covariance | 5.379 | 8.274 | 48.673 | 95.937 |

Target-degree scaling below uses CSR serial at the same largest sizes. Each entry is **ms / G interactions/s**, normalized by the actual degree at that point.

| Operator | Target 25 | Target 55 | Target 110 | Target 220 |
| --- | --- | --- | --- | --- |
| 2D density | 0.442 / 47.47 | 1.238 / 47.42 | 2.376 / 47.66 | 6.261 / 36.84 |
| 2D gradient | 0.612 / 34.25 | 1.606 / 36.56 | 3.404 / 33.26 | 8.170 / 28.23 |
| 2D covariance | 0.790 / 26.56 | 2.137 / 27.48 | 4.510 / 25.11 | 9.836 / 23.45 |
| 2D crk_moments | 1.037 / 20.23 | 2.347 / 25.02 | 4.482 / 25.26 | 9.901 / 23.30 |
| 3D density | 1.097 / 49.71 | 2.275 / 51.61 | 4.012 / 48.09 | 12.224 / 42.89 |
| 3D gradient | 1.598 / 34.13 | 3.566 / 32.94 | 5.988 / 32.22 | 16.498 / 31.78 |
| 3D covariance | 2.323 / 23.48 | 5.379 / 21.83 | 9.381 / 20.57 | 26.250 / 19.97 |
| 3D crk_moments | 4.732 / 11.52 | 9.526 / 12.33 | 15.721 / 12.27 | 43.042 / 12.18 |

Frozen correction-consumer times below use CSR serial and target 55. The 2D N is 1,048,576; the 3D N is 262,144. Factor construction is excluded from every corrected-consumer column.

| Consumer | Plain ms | Frozen renorm ms | Frozen CRK ms |
| --- | --- | --- | --- |
| 2D gradient | 1.606 | 1.498 | 1.584 |
| 2D divergence | 1.576 | 1.539 | 1.629 |
| 2D curl | 1.626 | 1.544 | 1.634 |
| 2D laplacian | 1.908 | 1.742 | 1.859 |
| 3D gradient | 0.588 | not measured | 0.631 |
| 3D divergence | 0.576 | not measured | 0.630 |
| 3D curl | 0.714 | not measured | 0.643 |
| 3D laplacian | 0.645 | not measured | 0.731 |

The lane/traversal ordering depends on N and neighbor work. Serial kernels use 256 threads/block and approximately ceil(N/256) blocks; tiled kernels use one block per query particle with 32 threads/block. Small serial grids can leave most of the GPU unused, while large tiled grids have many small blocks. Register limits, finite grid size, candidate work and memory access all matter; occupancy alone cannot choose a lane setting.

These prebuilt-topology results cannot establish whether an evolving simulation should rebuild CSR or reuse a grid. They also do not characterize its clustering, support asymmetry or boundary workload. Such decisions need the actual scheme's topology lifetime and calling pattern.

## Registers and launch diagnostics

The offline SM120 surveys compile actual production overloads with Warp and ptxas 13.0.48, without executing GPU kernels. The runtime CSR/grid and correction branches share a compiled overload. Registers below are allocated physical registers per thread, not PTX virtual declarations. The live driver queries independently match the sampled FP32 register allocations, including 3D covariance at 99/113 and CRK moments at 128/128. The failed 2D tiled Laplacian numerical gate prevented its live trace; its 72-register value remains an offline compiler result.

| Kernel | 2D registers serial / 32 lanes | 3D registers serial / 32 lanes |
| --- | --- | --- |
| density | 64 / 64 | 64 / 68 |
| interpolate | 68 / 64 | 71 / 64 |
| gradient | 72 / 75 | 83 / 95 |
| divergence | 72 / 78 | 87 / 94 |
| curl | 72 / 80 | 96 / 96 |
| laplacian | 72 / 72 | 79 / 80 |
| covariance | 80 / 80 | 99 / 113 |
| crk_volume | 70 | 72 |
| crk_moments | 91 / 95 | 128 / 128 |
| crk_density | 71 / 72 | 72 / 76 |
| pinv_symmetric | 40 | 40 |
| apply_pinv | 40 | 40 |
| pinv_policy | 30 | Torch |

No compiler-reported spill stores or loads appear in the surveyed forward kernels. This does not imply zero local-memory traffic: live local allocations are 88 bytes/thread in the basic pair kernels, 112 bytes for 2D moments, 184 bytes for 3D moments and 72 bytes for native inverse/application. Stack/local allocation and executed spill traffic are distinct measurements.

The preliminary report's occupancy values are optimistic resource ceilings that omit allocation granularity, partitions, shared-memory restrictions and finite grid size. The CUDA trace also supplies driver occupancy-calculator results, grid/block sizes, register/shared-memory allocation and limiting factors. For example, the serial density kernel permits four active blocks/SM in the driver calculation, while its N=65,536 grid has only 256 total blocks. Kineto's field named `est. achieved occupancy %` is an estimate; it is not a measured hardware counter. Keep both separate from achieved occupancy collected by Nsight.

The following are **actual CUDA kernel launches per public call** from separate Torch/CUDA traces, including initialization and vendor kernels. They are not the offline Torch-dispatch counts. The fixtures are 2D N=65,536 and 3D N=512, CSR serial, target 55. Instrumented durations are diagnostic only and do not supply scaling samples.

| Core API | 2D CUDA kernels/call | 3D CUDA kernels/call |
| --- | --- | --- |
| density | 2 | 2 |
| interpolate | 2 | 2 |
| gradient | 2 | 2 |
| divergence | 2 | 2 |
| curl | 2 | 2 |
| laplacian | 2 | 2 |
| covariance | 3 | 3 |
| crk_volume | 2 | 2 |
| crk_moments | 8 | 8 |
| crk_density | 2 | 2 |
| renorm | 10 | 18,010 |
| crk_factors | 136 | 122 |
| pinv_symmetric | 1 | 1 |
| apply_pinv | 1 | 1 |
| pinv_policy | 3 | 18,099 |
| torch_pinv | 17 | 16 |
| crk_terms | 124 | 110 |

Basic public pair calls include output initialization, so one Warp traversal can correspond to two or more CUDA launches. Frozen CRK moments initialize multiple result arrays. The 3D inverse policy's roughly 18,000 kernels for only 512 matrices identifies a wrapper/vendor-batching cost that cannot be inferred from the covariance kernel's occupancy. The core CRK Torch terms likewise emit many kernels, but ordinary frontend CRK local assembly already uses its fused kernel.

## Hardware counter capture

The first Nsight Compute 2025.3 attempts returned `ERR_NVGPUCTRPERM`. After the user configured R610 profiling capabilities, this account could open the profiler-device, profiler-context and trace-device nodes, but 2025.3 still rejected collection. The 2026.3.1 profiler then successfully collected counters. The persistent system installation is `/usr/local/NVIDIA-Nsight-Compute/ncu`; the earlier standalone extraction was removed after collection. Exact logs and capture commands are retained, and no occupancy or bandwidth value is inferred from elapsed time. [NVIDIA's permission documentation](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters) describes the legacy driver restriction and R610 capability grants.

Successful FP32 captures contain **253 matched Warp kernel records and 607 total CUDA kernel records**, including auxiliary initialization kernels. They cover the seven core operators at small N and N=262,144, targets 55/220, CSR/grid and requested lanes 1/32; and CSR CRK stages/native matrix APIs at 2D N=65,536 and 3D N=32,768. Eight cases failed the numerical gate before capture and have no counters. The 2D renormalization capture includes both covariance and inverse kernels. The 3D launch-heavy Torch inverse policy is described by its separate launch trace rather than replaying its thousands of vendor kernels.

The system profiler is `/usr/local/NVIDIA-Nsight-Compute/ncu`, reporting **2026.3.1**. Core captures used the identical-version standalone distribution before switching to that persistent installation for the matrix and bandwidth captures. Commands, sections, reports and raw CSV exports are retained. `--clock-control none` preserves workstation clocks; normal kernel replay and default cache control apply. Capture durations differ from ordinary execution and are never substituted for the scaling samples.

The next table uses CSR at N=262,144 and target 55. Entries are **serial / requested 32 lanes**. Achieved occupancy is the active-warp counter relative to hardware capacity during active cycles; it is not whole-device utilization. Eligible warps are per scheduler per active cycle. L2% is a composite pipeline utilization metric, and DRAM% is its own throughput metric.


| Core operator | Registers | Theoretical occupancy % | Achieved occupancy % | Eligible warps | L2 % | DRAM % |
| --- | --- | --- | --- | --- | --- | --- |
| 2D density | 64 / 64 | 66.7 / 50.0 | 59.6 / 48.5 | 0.48 / 0.63 | 74.2 / 78.7 | 13.6 / 14.7 |
| 2D interpolate | 68 / 64 | 50.0 / 50.0 | 46.6 / 48.7 | 0.47 / 0.58 | 89.6 / 80.6 | 14.2 / 12.4 |
| 2D gradient | 72 / 75 | 50.0 / 50.0 | 47.6 / 48.5 | 0.56 / 0.69 | 82.9 / 80.9 | 11.6 / 10.4 |
| 2D divergence | 72 / 78 | 50.0 / 50.0 | 48.2 / 48.6 | 0.55 / 0.70 | 84.9 / 75.3 | 11.8 / 10.3 |
| 2D curl | 72 / 80 | 50.0 / 50.0 | 48.3 / 48.6 | 0.62 / 0.74 | 82.9 / 75.7 | 11.1 / 10.2 |
| 2D laplacian | 72 / — | 50.0 / — | 45.4 / — | 0.69 / — | 84.0 / — | 9.9 / — |
| 2D covariance | 80 / 80 | 50.0 / 50.0 | 45.8 / 48.4 | 0.62 / 0.76 | 82.0 / 86.5 | 7.8 / 7.0 |
| 3D density | 64 / 68 | 66.7 / 50.0 | 58.7 / 48.9 | 0.57 / 0.48 | 75.7 / 91.2 | 13.1 / 10.6 |
| 3D interpolate | 71 / 64 | 50.0 / 50.0 | 47.2 / 49.0 | 0.59 / 0.50 | 87.3 / 92.1 | 14.5 / 9.8 |
| 3D gradient | 83 / 95 | 33.3 / 41.7 | 31.1 / 40.5 | 0.46 / 0.52 | 69.9 / 87.6 | 10.4 / 7.7 |
| 3D divergence | 87 / 94 | 33.3 / 41.7 | 30.8 / 40.5 | 0.47 / 0.56 | 64.9 / 82.8 | 10.6 / 7.5 |
| 3D curl | 96 / 96 | 33.3 / 41.7 | 31.5 / 40.5 | 0.48 / 0.51 | 77.5 / 91.0 | 10.2 / 7.8 |
| 3D laplacian | 79 / 80 | 50.0 / 50.0 | 46.3 / 48.7 | 0.82 / 0.74 | 83.6 / 92.1 | 8.6 / 6.2 |
| 3D covariance | 99 / 113 | 33.3 / 33.3 | 31.7 / 32.6 | 0.60 / 0.56 | 76.5 / 87.3 | 6.4 / 4.5 |


At target 55, the 2D CSR serial kernels show long-scoreboard waits as their largest stall category (roughly 4–14 cycles per issued instruction), with fewer than one eligible warp per scheduler. Their L2 composite utilization is high while DRAM utilization is much lower. This supports examining cache/request traffic and memory dependencies, rather than assuming DRAM bandwidth saturation. At large N, 3D covariance reaches about 32% achieved occupancy against a 33% theoretical limit: register/resource limits are visible even in FP32.

High occupancy also accompanies slow grid cases. For example, 2D gradient on grid at requested 32 lanes reaches about 48% achieved occupancy, similar to CSR, while the ordinary scaling measurements show substantially lower useful physical-pair throughput. Occupancy does not account for how much candidate/instruction work is performed per useful interaction.

At the small fixtures, serial density has only four blocks in 2D or two in 3D on a device with **188 SMs**. Achieved occupancy during active cycles can still be about 16%, while SM throughput over elapsed cycles is below 1%. Finite grid size and launch overhead must therefore be distinguished from resource occupancy.

CSR matrix-stage resource results below retain their own fixture sizes. CRK volume has no tiled implementation, so its requested-lane overrides launch the same serial kernel.


| Stage | N | Registers | Theoretical occupancy % | Achieved occupancy % | Eligible warps |
| --- | --- | --- | --- | --- | --- |
| 2D crk_volume | 65,536 | 70 / 70 | 50.0 / 50.0 | 22.9 / 22.9 | 0.27 / 0.27 |
| 2D crk_moments | 65,536 | 91 / 95 | 33.3 / 41.7 | 22.1 / 39.8 | 0.35 / 0.68 |
| 2D crk_density | 65,536 | 71 / 72 | 50.0 / 50.0 | 22.9 / 48.0 | 0.32 / 0.56 |
| 2D pinv_symmetric | 65,536 | 40 / — | 100.0 / — | 22.7 / — | 0.59 / — |
| 2D apply_pinv | 65,536 | 40 / — | 100.0 / — | 22.6 / — | 0.57 / — |
| 3D crk_volume | 32,768 | 72 / 72 | 50.0 / 50.0 | 16.2 / 16.2 | 0.23 / 0.23 |
| 3D crk_moments | 32,768 | 128 / 128 | 33.3 / 33.3 | 15.0 / 32.1 | 0.29 / 0.43 |
| 3D crk_density | 32,768 | 72 / 76 | 50.0 / 50.0 | 16.3 / 46.8 | 0.27 / 0.44 |
| 3D pinv_symmetric | 32,768 | 40 / — | 100.0 / — | 15.7 / — | 0.47 / — |
| 3D apply_pinv | 32,768 | 40 / — | 100.0 / — | 15.8 / — | 0.46 / — |


Nsight reports zero executed register-spilling bytes for the sampled Warp kernels. This confirms the compiler spill result for these captures, while leaving stack/local allocation as a separate resource statistic.

![FP32 registers and measured occupancy](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/fp32-kernel-resources.png)

### Absolute L2 rates

Nsight's L2 composite percentage is the largest normalized utilization among its constituent pipeline counters. In these probes the incoming crossbar counter sets that composite; it is not directly a percentage of byte bandwidth. A separate FP32 CSR capture at N=262,144 and target 55 records the requested-byte rate and its clock-dependent peak reference explicitly. TB/s here is decimal, and counts bytes requested at L2 rather than unique bytes reaching DRAM. [NVIDIA metric definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#metrics-overview).


| Operator | L2 requested TB/s | Peak reference TB/s | L2 byte-rate % | Composite L2 % | DRAM GB/s |
| --- | --- | --- | --- | --- | --- |
| density | 6.83 | 9.57 | 71.4 | 73.5 | 229.4 |
| gradient | 7.77 | 9.55 | 81.3 | 84.3 | 197.6 |
| covariance | 5.96 | 9.71 | 61.4 | 81.9 | 138.1 |


The peak reference is approximately **9.6 TB/s at the clocks observed in these captures**; it is not a separately measured streaming-cache maximum or a fixed marketing specification. The composite and byte-rate percentages differ most clearly for covariance. Raw rates and units are in [the bandwidth CSV](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-l2-reference.csv).

Full per-case metrics, attribution checks, report hashes and units are in [the counter analysis](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-counters-analysis.json). The four capture JSONs and compressed `.ncu-repz` reports are indexed there; [system capture commands](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/counter-system-manifest.json) use the persistent installation.

## Bottlenecks to investigate in used paths

1. **Public-call overhead at small N.** Compare the wrapper interval, output initialization, launch gaps and actual Warp kernel duration for each operator. The launch inventory establishes which calls have multiple auxiliary kernels. Prepared execution and timestep reuse need their own application measurements.
2. **CSR neighbor consumers across size and degree.** Inspect the seven basic operators as a family. Their large-N lane ordering differs from small launches, and grid candidate work differs from CSR physical pairs. Use the real simulation's traversal and topology lifetime when interpreting this comparison.
3. **3D renormalization when enabled.** The core wrapper pays for diagnostic general eigenvalues, sorting and a selected symmetric inverse. Both its measured scaling and launch census warrant attribution to the exact vendor/host operations in the configurations that call it. This is a feature-specific finding; it does not generalize to schemes that do not build these matrices.
4. **Matrix-heavy CSR kernels.** Covariance and CRK moments have higher FP32 register/local allocation than scalar operators. Relate their driver occupancy limits to measured stalls, issue activity and memory traffic before assigning a compute or memory bottleneck. Core Torch CRK factor assembly is kept as API/AD coverage because ordinary frontend local assembly differs.
5. **Numerical qualification gaps.** Preserve and investigate the flagged Laplacian and reduction-order cases with scale-aware diagnostics before treating their missing timing points as performance outcomes. The benchmark's componentwise parity gate is stricter than an error norm and should not be silently widened.

No optimization experiment or source replacement is included. Grid CRK moments are not a simulation optimization target.

## Reproducibility and validation

Generated evidence is saved in [the persistent development archive](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/) in accordance with the [development artifact policy](../docs/development-artifacts.md). The [inventory summary](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/inventory-summary.json), raw datasets and graph manifests retain commands, configuration, source hashes, individual samples, work counts, errors and skip reasons. The [FP32 follow-up manifest](/home/lu26029/dev/warpSPHDevelopment/core/experiments/core-operator-inventory-2026-10-21/fp32-inventory-manifest.json) retains exact job commands and exit codes; the one rejected trace was rerun with its final script and passed the source guard. The original interrupted 3D FP64 sweep and exploratory runs are not plotted as completed series.

The suite was established and CPU-qualified before the GPU sweeps: 660 baseline CPU cases plus 288 targeted option cases. Fresh GPU qualification occurs before every timed call. Final benchmark tests: **82 passed**, with 14 pre-existing Torch deprecation warnings. Raw sample medians, interaction counts and reported rates were independently checked by the scaling analyzer. Historical 15 October evidence remains separately labeled in the [preparation report](core_operator_evaluation_2026_10_21.md).

Retained tools:

- [Workload and option definitions](../../scripts/benchmarks/_operator_cases.py), [qualification/timing runner](../../scripts/benchmarks/bench_core_operators.py) and [phase driver](../../scripts/benchmarks/run_operator_evaluation.py).
- [Offline resources](../../scripts/benchmarks/survey_operator_resources.py), [untimed dispatch census](../../scripts/benchmarks/survey_operator_dispatch.py) and [live launch profiler](../../scripts/benchmarks/profile_sph_stages.py).
- [Plotter](../../scripts/benchmarks/plot_core_operators.py), [sample/rate analyzer](../../scripts/benchmarks/analyze_core_scaling.py), [counter association/export tool](../../scripts/benchmarks/analyze_sph_counters.py) and [benchmark contract tests](../../tests/benchmarks/test_core_operator_suite.py).

For a bounded FP32 repeat:

```bash
PYTHON=/home/lu26029/miniconda3/envs/warp/bin/python
OUT=.tmp/operator-evaluation/repeat
WARP_CACHE_PATH="$PWD/.tmp/operator-evaluation/pytest-warp-cache"   "$PYTHON" -m pytest tests/benchmarks -q
warpSPHCore_PRECISION=float32 "$PYTHON" scripts/benchmarks/bench_core_operators.py benchmark   --device cuda:0 --dim 2 --ns 1024 4096 16384 65536 262144 1048576   --neighbors 25 55 110 220 --operations density interpolate gradient divergence curl laplacian covariance   --traversals adjacency grid --lanes 1 32 --keep-going --max-estimated-call-ms 1000   --output "$OUT/core-2d.json"
"$PYTHON" scripts/benchmarks/plot_core_operators.py "$OUT/core-2d.json"   --neighbors 55 --output-dir "$OUT/figures"
```

Use `--traversals adjacency` for the simulation-relevant CRK inventory. Counter runs use `--profile-range` and store no ordinary performance samples. The full recorded commands include separate correction/formulation options and profiling conditions. For a separate diagnostic capture with the persistent profiler:

```bash
NCU=/usr/local/NVIDIA-Nsight-Compute/ncu
"$NCU" --section-folder /usr/local/NVIDIA-Nsight-Compute/sections   --clock-control none --profile-from-start off   --section LaunchStats --section Occupancy --section SpeedOfLight   --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats   --export "$OUT/density-counters"   "$PYTHON" scripts/benchmarks/bench_core_operators.py benchmark --device cuda:0   --profile-range --ns 65536 --neighbors 55 --operations density   --traversals adjacency --lanes 1 32 --output "$OUT/density-counters.json"
"$PYTHON" scripts/benchmarks/analyze_sph_counters.py "$OUT/density-counters.json"   --ncu "$NCU" --output "$OUT/density-counter-analysis.json"
```

Source implementation files are unchanged.
