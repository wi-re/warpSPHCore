# Native operator accumulator catalog

This catalog accompanies the accumulator survey. D is dimension, F is field component count, and P is polynomial basis size. Counts describe reduced state rather than total registers. Each record groups related range, adjacency and launch definitions where applicable. Source links point to the reviewed checkouts.

## Core operations

### density

[src/warpSPHCore/coreOperations/native/wp_density.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_density.py:33)

- Accumulates: density.
- Components: 1.
- Combines: componentwise sum.
- Finalizes: identity.
- Stores: one output field.
- Loads: mass and support; position/kind; optional CRK query coefficients and renormalization where used.
- Primary pattern and complexity: sum; low.

### interpolate

[src/warpSPHCore/coreOperations/native/wp_interpolate.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_interpolate.py:29)

- Accumulates: field.
- Components: F.
- Combines: componentwise sum.
- Finalizes: identity.
- Stores: one output field.
- Loads: field, volume or mass/density; optional omega; optional CRK query coefficients and renormalization where used.
- Primary pattern and complexity: sum; low.

### gradient

[src/warpSPHCore/coreOperations/native/wp_gradient.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_gradient.py:40)

- Accumulates: field gradient.
- Components: F*D.
- Combines: componentwise sum.
- Finalizes: identity.
- Stores: one output field.
- Loads: field, volume or mass/density; optional omega; optional CRK query coefficients and renormalization where used.
- Primary pattern and complexity: sum; low.

### divergence

[src/warpSPHCore/coreOperations/native/wp_divergence.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_divergence.py:24)

- Accumulates: field contraction.
- Components: output component count (vector input: 1).
- Combines: componentwise sum.
- Finalizes: identity.
- Stores: one output field.
- Loads: field, volume or mass/density; optional omega; optional CRK query coefficients and renormalization where used.
- Primary pattern and complexity: sum; low.

### curl

[src/warpSPHCore/coreOperations/native/wp_curl.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_curl.py:25)

- Accumulates: curl.
- Components: output component count (2D vector: 1; 3D vector: 3).
- Combines: componentwise sum.
- Finalizes: identity.
- Stores: one output field.
- Loads: field, volume or mass/density; optional omega; optional CRK query coefficients and renormalization where used.
- Primary pattern and complexity: sum; low.

### laplacian

[src/warpSPHCore/coreOperations/native/wp_laplacian.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_laplacian.py:26)

- Accumulates: field Laplacian.
- Components: output component count (ordinary scalar/vector: F).
- Combines: componentwise sum.
- Finalizes: identity.
- Stores: one output field.
- Loads: field, volume or mass/density; optional omega; scheme-specific derivatives; optional CRK query coefficients and renormalization where used.
- Primary pattern and complexity: sum; medium.

### covariance

[src/warpSPHCore/coreOperations/native/wp_covariance.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_covariance.py:39)

- Accumulates: matrix and neighbor count.
- Components: D*D floating + 1 integer.
- Combines: componentwise sum.
- Finalizes: identity.
- Stores: matrix and integer count.
- Loads: volume or mass/density; displacement, support and kind; optional CRK query coefficients and renormalization where used.
- Primary pattern and complexity: sum; low.
- Detail: Counts are accumulated even when the public caller requests only the covariance matrix.

### density JVP

[src/warpSPHCore/coreOperations/wp_densityJVP.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_densityJVP.py:44)

- Accumulates: tangent output.
- Components: 1.
- Combines: componentwise sum.
- Finalizes: identity.
- Stores: one tangent field.
- Loads: primal and tangent positions, support, relevant fields, volumes and correction data.
- Primary pattern and complexity: sum; medium.
- Detail: Separate JVP pass; do not assume every primal rank and scheme is supported.

### interpolate JVP

[src/warpSPHCore/coreOperations/wp_interpolateJVP.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_interpolateJVP.py:44)

- Accumulates: tangent output.
- Components: F.
- Combines: componentwise sum.
- Finalizes: identity.
- Stores: one tangent field.
- Loads: primal and tangent positions, support, relevant fields, volumes and correction data.
- Primary pattern and complexity: sum; medium.
- Detail: Separate JVP pass; do not assume every primal rank and scheme is supported.

### gradient JVP

[src/warpSPHCore/coreOperations/wp_gradientJVP.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_gradientJVP.py:54)

- Accumulates: tangent output.
- Components: D for the scalar-field path.
- Combines: componentwise sum.
- Finalizes: identity.
- Stores: one tangent field.
- Loads: primal and tangent positions, support, relevant fields, volumes and correction data.
- Primary pattern and complexity: sum; medium.
- Detail: Separate JVP pass; do not assume every primal rank and scheme is supported.

### divergence JVP

[src/warpSPHCore/coreOperations/wp_divergenceJVP.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_divergenceJVP.py:43)

- Accumulates: tangent output.
- Components: 1 for the vector-field path.
- Combines: componentwise sum.
- Finalizes: identity.
- Stores: one tangent field.
- Loads: primal and tangent positions, support, relevant fields, volumes and correction data.
- Primary pattern and complexity: sum; medium.
- Detail: Separate JVP pass; do not assume every primal rank and scheme is supported.

### curl JVP

[src/warpSPHCore/coreOperations/wp_curlJVP.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_curlJVP.py:44)

- Accumulates: tangent output.
- Components: 1 for the 2D vector-field path.
- Combines: componentwise sum.
- Finalizes: identity.
- Stores: one tangent field.
- Loads: primal and tangent positions, support, relevant fields, volumes and correction data.
- Primary pattern and complexity: sum; medium.
- Detail: Separate JVP pass; do not assume every primal rank and scheme is supported.

### covariance JVP

[src/warpSPHCore/coreOperations/wp_covarianceJVP.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_covarianceJVP.py:52)

- Accumulates: tangent output.
- Components: D*D.
- Combines: componentwise sum.
- Finalizes: identity.
- Stores: one tangent field.
- Loads: primal and tangent positions, support, relevant fields, volumes and correction data.
- Primary pattern and complexity: sum; medium.
- Detail: Separate JVP pass; do not assume every primal rank and scheme is supported.

### Laplacian Brookshaw JVP

[src/warpSPHCore/coreOperations/wp_laplacianJVP.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_laplacianJVP.py:233)

- Accumulates: tangent output.
- Components: output component count.
- Combines: componentwise sum.
- Finalizes: identity.
- Stores: one tangent field.
- Loads: primal/tangent pair geometry, fields, volumes and corrections.
- Primary pattern and complexity: sum; medium.

### Laplacian Naive JVP

[src/warpSPHCore/coreOperations/wp_laplacianJVP.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_laplacianJVP.py:536)

- Accumulates: tangent output.
- Components: output component count.
- Combines: componentwise sum.
- Finalizes: identity.
- Stores: one tangent field.
- Loads: primal/tangent pair geometry, fields, volumes and corrections.
- Primary pattern and complexity: sum; medium.

### Laplacian Dot JVP

[src/warpSPHCore/coreOperations/wp_laplacianJVP.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_laplacianJVP.py:825)

- Accumulates: tangent output.
- Components: output component count.
- Combines: componentwise sum.
- Finalizes: identity.
- Stores: one tangent field.
- Loads: primal/tangent pair geometry, fields, volumes and corrections.
- Primary pattern and complexity: sum; medium.

### Laplacian Default JVP

[src/warpSPHCore/coreOperations/wp_laplacianJVP.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_laplacianJVP.py:1182)

- Accumulates: tangent output.
- Components: output component count.
- Combines: componentwise sum.
- Finalizes: identity.
- Stores: one tangent field.
- Loads: primal/tangent pair geometry, fields, volumes and corrections.
- Primary pattern and complexity: sum; medium.

### density Hessian edge terms

[src/warpSPHCore/coreOperations/wp_densityHVP.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_densityHVP.py:129)

- Accumulates: none across neighbors.
- Components: 0.
- Combines: none.
- Finalizes: evaluate one pair Hessian.
- Stores: D*D per explicit edge.
- Loads: query/reference positions and support.
- Primary pattern and complexity: edge; medium.
- Detail: Only the explicit Warp edge kernel is counted here; subsequent HVP assembly is Python/Torch scaffolding.

### CRK density

[src/warpSPHCore/crk/crk_density.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/crk_density.py:30)

- Accumulates: mass-weighted volume sum; squared-volume sum.
- Components: 2.
- Combines: sum.
- Finalizes: guarded numerator/denominator.
- Stores: one density.
- Loads: mass, apparent volume, geometry/kernel.
- Primary pattern and complexity: finalize; low.

### CRK volume

[src/warpSPHCore/crk/crk_volume.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/crk_volume.py:28)

- Accumulates: kernel sum.
- Components: 1.
- Combines: sum.
- Finalizes: guarded reciprocal.
- Stores: one volume.
- Loads: geometry/kernel.
- Primary pattern and complexity: finalize; low.

### CRK moments

[src/warpSPHCore/crk/crk_moments.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/crk_moments.py:37)

- Accumulates: m0, m1, m2, gradient(m0), gradient(m1), gradient(m2), neighbor count.
- Components: 1 + 2*D + 2*D*D + D*D*D floating + 1 integer (3D: 52 + 1).
- Combines: componentwise sum.
- Finalizes: identity; coefficient solve is a later stage.
- Stores: six moment arrays and one count array.
- Loads: volume, displacement, kernel value and gradient.
- Primary pattern and complexity: sum; medium.
- Detail: Largest fixed SPH moment state in this survey; GPU scheduling must account for register pressure.

### CRK volume JVP

[src/warpSPHCore/crk/crk_volume_jvp.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/crk_volume_jvp.py:43)

- Accumulates: tangent kernel sum.
- Components: 1.
- Combines: sum.
- Finalizes: -d(sum W)*V*V using primal volume.
- Stores: one tangent volume.
- Loads: primal/tangent geometry and support.
- Primary pattern and complexity: finalize; medium.

### CRK moments JVP

[src/warpSPHCore/crk/crk_moments_jvp.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/crk_moments_jvp.py:49)

- Accumulates: six tangent moment tensors.
- Components: 1 + 2*D + 2*D*D + D*D*D (3D: 52).
- Combines: componentwise sum.
- Finalizes: identity.
- Stores: six tangent moment arrays.
- Loads: primal/tangent geometry, volumes and kernel derivatives.
- Primary pattern and complexity: sum; medium.

### MFM moments

[src/warpSPHCore/mfm/wp_mfm.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:316)

- Accumulates: omega, E, count.
- Components: 1 + D*D floating + 1 integer.
- Combines: sum.
- Finalizes: add self count of one.
- Stores: omega, matrix, count.
- Loads: positions and supports.
- Primary pattern and complexity: sum; low.

### MFM gradients

[src/warpSPHCore/mfm/wp_mfm.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:393)

- Accumulates: density gradient sum, velocity Jacobian sum, pressure gradient sum.
- Components: 2*D + D*D (3D: 15).
- Combines: sum.
- Finalizes: multiply by query inverse moment matrix; zero deficient queries.
- Stores: two gradient vectors and a Jacobian.
- Loads: rho, velocity, pressure, geometry/kernel.
- Primary pattern and complexity: finalize; medium.

### MFM limiter

[src/warpSPHCore/mfm/wp_mfm.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:474)

- Accumulates: neighbor and reconstructed min/max for rho, P and each velocity component.
- Components: 8 + 4*D (3D: 20).
- Combines: mixed componentwise min/max, query-value identities.
- Finalizes: alphaScalar ratios/clamps with query beta.
- Stores: rho limiter, velocity limiters, pressure limiter.
- Loads: rho, velocity, pressure, support and position; query gradients.
- Primary pattern and complexity: extrema; medium.

### MFM closure residual

[src/warpSPHCore/mfm/wp_mfm.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:587)

- Accumulates: face-vector sum and squared-face-norm sum.
- Components: D + 1.
- Combines: sum.
- Finalizes: identity.
- Stores: vector and scalar.
- Loads: positions, support, volumes, inverse moments, closure data.
- Primary pattern and complexity: sum; medium.

### MFM closure weights

[src/warpSPHCore/mfm/wp_mfm.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:625)

- Accumulates: none across neighbors.
- Components: 0.
- Combines: none.
- Finalizes: pair face weight and cap.
- Stores: one kappa per CSR edge.
- Loads: geometry, volumes, inverse moments.
- Primary pattern and complexity: edge; high.
- Detail: Following matvec and coarse-stencil kernels read kappa by edge offset; list-free execution needs recomputation or alternate storage.

### MFM closure matvec

[src/warpSPHCore/mfm/wp_mfm.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:657)

- Accumulates: Laplacian action and diagonal.
- Components: 2.
- Combines: sum.
- Finalizes: identity.
- Stores: y and diagonal.
- Loads: p[j] plus kappa at CSR edge offset.
- Primary pattern and complexity: sum; high.
- Detail: Arithmetic is simple; retained edge-weight lifecycle is the integration issue.

### MFM closure CSR matvec

[src/warpSPHCore/mfm/wp_closure.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_closure.py:24)

- Accumulates: Laplacian action and diagonal.
- Components: 2.
- Combines: sum.
- Finalizes: identity.
- Stores: y and diagonal.
- Loads: p[j], precomputed edge weights.
- Primary pattern and complexity: sum; high.

### MFM flux

[src/warpSPHCore/mfm/wp_mfm.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:765)

- Accumulates: mass, momentum and total-energy rates.
- Components: D + 2.
- Combines: sum.
- Finalizes: identity.
- Stores: dm, dp, de.
- Loads: primitive states, gradients, limiters, volumes, inverse moments, closure, material and Riemann data.
- Primary pattern and complexity: sum; high.
- Detail: Large pair computation, but only D+2 accumulated components; uses canonical pair orientation.

### MFM coarse stencil

[src/warpSPHCore/mfm/wp_mfm.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:992)

- Accumulates: one weighted sum per neighboring cell offset.
- Components: 3**D (3D: 27).
- Combines: componentwise sum into selected slot.
- Finalizes: identity.
- Stores: one fixed stencil vector.
- Loads: cell IDs and kappa indexed by CSR edge.
- Primary pattern and complexity: sum; high.

### MFM timestep

[src/warpSPHCore/mfm/wp_mfm.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:1064)

- Accumulates: maximum signal speed.
- Components: 1.
- Combines: max, initialized with twice query sound speed.
- Finalizes: 2*CFL*h/maxSignal.
- Stores: one timestep.
- Loads: rho, pressure, velocity, support and position.
- Primary pattern and complexity: extrema; low.

### multigrid segment sum

[src/warpSPHCore/mfm/multigrid.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/multigrid.py:68)

- Accumulates: one component of a coarse-cell sum.
- Components: 1 per output component.
- Combines: sum.
- Finalizes: identity.
- Stores: out[coarse cell, component].
- Loads: source array and permutation.
- Primary pattern and complexity: global; low.
- Detail: Segmented aggregation, not a particle neighborhood traversal.

### polynomial moments

[src/warpSPHCore/polyfit/wp_polyfit.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/polyfit/wp_polyfit.py:135)

- Accumulates: one matrix coefficient and count.
- Components: 1 floating + 1 integer per (query, matrix coefficient).
- Combines: sum.
- Finalizes: identity.
- Stores: matrix coefficient and, in primal, count from coefficient zero.
- Loads: scaled displacement, volume/kernel, exponents and sector; tangents in JVP.
- Primary pattern and complexity: sum; high.
- Detail: Full output size depends on basis. Preserve component/block scheduling rather than putting the entire polynomial matrix in each lane.

### polynomial RHS

[src/warpSPHCore/polyfit/wp_polyfit.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/polyfit/wp_polyfit.py:288)

- Accumulates: one RHS coefficient.
- Components: 1 per (query, basis, field component).
- Combines: sum.
- Finalizes: identity.
- Stores: RHS coefficient.
- Loads: field, scaled displacement, volume/kernel, exponents and sector; tangents in JVP.
- Primary pattern and complexity: sum; medium.

### polynomial moments JVP

[src/warpSPHCore/polyfit/wp_polyfitJVP.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/polyfit/wp_polyfitJVP.py:115)

- Accumulates: one matrix coefficient.
- Components: 1 floating per (query, matrix coefficient).
- Combines: sum.
- Finalizes: identity.
- Stores: matrix coefficient and, in primal, count from coefficient zero.
- Loads: scaled displacement, volume/kernel, exponents and sector; tangents in JVP.
- Primary pattern and complexity: sum; high.
- Detail: Full output size depends on basis. Preserve component/block scheduling rather than putting the entire polynomial matrix in each lane.

### polynomial RHS JVP

[src/warpSPHCore/polyfit/wp_polyfitJVP.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/polyfit/wp_polyfitJVP.py:193)

- Accumulates: one RHS coefficient.
- Components: 1 per (query, basis, field component).
- Combines: sum.
- Finalizes: identity.
- Stores: RHS coefficient.
- Loads: field, scaled displacement, volume/kernel, exponents and sector; tangents in JVP.
- Primary pattern and complexity: sum; medium.

### direct radius count

[src/warpSPHCore/radiusSearch/small/wp_radius_small.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/small/wp_radius_small.py:10)

- Accumulates: accepted neighbor count.
- Components: 1 integer.
- Combines: integer sum.
- Finalizes: identity.
- Stores: count per query.
- Loads: positions/supports.
- Primary pattern and complexity: construction; low.

### direct radius collect

[src/warpSPHCore/radiusSearch/small/wp_radius_small.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/small/wp_radius_small.py:55)

- Accumulates: row write cursor.
- Components: 1 offset integer.
- Combines: ordered append into allocated row.
- Finalizes: identity.
- Stores: i,j per accepted edge.
- Loads: positions/supports.
- Primary pattern and complexity: construction; medium.

### compact hash count

[src/warpSPHCore/radiusSearch/compactHash/wp_countNeighbors.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/wp_countNeighbors.py:16)

- Accumulates: accepted neighbor count.
- Components: 1 integer.
- Combines: integer sum.
- Finalizes: identity.
- Stores: count per query.
- Loads: positions/supports, cell/hash tables; periodic deduplication.
- Primary pattern and complexity: construction; medium.

### compact hash collect

[src/warpSPHCore/radiusSearch/compactHash/wp_collectNeighbors.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/wp_collectNeighbors.py:22)

- Accumulates: row write cursor.
- Components: 1 offset integer.
- Combines: ordered append into allocated row.
- Finalizes: identity.
- Stores: reference IDs and optional query IDs per accepted edge.
- Loads: positions/supports, sorted IDs, cell/hash tables; periodic deduplication.
- Primary pattern and complexity: construction; high.

### CSR row expansion

[src/warpSPHCore/radiusSearch/compactHash/wp_collectNeighbors.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/wp_collectNeighbors.py:239)

- Accumulates: strided write cursor.
- Components: 1 offset integer.
- Combines: disjoint row fill.
- Finalizes: identity.
- Stores: one query ID per edge.
- Loads: counts and offsets.
- Primary pattern and complexity: construction; low.

### Verlet count

[src/warpSPHCore/radiusSearch/verlet/wp_countVerlet.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/verlet/wp_countVerlet.py:15)

- Accumulates: accepted count.
- Components: 1 integer.
- Combines: integer sum.
- Finalizes: identity.
- Stores: count.
- Loads: candidate IDs and current geometry/support.
- Primary pattern and complexity: construction; low.

### Verlet collect

[src/warpSPHCore/radiusSearch/verlet/wp_updateVerlet.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/verlet/wp_updateVerlet.py:14)

- Accumulates: accepted count/cursor.
- Components: 1 integer.
- Combines: ordered append.
- Finalizes: identity.
- Stores: accepted i,j edges.
- Loads: candidate IDs and current geometry/support.
- Primary pattern and complexity: construction; medium.

### Verlet validity

[src/warpSPHCore/radiusSearch/verlet/wp_validity.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/verlet/wp_validity.py:30)

- Accumulates: movement max, support-change max, support min for query/reference sets.
- Components: 6 global scalar reduction slots.
- Combines: atomic min over signed maxima/minima in current Warp.
- Finalizes: combine bounds and compare with buffer.
- Stores: four diagnostics and one rebuild flag.
- Loads: current/prior positions and supports.
- Primary pattern and complexity: global; medium.
- Detail: A real global safety condition for neighbor-cache validity; separate from per-query SPH reductions.

## Frontend operations

### grad h omega

[src/warpSPH/modules/adaptiveSupport/wp_omega.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/adaptiveSupport/wp_omega.py:25)

- Accumulates: one scalar/vector/tensor sum.
- Components: 1.
- Combines: sum.
- Finalizes: 1 - sum.
- Stores: one query field.
- Loads: mass, density, support and grad-h kernel derivative.
- Primary pattern and complexity: finalize; low.

### delta density diffusion

[src/warpSPH/modules/deltaSPH/wp_densityDelta.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/deltaSPH/wp_densityDelta.py:110)

- Accumulates: one scalar/vector/tensor sum.
- Components: 1.
- Combines: sum.
- Finalizes: identity.
- Stores: one query field.
- Loads: rho or selected field, density gradients, support/corrections, body force.
- Primary pattern and complexity: sum; medium.

### delta viscosity

[src/warpSPH/modules/deltaSPH/wp_viscosityDelta.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/deltaSPH/wp_viscosityDelta.py:58)

- Accumulates: one scalar/vector/tensor sum.
- Components: D.
- Combines: sum.
- Finalizes: identity.
- Stores: one query field.
- Loads: velocity, mass/density, supports/corrections.
- Primary pattern and complexity: sum; low.

### thermal conductivity

[src/warpSPH/modules/dissipation/wp_conductivity.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/wp_conductivity.py:28)

- Accumulates: one scalar/vector/tensor sum.
- Components: 1.
- Combines: sum.
- Finalizes: multiply by thermalConductivity.
- Stores: one query field.
- Loads: energy, velocity, sound speed, pressure, alpha and viscosity parameters.
- Primary pattern and complexity: finalize; low.

### viscosity diffusion

[src/warpSPH/modules/dissipation/wp_diffusion.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/wp_diffusion.py:28)

- Accumulates: one scalar/vector/tensor sum.
- Components: D.
- Combines: sum.
- Finalizes: identity.
- Stores: one query field.
- Loads: velocity, density, sound speed, alpha, pressure; optional velocity/state gradients and Balsara field.
- Primary pattern and complexity: sum; medium.

### thermal dissipation

[src/warpSPH/modules/dissipation/wp_dissipation.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/wp_dissipation.py:29)

- Accumulates: one scalar/vector/tensor sum.
- Components: 1.
- Combines: sum.
- Finalizes: identity.
- Stores: one query field.
- Loads: energy, velocity, sound speed, alpha, pressure; optional reconstructed gradients.
- Primary pattern and complexity: sum; medium.

### wall moment

[src/warpSPH/modules/incompressible/wp_wallMoment.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/incompressible/wp_wallMoment.py:45)

- Accumulates: one scalar/vector/tensor sum.
- Components: D.
- Combines: sum.
- Finalizes: identity.
- Stores: one query field.
- Loads: rho, volume, position and kernel.
- Primary pattern and complexity: sum; low.

### surface aware pressure

[src/warpSPH/modules/pressure/wp_surfaceAware.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/pressure/wp_surfaceAware.py:34)

- Accumulates: one scalar/vector/tensor sum.
- Components: D.
- Combines: sum.
- Finalizes: -sum / query density.
- Stores: one query field.
- Loads: pressure, surface mask, volume, geometry/corrections.
- Primary pattern and complexity: finalize; low.

### raw delta shift

[src/warpSPH/modules/shifting/wp_deltaShiftRaw.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shifting/wp_deltaShiftRaw.py:22)

- Accumulates: one scalar/vector/tensor sum.
- Components: D.
- Combines: sum.
- Finalizes: identity.
- Stores: one query field.
- Loads: mass/density, geometry, kernel value/gradient, query parameters.
- Primary pattern and complexity: sum; low.

### shock moment M

[src/warpSPH/modules/shockCapturing/wp_computeM.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shockCapturing/wp_computeM.py:29)

- Accumulates: one scalar/vector/tensor sum.
- Components: D*D.
- Combines: sum.
- Finalizes: identity.
- Stores: one query field.
- Loads: volume, displacement, kernel gradient/corrections.
- Primary pattern and complexity: sum; low.

### Barecasco cover

[src/warpSPH/modules/surfaceDetection/wp_barecascoCover.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_barecascoCover.py:21)

- Accumulates: one scalar/vector/tensor sum.
- Components: D.
- Combines: sum.
- Finalizes: identity; caller may normalize.
- Stores: one query field.
- Loads: positions, support and kind.
- Primary pattern and complexity: sum; low.

### Barecasco cone count

[src/warpSPH/modules/surfaceDetection/wp_barecascoCone.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_barecascoCone.py:21)

- Accumulates: one scalar/vector/tensor sum.
- Components: 1.
- Combines: sum.
- Finalizes: identity.
- Stores: one query field.
- Loads: positions, supports and query cover axis.
- Primary pattern and complexity: sum; low.

### surface dilation sum

[src/warpSPH/modules/surfaceDetection/wp_dilate.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_dilate.py:26)

- Accumulates: one scalar/vector/tensor sum.
- Components: 1.
- Combines: sum.
- Finalizes: identity; thresholding is a caller operation.
- Stores: one query field.
- Loads: surface mask, positions/support/kind.
- Primary pattern and complexity: sum; low.

### Maronne count

[src/warpSPH/modules/surfaceDetection/wp_maronne.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_maronne.py:77)

- Accumulates: one scalar/vector/tensor sum.
- Components: 1.
- Combines: sum.
- Finalizes: identity.
- Stores: one query field.
- Loads: normals, positions/support and kind.
- Primary pattern and complexity: sum; low.

### frontend neighbor count

[src/warpSPH/modules/util/wp_numNeighbors.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/util/wp_numNeighbors.py:29)

- Accumulates: one scalar/vector/tensor sum.
- Components: 1 integer.
- Combines: sum.
- Finalizes: identity.
- Stores: one query field.
- Loads: positions, support, kind.
- Primary pattern and complexity: sum; low.

### frontend neighbor field sum

[src/warpSPH/modules/util/wp_sum.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/util/wp_sum.py:27)

- Accumulates: one scalar/vector/tensor sum.
- Components: field component count.
- Combines: sum.
- Finalizes: identity.
- Stores: one query field.
- Loads: field, position, support, kind.
- Primary pattern and complexity: sum; low.

### sample delta shift

[src/warpSPH/sample/wp_deltaShift.py](/home/lu26029/dev/warpSPH/src/warpSPH/sample/wp_deltaShift.py:35)

- Accumulates: one scalar/vector/tensor sum.
- Components: D.
- Combines: sum.
- Finalizes: identity.
- Stores: one query field.
- Loads: mass/density, geometry/corrections and shift parameters.
- Primary pattern and complexity: sum; low.

### material support moments

[src/warpSPH/modules/adaptiveSupport/wp_materialPsi.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/adaptiveSupport/wp_materialPsi.py:21)

- Accumulates: kernel sum and gradient-magnitude sum.
- Components: 2.
- Combines: sum.
- Finalizes: separate dimension roots.
- Stores: two support statistics.
- Loads: material IDs, mass/rho, geometry/kernel; optional tabulated gradient.
- Primary pattern and complexity: finalize; medium.

### support psi moments

[src/warpSPH/modules/adaptiveSupport/wp_psi0.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/adaptiveSupport/wp_psi0.py:24)

- Accumulates: kernel sum and gradient-magnitude sum.
- Components: 2.
- Combines: sum.
- Finalizes: separate dimension roots after all cells/lanes.
- Stores: psi0 and psi0h.
- Loads: geometry/kernel; relevant optional corrections.
- Primary pattern and complexity: finalize; low.

### support lookup construction

[src/warpSPH/modules/adaptiveSupport/wp_psi.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/adaptiveSupport/wp_psi.py:22)

- Accumulates: lattice kernel sum, derivative sum, count; separate shell-integral sum.
- Components: 3 floating + 1 integer, excluding induction variables.
- Combines: sum within distinct lattice/shell loops.
- Finalizes: rescaling, absolute value and dimension roots.
- Stores: n_h, psi, psiH, N_H, psiHLattice.
- Loads: synthetic lattice and kernel functions.
- Primary pattern and complexity: construction; medium.
- Detail: No simulation adjacency. Count is computed but N_H stores analytic support volume / particle volume.

### JST switch

[src/warpSPH/modules/artificialCompressible/wp_jstSwitch.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/wp_jstSwitch.py:47)

- Accumulates: numerator and denominator.
- Components: 2.
- Combines: sum.
- Finalizes: Python/Torch divides by clamped denominator.
- Stores: Warp stores two raw sums; public wrapper returns one switch.
- Loads: pressure, volume, geometry/kernel.
- Primary pattern and complexity: finalize; low.
- Detail: Native finalize can remove the extra Torch division/clamp pass while preserving denominator semantics.

### artificial compressible residual

[src/warpSPH/modules/artificialCompressible/wp_residual.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/wp_residual.py:18)

- Accumulates: divergence, pressure gradient, viscosity, gradient-divergence.
- Components: 1 + 3*D (3D: 10).
- Combines: sum.
- Finalizes: pressure gradient multiplied by -1/rho_i.
- Stores: one scalar and three vectors.
- Loads: velocity, pressure, mass/rho, fluid divergence and corrections.
- Primary pattern and complexity: finalize; medium.

### IISPH contact pressure diagonal

[src/warpSPH/modules/iisph/wp_contactPressureDiagonal.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_contactPressureDiagonal.py:18)

- Accumulates: one sum.
- Components: 1.
- Combines: sum.
- Finalizes: subtract query contact term.
- Stores: one query scalar/vector.
- Loads: rest volume, wall gradient, contact point, denominator.
- Primary pattern and complexity: finalize; low.

### IISPH density rate

[src/warpSPH/modules/iisph/wp_densityRate.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_densityRate.py:18)

- Accumulates: one sum.
- Components: 1.
- Combines: sum.
- Finalizes: add dot(query field, query wallGradient).
- Stores: one query scalar/vector.
- Loads: rest volume, field, kernel gradient.
- Primary pattern and complexity: finalize; low.

### IISPH gradient sum

[src/warpSPH/modules/iisph/wp_gradientSum.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_gradientSum.py:18)

- Accumulates: one sum.
- Components: D.
- Combines: sum.
- Finalizes: add query wallGradient.
- Stores: one query scalar/vector.
- Loads: rest volume, kernel gradient.
- Primary pattern and complexity: finalize; low.

### IISPH pressure acceleration

[src/warpSPH/modules/iisph/wp_pressureAcceleration.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_pressureAcceleration.py:20)

- Accumulates: one sum.
- Components: D.
- Combines: sum.
- Finalizes: add query pressure and contact-wall terms.
- Stores: one query scalar/vector.
- Loads: rest volume, density ratio, normalized pressure, query contact state.
- Primary pattern and complexity: finalize; low.

### IISPH pressure diagonal

[src/warpSPH/modules/iisph/wp_pressureDiagonal.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_pressureDiagonal.py:18)

- Accumulates: one sum.
- Components: 1.
- Combines: sum.
- Finalizes: combine with query gradientSum squared and density ratio.
- Stores: one query scalar/vector.
- Loads: rest volume, kernel gradient.
- Primary pattern and complexity: finalize; low.

### IISPH sampled gravity

[src/warpSPH/modules/iisph/wp_sampledGravity.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_sampledGravity.py:18)

- Accumulates: one sum.
- Components: D.
- Combines: sum.
- Finalizes: identity.
- Stores: one query scalar/vector.
- Loads: rest volume, contact point, query gravity and geometry.
- Primary pattern and complexity: sum; low.

### IISPH wall gradient

[src/warpSPH/modules/iisph/wp_wallGradient.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_wallGradient.py:18)

- Accumulates: one sum.
- Components: D.
- Combines: sum.
- Finalizes: identity.
- Stores: one query scalar/vector.
- Loads: rest volume and kernel gradient.
- Primary pattern and complexity: sum; low.

### incompressible alpha

[src/warpSPH/modules/incompressible/wp_alpha.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/incompressible/wp_alpha.py:49)

- Accumulates: gradient vector sum A and scalar squared-gradient sum B.
- Components: D + 1.
- Combines: sum before intended neighborhood finalization.
- Finalizes: area_i/m_i * dot(A,A) + area_i*B.
- Stores: one alpha.
- Loads: reference apparent area, mass, kind, geometry and gradient corrections.
- Primary pattern and complexity: finalize; medium.
- Detail: CURRENT GRID SEMANTICS: adjacency applies this nonlinear expression separately per cell and sums it. CSR applies it once. Preserve or fix explicitly; do not silently change on port.

### Liu matrices

[src/warpSPH/modules/liu/wp_mat.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/liu/wp_mat.py:27)

- Accumulates: Shepard sum, augmented vector, augmented matrix, neighbor count.
- Components: 1 + (D+1) + (D+1)**2 floating + 1 integer (3D: 21 + 1).
- Combines: sum.
- Finalizes: store Shepard from matrix[0,0], not the separately returned Shepard sum.
- Stores: Shepard scalar, augmented vector/matrix, count.
- Loads: reference quantity, query position, volume, kernel value/gradient.
- Primary pattern and complexity: sum; medium.

### mDBC no penetration shift

[src/warpSPH/modules/mdbc/wp_nopenshift.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/mdbc/wp_nopenshift.py:210)

- Accumulates: shift vector and per-axis integer contributor counts.
- Components: D floating + D integers.
- Combines: componentwise sum.
- Finalizes: divide each shift component by its count + epsilon.
- Stores: average shift vector and counts.
- Loads: ghost offsets defining boundary normals, velocity, mass/rho and geometry/corrections.
- Primary pattern and complexity: finalize; medium.

### characteristic shift speed

[src/warpSPH/modules/shifting/wp_michelUChar.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shifting/wp_michelUChar.py:61)

- Accumulates: found flag, best value, winning neighbor index.
- Components: 1 floating + 1 integer + 1 bool.
- Combines: argmax with explicit tie/empty policy.
- Finalizes: reload winning neighbor and evaluate/store selected quantity; preserve empty/self behavior.
- Stores: one speed.
- Loads: velocity and geometry.
- Primary pattern and complexity: select; medium.
- Detail: Current strict comparisons retain the first winning entry in traversal order. Selection state is small; tie and AD policy require explicit treatment.

### signal speed

[src/warpSPH/modules/shockCapturing/wp_vsig.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shockCapturing/wp_vsig.py:68)

- Accumulates: found flag, best value, winning neighbor index.
- Components: 1 floating + 1 integer + 1 bool.
- Combines: argmax with explicit tie/empty policy.
- Finalizes: reload winning neighbor and evaluate/store selected quantity; preserve empty/self behavior.
- Stores: one speed.
- Loads: velocity, optional sound speed and geometry.
- Primary pattern and complexity: select; medium.
- Detail: Current strict comparisons retain the first winning entry in traversal order. Selection state is small; tie and AD policy require explicit treatment.

### nearest surface normal

[src/warpSPH/modules/surfaceDetection/wp_nearestSurfaceNormal.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_nearestSurfaceNormal.py:58)

- Accumulates: found flag, best value, winning neighbor index.
- Components: 1 floating + 1 integer + 1 bool.
- Combines: argmin with explicit tie/empty policy.
- Finalizes: reload winning neighbor and evaluate/store selected quantity; preserve empty/self behavior.
- Stores: one distance and one normal.
- Loads: surface mask, normals and geometry.
- Primary pattern and complexity: select; medium.
- Detail: Current strict comparisons retain the first winning entry in traversal order. Selection state is small; tie and AD policy require explicit treatment.

### Read Hayfield signal speed

[src/warpSPH/modules/shockCapturing/ReadHayfield2012/kernels.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shockCapturing/ReadHayfield2012/kernels.py:58)

- Accumulates: found flag, best value, winning neighbor index.
- Components: 1 floating + 1 integer + 1 bool.
- Combines: argmax with explicit tie/empty policy.
- Finalizes: reload winning neighbor and evaluate/store selected quantity; preserve empty/self behavior.
- Stores: one speed.
- Loads: velocity, sound speed and geometry.
- Primary pattern and complexity: select; medium.
- Detail: Current strict comparisons retain the first winning entry in traversal order. Selection state is small; tie and AD policy require explicit treatment.

### Barecasco cover pass

[src/warpSPH/modules/surfaceDetection/wp_barecasco.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_barecasco.py:51)

- Accumulates: cover vector.
- Components: D.
- Combines: sum.
- Finalizes: normalize complete cover vector.
- Stores: query normal and input to second traversal.
- Loads: positions and supports.
- Primary pattern and complexity: multipass; medium.

### Barecasco cone pass

[src/warpSPH/modules/surfaceDetection/wp_barecasco.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_barecasco.py:203)

- Accumulates: cone count.
- Components: 1.
- Combines: sum.
- Finalizes: identity.
- Stores: surface detection count.
- Loads: positions/supports and finalized cover axis.
- Primary pattern and complexity: multipass; medium.
- Detail: Second full traversal depends on first-pass finalization; cannot obtain both by one ordinary pair reduction.

### implicit shift pair terms

[src/warpSPH/modules/shifting/wp_implicitShifting.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shifting/wp_implicitShifting.py:27)

- Accumulates: none across neighbors.
- Components: 0.
- Combines: none.
- Finalizes: evaluate W, gradient and Hessian.
- Stores: 1 + D + D*D per explicit edge.
- Loads: explicit i,j geometry/support.
- Primary pattern and complexity: edge; high.
- Detail: Downstream solver consumes edge tensors. Matrix-free recomputation is a separate integration choice.

### per side pressure

[src/warpSPH/modules/pressure/wp_perSidePressure.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/pressure/wp_perSidePressure.py:33)

- Accumulates: acceleration and unscaled work.
- Components: D + 1.
- Combines: sum.
- Finalizes: multiply completed work by query f_i.
- Stores: acceleration and dudt.
- Loads: pressure, velocity, mass/rho, support and two kernel gradients.
- Primary pattern and complexity: finalize; low.

### Godunov rates

[src/warpSPH/modules/godunov/wp_gsph.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/godunov/wp_gsph.py:92)

- Accumulates: acceleration and work.
- Components: D + 1.
- Combines: sum.
- Finalizes: identity.
- Stores: acceleration and dudt.
- Loads: primitive states, gradients, sound speed, limited Riemann states, geometry.
- Primary pattern and complexity: sum; high.

### Gaussian density/gradient

[src/warpSPH/modules/godunov/wp_inutsuka.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/godunov/wp_inutsuka.py:57)

- Accumulates: density and density gradient.
- Components: 1 + D.
- Combines: sum.
- Finalizes: identity.
- Stores: density and gradient.
- Loads: mass, support and geometry.
- Primary pattern and complexity: sum; medium.

### Inutsuka rates

[src/warpSPH/modules/godunov/wp_inutsuka.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/godunov/wp_inutsuka.py:202)

- Accumulates: acceleration and work.
- Components: D + 1.
- Combines: sum.
- Finalizes: identity.
- Stores: acceleration and dudt.
- Loads: primitive states, gradients, reconstruction, volume integral, Riemann data.
- Primary pattern and complexity: sum; high.

### compressible acceleration

[src/warpSPH/modules/compSPH/accel.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/accel.py:31)

- Accumulates: acceleration vector and optional heating scalar.
- Components: D + 1.
- Combines: sum for query outputs; independent edge writes when retained.
- Finalizes: identity.
- Stores: query acceleration/heating; optionally pressure and viscosity vectors per edge.
- Loads: velocity, energy, pressure, sound speed, alpha, correction data; CRK also reference coefficients and velocity gradients.
- Primary pattern and complexity: edge; high.
- Detail: retainPairs gates edge arrays. Pair formula is already factored and shared with recomputing work path; O(N) force-only modes exist.

### compressible thermal rate

[src/warpSPH/modules/compSPH/dudt.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/dudt.py:28)

- Accumulates: thermal rate.
- Components: 1.
- Combines: sum.
- Finalizes: identity.
- Stores: dudt.
- Loads: velocity, energy, pressure, sound speed, alpha and corrections.
- Primary pattern and complexity: sum; high.

### compressible stage work

[src/warpSPH/modules/compSPH/work.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/work.py:32)

- Accumulates: energy rate in reduce mode; otherwise no query sum.
- Components: 1 floating or 0.
- Combines: sum for rate; write/add by stable edge slot in retained mode.
- Finalizes: identity.
- Stores: rate per query OR weighted vector per edge.
- Loads: recomputed force pair, stage half velocities, final half velocity, material/entropy data.
- Primary pattern and complexity: multipass; high.
- Detail: Existing Warp code already recomputes pair force after stage acceleration. It has both rate-only and retained-work modes; all traversal is currently CSR.

### CRK acceleration

[src/warpSPH/modules/crk/accel.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/accel.py:32)

- Accumulates: acceleration vector and optional heating scalar.
- Components: D + 1.
- Combines: sum for query outputs; independent edge writes when retained.
- Finalizes: identity.
- Stores: query acceleration/heating; optionally pressure and viscosity vectors per edge.
- Loads: velocity, energy, pressure, sound speed, alpha, correction data; CRK also reference coefficients and velocity gradients.
- Primary pattern and complexity: edge; high.
- Detail: retainPairs gates edge arrays. Pair formula is already factored and shared with recomputing work path; O(N) force-only modes exist.

### CRK thermal rate

[src/warpSPH/modules/crk/dudt.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/dudt.py:36)

- Accumulates: thermal rate.
- Components: 1.
- Combines: sum.
- Finalizes: identity.
- Stores: dudt.
- Loads: velocity, energy, pressure, sound speed, alpha and corrections.
- Primary pattern and complexity: sum; high.

### CRK stage work

[src/warpSPH/modules/crk/work.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/work.py:19)

- Accumulates: energy rate in reduce mode; otherwise no query sum.
- Components: 1 floating or 0.
- Combines: sum for rate; write/add by stable edge slot in retained mode.
- Finalizes: identity.
- Stores: rate per query OR weighted vector per edge.
- Loads: recomputed force pair, stage half velocities, final half velocity, material/entropy data.
- Primary pattern and complexity: multipass; high.
- Detail: Existing Warp code already recomputes pair force after stage acceleration. It has both rate-only and retained-work modes; all traversal is currently CSR.

### compressible balance fractions

[src/warpSPH/modules/compSPH/balance.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/balance.py:34)

- Accumulates: none across neighbors.
- Components: 0.
- Combines: none.
- Finalizes: entropy/work partition with clamp and exceptional values.
- Stores: one fraction f_ij per edge.
- Loads: stored pressure/viscosity accelerations and query/reference stage state.
- Primary pattern and complexity: edge; high.

### retained pair work reduction

[src/warpSPH/modules/compSPH/workReduction.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/workReduction.py:23)

- Accumulates: one rate sum.
- Components: 1.
- Combines: sum.
- Finalizes: optional entropy partition per interaction.
- Stores: one query rate.
- Loads: stored work at edge slot, velocity difference and entropy state.
- Primary pattern and complexity: sum; high.

### material CRK density

[src/warpSPH/modules/crk/materialDensity.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/materialDensity.py:28)

- Accumulates: numerator and denominator.
- Components: 2.
- Combines: sum.
- Finalizes: guarded ratio.
- Stores: density.
- Loads: mass, volume, material IDs, geometry/kernel.
- Primary pattern and complexity: finalize; low.

### CRK coefficient solve

[src/warpSPH/modules/crk/terms.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/terms.py:43)

- Accumulates: none across neighbors; local matrix contractions.
- Components: 0 neighborhood components.
- Combines: none.
- Finalizes: inverse moment matrix, determinant limiter, A/B and derivatives.
- Stores: A scalar, B vector, gradA vector, gradB matrix (1 + 2*D + D*D).
- Loads: query moment tensors only.
- Primary pattern and complexity: pointwise; medium.
- Detail: Separate stage over precomputed moments; not automatically fused if neighboring operations need coefficients from both sides.

### pressure entropy sums

[src/warpSPH/modules/pesph/operators.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/pesph/operators.py:47)

- Accumulates: number density, weighted sum, two support derivatives, mass density.
- Components: 5.
- Combines: sum.
- Finalizes: identity.
- Stores: five query scalars.
- Loads: weights, mass, geometry and value/grad-h kernel.
- Primary pattern and complexity: sum; medium.

### pressure entropy force

[src/warpSPH/modules/pesph/operators.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/pesph/operators.py:89)

- Accumulates: acceleration and thermal work.
- Components: D + 1.
- Combines: sum.
- Finalizes: identity.
- Stores: acceleration and work.
- Loads: velocity, weights, pressure-entropy state and two kernel gradients.
- Primary pattern and complexity: sum; medium.

### frontend MFM flux

[src/warpSPH/modules/meshless/flux.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/meshless/flux.py:25)

- Accumulates: mass, momentum and total-energy rates.
- Components: D + 2.
- Combines: sum.
- Finalizes: identity.
- Stores: dm, dp, de.
- Loads: primitive states, gradients/limiters, volume, inverse moments, closure and material/Riemann data.
- Primary pattern and complexity: sum; high.

### frontend MFM limiter

[src/warpSPH/modules/reconstruction/meshless.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/meshless.py:66)

- Accumulates: neighbor and reconstructed min/max for rho, P and velocity.
- Components: 8 + 4*D (3D: 20).
- Combines: mixed componentwise min/max with query-value identities.
- Finalizes: limiter ratios and clamps.
- Stores: rho, pressure and velocity limiter factors.
- Loads: primitive states, geometry; query gradients/beta.
- Primary pattern and complexity: extrema; medium.

### Read Hayfield entropy

[src/warpSPH/modules/shockCapturing/ReadHayfield2012/kernels.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shockCapturing/ReadHayfield2012/kernels.py:222)

- Accumulates: entropy diffusion.
- Components: 1.
- Combines: sum.
- Finalizes: identity.
- Stores: one entropy rate.
- Loads: velocity, sound speed, alpha, pressure, entropy, mass/rho and geometry.
- Primary pattern and complexity: sum; medium.

### Wadsley neighbor mean

[src/warpSPH/modules/shockCapturing/Wadsley2017/kernels.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shockCapturing/Wadsley2017/kernels.py:78)

- Accumulates: weighted numerator and weight sum.
- Components: 2.
- Combines: sum.
- Finalizes: guarded ratio and clamp [-1,1].
- Stores: one R value.
- Loads: ratio, mass and geometry/support.
- Primary pattern and complexity: finalize; low.

### Wadsley point fields

[src/warpSPH/modules/shockCapturing/Wadsley2017/kernels.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shockCapturing/Wadsley2017/kernels.py:29)

- Accumulates: none across neighbors; local tensor contractions.
- Components: 0 neighborhood components.
- Combines: none.
- Finalizes: gradient-derived divergence, tensor norm, ratio and pressure normal.
- Stores: D, Tnorm, ratio and normal.
- Loads: query velocity/pressure gradients.
- Primary pattern and complexity: pointwise; low.

### dynGPUSPH closures

[src/warpSPH/modules/artificialCompressible/dyngpusphOperators.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/dyngpusphOperators.py:35)

- Accumulates: weight, pressure, lambda, divergence; mean velocity/position, laplacian, normal; surface flag.
- Components: 4 + 4*D floating + 1 bool (implemented in 2D: 12 + bool).
- Combines: sum + OR.
- Finalizes: weighted means, wall/pressure closure, normal handling.
- Stores: pressure, normal, average velocity, divergence, laplacian.
- Loads: pressure, velocity, lambda, body force, mass/rho and geometry.
- Primary pattern and complexity: mixed; medium.

### dynGPUSPH moments

[src/warpSPH/modules/artificialCompressible/dyngpusphOperators.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/dyngpusphOperators.py:122)

- Accumulates: moment matrix, normal, pressure gradient, pressure laplacian.
- Components: D*D + 2*D + 1 (implemented in 2D: 9).
- Combines: sum.
- Finalizes: eigenvalue-derived lambda, normal normalization.
- Stores: lambda, normal, gradient, laplacian.
- Loads: pressure, mass/rho and geometry.
- Primary pattern and complexity: finalize; medium.

### dynGPUSPH residual

[src/warpSPH/modules/artificialCompressible/dyngpusphOperators.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/dyngpusphOperators.py:196)

- Accumulates: three scalar sums, five vector sums, wall-distance minimum and edge-vicinity state.
- Components: 4 + 5*D floating + 1 categorical scalar (2D: 14 + 1).
- Combines: sum + min + categorical OR.
- Finalizes: scale or freeze switch.
- Stores: divergence, pressure/viscous/acoustic/second/fourth vectors, wall distance, diffusion, switch, edge.
- Loads: velocity and laplacian, pressure and derivatives, wall states, edge flags and geometry.
- Primary pattern and complexity: mixed; high.

### dynGPUSPH shift

[src/warpSPH/modules/artificialCompressible/dyngpusphShifting.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/dyngpusphShifting.py:44)

- Accumulates: eight scalars, six vectors, moment matrix, two counts and vicinity flag.
- Components: 8 + 6*D + D*D floating + 2 integers + bool (2D: 24 + 2 + bool).
- Combines: sum + min + max + OR with query-dependent identities.
- Finalizes: ratios, matrix inverse, normalizations, collision correction, clamps and pressure/velocity updates.
- Stores: shift, recorded shift, pressure, velocity, active flag, gradP, diagonal gradV, lambda.
- Loads: lambda, edge/normal, pressure, velocity, mass/rho, geometry and optional source-cell coordinates.
- Primary pattern and complexity: mixed; high.
- Detail: Specialized 2D kernel with two support ranges; keep its mixed reduction and domain predicates explicit.

### dynGPUSPH surface fields

[src/warpSPH/modules/artificialCompressible/dyngpusphShifting.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/dyngpusphShifting.py:239)

- Accumulates: moment matrix and normal.
- Components: D*D + D (implemented 2D: 6).
- Combines: sum.
- Finalizes: lambda from matrix, normalize normal.
- Stores: lambda and normal.
- Loads: geometry and volume.
- Primary pattern and complexity: finalize; medium.

### dynGPUSPH cone

[src/warpSPH/modules/artificialCompressible/dyngpusphShifting.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/dyngpusphShifting.py:294)

- Accumulates: whether a qualifying cone neighbor exists.
- Components: 1 bool.
- Combines: OR.
- Finalizes: combine predicate with query lambda/edge policy.
- Stores: edge classification.
- Loads: geometry and normal.
- Primary pattern and complexity: extrema; low.

### dynGPUSPH vicinity

[src/warpSPH/modules/artificialCompressible/dyngpusphShifting.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/dyngpusphShifting.py:343)

- Accumulates: whether an edge neighbor exists.
- Components: 1 bool.
- Combines: OR.
- Finalizes: preserve raw query label or mark vicinity.
- Stores: edge classification.
- Loads: raw edge flags, kind, position and support.
- Primary pattern and complexity: extrema; low.

## File coverage

All literal wp files and every other file with a decorated definition are listed below. The function inventory contains individual definition locations and exact return/update/store expressions. A file with zero decorated definitions contributes no additional Warp operation.

### Core

| File | Definitions | Family | Accumulation and outputs |
| --- | ---: | --- | --- |
| [coreOperations/_jvpCommon.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/_jvpCommon.py) | 1 | pair math helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Evaluate pair kernels, derivatives, corrected gradients or tangent weights; custom gradients accumulate local input adjoints. Stores: return scalar/vector/matrix or tangent tuple; custom-gradient input adjoints. |
| [coreOperations/native/wp_covariance.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_covariance.py) | 4 | operation | Operation records: covariance. |
| [coreOperations/native/wp_curl.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_curl.py) | 4 | operation | Operation records: curl. |
| [coreOperations/native/wp_density.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_density.py) | 4 | operation | Operation records: density. |
| [coreOperations/native/wp_divergence.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_divergence.py) | 4 | operation | Operation records: divergence. |
| [coreOperations/native/wp_gradient.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_gradient.py) | 4 | operation | Operation records: gradient. |
| [coreOperations/native/wp_interpolate.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_interpolate.py) | 4 | operation | Operation records: interpolate. |
| [coreOperations/native/wp_laplacian.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_laplacian.py) | 4 | operation | Operation records: laplacian. |
| [coreOperations/wp_covariance.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_covariance.py) | 0 | alias or scaffolding | 0 neighborhood accumulators Compatibility import alias to the corresponding coreOperations/native/wp_ implementation; no additional operation. Stores: none in decorated Warp code. |
| [coreOperations/wp_covarianceJVP.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_covarianceJVP.py) | 3 | operation | Operation records: covariance JVP. |
| [coreOperations/wp_curl.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_curl.py) | 0 | alias or scaffolding | 0 neighborhood accumulators Compatibility import alias to the corresponding coreOperations/native/wp_ implementation; no additional operation. Stores: none in decorated Warp code. |
| [coreOperations/wp_curlJVP.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_curlJVP.py) | 3 | operation | Operation records: curl JVP. |
| [coreOperations/wp_density.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_density.py) | 0 | alias or scaffolding | 0 neighborhood accumulators Compatibility import alias to the corresponding coreOperations/native/wp_ implementation; no additional operation. Stores: none in decorated Warp code. |
| [coreOperations/wp_densityHVP.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_densityHVP.py) | 1 | operation | Operation records: density Hessian edge terms. |
| [coreOperations/wp_densityJVP.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_densityJVP.py) | 3 | operation | Operation records: density JVP. |
| [coreOperations/wp_divergence.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_divergence.py) | 0 | alias or scaffolding | 0 neighborhood accumulators Compatibility import alias to the corresponding coreOperations/native/wp_ implementation; no additional operation. Stores: none in decorated Warp code. |
| [coreOperations/wp_divergenceJVP.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_divergenceJVP.py) | 3 | operation | Operation records: divergence JVP. |
| [coreOperations/wp_gradient.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_gradient.py) | 0 | alias or scaffolding | 0 neighborhood accumulators Compatibility import alias to the corresponding coreOperations/native/wp_ implementation; no additional operation. Stores: none in decorated Warp code. |
| [coreOperations/wp_gradientJVP.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_gradientJVP.py) | 3 | operation | Operation records: gradient JVP. |
| [coreOperations/wp_interpolate.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_interpolate.py) | 0 | alias or scaffolding | 0 neighborhood accumulators Compatibility import alias to the corresponding coreOperations/native/wp_ implementation; no additional operation. Stores: none in decorated Warp code. |
| [coreOperations/wp_interpolateJVP.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_interpolateJVP.py) | 3 | operation | Operation records: interpolate JVP. |
| [coreOperations/wp_laplacian.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_laplacian.py) | 0 | alias or scaffolding | 0 neighborhood accumulators Compatibility import alias to the corresponding coreOperations/native/wp_ implementation; no additional operation. Stores: none in decorated Warp code. |
| [coreOperations/wp_laplacianJVP.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_laplacianJVP.py) | 15 | operation | Operation records: Laplacian Brookshaw JVP, Laplacian Naive JVP, Laplacian Dot JVP, Laplacian Default JVP. |
| [crk/crk_density.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/crk_density.py) | 4 | operation | Operation records: CRK density. |
| [crk/crk_moments.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/crk_moments.py) | 4 | operation | Operation records: CRK moments. |
| [crk/crk_moments_jvp.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/crk_moments_jvp.py) | 3 | operation | Operation records: CRK moments JVP. |
| [crk/crk_volume.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/crk_volume.py) | 3 | operation | Operation records: CRK volume. |
| [crk/crk_volume_jvp.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/crk_volume_jvp.py) | 3 | operation | Operation records: CRK volume JVP. |
| [crk/kernel.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/kernel.py) | 5 | pair math helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Evaluate pair kernels, derivatives, corrected gradients or tangent weights; custom gradients accumulate local input adjoints. Stores: return scalar/vector/matrix or tangent tuple; custom-gradient input adjoints. |
| [kernels/derivative.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/derivative.py) | 2 | pair math helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Evaluate pair kernels, derivatives, corrected gradients or tangent weights; custom gradients accumulate local input adjoints. Stores: return scalar/vector/matrix or tangent tuple; custom-gradient input adjoints. |
| [kernels/eval_kernel.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/eval_kernel.py) | 7 | pair math helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Evaluate pair kernels, derivatives, corrected gradients or tangent weights; custom gradients accumulate local input adjoints. Stores: return scalar/vector/matrix or tangent tuple; custom-gradient input adjoints. |
| [kernels/gradH.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/gradH.py) | 4 | pair math helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Evaluate pair kernels, derivatives, corrected gradients or tangent weights; custom gradients accumulate local input adjoints. Stores: return scalar/vector/matrix or tangent tuple; custom-gradient input adjoints. |
| [kernels/gradient.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/gradient.py) | 11 | pair math helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Evaluate pair kernels, derivatives, corrected gradients or tangent weights; custom gradients accumulate local input adjoints. Stores: return scalar/vector/matrix or tangent tuple; custom-gradient input adjoints. |
| [kernels/hessian.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/hessian.py) | 2 | pair math helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Evaluate pair kernels, derivatives, corrected gradients or tangent weights; custom gradients accumulate local input adjoints. Stores: return scalar/vector/matrix or tangent tuple; custom-gradient input adjoints. |
| [kernels/kernel.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernel.py) | 3 | pair math helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Evaluate pair kernels, derivatives, corrected gradients or tangent weights; custom gradients accumulate local input adjoints. Stores: return scalar/vector/matrix or tangent tuple; custom-gradient input adjoints. |
| [kernels/kernelFunctions/B7.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/B7.py) | 7 | radial kernel helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Seven scalar evaluators/constants: value, first/second/third derivative, dimensional normalization, scale and packing ratio. Stores: return scalar; no global particle output. |
| [kernels/kernelFunctions/B8.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/B8.py) | 7 | radial kernel helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Seven scalar evaluators/constants: value, first/second/third derivative, dimensional normalization, scale and packing ratio. Stores: return scalar; no global particle output. |
| [kernels/kernelFunctions/adhesionKernel.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/adhesionKernel.py) | 7 | radial kernel helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Seven scalar evaluators/constants: value, first/second/third derivative, dimensional normalization, scale and packing ratio. Stores: return scalar; no global particle output. |
| [kernels/kernelFunctions/cohesionKernel.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/cohesionKernel.py) | 7 | radial kernel helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Seven scalar evaluators/constants: value, first/second/third derivative, dimensional normalization, scale and packing ratio. Stores: return scalar; no global particle output. |
| [kernels/kernelFunctions/cubicSpline.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/cubicSpline.py) | 7 | radial kernel helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Seven scalar evaluators/constants: value, first/second/third derivative, dimensional normalization, scale and packing ratio. Stores: return scalar; no global particle output. |
| [kernels/kernelFunctions/gaussian.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/gaussian.py) | 7 | radial kernel helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Seven scalar evaluators/constants: value, first/second/third derivative, dimensional normalization, scale and packing ratio. Stores: return scalar; no global particle output. |
| [kernels/kernelFunctions/hoct4.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/hoct4.py) | 7 | radial kernel helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Seven scalar evaluators/constants: value, first/second/third derivative, dimensional normalization, scale and packing ratio. Stores: return scalar; no global particle output. |
| [kernels/kernelFunctions/poly6.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/poly6.py) | 7 | radial kernel helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Seven scalar evaluators/constants: value, first/second/third derivative, dimensional normalization, scale and packing ratio. Stores: return scalar; no global particle output. |
| [kernels/kernelFunctions/quarticSpline.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/quarticSpline.py) | 7 | radial kernel helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Seven scalar evaluators/constants: value, first/second/third derivative, dimensional normalization, scale and packing ratio. Stores: return scalar; no global particle output. |
| [kernels/kernelFunctions/quinticSpline.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/quinticSpline.py) | 7 | radial kernel helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Seven scalar evaluators/constants: value, first/second/third derivative, dimensional normalization, scale and packing ratio. Stores: return scalar; no global particle output. |
| [kernels/kernelFunctions/spiky.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/spiky.py) | 7 | radial kernel helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Seven scalar evaluators/constants: value, first/second/third derivative, dimensional normalization, scale and packing ratio. Stores: return scalar; no global particle output. |
| [kernels/kernelFunctions/viscosityKernel.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/viscosityKernel.py) | 7 | radial kernel helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Seven scalar evaluators/constants: value, first/second/third derivative, dimensional normalization, scale and packing ratio. Stores: return scalar; no global particle output. |
| [kernels/kernelFunctions/wendland2.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/wendland2.py) | 7 | radial kernel helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Seven scalar evaluators/constants: value, first/second/third derivative, dimensional normalization, scale and packing ratio. Stores: return scalar; no global particle output. |
| [kernels/kernelFunctions/wendland4.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/wendland4.py) | 7 | radial kernel helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Seven scalar evaluators/constants: value, first/second/third derivative, dimensional normalization, scale and packing ratio. Stores: return scalar; no global particle output. |
| [kernels/kernelFunctions/wendland6.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/wendland6.py) | 7 | radial kernel helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Seven scalar evaluators/constants: value, first/second/third derivative, dimensional normalization, scale and packing ratio. Stores: return scalar; no global particle output. |
| [kernels/kernelJVP.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelJVP.py) | 6 | pair math helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Evaluate pair kernels, derivatives, corrected gradients or tangent weights; custom gradients accumulate local input adjoints. Stores: return scalar/vector/matrix or tangent tuple; custom-gradient input adjoints. |
| [kernels/laplacian.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/laplacian.py) | 4 | pair math helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Evaluate pair kernels, derivatives, corrected gradients or tangent weights; custom gradients accumulate local input adjoints. Stores: return scalar/vector/matrix or tangent tuple; custom-gradient input adjoints. |
| [kernels/properties.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/properties.py) | 5 | pair math helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Evaluate pair kernels, derivatives, corrected gradients or tangent weights; custom gradients accumulate local input adjoints. Stores: return scalar/vector/matrix or tangent tuple; custom-gradient input adjoints. |
| [linalg/krylov/bicgstabWorkspace.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/krylov/bicgstabWorkspace.py) | 12 | solver workspace | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Vector updates and scalar decisions using external dot/norm reductions; global failure/convergence flags use atomic_max. Stores: solution/residual/direction vectors, scalar state and optional history. |
| [linalg/krylov/cgWorkspace.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/krylov/cgWorkspace.py) | 6 | solver workspace | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Vector updates and scalar decisions using external dot/norm reductions; global failure/convergence flags use atomic_max. Stores: solution/residual/direction vectors, scalar state and optional history. |
| [linalg/pinv.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/pinv.py) | 5 | pointwise linear algebra | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Small matrix pseudoinverse/eigenvalue routines or application to a local RHS. Stores: per-query inverse/eigenvalues or solved local vectors. |
| [linalg/preconditioner.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/preconditioner.py) | 1 | pointwise linear algebra | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Apply a local block preconditioner. Stores: one vector component per output row. |
| [math/wp_cross.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_cross.py) | 3 | local math helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Fixed-dimensional arithmetic, tensor contraction, constructors and custom local adjoints; internal component sums are not neighbor reductions. Stores: return scalar/vector/matrix; local component assignments or input adjoints. |
| [math/wp_delta.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_delta.py) | 1 | local math helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Fixed-dimensional arithmetic, tensor contraction, constructors and custom local adjoints; internal component sums are not neighbor reductions. Stores: return scalar/vector/matrix; local component assignments or input adjoints. |
| [math/wp_dim.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_dim.py) | 12 | local math helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Fixed-dimensional arithmetic, tensor contraction, constructors and custom local adjoints; internal component sums are not neighbor reductions. Stores: return scalar/vector/matrix; local component assignments or input adjoints. |
| [math/wp_distance.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_distance.py) | 12 | local math helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Fixed-dimensional arithmetic, tensor contraction, constructors and custom local adjoints; internal component sums are not neighbor reductions. Stores: return scalar/vector/matrix; local component assignments or input adjoints. |
| [math/wp_divdot.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_divdot.py) | 3 | local math helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Fixed-dimensional arithmetic, tensor contraction, constructors and custom local adjoints; internal component sums are not neighbor reductions. Stores: return scalar/vector/matrix; local component assignments or input adjoints. |
| [math/wp_dot.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_dot.py) | 1 | local math helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Fixed-dimensional arithmetic, tensor contraction, constructors and custom local adjoints; internal component sums are not neighbor reductions. Stores: return scalar/vector/matrix; local component assignments or input adjoints. |
| [math/wp_eps.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_eps.py) | 4 | local math helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Fixed-dimensional arithmetic, tensor contraction, constructors and custom local adjoints; internal component sums are not neighbor reductions. Stores: return scalar/vector/matrix; local component assignments or input adjoints. |
| [math/wp_eye.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_eye.py) | 3 | local math helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Fixed-dimensional arithmetic, tensor contraction, constructors and custom local adjoints; internal component sums are not neighbor reductions. Stores: return scalar/vector/matrix; local component assignments or input adjoints. |
| [math/wp_laplaciandot.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_laplaciandot.py) | 5 | local math helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Fixed-dimensional arithmetic, tensor contraction, constructors and custom local adjoints; internal component sums are not neighbor reductions. Stores: return scalar/vector/matrix; local component assignments or input adjoints. |
| [math/wp_matmul.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_matmul.py) | 3 | local math helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Fixed-dimensional arithmetic, tensor contraction, constructors and custom local adjoints; internal component sums are not neighbor reductions. Stores: return scalar/vector/matrix; local component assignments or input adjoints. |
| [math/wp_norm.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_norm.py) | 3 | local math helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Fixed-dimensional arithmetic, tensor contraction, constructors and custom local adjoints; internal component sums are not neighbor reductions. Stores: return scalar/vector/matrix; local component assignments or input adjoints. |
| [math/wp_normalize.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_normalize.py) | 18 | local math helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Fixed-dimensional arithmetic, tensor contraction, constructors and custom local adjoints; internal component sums are not neighbor reductions. Stores: return scalar/vector/matrix; local component assignments or input adjoints. |
| [math/wp_outerTensorProduct.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_outerTensorProduct.py) | 3 | local math helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Fixed-dimensional arithmetic, tensor contraction, constructors and custom local adjoints; internal component sums are not neighbor reductions. Stores: return scalar/vector/matrix; local component assignments or input adjoints. |
| [math/wp_pow.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_pow.py) | 3 | local math helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Fixed-dimensional arithmetic, tensor contraction, constructors and custom local adjoints; internal component sums are not neighbor reductions. Stores: return scalar/vector/matrix; local component assignments or input adjoints. |
| [math/wp_sqrt.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_sqrt.py) | 2 | local math helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Fixed-dimensional arithmetic, tensor contraction, constructors and custom local adjoints; internal component sums are not neighbor reductions. Stores: return scalar/vector/matrix; local component assignments or input adjoints. |
| [math/wp_vec1.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_vec1.py) | 0 | alias or scaffolding | 0 neighborhood accumulators Vector/matrix type declarations only. Stores: none in decorated Warp code. |
| [math/wp_zero.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py) | 38 | local math helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Fixed-dimensional arithmetic, tensor contraction, constructors and custom local adjoints; internal component sums are not neighbor reductions. Stores: return scalar/vector/matrix; local component assignments or input adjoints. |
| [mfm/multigrid.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/multigrid.py) | 1 | operation | Operation records: multigrid segment sum. |
| [mfm/wp_closure.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_closure.py) | 1 | operation | Operation records: MFM closure CSR matvec. |
| [mfm/wp_mfm.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py) | 32 | operation | Operation records: MFM moments, MFM gradients, MFM limiter, MFM closure residual, MFM closure weights, MFM closure matvec, MFM flux, MFM coarse stencil, MFM timestep. |
| [pinv/wp_pinv1x1.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/pinv/wp_pinv1x1.py) | 2 | pointwise linear algebra | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Small matrix pseudoinverse/eigenvalue routines or application to a local RHS. Stores: per-query inverse/eigenvalues or solved local vectors. |
| [pinv/wp_pinv2x2.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/pinv/wp_pinv2x2.py) | 2 | pointwise linear algebra | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Small matrix pseudoinverse/eigenvalue routines or application to a local RHS. Stores: per-query inverse/eigenvalues or solved local vectors. |
| [pinv/wp_pinv3x3.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/pinv/wp_pinv3x3.py) | 0 | alias or scaffolding | 0 neighborhood accumulators Commented prototype only; no executable Warp definitions. Stores: none in decorated Warp code. |
| [polyfit/wp_polyfit.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/polyfit/wp_polyfit.py) | 10 | operation | Operation records: polynomial moments, polynomial RHS. |
| [polyfit/wp_polyfitJVP.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/polyfit/wp_polyfitJVP.py) | 7 | operation | Operation records: polynomial moments JVP, polynomial RHS JVP. |
| [radiusSearch/compactHash/indexing.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/indexing.py) | 3 | indexing and traversal helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Index/hash/Morton computation, periodic cell/range lookup, lane partition and laneSum. Stores: indices/hashes or returned range; laneSum reduces arbitrary scalar/vector/tensor values. |
| [radiusSearch/compactHash/morton.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/morton.py) | 3 | indexing and traversal helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Index/hash/Morton computation, periodic cell/range lookup, lane partition and laneSum. Stores: indices/hashes or returned range; laneSum reduces arbitrary scalar/vector/tensor values. |
| [radiusSearch/compactHash/wp_collectNeighbors.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/wp_collectNeighbors.py) | 2 | operation | Operation records: compact-hash collect, CSR row expansion. |
| [radiusSearch/compactHash/wp_countNeighbors.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/wp_countNeighbors.py) | 3 | operation | Operation records: compact-hash count. |
| [radiusSearch/compactHash/wp_hashCells.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/wp_hashCells.py) | 3 | indexing and traversal helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Index/hash/Morton computation, periodic cell/range lookup, lane partition and laneSum. Stores: indices/hashes or returned range; laneSum reduces arbitrary scalar/vector/tensor values. |
| [radiusSearch/compactHash/wp_index.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/wp_index.py) | 3 | indexing and traversal helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Index/hash/Morton computation, periodic cell/range lookup, lane partition and laneSum. Stores: indices/hashes or returned range; laneSum reduces arbitrary scalar/vector/tensor values. |
| [radiusSearch/grid_util.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/grid_util.py) | 8 | indexing and traversal helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Index/hash/Morton computation, periodic cell/range lookup, lane partition and laneSum. Stores: indices/hashes or returned range; laneSum reduces arbitrary scalar/vector/tensor values. |
| [radiusSearch/small/wp_radius_small.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/small/wp_radius_small.py) | 2 | operation | Operation records: direct radius count, direct radius collect. |
| [radiusSearch/verlet/wp_countVerlet.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/verlet/wp_countVerlet.py) | 2 | operation | Operation records: Verlet count. |
| [radiusSearch/verlet/wp_updateVerlet.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/verlet/wp_updateVerlet.py) | 2 | operation | Operation records: Verlet collect. |
| [radiusSearch/verlet/wp_validity.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/verlet/wp_validity.py) | 2 | operation | Operation records: Verlet validity. |
| [type_config.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/type_config.py) | 1 | scalar conversion helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Convert to configured scalar type. Stores: return scalar. |
| [util/directionality.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/directionality.py) | 3 | state and optional loader helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Particle/correction record loading, direction checks, support conversion and optional/default access. Stores: return record or value; no neighborhood output. |
| [util/stateUtil.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py) | 29 | state and optional loader helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Particle/correction record loading, direction checks, support conversion and optional/default access. Stores: return record or value; no neighborhood output. |
| [util/support.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/support.py) | 4 | state and optional loader helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Particle/correction record loading, direction checks, support conversion and optional/default access. Stores: return record or value; no neighborhood output. |
| [util/wp_util.py](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/wp_util.py) | 0 | alias or scaffolding | 0 neighborhood accumulators Python utility/scaffolding only; no executable Warp definition. Stores: none in decorated Warp code. |

### Frontend

| File | Definitions | Family | Accumulation and outputs |
| --- | ---: | --- | --- |
| [modules/adaptiveSupport/owenHermite.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/adaptiveSupport/owenHermite.py) | 2 | lookup helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Sample a scalar radial derivative table and interpolate a scalar gradient moment. Stores: lookup samples or returned scalar. |
| [modules/adaptiveSupport/wp_materialPsi.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/adaptiveSupport/wp_materialPsi.py) | 3 | operation | Operation records: material support moments. |
| [modules/adaptiveSupport/wp_omega.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/adaptiveSupport/wp_omega.py) | 3 | operation | Operation records: grad-h omega. |
| [modules/adaptiveSupport/wp_psi.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/adaptiveSupport/wp_psi.py) | 1 | operation | Operation records: support lookup construction. |
| [modules/adaptiveSupport/wp_psi0.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/adaptiveSupport/wp_psi0.py) | 6 | operation | Operation records: support psi moments. |
| [modules/artificialCompressible/dyngpusphOperators.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/dyngpusphOperators.py) | 3 | operation | Operation records: dynGPUSPH closures, dynGPUSPH moments, dynGPUSPH residual. |
| [modules/artificialCompressible/dyngpusphShifting.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/dyngpusphShifting.py) | 6 | operation | Operation records: dynGPUSPH shift, dynGPUSPH surface fields, dynGPUSPH cone, dynGPUSPH vicinity. |
| [modules/artificialCompressible/wp_jstSwitch.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/wp_jstSwitch.py) | 3 | operation | Operation records: JST switch. |
| [modules/artificialCompressible/wp_residual.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/wp_residual.py) | 3 | operation | Operation records: artificial-compressible residual. |
| [modules/compSPH/accel.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/accel.py) | 5 | operation | Operation records: compressible acceleration. |
| [modules/compSPH/balance.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/balance.py) | 5 | operation | Operation records: compressible balance fractions. |
| [modules/compSPH/dudt.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/dudt.py) | 4 | operation | Operation records: compressible thermal rate. |
| [modules/compSPH/work.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/work.py) | 1 | operation | Operation records: compressible stage work. |
| [modules/compSPH/workReduction.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/workReduction.py) | 3 | operation | Operation records: retained pair work reduction. |
| [modules/crk/accel.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/accel.py) | 5 | operation | Operation records: CRK acceleration. |
| [modules/crk/dudt.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/dudt.py) | 4 | operation | Operation records: CRK thermal rate. |
| [modules/crk/materialDensity.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/materialDensity.py) | 3 | operation | Operation records: material CRK density. |
| [modules/crk/partition.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/partition.py) | 1 | pair entropy helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Compute/clamp entropy-based pair work partition; no neighbor reduction. Stores: return one fraction. |
| [modules/crk/terms.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/terms.py) | 2 | operation | Operation records: CRK coefficient solve. |
| [modules/crk/work.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/work.py) | 1 | operation | Operation records: CRK stage work. |
| [modules/deltaSPH/updates.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/deltaSPH/updates.py) | 2 | pointwise state update | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. EOS/material lookup, conserved-to-primitive conversion or assembled/masked particle updates. Stores: per-particle thermodynamic fields and update vectors. |
| [modules/deltaSPH/wp_densityDelta.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/deltaSPH/wp_densityDelta.py) | 4 | operation | Operation records: delta density diffusion. |
| [modules/deltaSPH/wp_viscosityDelta.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/deltaSPH/wp_viscosityDelta.py) | 4 | operation | Operation records: delta viscosity. |
| [modules/dissipation/pi/coefficients.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/coefficients.py) | 1 | pair viscosity helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Construct a pair record, choose switched coefficients and evaluate a viscosity/conductivity term; no neighbor loop. Stores: return pair record, coefficient tuple or scalar. |
| [modules/dissipation/pi/dispatch.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/dispatch.py) | 3 | pair viscosity helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Construct a pair record, choose switched coefficients and evaluate a viscosity/conductivity term; no neighbor loop. Stores: return pair record, coefficient tuple or scalar. |
| [modules/dissipation/pi/oneSided.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/oneSided.py) | 1 | pair viscosity helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Construct a pair record, choose switched coefficients and evaluate a viscosity/conductivity term; no neighbor loop. Stores: return pair record, coefficient tuple or scalar. |
| [modules/dissipation/pi/pair.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/pair.py) | 2 | pair viscosity helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Construct a pair record, choose switched coefficients and evaluate a viscosity/conductivity term; no neighbor loop. Stores: return pair record, coefficient tuple or scalar. |
| [modules/dissipation/pi/terms.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/terms.py) | 20 | pair viscosity helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Construct a pair record, choose switched coefficients and evaluate a viscosity/conductivity term; no neighbor loop. Stores: return pair record, coefficient tuple or scalar. |
| [modules/dissipation/wp_conductivity.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/wp_conductivity.py) | 3 | operation | Operation records: thermal conductivity. |
| [modules/dissipation/wp_diffusion.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/wp_diffusion.py) | 3 | operation | Operation records: viscosity diffusion. |
| [modules/dissipation/wp_dissipation.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/wp_dissipation.py) | 3 | operation | Operation records: thermal dissipation. |
| [modules/eos/conserved.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/eos/conserved.py) | 1 | pointwise state update | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. EOS/material lookup, conserved-to-primitive conversion or assembled/masked particle updates. Stores: per-particle thermodynamic fields and update vectors. |
| [modules/eos/materials.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/eos/materials.py) | 1 | pointwise state update | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. EOS/material lookup, conserved-to-primitive conversion or assembled/masked particle updates. Stores: per-particle thermodynamic fields and update vectors. |
| [modules/godunov/wp_gsph.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/godunov/wp_gsph.py) | 4 | operation | Operation records: Godunov rates. |
| [modules/godunov/wp_inutsuka.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/godunov/wp_inutsuka.py) | 6 | operation | Operation records: Gaussian density/gradient, Inutsuka rates. |
| [modules/iisph/wp_contactPressureDiagonal.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_contactPressureDiagonal.py) | 3 | operation | Operation records: IISPH contact pressure diagonal. |
| [modules/iisph/wp_densityRate.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_densityRate.py) | 3 | operation | Operation records: IISPH density rate. |
| [modules/iisph/wp_gradientSum.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_gradientSum.py) | 3 | operation | Operation records: IISPH gradient sum. |
| [modules/iisph/wp_pressureAcceleration.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_pressureAcceleration.py) | 3 | operation | Operation records: IISPH pressure acceleration. |
| [modules/iisph/wp_pressureDiagonal.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_pressureDiagonal.py) | 3 | operation | Operation records: IISPH pressure diagonal. |
| [modules/iisph/wp_sampledGravity.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_sampledGravity.py) | 3 | operation | Operation records: IISPH sampled gravity. |
| [modules/iisph/wp_wallGradient.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_wallGradient.py) | 3 | operation | Operation records: IISPH wall gradient. |
| [modules/incompressible/wp_alpha.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/incompressible/wp_alpha.py) | 3 | operation | Operation records: incompressible alpha. |
| [modules/incompressible/wp_wallMoment.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/incompressible/wp_wallMoment.py) | 4 | operation | Operation records: wall moment. |
| [modules/liu/wp_mat.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/liu/wp_mat.py) | 4 | operation | Operation records: Liu matrices. |
| [modules/mdbc/wp_nopenshift.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/mdbc/wp_nopenshift.py) | 4 | operation | Operation records: mDBC no-penetration shift. |
| [modules/meshless/flux.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/meshless/flux.py) | 2 | operation | Operation records: frontend MFM flux. |
| [modules/meshless/primitives.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/meshless/primitives.py) | 1 | pointwise state update | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. EOS/material lookup, conserved-to-primitive conversion or assembled/masked particle updates. Stores: per-particle thermodynamic fields and update vectors. |
| [modules/pesph/operators.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/pesph/operators.py) | 3 | operation | Operation records: pressure-entropy sums, pressure-entropy force. |
| [modules/pressure/wp_perSidePressure.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/pressure/wp_perSidePressure.py) | 3 | operation | Operation records: per-side pressure. |
| [modules/pressure/wp_surfaceAware.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/pressure/wp_surfaceAware.py) | 4 | operation | Operation records: surface-aware pressure. |
| [modules/reconstruction/diagnostics.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/diagnostics.py) | 1 | pair reconstruction helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Limit/reconstruct scalar, primitive or velocity pair states; diagnostics can write one explicit pair result. Stores: return reconstructed scalar/vector/tuple or one diagnostic per supplied pair. |
| [modules/reconstruction/limiters.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/limiters.py) | 5 | pair reconstruction helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Limit/reconstruct scalar, primitive or velocity pair states; diagnostics can write one explicit pair result. Stores: return reconstructed scalar/vector/tuple or one diagnostic per supplied pair. |
| [modules/reconstruction/meshless.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/meshless.py) | 7 | operation | Operation records: frontend MFM limiter. |
| [modules/reconstruction/pairState.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/pairState.py) | 6 | pair reconstruction helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Limit/reconstruct scalar, primitive or velocity pair states; diagnostics can write one explicit pair result. Stores: return reconstructed scalar/vector/tuple or one diagnostic per supplied pair. |
| [modules/reconstruction/pairVelocity.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/pairVelocity.py) | 5 | pair reconstruction helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Limit/reconstruct scalar, primitive or velocity pair states; diagnostics can write one explicit pair result. Stores: return reconstructed scalar/vector/tuple or one diagnostic per supplied pair. |
| [modules/riemann/meshless.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/riemann/meshless.py) | 9 | pair Riemann helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Compute wave/contact states and scalar/vector fluxes from two states. Stores: return wave/star/flux tuple; no particle reduction. |
| [modules/riemann/solvers.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/riemann/solvers.py) | 9 | pair Riemann helper | 0 across-particle-neighborhood accumulators; see local syntax for internal component/solver reductions. Compute wave/contact states and scalar/vector fluxes from two states. Stores: return wave/star/flux tuple; no particle reduction. |
| [modules/shifting/wp_deltaShiftRaw.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shifting/wp_deltaShiftRaw.py) | 4 | operation | Operation records: raw delta shift. |
| [modules/shifting/wp_implicitShifting.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shifting/wp_implicitShifting.py) | 1 | operation | Operation records: implicit shift pair terms. |
| [modules/shifting/wp_michelUChar.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shifting/wp_michelUChar.py) | 4 | operation | Operation records: characteristic shift speed. |
| [modules/shockCapturing/ReadHayfield2012/kernels.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shockCapturing/ReadHayfield2012/kernels.py) | 7 | operation | Operation records: Read-Hayfield signal speed, Read-Hayfield entropy. |
| [modules/shockCapturing/Wadsley2017/kernels.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shockCapturing/Wadsley2017/kernels.py) | 2 | operation | Operation records: Wadsley neighbor mean, Wadsley point fields. |
| [modules/shockCapturing/wp_computeM.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shockCapturing/wp_computeM.py) | 3 | operation | Operation records: shock moment M. |
| [modules/shockCapturing/wp_vsig.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shockCapturing/wp_vsig.py) | 4 | operation | Operation records: signal speed. |
| [modules/surfaceDetection/wp_barecasco.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_barecasco.py) | 6 | operation | Operation records: Barecasco cover pass, Barecasco cone pass. |
| [modules/surfaceDetection/wp_barecascoCone.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_barecascoCone.py) | 4 | operation | Operation records: Barecasco cone count. |
| [modules/surfaceDetection/wp_barecascoCover.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_barecascoCover.py) | 4 | operation | Operation records: Barecasco cover. |
| [modules/surfaceDetection/wp_dilate.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_dilate.py) | 4 | operation | Operation records: surface dilation sum. |
| [modules/surfaceDetection/wp_maronne.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_maronne.py) | 3 | operation | Operation records: Maronne count. |
| [modules/surfaceDetection/wp_nearestSurfaceNormal.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_nearestSurfaceNormal.py) | 4 | operation | Operation records: nearest surface normal. |
| [modules/util/wp_numNeighbors.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/util/wp_numNeighbors.py) | 4 | operation | Operation records: frontend neighbor count. |
| [modules/util/wp_sum.py](/home/lu26029/dev/warpSPH/src/warpSPH/modules/util/wp_sum.py) | 3 | operation | Operation records: frontend neighbor field sum. |
| [sample/wp_deltaShift.py](/home/lu26029/dev/warpSPH/src/warpSPH/sample/wp_deltaShift.py) | 4 | operation | Operation records: sample delta shift. |

## Definition coverage

Each decorated source definition is linked below. Operation links follow local call relationships to reviewed anchors; a local helper may have no operation link and is described by its file assessment. The JSON retains the exact evidence and distinguishes source roles. Overloads with the same name remain separate by line.

| Source definition | Decorator | Role | Linked passes |
| --- | --- | --- | --- |
| [core:coreOperations/_jvpCommon.py:98 gradientWeightsJVP](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/_jvpCommon.py:98) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:coreOperations/native/wp_covariance.py:39 computeCovariance_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_covariance.py:39) | wp.func | range/candidate accumulation or edge emission | covariance |
| [core:coreOperations/native/wp_covariance.py:113 computeCovariance_Func_Adjacency](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_covariance.py:113) | wp.func | adjacency/range orchestration | covariance |
| [core:coreOperations/native/wp_covariance.py:162 computeCovariance_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_covariance.py:162) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | covariance |
| [core:coreOperations/native/wp_covariance.py:203 computeCovariance_KernelTiled](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_covariance.py:203) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | covariance |
| [core:coreOperations/native/wp_curl.py:25 computeSPHCurlTensor_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_curl.py:25) | wp.func | range/candidate accumulation or edge emission | curl |
| [core:coreOperations/native/wp_curl.py:105 computeSPHCurlTensor_Func_Adjacency](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_curl.py:105) | wp.func | adjacency/range orchestration | curl |
| [core:coreOperations/native/wp_curl.py:158 computeSPHCurlTensor_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_curl.py:158) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | curl |
| [core:coreOperations/native/wp_curl.py:193 computeSPHCurlTensor_KernelTiled](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_curl.py:193) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | curl |
| [core:coreOperations/native/wp_density.py:33 computeSPHDensity_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_density.py:33) | wp.func | range/candidate accumulation or edge emission | density |
| [core:coreOperations/native/wp_density.py:74 computeSPHDensity_Func_Adjacency](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_density.py:74) | wp.func | adjacency/range orchestration | density |
| [core:coreOperations/native/wp_density.py:118 computeSPHDensity_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_density.py:118) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | density |
| [core:coreOperations/native/wp_density.py:148 computeSPHDensity_KernelTiled](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_density.py:148) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | density |
| [core:coreOperations/native/wp_divergence.py:24 computeSPHDivergenceTensor_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_divergence.py:24) | wp.func | range/candidate accumulation or edge emission | divergence |
| [core:coreOperations/native/wp_divergence.py:111 computeSPHDivergenceTensor_Func_Adjacency](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_divergence.py:111) | wp.func | adjacency/range orchestration | divergence |
| [core:coreOperations/native/wp_divergence.py:166 computeSPHDivergenceTensor_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_divergence.py:166) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | divergence |
| [core:coreOperations/native/wp_divergence.py:204 computeSPHDivergenceTensor_KernelTiled](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_divergence.py:204) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | divergence |
| [core:coreOperations/native/wp_gradient.py:40 computeSPHGradientTensor_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_gradient.py:40) | wp.func | range/candidate accumulation or edge emission | gradient |
| [core:coreOperations/native/wp_gradient.py:123 computeSPHGradientTensor_Func_Adjacency](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_gradient.py:123) | wp.func | adjacency/range orchestration | gradient |
| [core:coreOperations/native/wp_gradient.py:177 computeSPHGradientTensor_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_gradient.py:177) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | gradient |
| [core:coreOperations/native/wp_gradient.py:221 computeSPHGradientTensor_KernelTiled](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_gradient.py:221) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | gradient |
| [core:coreOperations/native/wp_interpolate.py:29 computeSPHInterpolation_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_interpolate.py:29) | wp.func | range/candidate accumulation or edge emission | interpolate |
| [core:coreOperations/native/wp_interpolate.py:91 computeSPHInterpolation_Func_Adjacency](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_interpolate.py:91) | wp.func | adjacency/range orchestration | interpolate |
| [core:coreOperations/native/wp_interpolate.py:139 computeSPHInterpolation_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_interpolate.py:139) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | interpolate |
| [core:coreOperations/native/wp_interpolate.py:172 computeSPHInterpolation_KernelTiled](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_interpolate.py:172) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | interpolate |
| [core:coreOperations/native/wp_laplacian.py:26 computeSPHLaplacianTensor_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_laplacian.py:26) | wp.func | range/candidate accumulation or edge emission | laplacian |
| [core:coreOperations/native/wp_laplacian.py:177 computeSPHLaplacianTensor_Func_Adjacency](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_laplacian.py:177) | wp.func | adjacency/range orchestration | laplacian |
| [core:coreOperations/native/wp_laplacian.py:231 computeSPHLaplacianTensor_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_laplacian.py:231) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | laplacian |
| [core:coreOperations/native/wp_laplacian.py:265 computeSPHLaplacianTensor_KernelTiled](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/native/wp_laplacian.py:265) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | laplacian |
| [core:coreOperations/wp_covarianceJVP.py:52 computeCovarianceJVP_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_covarianceJVP.py:52) | wp.func | range/candidate accumulation or edge emission | covariance JVP |
| [core:coreOperations/wp_covarianceJVP.py:104 computeCovarianceJVP_Func_Adjacency](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_covarianceJVP.py:104) | wp.func | adjacency/range orchestration | covariance JVP |
| [core:coreOperations/wp_covarianceJVP.py:144 computeCovarianceJVP_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_covarianceJVP.py:144) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | covariance JVP |
| [core:coreOperations/wp_curlJVP.py:44 computeSPHCurlJVP_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_curlJVP.py:44) | wp.func | range/candidate accumulation or edge emission | curl JVP |
| [core:coreOperations/wp_curlJVP.py:126 computeSPHCurlJVP_Func_Adjacency](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_curlJVP.py:126) | wp.func | adjacency/range orchestration | curl JVP |
| [core:coreOperations/wp_curlJVP.py:180 computeSPHCurlJVP_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_curlJVP.py:180) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | curl JVP |
| [core:coreOperations/wp_densityHVP.py:129 _computeSPHDensityHVP_PairKernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_densityHVP.py:129) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | density Hessian edge terms |
| [core:coreOperations/wp_densityJVP.py:44 computeSPHDensityJVP_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_densityJVP.py:44) | wp.func | range/candidate accumulation or edge emission | density JVP |
| [core:coreOperations/wp_densityJVP.py:84 computeSPHDensityJVP_Func_Adjacency](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_densityJVP.py:84) | wp.func | adjacency/range orchestration | density JVP |
| [core:coreOperations/wp_densityJVP.py:129 computeSPHDensityJVP_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_densityJVP.py:129) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | density JVP |
| [core:coreOperations/wp_divergenceJVP.py:43 computeSPHDivergenceJVP_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_divergenceJVP.py:43) | wp.func | range/candidate accumulation or edge emission | divergence JVP |
| [core:coreOperations/wp_divergenceJVP.py:120 computeSPHDivergenceJVP_Func_Adjacency](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_divergenceJVP.py:120) | wp.func | adjacency/range orchestration | divergence JVP |
| [core:coreOperations/wp_divergenceJVP.py:174 computeSPHDivergenceJVP_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_divergenceJVP.py:174) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | divergence JVP |
| [core:coreOperations/wp_gradientJVP.py:54 computeSPHGradientJVP_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_gradientJVP.py:54) | wp.func | range/candidate accumulation or edge emission | gradient JVP |
| [core:coreOperations/wp_gradientJVP.py:134 computeSPHGradientJVP_Func_Adjacency](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_gradientJVP.py:134) | wp.func | adjacency/range orchestration | gradient JVP |
| [core:coreOperations/wp_gradientJVP.py:188 computeSPHGradientJVP_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_gradientJVP.py:188) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | gradient JVP |
| [core:coreOperations/wp_interpolateJVP.py:44 computeSPHInterpolateJVP_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_interpolateJVP.py:44) | wp.func | range/candidate accumulation or edge emission | interpolate JVP |
| [core:coreOperations/wp_interpolateJVP.py:100 computeSPHInterpolateJVP_Func_Adjacency](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_interpolateJVP.py:100) | wp.func | adjacency/range orchestration | interpolate JVP |
| [core:coreOperations/wp_interpolateJVP.py:151 computeSPHInterpolateJVP_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_interpolateJVP.py:151) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | interpolate JVP |
| [core:coreOperations/wp_laplacianJVP.py:144 _laplacianGeometryChainJVP](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_laplacianJVP.py:144) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:coreOperations/wp_laplacianJVP.py:212 _laplacianPJVP](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_laplacianJVP.py:212) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:coreOperations/wp_laplacianJVP.py:233 computeSPHLaplacianBrookshawJVP_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_laplacianJVP.py:233) | wp.func | range/candidate accumulation or edge emission | Laplacian Brookshaw JVP |
| [core:coreOperations/wp_laplacianJVP.py:300 computeSPHLaplacianBrookshawJVP_Func_Adjacency](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_laplacianJVP.py:300) | wp.func | adjacency/range orchestration | Laplacian Brookshaw JVP |
| [core:coreOperations/wp_laplacianJVP.py:354 computeSPHLaplacianBrookshawJVP_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_laplacianJVP.py:354) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | Laplacian Brookshaw JVP |
| [core:coreOperations/wp_laplacianJVP.py:536 computeSPHLaplacianNaiveJVP_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_laplacianJVP.py:536) | wp.func | range/candidate accumulation or edge emission | Laplacian Naive JVP |
| [core:coreOperations/wp_laplacianJVP.py:593 computeSPHLaplacianNaiveJVP_Func_Adjacency](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_laplacianJVP.py:593) | wp.func | adjacency/range orchestration | Laplacian Naive JVP |
| [core:coreOperations/wp_laplacianJVP.py:647 computeSPHLaplacianNaiveJVP_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_laplacianJVP.py:647) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | Laplacian Naive JVP |
| [core:coreOperations/wp_laplacianJVP.py:798 _laplacianDotProjJVP](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_laplacianJVP.py:798) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:coreOperations/wp_laplacianJVP.py:825 computeSPHLaplacianDotJVP_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_laplacianJVP.py:825) | wp.func | range/candidate accumulation or edge emission | Laplacian Dot JVP |
| [core:coreOperations/wp_laplacianJVP.py:926 computeSPHLaplacianDotJVP_Func_Adjacency](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_laplacianJVP.py:926) | wp.func | adjacency/range orchestration | Laplacian Dot JVP |
| [core:coreOperations/wp_laplacianJVP.py:980 computeSPHLaplacianDotJVP_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_laplacianJVP.py:980) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | Laplacian Dot JVP |
| [core:coreOperations/wp_laplacianJVP.py:1182 computeSPHLaplacianDefaultJVP_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_laplacianJVP.py:1182) | wp.func | range/candidate accumulation or edge emission | Laplacian Default JVP |
| [core:coreOperations/wp_laplacianJVP.py:1263 computeSPHLaplacianDefaultJVP_Func_Adjacency](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_laplacianJVP.py:1263) | wp.func | adjacency/range orchestration | Laplacian Default JVP |
| [core:coreOperations/wp_laplacianJVP.py:1317 computeSPHLaplacianDefaultJVP_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/coreOperations/wp_laplacianJVP.py:1317) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | Laplacian Default JVP |
| [core:crk/crk_density.py:30 computeCRKDensity_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/crk_density.py:30) | wp.func | range/candidate accumulation or edge emission | CRK density |
| [core:crk/crk_density.py:95 computeCRKDensity_Func_Adjacency](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/crk_density.py:95) | wp.func | adjacency/range orchestration | CRK density |
| [core:crk/crk_density.py:145 computeCRKDensity_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/crk_density.py:145) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | CRK density |
| [core:crk/crk_density.py:181 computeCRKDensity_KernelTiled](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/crk_density.py:181) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | CRK density |
| [core:crk/crk_moments.py:37 computeCRKMoments_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/crk_moments.py:37) | wp.func | range/candidate accumulation or edge emission | CRK moments |
| [core:crk/crk_moments.py:114 computeCRKMoments_Func_Adjacency](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/crk_moments.py:114) | wp.func | adjacency/range orchestration | CRK moments |
| [core:crk/crk_moments.py:180 computeCRKMoments_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/crk_moments.py:180) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | CRK moments |
| [core:crk/crk_moments.py:232 computeCRKMoments_KernelTiled](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/crk_moments.py:232) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | CRK moments |
| [core:crk/crk_moments_jvp.py:49 computeCRKMomentsJVP_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/crk_moments_jvp.py:49) | wp.func | range/candidate accumulation or edge emission | CRK moments JVP |
| [core:crk/crk_moments_jvp.py:135 computeCRKMomentsJVP_Func_Adjacency](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/crk_moments_jvp.py:135) | wp.func | adjacency/range orchestration | CRK moments JVP |
| [core:crk/crk_moments_jvp.py:197 computeCRKMomentsJVP_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/crk_moments_jvp.py:197) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | CRK moments JVP |
| [core:crk/crk_volume.py:28 computeCRKVolume_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/crk_volume.py:28) | wp.func | range/candidate accumulation or edge emission | CRK volume |
| [core:crk/crk_volume.py:69 computeCRKVolume_Func_Adjacency](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/crk_volume.py:69) | wp.func | adjacency/range orchestration | CRK volume |
| [core:crk/crk_volume.py:118 computeCRKVolume_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/crk_volume.py:118) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | CRK volume |
| [core:crk/crk_volume_jvp.py:43 computeCRKVolumeJVP_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/crk_volume_jvp.py:43) | wp.func | range/candidate accumulation or edge emission | CRK volume JVP |
| [core:crk/crk_volume_jvp.py:79 computeCRKVolumeJVP_Func_Adjacency](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/crk_volume_jvp.py:79) | wp.func | adjacency/range orchestration | CRK volume JVP |
| [core:crk/crk_volume_jvp.py:121 computeCRKVolumeJVP_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/crk_volume_jvp.py:121) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | CRK volume JVP |
| [core:crk/kernel.py:10 correctGradientCRK](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/kernel.py:10) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:crk/kernel.py:45 correctGradientCRKJVP](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/kernel.py:45) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:crk/kernel.py:98 computeKernelGradientCRKJVP](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/kernel.py:98) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:crk/kernel.py:128 computeKernelGradientCRK](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/kernel.py:128) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:crk/kernel.py:152 computeKernelCRK](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/crk/kernel.py:152) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/derivative.py:17 sphKernelDerivative_](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/derivative.py:17) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/derivative.py:26 sphKernelDerivative](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/derivative.py:26) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/eval_kernel.py:11 eval_k](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/eval_kernel.py:11) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/eval_kernel.py:45 eval_dkdq](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/eval_kernel.py:45) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/eval_kernel.py:79 eval_d2kdq2](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/eval_kernel.py:79) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/eval_kernel.py:113 eval_d3kdq3](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/eval_kernel.py:113) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/eval_kernel.py:147 eval_C_d](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/eval_kernel.py:147) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/eval_kernel.py:181 eval_kernelScale](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/eval_kernel.py:181) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/eval_kernel.py:215 eval_packing](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/eval_kernel.py:215) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/gradH.py:16 sphKernelDkDh_](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/gradH.py:16) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/gradH.py:31 sphKernelDkDh](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/gradH.py:31) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/gradH.py:50 sphGradientDkDh_](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/gradH.py:50) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/gradH.py:80 sphGradientDkDh](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/gradH.py:80) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/gradient.py:30 sphGradient_1D](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/gradient.py:30) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/gradient.py:42 sphGradient_2D](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/gradient.py:42) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/gradient.py:54 sphGradient_3D](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/gradient.py:54) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/gradient.py:92 adj_sphGradient_1D](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/gradient.py:92) | wp.func_grad | custom local adjoint | local helper or file assessment |
| [core:kernels/gradient.py:98 adj_sphGradient_2D](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/gradient.py:98) | wp.func_grad | custom local adjoint | local helper or file assessment |
| [core:kernels/gradient.py:104 adj_sphGradient_3D](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/gradient.py:104) | wp.func_grad | custom local adjoint | local helper or file assessment |
| [core:kernels/gradient.py:112 sphGradient_](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/gradient.py:112) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/gradient.py:115 sphGradient_](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/gradient.py:115) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/gradient.py:118 sphGradient_](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/gradient.py:118) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/gradient.py:124 sphKernelGradient](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/gradient.py:124) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/gradient.py:144 sphKernelGradient_ij](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/gradient.py:144) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/hessian.py:16 sphKernelHessian_](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/hessian.py:16) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/hessian.py:40 sphKernelHessian](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/hessian.py:40) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernel.py:16 sphKernel_](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernel.py:16) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernel.py:26 sphKernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernel.py:26) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernel.py:43 sphKernel_ij](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernel.py:43) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/B7.py:27 B7_k](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/B7.py:27) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/B7.py:31 B7_dkdq](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/B7.py:31) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/B7.py:35 B7_d2kdq2](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/B7.py:35) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/B7.py:39 B7_d3kdq3](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/B7.py:39) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/B7.py:43 B7_C_d](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/B7.py:43) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/B7.py:49 B7_kernelScale](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/B7.py:49) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/B7.py:55 B7_packingRatio](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/B7.py:55) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/B8.py:18 B8_k](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/B8.py:18) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/B8.py:22 B8_dkdq](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/B8.py:22) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/B8.py:26 B8_d2kdq2](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/B8.py:26) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/B8.py:30 B8_d3kdq3](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/B8.py:30) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/B8.py:34 B8_C_d](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/B8.py:34) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/B8.py:40 B8_kernelScale](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/B8.py:40) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/B8.py:46 B8_packingRatio](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/B8.py:46) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/adhesionKernel.py:9 adhesionKernel_k](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/adhesionKernel.py:9) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/adhesionKernel.py:15 adhesionKernel_dkdq](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/adhesionKernel.py:15) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/adhesionKernel.py:21 adhesionKernel_d2kdq2](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/adhesionKernel.py:21) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/adhesionKernel.py:27 adhesionKernel_d3kdq3](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/adhesionKernel.py:27) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/adhesionKernel.py:33 adhesionKernel_C_d](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/adhesionKernel.py:33) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/adhesionKernel.py:39 adhesionKernel_kernelScale](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/adhesionKernel.py:39) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/adhesionKernel.py:43 adhesionKernel_packingRatio](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/adhesionKernel.py:43) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/cohesionKernel.py:9 cohesionKernel_k](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/cohesionKernel.py:9) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/cohesionKernel.py:16 cohesionKernel_dkdq](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/cohesionKernel.py:16) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/cohesionKernel.py:23 cohesionKernel_d2kdq2](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/cohesionKernel.py:23) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/cohesionKernel.py:30 cohesionKernel_d3kdq3](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/cohesionKernel.py:30) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/cohesionKernel.py:37 cohesionKernel_C_d](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/cohesionKernel.py:37) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/cohesionKernel.py:43 cohesionKernel_kernelScale](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/cohesionKernel.py:43) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/cohesionKernel.py:47 cohesionKernel_packingRatio](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/cohesionKernel.py:47) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/cubicSpline.py:9 cubicSpline_k](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/cubicSpline.py:9) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/cubicSpline.py:13 cubicSpline_dkdq](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/cubicSpline.py:13) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/cubicSpline.py:17 cubicSpline_d2kdq2](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/cubicSpline.py:17) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/cubicSpline.py:24 cubicSpline_d3kdq3](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/cubicSpline.py:24) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/cubicSpline.py:31 cubicSpline_C_d](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/cubicSpline.py:31) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/cubicSpline.py:37 cubicSpline_kernelScale](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/cubicSpline.py:37) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/cubicSpline.py:43 cubicSpline_packingRatio](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/cubicSpline.py:43) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/gaussian.py:19 Gaussian_k](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/gaussian.py:19) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/gaussian.py:23 Gaussian_dkdq](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/gaussian.py:23) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/gaussian.py:28 Gaussian_d2kdq2](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/gaussian.py:28) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/gaussian.py:33 Gaussian_d3kdq3](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/gaussian.py:33) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/gaussian.py:38 Gaussian_C_d](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/gaussian.py:38) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/gaussian.py:45 Gaussian_kernelScale](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/gaussian.py:45) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/gaussian.py:49 Gaussian_packingRatio](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/gaussian.py:49) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/hoct4.py:41 HOCT4_k](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/hoct4.py:41) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/hoct4.py:49 HOCT4_dkdq](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/hoct4.py:49) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/hoct4.py:55 HOCT4_d2kdq2](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/hoct4.py:55) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/hoct4.py:60 HOCT4_d3kdq3](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/hoct4.py:60) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/hoct4.py:65 HOCT4_C_d](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/hoct4.py:65) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/hoct4.py:71 HOCT4_kernelScale](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/hoct4.py:71) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/hoct4.py:77 HOCT4_packingRatio](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/hoct4.py:77) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/poly6.py:9 poly6_k](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/poly6.py:9) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/poly6.py:13 poly6_dkdq](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/poly6.py:13) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/poly6.py:17 poly6_d2kdq2](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/poly6.py:17) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/poly6.py:21 poly6_d3kdq3](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/poly6.py:21) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/poly6.py:25 poly6_C_d](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/poly6.py:25) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/poly6.py:31 poly6_kernelScale](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/poly6.py:31) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/poly6.py:35 poly6_packingRatio](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/poly6.py:35) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/quarticSpline.py:9 quarticSpline_k](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/quarticSpline.py:9) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/quarticSpline.py:13 quarticSpline_dkdq](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/quarticSpline.py:13) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/quarticSpline.py:17 quarticSpline_d2kdq2](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/quarticSpline.py:17) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/quarticSpline.py:21 quarticSpline_d3kdq3](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/quarticSpline.py:21) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/quarticSpline.py:25 quarticSpline_C_d](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/quarticSpline.py:25) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/quarticSpline.py:31 quarticSpline_kernelScale](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/quarticSpline.py:31) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/quarticSpline.py:37 quarticSpline_packingRatio](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/quarticSpline.py:37) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/quinticSpline.py:9 quinticSpline_k](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/quinticSpline.py:9) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/quinticSpline.py:13 quinticSpline_dkdq](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/quinticSpline.py:13) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/quinticSpline.py:17 quinticSpline_d2kdq2](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/quinticSpline.py:17) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/quinticSpline.py:21 quinticSpline_d3kdq3](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/quinticSpline.py:21) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/quinticSpline.py:25 quinticSpline_C_d](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/quinticSpline.py:25) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/quinticSpline.py:31 quinticSpline_kernelScale](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/quinticSpline.py:31) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/quinticSpline.py:37 quinticSpline_packingRatio](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/quinticSpline.py:37) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/spiky.py:9 spiky_k](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/spiky.py:9) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/spiky.py:13 spiky_dkdq](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/spiky.py:13) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/spiky.py:17 spiky_d2kdq2](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/spiky.py:17) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/spiky.py:21 spiky_d3kdq3](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/spiky.py:21) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/spiky.py:25 spiky_C_d](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/spiky.py:25) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/spiky.py:31 spiky_kernelScale](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/spiky.py:31) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/spiky.py:35 spiky_packingRatio](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/spiky.py:35) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/viscosityKernel.py:9 viscosityKernel_k](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/viscosityKernel.py:9) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/viscosityKernel.py:13 viscosityKernel_dkdq](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/viscosityKernel.py:13) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/viscosityKernel.py:17 viscosityKernel_d2kdq2](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/viscosityKernel.py:17) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/viscosityKernel.py:21 viscosityKernel_d3kdq3](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/viscosityKernel.py:21) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/viscosityKernel.py:25 viscosityKernel_C_d](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/viscosityKernel.py:25) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/viscosityKernel.py:31 viscosityKernel_kernelScale](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/viscosityKernel.py:31) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/viscosityKernel.py:35 viscosityKernel_packingRatio](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/viscosityKernel.py:35) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/wendland2.py:9 wendland2_k](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/wendland2.py:9) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/wendland2.py:15 wendland2_dkdq](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/wendland2.py:15) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/wendland2.py:22 wendland2_d2kdq2](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/wendland2.py:22) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/wendland2.py:29 wendland2_d3kdq3](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/wendland2.py:29) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/wendland2.py:36 wendland2_C_d](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/wendland2.py:36) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/wendland2.py:42 wendland2_kernelScale](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/wendland2.py:42) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/wendland2.py:48 wendland2_packingRatio](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/wendland2.py:48) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/wendland4.py:9 wendland4_k](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/wendland4.py:9) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/wendland4.py:16 wendland4_dkdq](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/wendland4.py:16) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/wendland4.py:23 wendland4_d2kdq2](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/wendland4.py:23) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/wendland4.py:30 wendland4_d3kdq3](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/wendland4.py:30) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/wendland4.py:37 wendland4_C_d](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/wendland4.py:37) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/wendland4.py:43 wendland4_kernelScale](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/wendland4.py:43) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/wendland4.py:49 wendland4_packingRatio](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/wendland4.py:49) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/wendland6.py:9 wendland6_k](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/wendland6.py:9) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/wendland6.py:16 wendland6_dkdq](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/wendland6.py:16) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/wendland6.py:23 wendland6_d2kdq2](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/wendland6.py:23) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/wendland6.py:30 wendland6_d3kdq3](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/wendland6.py:30) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/wendland6.py:37 wendland6_C_d](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/wendland6.py:37) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/wendland6.py:43 wendland6_kernelScale](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/wendland6.py:43) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelFunctions/wendland6.py:49 wendland6_packingRatio](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelFunctions/wendland6.py:49) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelJVP.py:36 sphKernelJVP_ij](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelJVP.py:36) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelJVP.py:68 sphKernelJVP](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelJVP.py:68) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelJVP.py:88 sphKernelGradientJVP_ij](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelJVP.py:88) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelJVP.py:134 sphKernelGradientJVP](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelJVP.py:134) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelJVP.py:152 sphKernelLaplacianJVP_ij](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelJVP.py:152) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/kernelJVP.py:203 sphKernelLaplacianJVP](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/kernelJVP.py:203) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/laplacian.py:16 sphKernelLaplacian_](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/laplacian.py:16) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/laplacian.py:38 sphKernelLaplacianGradient_](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/laplacian.py:38) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/laplacian.py:110 sphKernelLaplacianDkDh_](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/laplacian.py:110) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/laplacian.py:162 sphKernelLaplacian](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/laplacian.py:162) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/properties.py:13 sphKernelScale](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/properties.py:13) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/properties.py:17 sphKernelC_d](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/properties.py:17) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/properties.py:28 sphKernelN_H](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/properties.py:28) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/properties.py:38 sphKernel_xi](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/properties.py:38) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:kernels/properties.py:43 resolveNormalization](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/kernels/properties.py:43) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:linalg/krylov/bicgstabWorkspace.py:19 _divide](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/krylov/bicgstabWorkspace.py:19) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:linalg/krylov/bicgstabWorkspace.py:26 _initialize](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/krylov/bicgstabWorkspace.py:26) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [core:linalg/krylov/bicgstabWorkspace.py:44 _alpha](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/krylov/bicgstabWorkspace.py:44) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [core:linalg/krylov/bicgstabWorkspace.py:60 _halfCandidate](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/krylov/bicgstabWorkspace.py:60) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [core:linalg/krylov/bicgstabWorkspace.py:80 _halfDecision](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/krylov/bicgstabWorkspace.py:80) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [core:linalg/krylov/bicgstabWorkspace.py:96 _omega](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/krylov/bicgstabWorkspace.py:96) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [core:linalg/krylov/bicgstabWorkspace.py:101 _fullCandidate](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/krylov/bicgstabWorkspace.py:101) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [core:linalg/krylov/bicgstabWorkspace.py:123 _fullDecision](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/krylov/bicgstabWorkspace.py:123) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [core:linalg/krylov/bicgstabWorkspace.py:145 _direction](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/krylov/bicgstabWorkspace.py:145) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [core:linalg/krylov/bicgstabWorkspace.py:159 _finish](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/krylov/bicgstabWorkspace.py:159) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [core:linalg/krylov/bicgstabWorkspace.py:174 _commit](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/krylov/bicgstabWorkspace.py:174) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [core:linalg/krylov/bicgstabWorkspace.py:183 _diagonal](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/krylov/bicgstabWorkspace.py:183) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [core:linalg/krylov/cgWorkspace.py:21 _initialize](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/krylov/cgWorkspace.py:21) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [core:linalg/krylov/cgWorkspace.py:43 _candidate](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/krylov/cgWorkspace.py:43) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [core:linalg/krylov/cgWorkspace.py:65 _decide](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/krylov/cgWorkspace.py:65) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [core:linalg/krylov/cgWorkspace.py:91 _commit](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/krylov/cgWorkspace.py:91) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [core:linalg/krylov/cgWorkspace.py:100 _diagonal](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/krylov/cgWorkspace.py:100) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [core:linalg/krylov/cgWorkspace.py:106 _direction](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/krylov/cgWorkspace.py:106) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [core:linalg/pinv.py:42 _rotate](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/pinv.py:42) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:linalg/pinv.py:64 symmetricPinv3](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/pinv.py:64) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:linalg/pinv.py:87 loadPadded3x3](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/pinv.py:87) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:linalg/pinv.py:98 _pinvKernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/pinv.py:98) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [core:linalg/pinv.py:108 _applyPinvKernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/pinv.py:108) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [core:linalg/preconditioner.py:21 _blockApply](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/linalg/preconditioner.py:21) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [core:math/wp_cross.py:13 curlProduct](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_cross.py:13) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_cross.py:41 curlProduct](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_cross.py:41) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_cross.py:58 curlProduct](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_cross.py:58) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_delta.py:5 kroneckerDelta](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_delta.py:5) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_dim.py:6 get_dim](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_dim.py:6) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_dim.py:9 get_dim](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_dim.py:9) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_dim.py:12 get_dim](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_dim.py:12) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_dim.py:16 get_dim](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_dim.py:16) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_dim.py:19 get_dim](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_dim.py:19) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_dim.py:22 get_dim](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_dim.py:22) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_dim.py:26 get_dim](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_dim.py:26) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_dim.py:29 get_dim](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_dim.py:29) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_dim.py:32 get_dim](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_dim.py:32) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_dim.py:35 get_dim](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_dim.py:35) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_dim.py:38 get_dim](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_dim.py:38) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_dim.py:41 get_dim](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_dim.py:41) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_distance.py:10 mod_distance](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_distance.py:10) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_distance.py:39 computeCartesianDistance](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_distance.py:39) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_distance.py:52 mod_warp](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_distance.py:52) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_distance.py:57 moduloDistanceWarp](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_distance.py:57) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_distance.py:66 minimumImageDistanceWarp](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_distance.py:66) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_distance.py:80 computeDistance](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_distance.py:80) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_distance.py:87 mod_warp](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_distance.py:87) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_distance.py:92 project_mod](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_distance.py:92) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_distance.py:99 moduloDistanceComponent](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_distance.py:99) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_distance.py:112 minimumImageDistance](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_distance.py:112) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_distance.py:129 computeDistance](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_distance.py:129) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_distance.py:139 computeDistanceVec](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_distance.py:139) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_divdot.py:23 divergenceProduct](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_divdot.py:23) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_divdot.py:44 divergenceProduct](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_divdot.py:44) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_divdot.py:65 divergenceProduct](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_divdot.py:65) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_dot.py:9 positiveDotProduct](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_dot.py:9) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_eps.py:5 get_epsilon](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_eps.py:5) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_eps.py:9 get_epsilon](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_eps.py:9) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_eps.py:13 get_epsilon](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_eps.py:13) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_eps.py:17 get_epsilon](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_eps.py:17) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_eye.py:6 warp_eye](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_eye.py:6) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_eye.py:13 warp_eye](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_eye.py:13) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_eye.py:20 warp_eye](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_eye.py:20) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_laplaciandot.py:18 computeDotLaplacian](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_laplaciandot.py:18) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_laplaciandot.py:28 computeDotLaplacian](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_laplaciandot.py:28) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_laplaciandot.py:40 computeDotLaplacian](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_laplaciandot.py:40) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_laplaciandot.py:53 computeDotLaplacian](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_laplaciandot.py:53) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_laplaciandot.py:63 computeLaplacianDot2](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_laplaciandot.py:63) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_matmul.py:7 matmul](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_matmul.py:7) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_matmul.py:22 matmul](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_matmul.py:22) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_matmul.py:37 matmul](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_matmul.py:37) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_norm.py:11 norm_warp](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_norm.py:11) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_norm.py:15 norm_grad_warp](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_norm.py:15) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_norm.py:22 norm_hess_warp](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_norm.py:22) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_normalize.py:13 vectorNormalize_warp_1D](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_normalize.py:13) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_normalize.py:17 adj_vectorNormalize_warp_1D](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_normalize.py:17) | wp.func_grad | custom local adjoint | local helper or file assessment |
| [core:math/wp_normalize.py:28 vectorNormalize_warp_2D](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_normalize.py:28) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_normalize.py:32 adj_vectorNormalize_warp_2D](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_normalize.py:32) | wp.func_grad | custom local adjoint | local helper or file assessment |
| [core:math/wp_normalize.py:42 vectorNormalize_warp_3D](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_normalize.py:42) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_normalize.py:46 adj_vectorNormalize_warp_3D](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_normalize.py:46) | wp.func_grad | custom local adjoint | local helper or file assessment |
| [core:math/wp_normalize.py:57 vectorNormalize_warp](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_normalize.py:57) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_normalize.py:60 vectorNormalize_warp](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_normalize.py:60) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_normalize.py:63 vectorNormalize_warp](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_normalize.py:63) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_normalize.py:69 vectorNorm_warp_1D](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_normalize.py:69) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_normalize.py:73 adj_vectorNorm_warp](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_normalize.py:73) | wp.func_grad | custom local adjoint | local helper or file assessment |
| [core:math/wp_normalize.py:78 vectorNorm_warp_2D](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_normalize.py:78) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_normalize.py:82 adj_vectorNorm_warp](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_normalize.py:82) | wp.func_grad | custom local adjoint | local helper or file assessment |
| [core:math/wp_normalize.py:87 vectorNorm_warp_3D](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_normalize.py:87) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_normalize.py:91 adj_vectorNorm_warp](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_normalize.py:91) | wp.func_grad | custom local adjoint | local helper or file assessment |
| [core:math/wp_normalize.py:96 vectorNorm_warp](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_normalize.py:96) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_normalize.py:99 vectorNorm_warp](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_normalize.py:99) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_normalize.py:102 vectorNorm_warp](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_normalize.py:102) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_outerTensorProduct.py:7 outerTensorProduct](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_outerTensorProduct.py:7) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_outerTensorProduct.py:25 outerTensorProduct](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_outerTensorProduct.py:25) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_outerTensorProduct.py:43 outerTensorProduct](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_outerTensorProduct.py:43) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_pow.py:6 iPow](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_pow.py:6) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_pow.py:28 cpow_warp](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_pow.py:28) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_pow.py:32 bpow_warp](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_pow.py:32) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_sqrt.py:8 safe_sqrt](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_sqrt.py:8) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_sqrt.py:11 adj_safe_sqrt](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_sqrt.py:11) | wp.func_grad | custom local adjoint | local helper or file assessment |
| [core:math/wp_zero.py:10 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:10) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:15 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:15) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:20 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:20) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:25 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:25) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:30 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:30) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:35 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:35) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:40 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:40) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:45 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:45) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:50 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:50) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:56 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:56) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:61 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:61) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:66 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:66) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:72 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:72) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:77 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:77) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:83 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:83) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:88 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:88) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:93 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:93) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:100 zero_like_warp](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:100) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:108 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:108) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:113 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:113) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:118 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:118) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:123 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:123) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:129 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:129) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:134 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:134) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:139 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:139) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:145 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:145) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:150 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:150) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:155 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:155) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:162 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:162) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:167 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:167) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:172 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:172) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:177 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:177) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:183 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:183) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:188 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:188) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:193 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:193) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:199 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:199) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:204 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:204) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:math/wp_zero.py:209 zero_like](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/math/wp_zero.py:209) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:mfm/multigrid.py:68 _segmentSum_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/multigrid.py:68) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | multigrid segment sum |
| [core:mfm/wp_closure.py:24 mfmClosureMatvecCSR_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_closure.py:24) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | MFM closure CSR matvec |
| [core:mfm/wp_mfm.py:65 faceVector](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:65) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:mfm/wp_mfm.py:71 kernelDerivativeValue](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:71) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:mfm/wp_mfm.py:80 guardedFace](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:80) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:mfm/wp_mfm.py:124 alphaScalar](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:124) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:mfm/wp_mfm.py:140 _sgn](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:140) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:mfm/wp_mfm.py:149 pairLimit](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:149) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:mfm/wp_mfm.py:172 _speedsGaburov](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:172) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:mfm/wp_mfm.py:178 _speedsRoe](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:178) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:mfm/wp_mfm.py:194 _speedsPvrs](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:194) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:mfm/wp_mfm.py:209 _starFromSpeeds](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:209) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:mfm/wp_mfm.py:219 _okStar](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:219) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:mfm/wp_mfm.py:224 hllcStar](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:224) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:mfm/wp_mfm.py:246 _hllcSide](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:246) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:mfm/wp_mfm.py:267 faceFluxPair](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:267) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:mfm/wp_mfm.py:316 mfmMoments_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:316) | wp.func | range/candidate accumulation or edge emission | MFM moments |
| [core:mfm/wp_mfm.py:342 mfmMoments_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:342) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | MFM moments |
| [core:mfm/wp_mfm.py:376 kernel_self](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:376) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:mfm/wp_mfm.py:393 mfmGradients_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:393) | wp.func | range/candidate accumulation or edge emission | MFM gradients |
| [core:mfm/wp_mfm.py:420 mfmGradients_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:420) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | MFM gradients |
| [core:mfm/wp_mfm.py:474 mfmLimiter_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:474) | wp.func | range/candidate accumulation or edge emission | MFM limiter |
| [core:mfm/wp_mfm.py:515 mfmLimiter_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:515) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | MFM limiter |
| [core:mfm/wp_mfm.py:577 pairFace](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:577) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:mfm/wp_mfm.py:587 mfmClosureResidual_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:587) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | MFM closure residual |
| [core:mfm/wp_mfm.py:625 mfmClosureWeights_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:625) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | MFM closure weights |
| [core:mfm/wp_mfm.py:657 mfmClosureMatvec_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:657) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | MFM closure matvec |
| [core:mfm/wp_mfm.py:713 sideTimeDerivative](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:713) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:mfm/wp_mfm.py:726 reconstructSide](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:726) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:mfm/wp_mfm.py:765 mfmFlux_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:765) | wp.func | range/candidate accumulation or edge emission | MFM flux |
| [core:mfm/wp_mfm.py:920 mfmFlux_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:920) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | MFM flux |
| [core:mfm/wp_mfm.py:982 _wrapDiff](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:982) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:mfm/wp_mfm.py:992 mfmCoarseStencil_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:992) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | MFM coarse stencil |
| [core:mfm/wp_mfm.py:1064 mfmTimestep_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/mfm/wp_mfm.py:1064) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | MFM timestep |
| [core:pinv/wp_pinv1x1.py:39 pseudoInverse1x1](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/pinv/wp_pinv1x1.py:39) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:pinv/wp_pinv1x1.py:49 pseudoInverse1x1Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/pinv/wp_pinv1x1.py:49) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [core:pinv/wp_pinv2x2.py:79 matmul2](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/pinv/wp_pinv2x2.py:79) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:pinv/wp_pinv2x2.py:91 pinv2x2_warp](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/pinv/wp_pinv2x2.py:91) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [core:polyfit/wp_polyfit.py:53 polyIntPow](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/polyfit/wp_polyfit.py:53) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:polyfit/wp_polyfit.py:62 polyMonomial](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/polyfit/wp_polyfit.py:62) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:polyfit/wp_polyfit.py:80 polyInSector](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/polyfit/wp_polyfit.py:80) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:polyfit/wp_polyfit.py:101 polyFitWeight](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/polyfit/wp_polyfit.py:101) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:polyfit/wp_polyfit.py:135 computePolyMoments_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/polyfit/wp_polyfit.py:135) | wp.func | range/candidate accumulation or edge emission | polynomial moments |
| [core:polyfit/wp_polyfit.py:169 computePolyMoments_Func_Adjacency](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/polyfit/wp_polyfit.py:169) | wp.func | adjacency/range orchestration | polynomial moments |
| [core:polyfit/wp_polyfit.py:205 computePolyMoments_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/polyfit/wp_polyfit.py:205) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | polynomial moments |
| [core:polyfit/wp_polyfit.py:288 computePolyRHS_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/polyfit/wp_polyfit.py:288) | wp.func | range/candidate accumulation or edge emission | polynomial RHS |
| [core:polyfit/wp_polyfit.py:318 computePolyRHS_Func_Adjacency](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/polyfit/wp_polyfit.py:318) | wp.func | adjacency/range orchestration | polynomial RHS |
| [core:polyfit/wp_polyfit.py:356 computePolyRHS_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/polyfit/wp_polyfit.py:356) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | polynomial RHS |
| [core:polyfit/wp_polyfitJVP.py:45 polyIntPowD](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/polyfit/wp_polyfitJVP.py:45) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:polyfit/wp_polyfitJVP.py:53 polyMonomialJVP](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/polyfit/wp_polyfitJVP.py:53) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:polyfit/wp_polyfitJVP.py:71 polyFitWeightJVP](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/polyfit/wp_polyfitJVP.py:71) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:polyfit/wp_polyfitJVP.py:115 computePolyMomentsJVP_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/polyfit/wp_polyfitJVP.py:115) | wp.func | range/candidate accumulation or edge emission | polynomial moments JVP |
| [core:polyfit/wp_polyfitJVP.py:145 computePolyMomentsJVP_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/polyfit/wp_polyfitJVP.py:145) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | polynomial moments JVP |
| [core:polyfit/wp_polyfitJVP.py:193 computePolyRHSJVP_Func_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/polyfit/wp_polyfitJVP.py:193) | wp.func | range/candidate accumulation or edge emission | polynomial RHS JVP |
| [core:polyfit/wp_polyfitJVP.py:228 computePolyRHSJVP_Kernel](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/polyfit/wp_polyfitJVP.py:228) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | polynomial RHS JVP |
| [core:radiusSearch/compactHash/indexing.py:15 computeZOrderIndex](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/indexing.py:15) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:radiusSearch/compactHash/indexing.py:32 getLinearIndex64](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/indexing.py:32) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:radiusSearch/compactHash/indexing.py:83 getLinearIndex](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/indexing.py:83) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:radiusSearch/compactHash/morton.py:14 mortonPattern](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/morton.py:14) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:radiusSearch/compactHash/morton.py:76 splitBy3Bits64](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/morton.py:76) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:radiusSearch/compactHash/morton.py:85 computeZOrderIndex64](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/morton.py:85) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:radiusSearch/compactHash/wp_collectNeighbors.py:239 fillCSRRows](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/wp_collectNeighbors.py:239) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | CSR row expansion |
| [core:radiusSearch/compactHash/wp_collectNeighbors.py:22 collect](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/wp_collectNeighbors.py:22) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | compact-hash collect |
| [core:radiusSearch/compactHash/wp_countNeighbors.py:16 _countNeighborsCompactHashMapRow](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/wp_countNeighbors.py:16) | wp.func | range/candidate accumulation or edge emission | compact-hash count |
| [core:radiusSearch/compactHash/wp_countNeighbors.py:114 radiusSearchCountNeighborsCompactHashMap](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/wp_countNeighbors.py:114) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | compact-hash count |
| [core:radiusSearch/compactHash/wp_countNeighbors.py:118 radiusSearchCountNeighborsSpatial](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/wp_countNeighbors.py:118) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | compact-hash count |
| [core:radiusSearch/compactHash/wp_hashCells.py:26 hashGridIndex](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/wp_hashCells.py:26) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:radiusSearch/compactHash/wp_hashCells.py:37 hashGridVec3i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/wp_hashCells.py:37) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:radiusSearch/compactHash/wp_hashCells.py:51 hashCells](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/wp_hashCells.py:51) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [core:radiusSearch/compactHash/wp_index.py:16 clampCellIndex](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/wp_index.py:16) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:radiusSearch/compactHash/wp_index.py:25 wrapCellComponentPeriodic](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/wp_index.py:25) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:radiusSearch/compactHash/wp_index.py:33 indexCells](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/wp_index.py:33) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [core:radiusSearch/grid_util.py:15 wrapCellComponentPeriodic](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/grid_util.py:15) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:radiusSearch/grid_util.py:22 iterateCell](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/grid_util.py:22) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:radiusSearch/grid_util.py:40 checkOffset](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/grid_util.py:40) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:radiusSearch/grid_util.py:149 getIndexRange](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/grid_util.py:149) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:radiusSearch/grid_util.py:172 laneSubRange](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/grid_util.py:172) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:radiusSearch/grid_util.py:186 laneSlice](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/grid_util.py:186) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:radiusSearch/grid_util.py:198 getIndexRangeLane](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/grid_util.py:198) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:radiusSearch/grid_util.py:225 laneSum](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/grid_util.py:225) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:radiusSearch/small/wp_radius_small.py:10 warp_radius_search_kernel_direct_2](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/small/wp_radius_small.py:10) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | direct radius count |
| [core:radiusSearch/small/wp_radius_small.py:55 warp_radius_search_collect_kernel_direct_2](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/small/wp_radius_small.py:55) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | direct radius collect |
| [core:radiusSearch/verlet/wp_countVerlet.py:15 countNeighborsVerletFunc](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/verlet/wp_countVerlet.py:15) | wp.func | range/candidate accumulation or edge emission | Verlet count |
| [core:radiusSearch/verlet/wp_countVerlet.py:62 countNeighborsVerlet](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/verlet/wp_countVerlet.py:62) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | Verlet count |
| [core:radiusSearch/verlet/wp_updateVerlet.py:14 updateNeighborsVerletFunc](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/verlet/wp_updateVerlet.py:14) | wp.func | range/candidate accumulation or edge emission | Verlet collect |
| [core:radiusSearch/verlet/wp_updateVerlet.py:74 updateNeighborsVerlet](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/verlet/wp_updateVerlet.py:74) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | Verlet collect |
| [core:radiusSearch/verlet/wp_validity.py:30 _validityReduce](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/verlet/wp_validity.py:30) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | Verlet validity |
| [core:radiusSearch/verlet/wp_validity.py:57 _validityDecide](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/radiusSearch/verlet/wp_validity.py:57) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | Verlet validity |
| [core:type_config.py:187 scalar](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/type_config.py:187) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/directionality.py:4 checkDirectionality_j](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/directionality.py:4) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/directionality.py:41 checkDirectionality_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/directionality.py:41) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/directionality.py:78 checkDirectionality_Func](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/directionality.py:78) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:11 getParticle](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:11) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:18 getL_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:18) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:28 getVolume_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:28) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:34 getVolume_j](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:34) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:42 getGradH_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:42) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:50 getGradH_j](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:50) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:58 getCRK_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:58) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:67 getCRK_j](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:67) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:82 getRenormTangent_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:82) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:92 getVolumeTangent_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:92) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:101 getVolumeTangent_j](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:101) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:110 getGradHTangent_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:110) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:119 getGradHTangent_j](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:119) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:128 getCRKTangent_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:128) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:138 getCRKTangent_j](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:138) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:150 getParticleData](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:150) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:157 getParticleCorrectionData_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:157) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:172 getParticleCorrectionData_j](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:172) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:188 getParticleCorrectionTangentData_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:188) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:205 getParticleData](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:205) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:212 getParticleCorrectionData_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:212) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:227 getParticleCorrectionData_j](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:227) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:243 getParticleCorrectionTangentData_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:243) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:260 getParticleData](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:260) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:267 getParticleCorrectionData_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:267) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:282 getParticleCorrectionData_j](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:282) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:298 getParticleCorrectionTangentData_i](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:298) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:322 access_optional](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:322) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/stateUtil.py:329 ternary_helper](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/stateUtil.py:329) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/support.py:91 volumeToSupport_warp](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/support.py:91) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/support.py:105 computePairwiseSupport](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/support.py:105) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/support.py:117 computePairwiseSupportJVP](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/support.py:117) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [core:util/support.py:136 isInSupport](/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore/src/warpSPHCore/util/support.py:136) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/adaptiveSupport/owenHermite.py:21 _sample_gradient](/home/lu26029/dev/warpSPH/src/warpSPH/modules/adaptiveSupport/owenHermite.py:21) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [frontend:modules/adaptiveSupport/owenHermite.py:27 quadratic_gradient](/home/lu26029/dev/warpSPH/src/warpSPH/modules/adaptiveSupport/owenHermite.py:27) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/adaptiveSupport/wp_materialPsi.py:21 _sums](/home/lu26029/dev/warpSPH/src/warpSPH/modules/adaptiveSupport/wp_materialPsi.py:21) | wp.func | range/candidate accumulation or edge emission | material support moments |
| [frontend:modules/adaptiveSupport/wp_materialPsi.py:55 _kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/adaptiveSupport/wp_materialPsi.py:55) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | material support moments |
| [frontend:modules/adaptiveSupport/wp_materialPsi.py:67 _tiled](/home/lu26029/dev/warpSPH/src/warpSPH/modules/adaptiveSupport/wp_materialPsi.py:67) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | material support moments |
| [frontend:modules/adaptiveSupport/wp_omega.py:25 computeOmega_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/adaptiveSupport/wp_omega.py:25) | wp.func | range/candidate accumulation or edge emission | grad-h omega |
| [frontend:modules/adaptiveSupport/wp_omega.py:96 computeOmega_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/adaptiveSupport/wp_omega.py:96) | wp.func | adjacency/range orchestration | grad-h omega |
| [frontend:modules/adaptiveSupport/wp_omega.py:163 computeOmega_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/adaptiveSupport/wp_omega.py:163) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | grad-h omega |
| [frontend:modules/adaptiveSupport/wp_psi.py:22 computePsi](/home/lu26029/dev/warpSPH/src/warpSPH/modules/adaptiveSupport/wp_psi.py:22) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | support lookup construction |
| [frontend:modules/adaptiveSupport/wp_psi0.py:24 computePsi0_Func_i_raw](/home/lu26029/dev/warpSPH/src/warpSPH/modules/adaptiveSupport/wp_psi0.py:24) | wp.func | range/candidate accumulation or edge emission | support psi moments |
| [frontend:modules/adaptiveSupport/wp_psi0.py:119 computePsi0_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/adaptiveSupport/wp_psi0.py:119) | wp.func | range/candidate accumulation or edge emission | support psi moments |
| [frontend:modules/adaptiveSupport/wp_psi0.py:168 computePsi0_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/adaptiveSupport/wp_psi0.py:168) | wp.func | adjacency/range orchestration | support psi moments |
| [frontend:modules/adaptiveSupport/wp_psi0.py:195 computePsi0_Func_Adjacency_raw](/home/lu26029/dev/warpSPH/src/warpSPH/modules/adaptiveSupport/wp_psi0.py:195) | wp.func | adjacency/range orchestration | support psi moments |
| [frontend:modules/adaptiveSupport/wp_psi0.py:265 computePsi0_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/adaptiveSupport/wp_psi0.py:265) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | support psi moments |
| [frontend:modules/adaptiveSupport/wp_psi0.py:296 computePsi0_KernelTiled](/home/lu26029/dev/warpSPH/src/warpSPH/modules/adaptiveSupport/wp_psi0.py:296) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | support psi moments |
| [frontend:modules/artificialCompressible/dyngpusphOperators.py:35 _closures](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/dyngpusphOperators.py:35) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | dynGPUSPH closures |
| [frontend:modules/artificialCompressible/dyngpusphOperators.py:122 _moments](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/dyngpusphOperators.py:122) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | dynGPUSPH moments |
| [frontend:modules/artificialCompressible/dyngpusphOperators.py:196 _residual](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/dyngpusphOperators.py:196) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | dynGPUSPH residual |
| [frontend:modules/artificialCompressible/dyngpusphShifting.py:23 _unit](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/dyngpusphShifting.py:23) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/artificialCompressible/dyngpusphShifting.py:32 source_cell_pair](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/dyngpusphShifting.py:32) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/artificialCompressible/dyngpusphShifting.py:44 _kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/dyngpusphShifting.py:44) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | dynGPUSPH shift |
| [frontend:modules/artificialCompressible/dyngpusphShifting.py:239 _surface_fields](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/dyngpusphShifting.py:239) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | dynGPUSPH surface fields |
| [frontend:modules/artificialCompressible/dyngpusphShifting.py:294 _surface_cone](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/dyngpusphShifting.py:294) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | dynGPUSPH cone |
| [frontend:modules/artificialCompressible/dyngpusphShifting.py:343 _surface_vicinity](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/dyngpusphShifting.py:343) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | dynGPUSPH vicinity |
| [frontend:modules/artificialCompressible/wp_jstSwitch.py:47 computeJstSwitch_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/wp_jstSwitch.py:47) | wp.func | range/candidate accumulation or edge emission | JST switch |
| [frontend:modules/artificialCompressible/wp_jstSwitch.py:96 computeJstSwitch_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/wp_jstSwitch.py:96) | wp.func | adjacency/range orchestration | JST switch |
| [frontend:modules/artificialCompressible/wp_jstSwitch.py:157 computeJstSwitch_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/wp_jstSwitch.py:157) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | JST switch |
| [frontend:modules/artificialCompressible/wp_residual.py:18 _sums](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/wp_residual.py:18) | wp.func | range/candidate accumulation or edge emission | artificial-compressible residual |
| [frontend:modules/artificialCompressible/wp_residual.py:92 _kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/wp_residual.py:92) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | artificial-compressible residual |
| [frontend:modules/artificialCompressible/wp_residual.py:111 _tiled](/home/lu26029/dev/warpSPH/src/warpSPH/modules/artificialCompressible/wp_residual.py:111) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | artificial-compressible residual |
| [frontend:modules/compSPH/accel.py:31 computeCompSPHPair](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/accel.py:31) | wp.func | local math, pair evaluation, loading or indexing helper | compressible acceleration |
| [frontend:modules/compSPH/accel.py:135 computeCompSPHAccel_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/accel.py:135) | wp.func | range/candidate accumulation or edge emission | compressible acceleration |
| [frontend:modules/compSPH/accel.py:207 computeCompSPHAccel_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/accel.py:207) | wp.func | adjacency/range orchestration | compressible acceleration |
| [frontend:modules/compSPH/accel.py:300 computeCompSPHAccel_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/accel.py:300) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | compressible acceleration |
| [frontend:modules/compSPH/accel.py:347 computeCompSPHAccel_KernelTiled](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/accel.py:347) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | compressible acceleration |
| [frontend:modules/compSPH/balance.py:30 sgn](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/balance.py:30) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/compSPH/balance.py:34 computeCompSPHBalanceTerm_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/balance.py:34) | wp.func | range/candidate accumulation or edge emission | compressible balance fractions |
| [frontend:modules/compSPH/balance.py:180 computeCompSPHBalanceTerm_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/balance.py:180) | wp.func | adjacency/range orchestration | compressible balance fractions |
| [frontend:modules/compSPH/balance.py:266 computeCompSPHBalanceTerm_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/balance.py:266) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | compressible balance fractions |
| [frontend:modules/compSPH/balance.py:310 computeCompSPHBalanceTerm_KernelTiled](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/balance.py:310) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | compressible balance fractions |
| [frontend:modules/compSPH/dudt.py:28 computeCompSPHdudt_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/dudt.py:28) | wp.func | range/candidate accumulation or edge emission | compressible thermal rate |
| [frontend:modules/compSPH/dudt.py:158 computeCompSPHdudt_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/dudt.py:158) | wp.func | adjacency/range orchestration | compressible thermal rate |
| [frontend:modules/compSPH/dudt.py:243 computeCompSPHdudt_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/dudt.py:243) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | compressible thermal rate |
| [frontend:modules/compSPH/dudt.py:284 computeCompSPHdudt_KernelTiled](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/dudt.py:284) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | compressible thermal rate |
| [frontend:modules/compSPH/work.py:32 _compWorkKernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/work.py:32) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | compressible stage work |
| [frontend:modules/compSPH/workReduction.py:14 _pairDot](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/workReduction.py:14) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/compSPH/workReduction.py:23 _pairWorkRows](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/workReduction.py:23) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | retained pair work reduction |
| [frontend:modules/compSPH/workReduction.py:52 _referenceEntropyRows](/home/lu26029/dev/warpSPH/src/warpSPH/modules/compSPH/workReduction.py:52) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | retained pair work reduction |
| [frontend:modules/crk/accel.py:32 computeCrkSPHPair](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/accel.py:32) | wp.func | local math, pair evaluation, loading or indexing helper | CRK acceleration |
| [frontend:modules/crk/accel.py:158 computeCrkSPHAccel_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/accel.py:158) | wp.func | range/candidate accumulation or edge emission | CRK acceleration |
| [frontend:modules/crk/accel.py:241 computeCrkSPHAccel_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/accel.py:241) | wp.func | adjacency/range orchestration | CRK acceleration |
| [frontend:modules/crk/accel.py:340 computeCrkSPHAccel_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/accel.py:340) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | CRK acceleration |
| [frontend:modules/crk/accel.py:392 computeCrkSPHAccel_KernelTiled](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/accel.py:392) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | CRK acceleration |
| [frontend:modules/crk/dudt.py:36 computeCrkSPHdudt_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/dudt.py:36) | wp.func | range/candidate accumulation or edge emission | CRK thermal rate |
| [frontend:modules/crk/dudt.py:198 computeCrkSPHdudt_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/dudt.py:198) | wp.func | adjacency/range orchestration | CRK thermal rate |
| [frontend:modules/crk/dudt.py:289 computeCrkSPHdudt_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/dudt.py:289) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | CRK thermal rate |
| [frontend:modules/crk/dudt.py:334 computeCrkSPHdudt_KernelTiled](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/dudt.py:334) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | CRK thermal rate |
| [frontend:modules/crk/materialDensity.py:28 _materialDensitySums](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/materialDensity.py:28) | wp.func | range/candidate accumulation or edge emission | material CRK density |
| [frontend:modules/crk/materialDensity.py:62 _materialDensityKernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/materialDensity.py:62) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | material CRK density |
| [frontend:modules/crk/materialDensity.py:75 _materialDensityTiled](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/materialDensity.py:75) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | material CRK density |
| [frontend:modules/crk/partition.py:7 crkEntropyFraction](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/partition.py:7) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/crk/terms.py:33 _det](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/terms.py:33) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/crk/terms.py:43 _crkTermsKernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/terms.py:43) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | CRK coefficient solve |
| [frontend:modules/crk/work.py:19 _crkWorkKernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/crk/work.py:19) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | CRK stage work |
| [frontend:modules/deltaSPH/updates.py:16 _assemble](/home/lu26029/dev/warpSPH/src/warpSPH/modules/deltaSPH/updates.py:16) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [frontend:modules/deltaSPH/updates.py:35 _fluidMask](/home/lu26029/dev/warpSPH/src/warpSPH/modules/deltaSPH/updates.py:35) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [frontend:modules/deltaSPH/wp_densityDelta.py:110 computeDensityDiffusionDeltaSPH_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/deltaSPH/wp_densityDelta.py:110) | wp.func | range/candidate accumulation or edge emission | delta density diffusion |
| [frontend:modules/deltaSPH/wp_densityDelta.py:276 computeDensityDiffusionDeltaSPH_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/deltaSPH/wp_densityDelta.py:276) | wp.func | adjacency/range orchestration | delta density diffusion |
| [frontend:modules/deltaSPH/wp_densityDelta.py:370 computeDensityDiffusionDeltaSPH_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/deltaSPH/wp_densityDelta.py:370) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | delta density diffusion |
| [frontend:modules/deltaSPH/wp_densityDelta.py:411 computeDensityDiffusionDeltaSPH_KernelTiled](/home/lu26029/dev/warpSPH/src/warpSPH/modules/deltaSPH/wp_densityDelta.py:411) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | delta density diffusion |
| [frontend:modules/deltaSPH/wp_viscosityDelta.py:58 computeVelocityDiffusionDeltaSPH_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/deltaSPH/wp_viscosityDelta.py:58) | wp.func | range/candidate accumulation or edge emission | delta viscosity |
| [frontend:modules/deltaSPH/wp_viscosityDelta.py:177 computeVelocityDiffusionDeltaSPH_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/deltaSPH/wp_viscosityDelta.py:177) | wp.func | adjacency/range orchestration | delta viscosity |
| [frontend:modules/deltaSPH/wp_viscosityDelta.py:259 computeVelocityDiffusionDeltaSPH_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/deltaSPH/wp_viscosityDelta.py:259) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | delta viscosity |
| [frontend:modules/deltaSPH/wp_viscosityDelta.py:297 computeVelocityDiffusionDeltaSPH_KernelTiled](/home/lu26029/dev/warpSPH/src/warpSPH/modules/deltaSPH/wp_viscosityDelta.py:297) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | delta viscosity |
| [frontend:modules/dissipation/pi/coefficients.py:11 switchedCoefficients](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/coefficients.py:11) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/dissipation/pi/dispatch.py:18 computePi_term](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/dispatch.py:18) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/dissipation/pi/dispatch.py:56 computePi_pair](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/dispatch.py:56) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/dissipation/pi/dispatch.py:86 computePi_actual](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/dispatch.py:86) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/dissipation/pi/oneSided.py:14 computeFrontiereQ](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/oneSided.py:14) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/dissipation/pi/pair.py:59 pick](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/pair.py:59) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/dissipation/pi/pair.py:71 buildPair](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/pair.py:71) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/dissipation/pi/terms.py:27 _val](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/terms.py:27) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/dissipation/pi/terms.py:32 _muRaw](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/terms.py:32) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/dissipation/pi/terms.py:38 _scalingFactor](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/terms.py:38) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/dissipation/pi/terms.py:44 _muScaled](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/terms.py:44) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/dissipation/pi/terms.py:51 monaghanGingold1983](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/terms.py:51) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/dissipation/pi/terms.py:57 monaghan1992](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/terms.py:57) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/dissipation/pi/terms.py:66 wadsley2008](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/terms.py:66) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/dissipation/pi/terms.py:73 deltaSPH](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/terms.py:73) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/dissipation/pi/terms.py:82 defaultTerm](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/terms.py:82) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/dissipation/pi/terms.py:91 monaghan1997a](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/terms.py:91) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/dissipation/pi/terms.py:98 price2012Eq98Label](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/terms.py:98) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/dissipation/pi/terms.py:105 price2012](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/terms.py:105) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/dissipation/pi/terms.py:112 monaghan1997b](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/terms.py:112) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/dissipation/pi/terms.py:121 dukowicz](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/terms.py:121) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/dissipation/pi/terms.py:130 cleary1998](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/terms.py:130) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/dissipation/pi/terms.py:145 price2008](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/terms.py:145) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/dissipation/pi/terms.py:159 hopkins2015Conductivity](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/terms.py:159) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/dissipation/pi/terms.py:177 frontiere2017](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/terms.py:177) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/dissipation/pi/terms.py:203 riemannDissipation](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/terms.py:203) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/dissipation/pi/terms.py:245 evaluateTerm](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/pi/terms.py:245) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/dissipation/wp_conductivity.py:28 computeConductivity_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/wp_conductivity.py:28) | wp.func | range/candidate accumulation or edge emission | thermal conductivity |
| [frontend:modules/dissipation/wp_conductivity.py:135 computeConductivity_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/wp_conductivity.py:135) | wp.func | adjacency/range orchestration | thermal conductivity |
| [frontend:modules/dissipation/wp_conductivity.py:221 computeConductivity_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/wp_conductivity.py:221) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | thermal conductivity |
| [frontend:modules/dissipation/wp_diffusion.py:28 computeViscosity_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/wp_diffusion.py:28) | wp.func | range/candidate accumulation or edge emission | viscosity diffusion |
| [frontend:modules/dissipation/wp_diffusion.py:158 computeViscosity_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/wp_diffusion.py:158) | wp.func | adjacency/range orchestration | viscosity diffusion |
| [frontend:modules/dissipation/wp_diffusion.py:255 computeViscosity_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/wp_diffusion.py:255) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | viscosity diffusion |
| [frontend:modules/dissipation/wp_dissipation.py:29 computeThermalDissipation_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/wp_dissipation.py:29) | wp.func | range/candidate accumulation or edge emission | thermal dissipation |
| [frontend:modules/dissipation/wp_dissipation.py:178 computeThermalDissipation_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/wp_dissipation.py:178) | wp.func | adjacency/range orchestration | thermal dissipation |
| [frontend:modules/dissipation/wp_dissipation.py:278 computeThermalDissipation_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/dissipation/wp_dissipation.py:278) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | thermal dissipation |
| [frontend:modules/eos/conserved.py:18 _primitives](/home/lu26029/dev/warpSPH/src/warpSPH/modules/eos/conserved.py:18) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [frontend:modules/eos/materials.py:124 _evaluate](/home/lu26029/dev/warpSPH/src/warpSPH/modules/eos/materials.py:124) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [frontend:modules/godunov/wp_gsph.py:42 godunovStates](/home/lu26029/dev/warpSPH/src/warpSPH/modules/godunov/wp_gsph.py:42) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/godunov/wp_gsph.py:92 godunov_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/godunov/wp_gsph.py:92) | wp.func | range/candidate accumulation or edge emission | Godunov rates |
| [frontend:modules/godunov/wp_gsph.py:141 godunov_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/godunov/wp_gsph.py:141) | wp.func | adjacency/range orchestration | Godunov rates |
| [frontend:modules/godunov/wp_gsph.py:202 computeGodunov_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/godunov/wp_gsph.py:202) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | Godunov rates |
| [frontend:modules/godunov/wp_inutsuka.py:49 gaussian](/home/lu26029/dev/warpSPH/src/warpSPH/modules/godunov/wp_inutsuka.py:49) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/godunov/wp_inutsuka.py:57 gaussianDensity_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/godunov/wp_inutsuka.py:57) | wp.func | adjacency/range orchestration | Gaussian density/gradient |
| [frontend:modules/godunov/wp_inutsuka.py:116 gaussianDensity_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/godunov/wp_inutsuka.py:116) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | Gaussian density/gradient |
| [frontend:modules/godunov/wp_inutsuka.py:176 inutsukaVolumes](/home/lu26029/dev/warpSPH/src/warpSPH/modules/godunov/wp_inutsuka.py:176) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/godunov/wp_inutsuka.py:202 inutsuka_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/godunov/wp_inutsuka.py:202) | wp.func | adjacency/range orchestration | Inutsuka rates |
| [frontend:modules/godunov/wp_inutsuka.py:325 computeInutsuka_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/godunov/wp_inutsuka.py:325) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | Inutsuka rates |
| [frontend:modules/iisph/wp_contactPressureDiagonal.py:18 computeIISPHContactPressureDiagonal_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_contactPressureDiagonal.py:18) | wp.func | range/candidate accumulation or edge emission | IISPH contact pressure diagonal |
| [frontend:modules/iisph/wp_contactPressureDiagonal.py:47 computeIISPHContactPressureDiagonal_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_contactPressureDiagonal.py:47) | wp.func | adjacency/range orchestration | IISPH contact pressure diagonal |
| [frontend:modules/iisph/wp_contactPressureDiagonal.py:89 computeIISPHContactPressureDiagonal_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_contactPressureDiagonal.py:89) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | IISPH contact pressure diagonal |
| [frontend:modules/iisph/wp_densityRate.py:18 computeIISPHDensityRate_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_densityRate.py:18) | wp.func | range/candidate accumulation or edge emission | IISPH density rate |
| [frontend:modules/iisph/wp_densityRate.py:42 computeIISPHDensityRate_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_densityRate.py:42) | wp.func | adjacency/range orchestration | IISPH density rate |
| [frontend:modules/iisph/wp_densityRate.py:82 computeIISPHDensityRate_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_densityRate.py:82) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | IISPH density rate |
| [frontend:modules/iisph/wp_gradientSum.py:18 computeIISPHGradientSum_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_gradientSum.py:18) | wp.func | range/candidate accumulation or edge emission | IISPH gradient sum |
| [frontend:modules/iisph/wp_gradientSum.py:41 computeIISPHGradientSum_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_gradientSum.py:41) | wp.func | adjacency/range orchestration | IISPH gradient sum |
| [frontend:modules/iisph/wp_gradientSum.py:78 computeIISPHGradientSum_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_gradientSum.py:78) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | IISPH gradient sum |
| [frontend:modules/iisph/wp_pressureAcceleration.py:20 computeIISPHPressureAcceleration_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_pressureAcceleration.py:20) | wp.func | range/candidate accumulation or edge emission | IISPH pressure acceleration |
| [frontend:modules/iisph/wp_pressureAcceleration.py:46 computeIISPHPressureAcceleration_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_pressureAcceleration.py:46) | wp.func | adjacency/range orchestration | IISPH pressure acceleration |
| [frontend:modules/iisph/wp_pressureAcceleration.py:90 computeIISPHPressureAcceleration_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_pressureAcceleration.py:90) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | IISPH pressure acceleration |
| [frontend:modules/iisph/wp_pressureDiagonal.py:18 computeIISPHPressureDiagonal_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_pressureDiagonal.py:18) | wp.func | range/candidate accumulation or edge emission | IISPH pressure diagonal |
| [frontend:modules/iisph/wp_pressureDiagonal.py:41 computeIISPHPressureDiagonal_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_pressureDiagonal.py:41) | wp.func | adjacency/range orchestration | IISPH pressure diagonal |
| [frontend:modules/iisph/wp_pressureDiagonal.py:81 computeIISPHPressureDiagonal_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_pressureDiagonal.py:81) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | IISPH pressure diagonal |
| [frontend:modules/iisph/wp_sampledGravity.py:18 computeIISPHSampledGravity_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_sampledGravity.py:18) | wp.func | range/candidate accumulation or edge emission | IISPH sampled gravity |
| [frontend:modules/iisph/wp_sampledGravity.py:44 computeIISPHSampledGravity_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_sampledGravity.py:44) | wp.func | adjacency/range orchestration | IISPH sampled gravity |
| [frontend:modules/iisph/wp_sampledGravity.py:84 computeIISPHSampledGravity_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_sampledGravity.py:84) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | IISPH sampled gravity |
| [frontend:modules/iisph/wp_wallGradient.py:18 computeIISPHWallGradient_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_wallGradient.py:18) | wp.func | range/candidate accumulation or edge emission | IISPH wall gradient |
| [frontend:modules/iisph/wp_wallGradient.py:41 computeIISPHWallGradient_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_wallGradient.py:41) | wp.func | adjacency/range orchestration | IISPH wall gradient |
| [frontend:modules/iisph/wp_wallGradient.py:77 computeIISPHWallGradient_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/iisph/wp_wallGradient.py:77) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | IISPH wall gradient |
| [frontend:modules/incompressible/wp_alpha.py:49 computeAlpha_Func_i_first](/home/lu26029/dev/warpSPH/src/warpSPH/modules/incompressible/wp_alpha.py:49) | wp.func | range/candidate accumulation or edge emission | incompressible alpha |
| [frontend:modules/incompressible/wp_alpha.py:144 computeAlpha_Func_Adjacency_first](/home/lu26029/dev/warpSPH/src/warpSPH/modules/incompressible/wp_alpha.py:144) | wp.func | adjacency/range orchestration | incompressible alpha |
| [frontend:modules/incompressible/wp_alpha.py:215 computeAlpha_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/incompressible/wp_alpha.py:215) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | incompressible alpha |
| [frontend:modules/incompressible/wp_wallMoment.py:45 computeWallMoment_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/incompressible/wp_wallMoment.py:45) | wp.func | range/candidate accumulation or edge emission | wall moment |
| [frontend:modules/incompressible/wp_wallMoment.py:99 computeWallMoment_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/incompressible/wp_wallMoment.py:99) | wp.func | adjacency/range orchestration | wall moment |
| [frontend:modules/incompressible/wp_wallMoment.py:167 computeWallMoment_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/incompressible/wp_wallMoment.py:167) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | wall moment |
| [frontend:modules/incompressible/wp_wallMoment.py:194 computeWallMoment_KernelTiled](/home/lu26029/dev/warpSPH/src/warpSPH/modules/incompressible/wp_wallMoment.py:194) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | wall moment |
| [frontend:modules/liu/wp_mat.py:27 computeLiuMatrices_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/liu/wp_mat.py:27) | wp.func | range/candidate accumulation or edge emission | Liu matrices |
| [frontend:modules/liu/wp_mat.py:157 computeLiuMatrices_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/liu/wp_mat.py:157) | wp.func | adjacency/range orchestration | Liu matrices |
| [frontend:modules/liu/wp_mat.py:243 computeLiuMatrices_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/liu/wp_mat.py:243) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | Liu matrices |
| [frontend:modules/liu/wp_mat.py:289 computeLiuMatrices_KernelTiled](/home/lu26029/dev/warpSPH/src/warpSPH/modules/liu/wp_mat.py:289) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | Liu matrices |
| [frontend:modules/mdbc/wp_nopenshift.py:210 computeMdbcNoPenShift_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/mdbc/wp_nopenshift.py:210) | wp.func | range/candidate accumulation or edge emission | mDBC no-penetration shift |
| [frontend:modules/mdbc/wp_nopenshift.py:348 computeMdbcNoPenShift_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/mdbc/wp_nopenshift.py:348) | wp.func | adjacency/range orchestration | mDBC no-penetration shift |
| [frontend:modules/mdbc/wp_nopenshift.py:430 computeMdbcNoPenShift_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/mdbc/wp_nopenshift.py:430) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | mDBC no-penetration shift |
| [frontend:modules/mdbc/wp_nopenshift.py:470 computeMdbcNoPenShift_KernelTiled](/home/lu26029/dev/warpSPH/src/warpSPH/modules/mdbc/wp_nopenshift.py:470) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | mDBC no-penetration shift |
| [frontend:modules/meshless/flux.py:25 mfmFlux_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/meshless/flux.py:25) | wp.func | range/candidate accumulation or edge emission | frontend MFM flux |
| [frontend:modules/meshless/flux.py:188 flux](/home/lu26029/dev/warpSPH/src/warpSPH/modules/meshless/flux.py:188) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | frontend MFM flux |
| [frontend:modules/meshless/primitives.py:19 _diagnostics](/home/lu26029/dev/warpSPH/src/warpSPH/modules/meshless/primitives.py:19) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [frontend:modules/pesph/operators.py:27 _neighborRange](/home/lu26029/dev/warpSPH/src/warpSPH/modules/pesph/operators.py:27) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/pesph/operators.py:47 _pressureSums](/home/lu26029/dev/warpSPH/src/warpSPH/modules/pesph/operators.py:47) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | pressure-entropy sums |
| [frontend:modules/pesph/operators.py:89 _pressureForce](/home/lu26029/dev/warpSPH/src/warpSPH/modules/pesph/operators.py:89) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | pressure-entropy force |
| [frontend:modules/pressure/wp_perSidePressure.py:33 perSide_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/pressure/wp_perSidePressure.py:33) | wp.func | range/candidate accumulation or edge emission | per-side pressure |
| [frontend:modules/pressure/wp_perSidePressure.py:69 perSide_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/pressure/wp_perSidePressure.py:69) | wp.func | adjacency/range orchestration | per-side pressure |
| [frontend:modules/pressure/wp_perSidePressure.py:120 computePerSidePressure_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/pressure/wp_perSidePressure.py:120) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | per-side pressure |
| [frontend:modules/pressure/wp_surfaceAware.py:34 computePressureSurfaceAware_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/pressure/wp_surfaceAware.py:34) | wp.func | range/candidate accumulation or edge emission | surface-aware pressure |
| [frontend:modules/pressure/wp_surfaceAware.py:128 computePressureSurfaceAware_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/pressure/wp_surfaceAware.py:128) | wp.func | adjacency/range orchestration | surface-aware pressure |
| [frontend:modules/pressure/wp_surfaceAware.py:211 computePressureSurfaceAware_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/pressure/wp_surfaceAware.py:211) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | surface-aware pressure |
| [frontend:modules/pressure/wp_surfaceAware.py:247 computePressureSurfaceAware_KernelTiled](/home/lu26029/dev/warpSPH/src/warpSPH/modules/pressure/wp_surfaceAware.py:247) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | surface-aware pressure |
| [frontend:modules/reconstruction/diagnostics.py:32 _pairPhiKernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/diagnostics.py:32) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | local helper or file assessment |
| [frontend:modules/reconstruction/limiters.py:28 limiterVL](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/limiters.py:28) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/reconstruction/limiters.py:38 limiterPsi](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/limiters.py:38) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/reconstruction/limiters.py:64 sgn](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/limiters.py:64) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/reconstruction/limiters.py:71 computeVanLeer](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/limiters.py:71) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/reconstruction/limiters.py:150 crkLimiter](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/limiters.py:150) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/reconstruction/meshless.py:20 alphaScalar](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/meshless.py:20) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/reconstruction/meshless.py:36 _sgn](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/meshless.py:36) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/reconstruction/meshless.py:45 pairLimit](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/meshless.py:45) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/reconstruction/meshless.py:66 mfmLimiter_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/meshless.py:66) | wp.func | range/candidate accumulation or edge emission | frontend MFM limiter |
| [frontend:modules/reconstruction/meshless.py:107 mfmLimiter_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/meshless.py:107) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | frontend MFM limiter |
| [frontend:modules/reconstruction/meshless.py:164 sideTimeDerivative](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/meshless.py:164) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/reconstruction/meshless.py:177 reconstructSide](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/meshless.py:177) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/reconstruction/pairState.py:26 reconstructPairScalar](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/pairState.py:26) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/reconstruction/pairState.py:52 reconstructPairRiemannStates](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/pairState.py:52) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/reconstruction/pairState.py:68 pairRiemannStates](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/pairState.py:68) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/reconstruction/pairState.py:87 limitedIncrement](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/pairState.py:87) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/reconstruction/pairState.py:105 reconstructPairIncrements](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/pairState.py:105) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/reconstruction/pairState.py:119 reconstructPair1D](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/pairState.py:119) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/reconstruction/pairVelocity.py:29 rawPairVelocity](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/pairVelocity.py:29) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/reconstruction/pairVelocity.py:36 linearPairVelocity](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/pairVelocity.py:36) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/reconstruction/pairVelocity.py:52 limitedPairPhi](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/pairVelocity.py:52) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/reconstruction/pairVelocity.py:75 monotoneNormalJump](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/pairVelocity.py:75) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/reconstruction/pairVelocity.py:99 reconstructPairVelocity](/home/lu26029/dev/warpSPH/src/warpSPH/modules/reconstruction/pairVelocity.py:99) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/riemann/meshless.py:15 _speedsGaburov](/home/lu26029/dev/warpSPH/src/warpSPH/modules/riemann/meshless.py:15) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/riemann/meshless.py:21 _speedsRoe](/home/lu26029/dev/warpSPH/src/warpSPH/modules/riemann/meshless.py:21) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/riemann/meshless.py:40 _speedsPvrs](/home/lu26029/dev/warpSPH/src/warpSPH/modules/riemann/meshless.py:40) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/riemann/meshless.py:55 _okStar](/home/lu26029/dev/warpSPH/src/warpSPH/modules/riemann/meshless.py:55) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/riemann/meshless.py:60 hllcStarMaterial](/home/lu26029/dev/warpSPH/src/warpSPH/modules/riemann/meshless.py:60) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/riemann/meshless.py:82 _hllcSide](/home/lu26029/dev/warpSPH/src/warpSPH/modules/riemann/meshless.py:82) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/riemann/meshless.py:103 faceFluxPairMaterial](/home/lu26029/dev/warpSPH/src/warpSPH/modules/riemann/meshless.py:103) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/riemann/meshless.py:132 hllcStar](/home/lu26029/dev/warpSPH/src/warpSPH/modules/riemann/meshless.py:132) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/riemann/meshless.py:138 faceFluxPair](/home/lu26029/dev/warpSPH/src/warpSPH/modules/riemann/meshless.py:138) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/riemann/solvers.py:36 _soundSpeed](/home/lu26029/dev/warpSPH/src/warpSPH/modules/riemann/solvers.py:36) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/riemann/solvers.py:41 _acousticStar](/home/lu26029/dev/warpSPH/src/warpSPH/modules/riemann/solvers.py:41) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/riemann/solvers.py:51 _pvrsPressure](/home/lu26029/dev/warpSPH/src/warpSPH/modules/riemann/solvers.py:51) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/riemann/solvers.py:58 _trrsStar](/home/lu26029/dev/warpSPH/src/warpSPH/modules/riemann/solvers.py:58) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/riemann/solvers.py:71 _tsrsStar](/home/lu26029/dev/warpSPH/src/warpSPH/modules/riemann/solvers.py:71) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/riemann/solvers.py:85 hllcContact](/home/lu26029/dev/warpSPH/src/warpSPH/modules/riemann/solvers.py:85) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/riemann/solvers.py:96 _hllcWaveSpeeds](/home/lu26029/dev/warpSPH/src/warpSPH/modules/riemann/solvers.py:96) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/riemann/solvers.py:114 riemannStarState](/home/lu26029/dev/warpSPH/src/warpSPH/modules/riemann/solvers.py:114) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/riemann/solvers.py:163 hllcFlux](/home/lu26029/dev/warpSPH/src/warpSPH/modules/riemann/solvers.py:163) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/shifting/wp_deltaShiftRaw.py:22 computeDeltaShiftRaw_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shifting/wp_deltaShiftRaw.py:22) | wp.func | range/candidate accumulation or edge emission | raw delta shift |
| [frontend:modules/shifting/wp_deltaShiftRaw.py:98 computeDeltaShiftRaw_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shifting/wp_deltaShiftRaw.py:98) | wp.func | adjacency/range orchestration | raw delta shift |
| [frontend:modules/shifting/wp_deltaShiftRaw.py:174 computeDeltaShiftRaw_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shifting/wp_deltaShiftRaw.py:174) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | raw delta shift |
| [frontend:modules/shifting/wp_deltaShiftRaw.py:210 computeDeltaShiftRaw_KernelTiled](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shifting/wp_deltaShiftRaw.py:210) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | raw delta shift |
| [frontend:modules/shifting/wp_implicitShifting.py:27 computeShiftingPairTerms_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shifting/wp_implicitShifting.py:27) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | implicit shift pair terms |
| [frontend:modules/shifting/wp_michelUChar.py:35 computeUChar_valueAt](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shifting/wp_michelUChar.py:35) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/shifting/wp_michelUChar.py:61 computeUChar_Func_i_argmax](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shifting/wp_michelUChar.py:61) | wp.func | range/candidate accumulation or edge emission | characteristic shift speed |
| [frontend:modules/shifting/wp_michelUChar.py:124 computeUChar_Func_Adjacency_argmax](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shifting/wp_michelUChar.py:124) | wp.func | adjacency/range orchestration | characteristic shift speed |
| [frontend:modules/shifting/wp_michelUChar.py:192 computeUChar_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shifting/wp_michelUChar.py:192) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | characteristic shift speed |
| [frontend:modules/shockCapturing/ReadHayfield2012/kernels.py:43 rhVsig_valueAt](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shockCapturing/ReadHayfield2012/kernels.py:43) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/shockCapturing/ReadHayfield2012/kernels.py:58 rhVmax_Func_i_argmax](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shockCapturing/ReadHayfield2012/kernels.py:58) | wp.func | range/candidate accumulation or edge emission | Read-Hayfield signal speed |
| [frontend:modules/shockCapturing/ReadHayfield2012/kernels.py:95 rhVmax_Func_Adjacency_argmax](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shockCapturing/ReadHayfield2012/kernels.py:95) | wp.func | adjacency/range orchestration | Read-Hayfield signal speed |
| [frontend:modules/shockCapturing/ReadHayfield2012/kernels.py:143 computeReadHayfieldVmax_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shockCapturing/ReadHayfield2012/kernels.py:143) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | Read-Hayfield signal speed |
| [frontend:modules/shockCapturing/ReadHayfield2012/kernels.py:222 rhEntropy_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shockCapturing/ReadHayfield2012/kernels.py:222) | wp.func | range/candidate accumulation or edge emission | Read-Hayfield entropy |
| [frontend:modules/shockCapturing/ReadHayfield2012/kernels.py:273 rhEntropy_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shockCapturing/ReadHayfield2012/kernels.py:273) | wp.func | adjacency/range orchestration | Read-Hayfield entropy |
| [frontend:modules/shockCapturing/ReadHayfield2012/kernels.py:322 computeReadHayfieldEntropy_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shockCapturing/ReadHayfield2012/kernels.py:322) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | Read-Hayfield entropy |
| [frontend:modules/shockCapturing/Wadsley2017/kernels.py:29 wadsleyPointKernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shockCapturing/Wadsley2017/kernels.py:29) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | Wadsley point fields |
| [frontend:modules/shockCapturing/Wadsley2017/kernels.py:78 wadsleyNeighbourMeanKernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shockCapturing/Wadsley2017/kernels.py:78) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | Wadsley neighbor mean |
| [frontend:modules/shockCapturing/wp_computeM.py:29 computeM_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shockCapturing/wp_computeM.py:29) | wp.func | range/candidate accumulation or edge emission | shock moment M |
| [frontend:modules/shockCapturing/wp_computeM.py:109 computeM_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shockCapturing/wp_computeM.py:109) | wp.func | adjacency/range orchestration | shock moment M |
| [frontend:modules/shockCapturing/wp_computeM.py:176 computeM_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shockCapturing/wp_computeM.py:176) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | shock moment M |
| [frontend:modules/shockCapturing/wp_vsig.py:30 computeVsig_valueAt](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shockCapturing/wp_vsig.py:30) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/shockCapturing/wp_vsig.py:68 computeVsig_Func_i_argmax](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shockCapturing/wp_vsig.py:68) | wp.func | range/candidate accumulation or edge emission | signal speed |
| [frontend:modules/shockCapturing/wp_vsig.py:155 computeVsig_Func_Adjacency_argmax](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shockCapturing/wp_vsig.py:155) | wp.func | adjacency/range orchestration | signal speed |
| [frontend:modules/shockCapturing/wp_vsig.py:235 computeVsig_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/shockCapturing/wp_vsig.py:235) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | signal speed |
| [frontend:modules/surfaceDetection/wp_barecasco.py:51 computeBarecascoSurfaceDetection_Func_i_first](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_barecasco.py:51) | wp.func | range/candidate accumulation or edge emission | Barecasco cover pass |
| [frontend:modules/surfaceDetection/wp_barecasco.py:137 computeBarecascoSurfaceDetection_Func_Adjacency_first](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_barecasco.py:137) | wp.func | adjacency/range orchestration | Barecasco cover pass |
| [frontend:modules/surfaceDetection/wp_barecasco.py:203 computeBarecascoSurfaceDetection_Func_i_second](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_barecasco.py:203) | wp.func | range/candidate accumulation or edge emission | Barecasco cone pass |
| [frontend:modules/surfaceDetection/wp_barecasco.py:299 computeBarecascoSurfaceDetection_Func_Adjacency_second](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_barecasco.py:299) | wp.func | adjacency/range orchestration | Barecasco cone pass |
| [frontend:modules/surfaceDetection/wp_barecasco.py:369 computeBarecascoSurfaceDetection_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_barecasco.py:369) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | Barecasco cone pass, Barecasco cover pass |
| [frontend:modules/surfaceDetection/wp_barecasco.py:413 computeBarecascoSurfaceDetection_KernelTiled](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_barecasco.py:413) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | Barecasco cone pass, Barecasco cover pass |
| [frontend:modules/surfaceDetection/wp_barecascoCone.py:21 computeBarecascoConeCount_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_barecascoCone.py:21) | wp.func | range/candidate accumulation or edge emission | Barecasco cone count |
| [frontend:modules/surfaceDetection/wp_barecascoCone.py:86 computeBarecascoConeCount_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_barecascoCone.py:86) | wp.func | adjacency/range orchestration | Barecasco cone count |
| [frontend:modules/surfaceDetection/wp_barecascoCone.py:160 computeBarecascoConeCount_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_barecascoCone.py:160) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | Barecasco cone count |
| [frontend:modules/surfaceDetection/wp_barecascoCone.py:194 computeBarecascoConeCount_KernelTiled](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_barecascoCone.py:194) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | Barecasco cone count |
| [frontend:modules/surfaceDetection/wp_barecascoCover.py:21 computeBarecascoCover_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_barecascoCover.py:21) | wp.func | range/candidate accumulation or edge emission | Barecasco cover |
| [frontend:modules/surfaceDetection/wp_barecascoCover.py:81 computeBarecascoCover_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_barecascoCover.py:81) | wp.func | adjacency/range orchestration | Barecasco cover |
| [frontend:modules/surfaceDetection/wp_barecascoCover.py:151 computeBarecascoCover_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_barecascoCover.py:151) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | Barecasco cover |
| [frontend:modules/surfaceDetection/wp_barecascoCover.py:181 computeBarecascoCover_KernelTiled](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_barecascoCover.py:181) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | Barecasco cover |
| [frontend:modules/surfaceDetection/wp_dilate.py:26 dilateSurfaceMask_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_dilate.py:26) | wp.func | range/candidate accumulation or edge emission | surface dilation sum |
| [frontend:modules/surfaceDetection/wp_dilate.py:106 dilateSurfaceMask_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_dilate.py:106) | wp.func | adjacency/range orchestration | surface dilation sum |
| [frontend:modules/surfaceDetection/wp_dilate.py:181 dilateSurfaceMask_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_dilate.py:181) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | surface dilation sum |
| [frontend:modules/surfaceDetection/wp_dilate.py:213 dilateSurfaceMask_KernelTiled](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_dilate.py:213) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | surface dilation sum |
| [frontend:modules/surfaceDetection/wp_maronne.py:77 computeMaronneSurfaceDetection_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_maronne.py:77) | wp.func | range/candidate accumulation or edge emission | Maronne count |
| [frontend:modules/surfaceDetection/wp_maronne.py:186 computeMaronneSurfaceDetection_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_maronne.py:186) | wp.func | adjacency/range orchestration | Maronne count |
| [frontend:modules/surfaceDetection/wp_maronne.py:254 computeMaronneSurfaceDetection_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_maronne.py:254) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | Maronne count |
| [frontend:modules/surfaceDetection/wp_nearestSurfaceNormal.py:41 computeNearestSurfaceNormal_valueAt](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_nearestSurfaceNormal.py:41) | wp.func | local math, pair evaluation, loading or indexing helper | local helper or file assessment |
| [frontend:modules/surfaceDetection/wp_nearestSurfaceNormal.py:58 computeNearestSurfaceNormal_Func_i_argmin](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_nearestSurfaceNormal.py:58) | wp.func | range/candidate accumulation or edge emission | nearest surface normal |
| [frontend:modules/surfaceDetection/wp_nearestSurfaceNormal.py:113 computeNearestSurfaceNormal_Func_Adjacency_argmin](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_nearestSurfaceNormal.py:113) | wp.func | adjacency/range orchestration | nearest surface normal |
| [frontend:modules/surfaceDetection/wp_nearestSurfaceNormal.py:175 computeNearestSurfaceNormal_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/surfaceDetection/wp_nearestSurfaceNormal.py:175) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | nearest surface normal |
| [frontend:modules/util/wp_numNeighbors.py:29 countNeighbors_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/util/wp_numNeighbors.py:29) | wp.func | range/candidate accumulation or edge emission | frontend neighbor count |
| [frontend:modules/util/wp_numNeighbors.py:108 countNeighbors_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/util/wp_numNeighbors.py:108) | wp.func | adjacency/range orchestration | frontend neighbor count |
| [frontend:modules/util/wp_numNeighbors.py:180 countNeighbors_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/util/wp_numNeighbors.py:180) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | frontend neighbor count |
| [frontend:modules/util/wp_numNeighbors.py:210 countNeighbors_KernelTiled](/home/lu26029/dev/warpSPH/src/warpSPH/modules/util/wp_numNeighbors.py:210) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | frontend neighbor count |
| [frontend:modules/util/wp_sum.py:27 warpSum_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/modules/util/wp_sum.py:27) | wp.func | range/candidate accumulation or edge emission | frontend neighbor field sum |
| [frontend:modules/util/wp_sum.py:107 warpSum_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/modules/util/wp_sum.py:107) | wp.func | adjacency/range orchestration | frontend neighbor field sum |
| [frontend:modules/util/wp_sum.py:178 warpSum_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/modules/util/wp_sum.py:178) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | frontend neighbor field sum |
| [frontend:sample/wp_deltaShift.py:35 computeDeltaShift_Func_i](/home/lu26029/dev/warpSPH/src/warpSPH/sample/wp_deltaShift.py:35) | wp.func | range/candidate accumulation or edge emission | sample delta shift |
| [frontend:sample/wp_deltaShift.py:157 computeDeltaShift_Func_Adjacency](/home/lu26029/dev/warpSPH/src/warpSPH/sample/wp_deltaShift.py:157) | wp.func | adjacency/range orchestration | sample delta shift |
| [frontend:sample/wp_deltaShift.py:235 computeDeltaShift_Kernel](/home/lu26029/dev/warpSPH/src/warpSPH/sample/wp_deltaShift.py:235) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | sample delta shift |
| [frontend:sample/wp_deltaShift.py:271 computeDeltaShift_KernelTiled](/home/lu26029/dev/warpSPH/src/warpSPH/sample/wp_deltaShift.py:271) | wp.kernel | kernel entry; inspect linked pass or file assessment for output scope | sample delta shift |
