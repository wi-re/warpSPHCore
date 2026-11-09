# First-order residual and implicit adjoints

`warpSPHCore.linalg.ResidualActions(residual, state, parameters)` linearizes a
flat square residual `R(z, theta)`. Its state JVP and VJP supply a
`LinearOperator` with an explicit transpose. Parameter products use independent
argument slots, so aliased parameters are summed exactly once by the enclosing
PyTorch autograd graph. Products are first-order; they return detached tensors.
Native first-order custom operators are supported; dense Jacobians are not used.

`implicitSolve(residual, initial, parameters, primalSolve, options=...)` returns
`(z, report)`. The caller supplies the existing primal algorithm. Both a Tensor
and a `SolveResult` are accepted. The returned residual is measured independently;
a negative solver status, nonfinite residual or failed declared tolerance raises
before exposing an implicit derivative. The warm-start derivative is zero.
The residual must describe a locally unique, differentiable root on its declared
fixed active set. A small residual alone cannot establish uniqueness.

Reverse mode solves `R_z.T lambda = output_cotangent` and returns
`-R_theta.T lambda`. Forward mode solves
`R_z tangent = -R_theta parameter_direction`. Separate optional `adjointSolve`
and `tangentSolve` callbacks receive `(operator, rhs)`; the primal solver or its
preconditioner is never implicitly reused. The default is general GMRES with
FP64 orthogonalization. Every linear solve measures the original-system residual
and checks its status, including custom solvers. Reports retain actual residuals,
thresholds and statuses. An iteration budget is not a convergence certificate.

`rootAtol`, `rootRtol` and an explicit `residualScale` define the root certificate;
linear solves use separate absolute/relative tolerances. For mixed physical
units, normalize residual blocks before handing them to this API. Qualify both
root and adjoint tolerance sensitivity for each application. The generic API
cannot infer omitted equations or determine the accuracy of wall closures.

`fixedIterations(iteration, initial, parameters, count)` instead differentiates
exactly that finite algorithm, including its initial state. Its derivative is
meaningful even when the root is unconverged. It neither calls the implicit rule
nor interprets a stopping metric as a root certificate.

## Constraints and controls

All continuous coefficients, RHS, geometry, timestep, prescribed boundary data
and history must be explicit parameters. Fixed topology, masks, active sets and
gauges can be supplied as versioned `controls` and an optional `controlToken`
callback that also detects replacement. Do not close over differentiable data
without listing it as a parameter. Input changes invalidate the linearization.

Use `RestrictedSystem` for actual fixed Dirichlet rows and lift their prescribed
values as continuous inputs. A pressure pin does not prove that a constant mode
is a nullspace. For a closed singular system, validate explicit left/right bases
with `NullspaceSystem`, check RHS compatibility, and use a consistent reduced or
bordered residual with a declared gauge. Verify the original physical residual
as well as the gauge; a nonzero bordering multiplier can conceal incompatibility.
Changing nullspace bases or rank transitions need their own derivative contract.
The regression covers distinct left/right modes, projected RHS and a continuous
gauge; it does not infer symmetry from equal dimensions or a primal solver.

## Execution and cost

Implicit convergence checks are eager and perform host reads. Whole-solve CUDA
graph capture is explicitly rejected. Separately prepared residual/native action
capture remains an owner-specific capability. The reverse state graph is cached
per linearization, parameter pullbacks are evaluated once per adjoint, and
GMRES uses O(N*restart) workspace. This avoids storing every primal iteration;
it does not promise that a transpose solve costs as little as the forward solve.

Tests cover CPU/CUDA float32/float64, a nonlinear nonsymmetric root, explicit
Jacobians, independent differences, controlled unrolling, aliasing and warm
starts, control lifetimes, failed primal/transpose/budget cases, distinct
nullspaces/gauges and root/adjoint tolerance sensitivity. Device qualification
and installed/no-Warp evidence are recorded by the frontend follow-up handoff.
