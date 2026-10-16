# Linear algebra

[Runnable examples](examples.md) compose these APIs for constrained diffusion,
disconnected singular graphs, a vector nonsymmetric block-preconditioned
GMRES solve, a saddle-point MINRES solve and Richardson fallback.

`warpSPHCore.linalg` provides matrix-free CG, BiCG, BiCGStab, restarted GMRES,
short-recurrence MINRES and bounded Richardson iteration, a shared
stationary iteration driver, and native symmetric pseudoinverses for small
batches. The iterative solvers default to predicated Torch updates; CG and
BiCGStab also offer opt-in native Warp updates with reusable scratch. Their
checkpoint and verification reads make them eager solves; returning device
tensors does not make a complete solve CUDA-graph capturable.

## Matrix free solves

```python
import torch
from warpSPHCore.linalg import cg

A = torch.diag(torch.tensor([2.0, 3.0]))
b = torch.tensor([1.0, 2.0])
result = cg(lambda v: A @ v, b, precond=1 / A.diag(), rtol=1e-6)
assert result.statusInt() >= 0
torch.testing.assert_close(result.x, torch.linalg.solve(A, b))
```

Inputs are flat real floating-point vectors. `x0`, operator outputs and
preconditioner outputs must match `b` in shape, dtype and device. Closures
must be linear, deterministic for the fixed solve, and return new output
tensors without mutating the input vector. CUDA closures must order their
work on Torch's current stream or explicitly establish equivalent ordering.

| Method | Operator requirement | Preconditioner |
|---|---|---|
| `cg(matvec, b, x0=None, ...)` | Symmetric positive definite on the solved subspace | Symmetric positive definite inverse action |
| `bicg(matvec, matvecT, b, x0=None, ...)` | A genuine transpose for a nonsymmetric operator | Left inverse action `P`; supply `precondT` for `P.T` if nonsymmetric |
| `bicgstab(matvec, b, x0=None, ...)` | General real linear operator | Right inverse action |
| `gmres(matvec, b, x0=None, ...)` | General real linear operator | Right (default) or left fixed linear inverse action |
| `minres(matvec, b, x0=None, symmetric=True, ...)` | Explicitly declared symmetric; indefinite and compatible singular systems allowed | Symmetric positive definite inverse action on the solved subspace |
| `richardson(matvec, b, x0=None, ...)` | Fixed real linear action; convergence is scenario-dependent | No inverse preconditioner; scalar fixed/adaptive steps |

`precond=None` is identity; a flat tensor multiplies elementwise (the
**inverse** diagonal), and a callable applies the inverse preconditioner.
For BiCG, omitting `precondT` asserts that the supplied action is symmetric.
It is not an automatic transpose of an arbitrary callable.

Success certifies the original system:

$$
\|b-Ax\|_2 \leq \max(\mathrm{tol},\mathrm{atol},\mathrm{rtol}\|b\|_2).
$$

All tolerances must be finite and nonnegative. Exact zero residual is accepted
when all tolerances are zero. A zero source does not bypass a warm start's
residual; an already-valid nullspace solution is preserved. BiCG history and
stopping norms describe the original system, independent of preconditioner
scaling.

For CG, BiCG and BiCGStab, recurrence convergence is verified with a direct residual. If it fails,
`maxRestarts` (default 2) bounds fresh-residual passes. `maxiter` bounds total
scheduled loop iterations across these passes, including frozen iterations.
`maxiter=0` tests only the initial guess. `checkEvery` (default 8) must be
positive; the solution freezes when a device stopping test fires, while the
host checks only at those intervals. Closures can still be called during
frozen work.

`threshold` is an optional finite nonnegative per-entry norm bound, grouping
`dim` consecutive components. The vector length must be divisible by `dim`.
The threshold takes precedence over convergence at the initial iterate and
accepted full or half updates. It reports a failure with the candidate that
exceeded the bound; it does not clamp the solution.

## Results and failures

`SolveResult` contains:

- `x`, device `status`, `trueResidual` and device integer `iterations`.
  `iterations` counts completed updates, including accepted half steps;
  GMRES counts completed Arnoldi steps instead.
- `loopIterations`, `matvecs`, `adjointMatvecs`,
  `preconditionerApplications`, `verifications` and `restarts` as host counts.
  Work counts include frozen work and verification. `verifications` counts
  direct original residual evaluations, including the initial one.
- `setupIterations` counts executed capture warmup updates. They are excluded
  from the numerical budget but included in operator/preconditioner work counts.
- Richardson adds `trialSteps` (attempted candidates, including rejected and
  nonfinite trials), `backtracks` (actual halvings), `bestIteration` (accepted
  update index of the best iterate) and `stepSize` (last accepted signed device
  scalar, or None). Other methods leave these fields at their defaults.
- Device `norms` and `valid` history arrays. `convergence()` materializes the
  legacy history; CG/BiCG include initial norms, while BiCGStab records full
  updates and appends its verified residual. Restarts can add initial entries.

`statusInt()` is cached. A nonnegative status means verified convergence, but
its integer value retains the legacy indexing: CG/BiCG updates are zero-based,
and BiCGStab's half/full exits differ. GMRES, MINRES and Richardson use completed
step counts on success.
Use `iterations` for completed work.

| Status | Meaning |
|---|---|
| `RHO_BREAKDOWN` (`-10`) | Unusable dual-residual inner product |
| `DENOMINATOR_BREAKDOWN` (`-11`) | Unusable scalar update or nonfinite candidate |
| `THRESHOLD` (`-12`) | Per-entry bound exceeded |
| `STAGNATION` (`-13`) | GMRES full cycle made insufficient true-residual progress |
| `BUDGET` (`-14`) | Scheduled iteration budget exhausted |
| `RICHARDSON_STAGNATION` (`-15`) | Bounded search, zero step/direction, or the fixed-step stagnation window could not progress |
| `INDEFINITE` (`-16`) | CG curvature failure, or CG/MINRES encountered a nonpositive/nonfinite active inverse quadratic form |
| `RESIDUAL_MISMATCH` (`-17`) | Recurrence passed but bounded verification/restarts did not |

`legacy()` returns `(x, status_int, convergence_list)`. CG/BiCG's `stamp=False`
suppresses appending a failed residual to that legacy list; it never disables
verification or `trueResidual`. BiCGStab always appends a verified final
residual except for an already-solved zero source. GMRES always stamps its
verified residual. MINRES and Richardson also stamp the final original residual
(MINRES may omit the stamp for a trivial zero-source solve).

### Restarted GMRES

```python
from warpSPHCore.linalg import gmres

A = torch.tensor([[3., 1.], [-1., 2.]], dtype=torch.float64)
b = torch.tensor([1., 2.], dtype=A.dtype)
result = gmres(lambda v: A @ v, b, restart=20, maxiter=100,
               precond=1 / A.diag(), preconditionerSide='right', rtol=1e-10)
assert result.statusInt() >= 0
torch.testing.assert_close(result.x, torch.linalg.solve(A, b))
```

The Arnoldi construction uses two projection passes (CGS2); incremental
Givens rotations solve the small least-squares problem without normal
equations. `restart` defaults to 30 and is capped by the vector length and
remaining budget. The one-dimensional case is supported. Happy breakdown
also requires original-residual verification; it is not automatic success.

Both `preconditionerSide='right'` and `'left'` accept a fixed linear inverse
action. Flexible or changing preconditioners are outside this contract.
`maxiter` bounds scheduled Arnoldi steps across every cycle, including frozen
work. `iterations` counts real Arnoldi columns; candidates are materialized at
checkpoints and cycle boundaries, or every iteration when a threshold is set.
`checkEvery` retains the usual device freeze and delayed host-check semantics.

Each cycle verifies the original residual. Ordinary restarts are limited by
the total budget; `maxRestarts` (default 2) separately bounds repairs after
an optimistic convergence estimate or happy breakdown fails verification.
The result's `restarts` counts all fresh cycles. `stagnationTol` (default
0.001) requires that fractional decrease in true residual over a full cycle;
zero disables this guard. Choose a positive absolute tolerance for a zero RHS
if roundoff-level residuals are acceptable; relative tolerance alone requires
an exactly zero residual.

History contains the initial true norm and valid projected QR residual
estimates, followed by the stamped final true norm. Left-preconditioned
estimates are rescaled to the cycle's initial original norm; they remain
triggers, not certificates. `history=False` omits the recurrence arrays and
retains the final certificate and work counters.

GMRES follows the vector's float32/float64 dtype independently of process
precision. `accumulationDtype=torch.float64` promotes projections, QR,
triangular solves and candidate accumulation while callback vectors and stored
basis vectors retain the input dtype. It does not recover operator precision.
Without an autograd trace, basis, preconditioned directions and temporary stacks use O(n * restart)
memory; QR uses O(restart squared). Work remains an eager Torch baseline:
there is no reusable GMRES workspace or captured complete solve yet. An
implicit differentiation interface is not provided or qualified here.

### Short-recurrence MINRES

```python
from warpSPHCore.linalg import minres, DiagonalPreconditioner

A = torch.diag(torch.tensor([-2., 1., 3.], dtype=torch.float64))
b = torch.tensor([1., -2., 3.], dtype=A.dtype)
inverse = DiagonalPreconditioner(torch.tensor([2., 1., 3.], dtype=A.dtype), requireSPD=True)
result = minres(lambda v: A @ v, b, symmetric=True, precond=inverse, rtol=1e-11)
assert result.statusInt() >= 0
torch.testing.assert_close(result.x, torch.linalg.solve(A, b))
```

MINRES requires the explicit declaration `symmetric=True`; it does not infer
self-adjointness from a callback or probe. A can be indefinite or singular.
Compatible singular systems are supported, while incompatible requests cannot
return successful exact-solve status. This interface does not provide
least-squares/MINRES-QLP minimum-length success.

`precond` follows the same **inverse action** convention as CG. None is
identity. A flat inverse diagonal is checked finite and strictly positive.
An SPD-certified helper (`spdVerified=True`) is accepted directly. A general
callable/adapter requires `preconditionerSPD=True`, declaring a fixed symmetric
positive definite inverse on the solved subspace. In particular a projected
inverse from `NullspaceSystem.projectPreconditioner` needs that declaration;
it is semidefinite on the full space. The algorithm checks active inverse
quadratic forms, but those checks do not certify symmetry or global SPD.
Negative operator eigenvalues are valid, so they do not by themselves imply
`INDEFINITE` failure. An indefinite inverse is invalid.

The [Paige–Saunders](https://web.stanford.edu/group/SOL/software/minres/)
Lanczos/Givens recurrence retains a constant number of vectors, with a matching
A-times-direction recurrence for the **original residual**. Stopping does not
compare a scaled preconditioned norm with an original tolerance. Direct
original-system verification and bounded `maxRestarts` residual-replacement
passes guard recurrence drift and optimistic invariant-subspace termination.
The final Lanczos update is processed before treating a zero next direction
as termination. There is no growing basis or triangular solve.

`tol`, `rtol`, `atol`, `threshold`, `dim`, `checkEvery`, `maxiter` and
`maxRestarts` follow the shared contracts. Inactive iterates freeze while
remaining work up to a checkpoint is counted; `maxiter` includes that work
across repair passes. Success certifies `||b-Ax||` and `iterations` counts
accepted updates. `history=False` omits recurrence arrays while retaining
the final certificate/work counts. History otherwise tracks original-residual
recurrence norms and appends direct verification. Choose positive `atol` for
zero sources when a rounding-sized residual is acceptable; zero tolerances
require exact zero and can return failure after accurate finite iterations.

Storage is O(n) without an autograd trace, plus optional O(maxiter) scalar
history. Vectors follow CPU/CUDA float32/float64 dtype independently of process
precision. There is no native workspace or externally capturable complete
solve, and implicit solve differentiation is not provided. The existing
frontend MINRES congruence adapter is a separate migration; its diagonal
semantics must be mapped explicitly rather than assumed interchangeable.

### Bounded Richardson iteration

```python
from warpSPHCore.linalg import richardson

A = torch.diag(torch.tensor([1., 2., 3.], dtype=torch.float64))
b = torch.tensor([1., -2., 3.], dtype=A.dtype)
result = richardson(lambda v: A @ v, b, rtol=1e-11, maxiter=200)
assert result.statusInt() >= 0
torch.testing.assert_close(result.x, torch.linalg.solve(A, b))
```

Richardson applies `x <- x + step * (b - A x)` with **no inverse
preconditioner**. This preserves its role as a last fallback when a production
diagonal is useful in Krylov but unsuitable for a stationary update.
`omega=<finite nonzero scalar>` selects a fixed signed step. With
`omega=None`, each step selects the scalar that minimizes the linear residual
along r, `(r.T A r) / ||A r||^2`, using a scaled A r to avoid squaring very
small/large actions. The sign is automatic. General indefinite/nonsymmetric
systems can still stagnate; this is not a convergence guarantee.

`backtracking=True` accepts a direct original-residual decrease or convergence,
halving the step at most `maxBacktracks=8` times. Every trial starts from the
same accepted iterate. Rejected trials do not consume accepted-update budget,
and threshold checks apply to accepted candidates rather than transient
overshoots. `backtracking=False` applies literal fixed/adaptive updates, including
possible residual increases. `minStep=0` and optional positive `maxStep` bound
step magnitudes. Automatic steps are clipped to maxStep; a supplied fixed omega
outside the bounds is rejected. A step that is zero or below minStep stagnates.

`maxiter` bounds outer update/search attempts; `iterations` counts accepted
updates and `loopIterations` includes a final unsuccessful search. A search
has at most maxBacktracks+1 candidate attempts, also capped by minStep.
`stagnationWindow=8` counts accepted updates without improving the best norm;
zero disables that window. Search exhaustion or a zero minimizing direction
reports `RICHARDSON_STAGNATION`; wholly nonfinite candidates/directions report
`DENOMINATOR_BREAKDOWN`.

On budget, stagnation or nonfinite failure, the result returns the best finite
accepted iterate (including the warm start). Threshold failure returns the
accepted violating candidate, following the other solvers. The final stamped
residual always describes the returned iterate, so an owning frontend can
compare fallback candidates by actual residual. `iterations` can exceed
`bestIteration` when a previous iterate was retained. History contains initial
and accepted direct norms plus the final stamp; rejected trials are represented
by work counters. `history=False` skips the arrays.

Every evaluated trial and the final returned iterate use a direct original
residual. Adaptive direction applications count as matvecs, not verifications;
rejected finite trial residuals count as both. Trial acceptance reads the host:
this is an eager numerical baseline without a workspace or capture promise.
The step policy intentionally differs from the older frontend power-iteration
seed; migration and fallback-chain selection remain application-owned.
Projected/clamped pressure iteration stays in the separate stationary interface.

`scripts/examples/solver_capabilities.py` exercises MINRES on a singular
velocity/pressure saddle-point model and Richardson as a warm fallback on a
translation-nullspace Hessian model, independently checking physical residuals:

```bash
python scripts/examples/solver_capabilities.py --device cpu --dtype float64
python scripts/examples/solver_capabilities.py --device cuda --dtype float32
```

### Explicit operator metadata

```python
from warpSPHCore.linalg import LinearOperator

operator = LinearOperator(A.shape, A.dtype, A.device,
                          matvec=lambda v: A @ v,
                          adjoint=lambda v: A.T @ v)
result = gmres(operator, b, rtol=1e-10)
out = torch.empty_like(b)
operator.apply(b, out)  # writes caller storage
transpose_product = operator.applyAdjoint(b)
```

`LinearOperator` validates shape, dtype and device. It continues to work as a
simple callable with existing solvers. An optional `matvecOut(v, out)` writes
every supplied output entry and returns `None` or that exact output object;
it must preserve storage/metadata and cannot overlap the input. An operator
can supply only this callback; ordinary calls then allocate fresh output.
Without it, `apply(v, out)` copies an ordinary result. Callbacks own stream
ordering and derivative support, as with ordinary solver callbacks.

The adapter has no coefficient cache and infers neither symmetry nor an
adjoint. Coordinate restriction and Dirichlet lifting are provided by
`RestrictedSystem` below. `NullspaceSystem` provides explicit singular-system
compatibility, projection and gauge helpers.

### Restricted systems and prescribed values

```python
from warpSPHCore.linalg import LinearOperator, RestrictedSystem, DiagonalPreconditioner, cg

A = torch.tensor([[2., -1., 0.], [-1., 2., -1.], [0., -1., 2.]], dtype=torch.float64)
operator = LinearOperator(A.shape, A.dtype, A.device,
                          lambda v: A @ v, adjoint=lambda v: A.T @ v)
free = torch.tensor([1], device=A.device)
system = RestrictedSystem(operator, free)
b = torch.tensor([0., 2., 0.], dtype=A.dtype)
prescribed = torch.tensor([1., 0., 3.], dtype=A.dtype)
rhs = system.liftRhs(b, prescribed)  # [6], including both boundary contributions
precond = DiagonalPreconditioner(A.diag()[free], requireSPD=True)
result = cg(system, rhs, precond=precond, rtol=1e-12)
assert result.statusInt() >= 0
x = system.reconstruct(result.x, prescribed)  # [1, 3, 3]
torch.testing.assert_close(system.restrict(b - A @ x), torch.zeros_like(rhs))
```

For injection E into free scalar coordinates and R = E.T, the reduced action
is R A E, the lifted RHS is R(b - A g), and reconstruction is E y + g. The
boundary contribution is part of the RHS, so the reduced action remains linear.
The original constrained rows may carry reaction forces; certify the original
**free-row** residual rather than requiring the full residual to vanish.
Finite-precision reconstruction can round differently from the reduced action,
so check this residual separately when the physical consumer requires it.

`RestrictedSystem(operator, freeIndices)` requires a square CPU/CUDA
`LinearOperator` and a unique flat int32/int64 index tensor on the same device.
Indices retain caller order; duplicates and out-of-range coordinates are
rejected. Coordinates address scalar entries of the solver's flat vectors.
For particle-major d-component data, explicitly expand each selected particle
to its d scalar indices in the required component order; partial blocks are
also permitted. All-free, no-free and empty full systems are supported.

`restrict(v, out=None)` gathers a full vector; `prolong(v, out=None)` injects
a reduced vector with zeros elsewhere. `liftRhs(b, prescribed=None, out=None)`
and `reconstruct(v, prescribed=None, out=None)` take a full prescribed vector
g that must be finite and exactly zero on free coordinates. None means
homogeneous values. Prescribed values are validated per call and never retained;
changed RHS/boundaries reuse the same reduced operator. No-free systems return
empty reduced actions/RHSs without invoking the full operator.

The helper is callable and offers `apply(v, out=None)`, `applyAdjoint(v)` and
a stable `asLinearOperator()`. The reduced adjoint is R A.T E and is available
only when the full operator explicitly supplied one. Restriction does not
certify symmetry or positive definiteness. An ordinary reduced action supports
input-vector reverse/forward AD when its full callback does; this does not
provide implicit differentiation of the solve.

The helper owns its index map; `freeIndices` and `constrainedIndices` return
diagnostic copies (the complement is sorted). Full coefficients remain owned
by the original operator. Updates between solves remain visible; changing
the action during a solve violates the solver contract. Ordinary outputs own
their storage, and no shared application scratch is retained. Optional outputs
must match shape/dtype/device and must not overlap any input; they are intended
for inference. Setup and prescribed-vector validation may synchronize and must
run outside CUDA capture. Ordinary reduced applications use Torch's current
stream and can be captured if the full action supports it; a complete solve
still has checkpoint and verification reads.

Preconditioners passed to the reduced solver must act on reduced vectors.
To build reduced diagonal/block Jacobi, select the appropriate coefficients
and construct a new helper; partially selected blocks require constructing
their selected subblocks. To explicitly restrict an existing full inverse
action P, use `system.restrictInverseAction(fullInverse.asLinearOperator())`.
This returns R P E with any explicitly supplied adjoint and live coefficient
updates. In general it differs from inverse(R A E): for
`A = [[2, 1], [1, 2]]` with only coordinate 0 free, the two inverse actions
are 2/3 and 1/2, respectively. No inverse construction is silently substituted.

A complete manufactured Poisson example is in
`scripts/examples/constrained_poisson.py`. Run it from the repository root:

```bash
python scripts/examples/constrained_poisson.py --device cpu --dtype float64
python scripts/examples/constrained_poisson.py --device cuda --dtype float32
```

It verifies nonzero prescribed values, the reconstructed free-row residual,
and the analytic quadratic solution, independently of the solver's estimate.

### Nullspaces, source compatibility and gauges

```python
from warpSPHCore.linalg import LinearOperator, NullspaceSystem, cg

A = torch.tensor([[1., -1.], [-1., 1.]], dtype=torch.float64)
operator = LinearOperator(A.shape, A.dtype, A.device,
                          lambda v: A @ v, adjoint=lambda v: A.T @ v)
Q = torch.ones((2, 1), dtype=A.dtype) / 2**.5
system = NullspaceSystem(operator, rightBasis=Q, leftBasis=Q)
prepared = system.prepareRhs(torch.tensor([1., -1.], dtype=A.dtype))
result = cg(system, prepared.rhs, rtol=1e-11)
assert result.statusInt() >= 0
x = system.gauge(result.x)  # zero arithmetic mean for this constant basis
report = system.verify(x, prepared, rtol=1e-11)
assert report.requestedConverged
torch.testing.assert_close(x, torch.tensor([.5, -.5], dtype=A.dtype))
```

`NullspaceSystem(operator, rightBasis, leftBasis, basisTolerance=None,
nullTolerance=None)` takes a square CPU/CUDA `LinearOperator` and two explicit
real `(n, k)` bases of the same dtype/device. Both bases must have orthonormal,
independent Euclidean columns. Right modes describe solution freedom; left
modes describe source compatibility. For a nonsymmetric operator these spaces
can differ. Equal bases never assert symmetry or supply a missing adjoint.
Nonempty bases require an explicit adjoint for left-mode validation. Empty
bases and empty/full-nullspace systems are supported.

Setup copies both bases, checks `||Q.T Q - I||_inf <= basisTolerance` and
checks each `||A q_right||_2` / `||A.T q_left||_2 <= nullTolerance`. Both
defaults are `64 * eps(dtype)`; basis tolerance must be less than one. The
null tolerance is **absolute, in action units**; specify it explicitly for
scaled or approximate actions. These tests validate the declared columns;
they cannot prove the basis is complete. The caller must declare every
relevant mode, including one constant mode per disconnected graph component.

With `P_left = I - Q_left Q_left.T` and similarly `P_right`, the callable
action is `P_left A P_right` and its explicit adjoint is
`P_right A.T P_left`. `projectLeft(v, out=None)` and
`projectRight(v, out=None)` expose those Euclidean projections. Project a warm
start with `projectRight` when an orthogonal representative is wanted.
The helper also offers `apply(v, out=None)`, `applyAdjoint(v)` and a stable
`asLinearOperator()`. Ordinary results own their storage; supplied outputs
must match metadata and cannot overlap the input. Projections are accurate
to the working precision and the accepted basis tolerance.

`prepareRhs(b, policy='error', rtol=None, atol=0)` measures the left component
`Q_left (Q_left.T b)`. Compatibility means its norm is at most
`max(atol, rtol * ||b||)`, with default rtol `64 * eps(dtype)`.
The default policy raises for an incompatible source and returns an unchanged
snapshot otherwise; even a tolerated tiny component is not silently removed.
`policy='project'` explicitly selects `P_left b` as the source. The resulting
`NullspaceRhs` contains owned `requested`, `rhs` and `removed` vectors,
`incompatibleNorm`, `tolerance`, `compatible` and `policy`. Treat those
snapshots as read-only while solving and verifying.

Projection can leave a roundoff-sized selected source, even when the exact
compatible component is zero. Choose an explicit absolute solve/verification
tolerance for that scale; the helper does not silently round the source to zero.

After solving and choosing a gauge, `verify(x, prepared, rtol=1e-5, atol=0)`
returns device scalar residual norms and convergence flags:

| Fields | Equation checked |
|---|---|
| `requestedResidual`, `requestedConverged` | Original A and the requested source |
| `solvedResidual`, `solvedConverged` | Original A and the selected source |
| `projectedResidual`, `projectedConverged` | Projected action and the selected source |
| `removedNorm` | Norm of requested minus selected source |

Each equation uses its respective source norm in the stopping target. The
helper never changes a solver status: success with a projected source does
not certify the original request. Inexact declared modes or large gauge
shifts can also make original-action verification fail even when the projected
solve passed. These checks are needed even after a successful solver return.

For preconditioning, wrap an explicit full inverse action using
`system.projectPreconditioner(inverse.asLinearOperator())`. It applies
`P_right M_inverse P_left` and supplies its transpose only if declared. When
A is self-adjoint, both nullspaces coincide, and the inverse is SPD, this
keeps preconditioning symmetric and positive on the solved subspace. It is
singular on the full space, so it is not a global SPD certificate. Changing
preconditioners/flexible methods remain outside the existing solver contracts.

Gauges add right-nullspace components after the solve:

- `gauge(x, coefficients=None)` sets `Q_right.T x` to the chosen coefficients,
  defaulting to zero. For componentwise constant modes this is zero arithmetic
  mean on each component; arbitrary modes do not imply a physical mean.
- `weightedGauge(x, weights, values=None)` sets k weighted means. Weights have
  shape `(n, k)`, finite nonnegative entries and positive column totals. Each
  column is normalized to sum one; values default to zero.
- `pinGauge(x, indices, values=None)` fixes k scalar coordinates, defaulting to
  zero. Indices must be unique int32/int64 coordinates on the operator device.

The weighted/pinned functionals must independently constrain all right modes;
for example, two pins in one disconnected component cannot fix another
component's constant. Gauges solve the small constraint system in the declared
nullspace and reject singular/poorly conditioned coupling using basisTolerance
as the relative singular-value cutoff. They do not insert weighted/oblique
projections into CG or change its Euclidean symmetry contract. Verify the
original action after gauge reconstruction.

Basis setup rejects coefficient reverse/forward AD. Ordinary vector actions
and projections preserve AD if the full callbacks do; differentiation through
the solve is not supplied. Coefficients stay owned by the full operator;
after changing them, call `validate()` to recheck both spaces. The operator
must stay fixed throughout a solve. Restricting/pinning out coordinates can
remove null modes; construct and validate a new declaration on the reduced
operator rather than slicing and reusing the previous basis automatically.

Setup/validation, RHS policies and gauges can synchronize and must run outside
CUDA capture. Ordinary projected actions/preconditioners use Torch's current
stream and can be captured when the full callbacks support it. Diagnostic basis
properties return copies, and no shared application scratch is retained.

`scripts/examples/singular_graph.py` solves two disconnected path graphs,
checks zero-mean and weighted gauges, and reports both physical residuals:

```bash
python scripts/examples/singular_graph.py --device cpu --dtype float64
python scripts/examples/singular_graph.py --device cuda --dtype float32 --project-incompatible
```

### Diagonal and block-Jacobi helpers

```python
from warpSPHCore.linalg import DiagonalPreconditioner, BlockJacobiPreconditioner

blocks = torch.tensor([[[4., 1.], [1., 3.]]], dtype=torch.float64)
A, b = blocks[0], torch.ones(2, dtype=blocks.dtype)
diagonal = DiagonalPreconditioner(A.diag(), requireSPD=True)
precond = BlockJacobiPreconditioner(blocks, requireSPD=True)
result = cg(lambda v: A @ v, b, precond=precond, rtol=1e-10)
assert result.statusInt() >= 0
precond.update(blocks * 2)  # same shape, dtype, device and coefficient addresses
```

These constructors take the operator's diagonal or `(n, d, d)` diagonal
blocks, **not** already-inverted coefficients. Blocks act on particle-major
flat vectors of length `n*d`. Setup follows the coefficient's float32/float64
dtype independently of process precision. General nonsymmetric blocks use
SVD; an explicit `requireSPD=True` uses signed symmetric eigenvalues, requires
exactly symmetric blocks and certifies the resulting inverse as positive
definite. Symmetrize input explicitly if that is the intended approximation.
The certificate concerns the preconditioner, not the system operator.

All regularization and singular handling are explicit:

| Option | Meaning |
|---|---|
| `shift=0` | Add this signed scalar to each diagonal before inversion; no automatic ridge |
| `atol=0`, `rtol=None` | Discard magnitudes at most `max(atol, rtol*scale)`; default `rtol=d*eps(dtype)` |
| `singular='error'` | Reject an unusable entry/block, including an overflowing reciprocal |
| `singular='identity'` | Replace an unusable scalar with inverse one, or an entire unusable block with identity |
| `singular='pinv'` | Zero only the discarded inverse components; this can produce a semidefinite action |
| `requireSPD=False` | No SPD certificate; when true, reject a nonpositive or rank-deficient resulting action |

Block scale is its largest singular/eigenvalue magnitude. Diagonal entries
are independent 1x1 blocks: one large row cannot discard another small row.
Negative retained values keep their sign. `requireSPD=True` permits identity
fallback but rejects rank-deficient pseudoinverses. `fallbackCount` counts
unusable entries/blocks in the last successful setup.

Both helpers are callable and provide `apply(v, out=None)`,
`applyAdjoint(v, out=None)` and a stable `asLinearOperator()` adapter. For
nonsymmetric block preconditioning with BiCG, explicitly pass
`precondT=precond.applyAdjoint`; the transpose is not the forward action.
Ordinary calls allocate independent outputs; supplied outputs must not alias
inputs. `inverse` returns a diagnostic copy; mutating it or the original
constructor input does not change the helper. `nbytes` counts owned inverse
storage.

`update(coefficients)` validates the complete new action before copying it
into fixed-address storage. Failed setup leaves the previous action,
`fallbackCount` and `generation` unchanged. Successful updates increment
`generation`; construction policies and metadata remain fixed. Use helpers
serially on Torch's current stream, or establish dependencies yourself.
Updates between solves are visible to captured applications and cached CG
without clearing its capture cache. Updating during a solve or concurrently
with an application violates the fixed-preconditioner contract.

Setup/update may synchronize and must run outside CUDA capture. They reject
coefficient reverse/forward AD rather than silently detach a differentiable
coefficient. Ordinary application supports reverse/forward derivatives with
respect to the input vector, treating stored coefficients as constants.
Supplied-output application is for inference.

Block application defaults to Torch. `application='native'` opts into a Warp
CUDA kernel for 1x1–3x3 blocks, following Torch's current stream and the vector
dtype. CPU and vector-AD calls retain Torch. Warm the chosen action before
capture. This option accelerates repeated small block actions; eager Python
launch costs and setup must be measured for the intended workload. Diagonal
actions remain a Torch elementwise multiply; larger blocks retain Torch.
The helpers use the solvers' callable-preconditioner path. The existing
`CGWorkspace` tensor-diagonal specialization remains available for an already
inverted tensor; a helper's `inverse` is a snapshot and will not receive later
helper updates.

### Optional CG fusion

`WARPSPHCORE_COMPILE_GLUE=1` enables compilation of CG's pure Torch update
chains through the existing `compileGlue` facility. Matvecs and all Krylov
inner products remain outside the compiled helpers; keeping the inner
products on the same reduction implementation preserves exact identity
solves. Final residual verification remains enabled.

Warm compilation before measuring or capturing individual helpers. Fusion
can change floating-point rounding and stopping iterations, and compilation
failure warns and falls back to eager execution. The whole solve still has
host checkpoints and verification reads. This option does not provide a
native Warp workspace or a captured complete solve.

### Native CG workspace

```python
from warpSPHCore.linalg import CGWorkspace, cg

workspace = CGWorkspace(b, maxiter=128, history=False)
result = cg(lambda v: A @ v, b, maxiter=128, workspace=workspace)
# Reuse the same scratch with another RHS of the same shape/dtype/device.
second = cg(lambda v: A @ v, b * 2, maxiter=128, workspace=workspace)
```

This path supports flat CPU/CUDA float32 and float64 tensors, independently of
SPH process precision. It fuses candidate updates and finite/threshold tests,
uses explicit device outputs for Warp reductions, and prebinds update launches
once per residual pass. It retains host checkpoints, bounded residual
replacement and final original-residual verification. It is an eager solve;
captured complete solves remain future work.

`history=False` returns empty recurrence `norms`/`valid` arrays. Verified
residuals, status, completed updates and work counts remain available. Failure
stamping follows the usual `stamp` argument. Workspace capacity must cover the
requested `maxiter` (including its default); scratch addresses stay fixed.
`workspace.nbytes` counts owned tensor buffers, excluding callback allocations,
Warp's internal reduction scratch and returned result copies.
Captured graph pools are also excluded from this owned-buffer count.

Results own their storage: reusing the workspace cannot overwrite an earlier
solution or history. Reuse it serially on the same Torch stream, or establish
the dependency between streams yourself. Concurrent solves need separate
workspaces. Native updates reject inputs or callback outputs that require
gradients or carry forward AD tangents; use the default Torch path for those.

An optional `matvecOut(v, out)` callback writes the supplied workspace buffer
and returns `None` or that exact output object. It must implement the same
linear action as `matvec`, write every entry, preserve the supplied output's
storage/metadata and leave `v` unchanged. Ordinary
`matvec` is still used for initial/final verification. Without `matvecOut`, the
workspace copies an ordinary callback's output into its fixed buffer.
Elementwise inverse diagonals use a native kernel; callable preconditioners
remain supported. Native reduction order and fused arithmetic can change
stopping iterations relative to Torch, so compare verified accuracy rather
than bitwise history. Compilation and setup should be warmed before timing.

### Cached CG chunks

```python
# CUDA only. Bind the callback once so subsequent solves reuse its graphs.
matvec = lambda v: A @ v
workspace = CGWorkspace(b, maxiter=128, history=False, capture=True)
first = cg(matvec, b, workspace=workspace, maxiter=128)
second = cg(matvec, b * 2, workspace=workspace, maxiter=128)
```

This caches a CUDA graph for `checkEvery` guarded updates and a shorter tail
when needed. Device stopping predicates still freeze the iterate immediately;
host checks run between chunks. The last chunk respects the remaining budget.
Initial/final verification and bounded residual replacement remain eager.
This option does **not** make the complete solve callable inside an outer
CUDA capture, and it does not require conditional CUDA graph support.

Callbacks must be graph-safe, deterministic and use Torch's current stream.
The first use executes one warmup update per new chunk size, restores the
numerical state, then records the graph. Python callback invocations during
recording do not count as executed matvecs. Later replays update work counts
without calling Python callbacks. Initial/final verification still invokes
the ordinary Python action. Warmup work is reported in `setupIterations`.

Use stable callable objects and perform first-use capture setup serially
before starting concurrent GPU work. The implementation orders its setup
stream explicitly, registers Warp's external capture resources and avoids
device-wide synchronization. It suspends cyclic GC during recording and
restores the caller's GC state afterward.

RHS, tolerance, initial-guess and same-address coefficient/preconditioner
content changes are visible on replay. Callback identity, preconditioner
tensor storage/strides, threshold, dimension or check interval changes
invalidate automatically. Replacing storage or geometry hidden inside the
same callback requires `workspace.clearCapture()` before the next solve.
This includes adjacency replacement/rebuilds that change captured bindings.
The cache holds at most two chunk sizes per binding. `captureBuilds` and
`captureMilliseconds` are cumulative setup diagnostics; the latter measures
recording time, excluding kernel warmup and state snapshots.

Upstream `warp.optim.linear.CG` is currently a comparison backend, not a
public alternative to `cg`: its tolerance floor, zero-budget meaning,
breakdown/finite-iterate behavior and diagnostics differ. It can also stop
on a recurrence residual that fails independent original-residual verification.

### Native BiCGStab workspace

```python
from warpSPHCore.linalg import BiCGStabWorkspace, bicgstab

workspace = BiCGStabWorkspace(b, maxiter=128, history=False)
result = bicgstab(lambda v: A @ v, b, workspace=workspace, maxiter=128)
```

Native BiCGStab supports flat CPU/CUDA float32/float64 vectors independently
of SPH process precision. Its ten fixed-address vectors are reused across
solves; returned solutions, counters and histories own their storage.
`history=False` omits recurrence arrays while keeping the verified residual
and work counts. Native updates reject reverse/forward AD inputs and callback
outputs; the default Torch implementation remains available for those.

The native state machine preserves right preconditioning, half/full exits,
relative rho/denominator guards, finite-iterate handling and threshold
precedence. Recurrence reductions write to fixed scalar outputs; their internal
Warp scratch is not included in `workspace.nbytes`. Native reduction order
and fused arithmetic can change rounding/iterations; compare original-residual
accuracy rather than bitwise histories. Initial/final verification and bounded
residual replacement retain the existing eager solver contract.

Optional `matvecOut(v, out)` and `precondOut(v, out)` callbacks fully write the
supplied destination and return `None` or that exact object. They preserve
storage/shape/strides/dtype/device and must not mutate `v`. `precondOut` requires
an ordinary `precond` representing the same fixed inverse action. Tensor
preconditioners already use a native multiply; callable preconditioners can
avoid allocating results by supplying this callback. For example:

```python
# Bind actions once if the workspace captures them.
action = lambda v: A @ v
into = lambda v, out: torch.mv(A, v, out=out)
precond = BlockJacobiPreconditioner(A[None], application='native')
preconditionInto = precond.apply
workspace = BiCGStabWorkspace(b, 128, history=False, capture=b.is_cuda)
result = bicgstab(action, b, precond=precond, workspace=workspace, maxiter=128,
                  matvecOut=into, precondOut=preconditionInto)
```

Ordinary `matvec` is still used for original-system verification. A supplied
action that disagrees with it cannot manufacture verified success. Helpers'
same-address coefficient updates are visible to replay; preconditioners must
remain fixed during each solve.

`capture=True` caches guarded chunks of `checkEvery` iterations and a strict
budget tail. It shares CG's cache lifecycle, stream ordering, GC restoration,
two-size cache bound and explicit `clearCapture()` rule for hidden storage
replacement. Replacing a supplied preconditioner callback also invalidates.
Warmup and recording occur on an explicitly ordered setup stream; no
device-wide synchronization is used by solver setup. This is not a captured
complete solve and cannot run inside an outer capture.

Every scheduled iteration executes **two operator actions and two supplied
preconditioner actions** (zero preconditioner actions for `precond=None`),
including frozen work after a half-step exit.
Native cached first use also executes one warmup iteration per new chunk
size, reported in `setupIterations` and included in action counts. Python
calls made only while recording do not count as executed work. The number of
native recurrence reductions is one initial squared norm per residual pass
plus eight per scheduled/warmup iteration. Original-system certificate
reductions and callback work are separate. Conditional omission of the
unneeded half-step tail remains a future scheduling experiment.

## Small symmetric matrices

```python
import warp as wp
from warpSPHCore.linalg import pinvSymmetric, applyPinv

wp.init()
M = torch.diag(torch.tensor([2.0, 3.0]))[None]
P = pinvSymmetric(M)
V = torch.ones(1, 4, 2)
corrected = applyPinv(V, M)  # V @ pinv(M), fused in one native launch
```

`M` has shape `(n, d, d)`, `1 <= d <= 3`; `V` has shape `(n, p, d)` and
matches its dtype and device. Native kernels use the symmetric part of `M`
and signed eigenvalue reciprocals. Use a general matrix routine for
nonsymmetric matrices.

The cutoff drops eigenvalues with magnitude at most `rtol * max(abs(lambda))`.
Default `rtol` is `d * eps(dtype)`. Native inputs must match the process
precision, selected with `warpSPHCore_PRECISION` before importing the package.
The supported public dtypes are float32 and float64. Empty batches and inputs
carrying reverse or forward AD use a Torch symmetric spectral implementation
with a fixed-retained-set derivative. Reverse AD, Torch duals and `torch.func.jvp`
include retained/discarded subspace rotation even for a nonzero discarded
eigenvalue; first and second reverse derivatives are qualified. Both native
and AD paths use the symmetric part. Cutoff crossings remain nonsmooth.
These fallback operations can incur Torch dense-solver synchronization.

Native strided inputs are copied as necessary; outputs are contiguous and
newly allocated. CUDA launches use Torch's current stream and can be captured
after initialization and kernel warmup. `symmetricPinv3` and `loadPadded3x3`
are exported Warp functions for callers that need the arithmetic inline.

This API is distinct from `pinv_warp`, which also applies renormalization
policies such as the low-neighbor identity fallback and eigenvalue diagnostics.

## Stationary iteration

`runStationary(step, x0, maxIters=..., tolerance=..., schedule=...)` calls
`step(x, i) -> (xNext, stat, rNorm)`. A state can be a tensor or a fixed
pytree of tensors, such as `(pressure, residual)`. Steps must not mutate the
supplied state. `CheckSchedule` supports `every` and history-based `adaptive`
checkpoints; `minIters` retains the legacy zero-based eligible iteration index.

The absolute test is `stat < tolerance` (`<=` with `inclusive=True`), with an
optional disjunction `rNorm <= relTarget`. The step owns the statistic's
meaning: it may describe the residual before updating. `converged` therefore
does not certify a residual of `xNext`.

Adaptive scheduling continues updating across skipped convergence points;
nonmonotone statistics can worsen. `firstConvergedIteration` records the first
eligible supplied statistic that passed, even if no host check occurred then.
`termination` distinguishes `converged`, `budget` and `nonfinite`. Nonfinite
iterates/statistics freeze the last accepted state until the next checkpoint.

`residualStatistic` preserves the pressure-specific `flooredOneSided`,
`oneSided` and `meanAbsolute` definitions. An empty selected set has no active
equations and returns zero. Histories and optional tracked vectors are copied
to the host together at solve completion.
