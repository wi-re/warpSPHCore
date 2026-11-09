"""First-order residual actions and independently verified implicit solves.

A residual must be square, flat, deterministic at fixed controls, and expose
*every continuous dependency* as a tensor parameter. Discrete topology, active
sets and gauges are fixed controls. A caller supplies its primal algorithm;
this module never changes that algorithm's stopping rule or accepts its status
as a residual certificate. Residual normalization is the caller's responsibility.
Implicit solves are eager (host convergence checks), not CUDA graph capturable.
"""
from dataclasses import dataclass, field
from contextlib import nullcontext
import math
import torch
from torch.autograd import forward_ad
from torch.autograd.function import once_differentiable
from .operator import LinearOperator
from .krylov import gmres, SolveResult


@dataclass(frozen=True)
class AdjointOptions:
    rootAtol: float = 1e-7
    rootRtol: float = 1e-5
    residualScale: float = 1.0
    linearAtol: float = 1e-8
    linearRtol: float = 1e-5
    maxiter: int = 200
    restart: int = 30

    def __post_init__(self):
        for name in ('rootAtol', 'rootRtol', 'residualScale', 'linearAtol', 'linearRtol'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name in ('maxiter', 'restart'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')


@dataclass
class AdjointReport:
    rootResidual: float = math.inf
    rootThreshold: float = 0.0
    primalStatus: object = None
    adjoint: list = field(default_factory=list)
    tangent: list = field(default_factory=list)


def _norm(value):
    return float(torch.linalg.vector_norm(value.detach().double()))


def _eager(value):
    if value.is_cuda and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError('Implicit adjoints require eager convergence verification')


def _signature(controls):
    return tuple((id(t), t._version, t.data_ptr(), tuple(t.shape), tuple(t.stride())) for t in controls)


class ResidualActions:
    """Independent state/parameter JVP and VJP slots at a fixed linearization.

    Parameters that alias are independent slots here. Passing the same tensor
    into multiple slots of ``implicitSolve`` correctly sums their pullbacks in
    PyTorch. No dense Jacobian or assumed symmetric transpose is used. Actions
    return detached first-order products; differentiating the actions again is
    outside this contract. Residuals may use native first-order autograd kernels.
    """
    def __init__(self, residual, state, parameters=(), *, controls=(), controlToken=None):
        if state.ndim != 1 or state.dtype not in (torch.float32, torch.float64):
            raise ValueError('Residual state must be a flat float32/float64 tensor')
        self.residual = residual
        self.state = state.detach()
        self.parameters = tuple(p.detach() for p in parameters)
        if any(not p.is_floating_point() or p.device != state.device or p.dtype != state.dtype for p in self.parameters):
            raise ValueError('Residual parameters must share the state floating dtype and device')
        self.controls = tuple(controls)
        self.signature = _signature(self.controls)
        self.controlToken = controlToken
        self.token = controlToken() if controlToken else None
        self.versions = _signature((self.state, *self.parameters))
        self._reverse = None

    def check(self):
        if _signature(self.controls) != self.signature or (self.controlToken and self.controlToken() != self.token):
            raise RuntimeError('Residual controls changed after linearization')
        if _signature((self.state, *self.parameters)) != self.versions:
            raise RuntimeError('Residual inputs changed after linearization')

    def _value(self, z, parameters):
        self.check()
        out = self.residual(z, *parameters)
        if out.shape != self.state.shape or out.dtype != self.state.dtype or out.device != self.state.device:
            raise ValueError('Residual must preserve state shape, dtype and device')
        return out

    def value(self):
        with torch.no_grad():
            return self._value(self.state, self.parameters).detach()

    def jvp(self, stateDirection=None, parameterDirections=None):
        directions = (None,) * len(self.parameters) if parameterDirections is None else tuple(parameterDirections)
        if len(directions) != len(self.parameters):
            raise ValueError('Parameter direction count mismatch')
        with torch.no_grad(), forward_ad._set_fwd_grad_enabled(True), (forward_ad.dual_level() if forward_ad._current_level < 0 else nullcontext()):
            z = self.state if stateDirection is None else forward_ad.make_dual(self.state, stateDirection)
            parameters = tuple(p if d is None else forward_ad.make_dual(p, d) for p, d in zip(self.parameters, directions))
            result = forward_ad.unpack_dual(self._value(z, parameters)).tangent
            return torch.zeros_like(self.state) if result is None else result.detach()

    def stateVjp(self, cotangent):
        with torch.enable_grad():
            self.check()
            if self._reverse is None:
                z = self.state.detach().requires_grad_()
                self._reverse = (z, self._value(z, self.parameters))
            z, value = self._reverse
            result = torch.autograd.grad(value, z, cotangent, retain_graph=True, allow_unused=True)[0] if value.requires_grad else None
            return torch.zeros_like(z) if result is None else result.detach()

    def parameterVjp(self, cotangent):
        with torch.enable_grad():
            # Independent leaves per argument, even if the caller aliased them.
            parameters = tuple(p.detach().requires_grad_() for p in self.parameters)
            value = self._value(self.state, parameters)
            if not parameters:
                return ()
            result = torch.autograd.grad(value, parameters, cotangent, allow_unused=True) if value.requires_grad else (None,) * len(parameters)
            return tuple(torch.zeros_like(p) if g is None else g.detach() for p, g in zip(parameters, result))

    def operator(self, *, transpose=False):
        n = self.state.numel()
        forward = lambda v: self.jvp(stateDirection=v)
        reverse = self.stateVjp
        return LinearOperator((n, n), self.state.dtype, self.state.device,
                              matvec=reverse if transpose else forward,
                              adjoint=forward if transpose else reverse)


def _linear(action, rhs, options, solver, reports, role):
    _eager(rhs)
    with torch.no_grad():
        if solver is None:
            result = gmres(action, rhs, atol=options.linearAtol, rtol=options.linearRtol,
                           maxiter=options.maxiter, restart=options.restart, checkEvery=1,
                           stagnationTol=0, accumulationDtype=torch.float64, history=False)
        else:
            # A separate callback for each direction; primal preconditioners are
            # never reused implicitly for the transpose system.
            result = solver(action, rhs)
        x = result.x if isinstance(result, SolveResult) else result
        status = result.statusInt() if isinstance(result, SolveResult) else None
        error = _norm(action(x) - rhs)
        threshold = max(options.linearAtol, options.linearRtol * _norm(rhs))
        reports.append(dict(residual=error, threshold=threshold, status=status))
        if (status is not None and status < 0) or not math.isfinite(error) or error > threshold:
            raise RuntimeError(f'{role} solve did not converge: residual={error}, threshold={threshold}, status={status}')
        return x.detach()


@dataclass
class _Problem:
    residual: object
    primalSolve: object
    options: AdjointOptions
    report: AdjointReport
    controls: tuple
    controlToken: object
    adjointSolve: object
    tangentSolve: object


class _Implicit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, problem, initial, *parameters):
        _eager(initial)
        result = problem.primalSolve(initial.detach(), *(p.detach() for p in parameters))
        z = result.x if isinstance(result, SolveResult) else result
        status = result.statusInt() if isinstance(result, SolveResult) else None
        actions = ResidualActions(problem.residual, z, parameters, controls=problem.controls, controlToken=problem.controlToken)
        error = _norm(actions.value())
        options = problem.options
        threshold = max(options.rootAtol, options.rootRtol * options.residualScale)
        problem.report.rootResidual, problem.report.rootThreshold, problem.report.primalStatus = error, threshold, status
        if (status is not None and status < 0) or not math.isfinite(error) or error > threshold:
            raise RuntimeError(f'Primal residual is not converged for implicit differentiation: residual={error}, threshold={threshold}, status={status}')
        ctx.problem = problem
        ctx.actions = actions
        ctx.save_for_backward(z, initial, *parameters)
        ctx.save_for_forward(z, initial, *parameters)
        return z.clone()

    @staticmethod
    @once_differentiable
    def backward(ctx, bar):
        ctx.saved_tensors  # PyTorch version checks for every continuous input.
        p, actions = ctx.problem, ctx.actions
        actions.check()
        lam = _linear(actions.operator(transpose=True), bar, p.options, p.adjointSolve, p.report.adjoint, 'Adjoint')
        grads = actions.parameterVjp(-lam)
        return None, torch.zeros_like(ctx.saved_tensors[1]), *grads

    @staticmethod
    def jvp(ctx, _problem, _initial, *directions):
        ctx.saved_tensors
        p, actions = ctx.problem, ctx.actions
        rhs = -actions.jvp(parameterDirections=directions)
        return _linear(actions.operator(), rhs, p.options, p.tangentSolve, p.report.tangent, 'Tangent')


def implicitSolve(residual, initial, parameters, primalSolve, *, options=None,
                  controls=(), controlToken=None, adjointSolve=None, tangentSolve=None):
    """Return ``(root, report)`` with a verified converged-root first derivative.

    The warm start has zero root derivative. For a finite iteration algorithm
    use ``fixedIterations`` instead. Fixed restrictions/gauges must be encoded
    in the residual (e.g. via RestrictedSystem/NullspaceSystem); validate the
    original physical residual separately when reducing a singular system.
    Reports store measured primal, adjoint and tangent residuals, not an
    iteration-cap heuristic. Custom linear solvers receive (operator, rhs) and
    their returned solutions are independently verified in the original system.
    """
    report = AdjointReport()
    problem = _Problem(residual, primalSolve, options or AdjointOptions(), report,
                       tuple(controls), controlToken, adjointSolve, tangentSolve)
    return _Implicit.apply(problem, initial, *tuple(parameters)), report


def fixedIterations(iteration, initial, parameters, count):
    """Differentiate exactly ``count`` updates, including the initial state.

    No root convergence or implicit gradient is inferred. Branches and history
    dependencies have the semantics of the supplied iteration's ordinary AD.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError('Iteration count must be a nonnegative integer')
    z = initial
    for _ in range(count):
        z = iteration(z, *parameters)
    return z
