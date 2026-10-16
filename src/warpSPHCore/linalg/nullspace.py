"""Declared Euclidean nullspaces, explicit RHS policies and solution gauges."""

from dataclasses import dataclass, field

import torch
from torch.autograd import forward_ad

from .krylov.common import absoluteTolerance, converged, finiteScalar
from .operator import LinearOperator

__all__ = ['NullspaceSystem', 'NullspaceRhs', 'NullspaceResiduals']


def _tolerance(name, value):
    finiteScalar(name, value)
    return float(value)


@dataclass(frozen=True)
class NullspaceRhs:
    """Owned RHS snapshots; projection is performed only under policy='project'.

    ``compatible`` describes the requested source under the preparation
    tolerance. It does not certify an operator solve. Treat snapshots as
    read-only while solving/verifying.
    """

    requested: torch.Tensor
    rhs: torch.Tensor
    removed: torch.Tensor
    incompatibleNorm: torch.Tensor
    tolerance: torch.Tensor
    compatible: bool
    policy: str
    _system: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class NullspaceResiduals:
    """Device scalar norms/flags; no solver status is relabeled by this report.

    requested/solved use the ORIGINAL action with the requested/selected source.
    projected uses the projected action with the selected source. Tolerances
    scale with each respective source. Gauge reconstruction should precede
    this check.
    """

    requestedResidual: torch.Tensor
    solvedResidual: torch.Tensor
    projectedResidual: torch.Tensor
    removedNorm: torch.Tensor
    requestedConverged: torch.Tensor
    solvedConverged: torch.Tensor
    projectedConverged: torch.Tensor


class NullspaceSystem:
    """Project a square action using explicitly supplied orthonormal bases.

    ``rightBasis`` and ``leftBasis`` have shape (n, k), with equal k, and
    orthonormal columns in the Euclidean inner product. Empty bases are allowed.
    The helper owns copies and validates A Q_right and A.T Q_left by applying
    the supplied callbacks. A nonempty basis requires an explicit adjoint;
    equal bases do not imply symmetry. Basis completeness is the caller's
    declaration: matrix-free checks cannot certify that no modes are missing.

    The callable action is P_left A P_right, with P = I - Q Q.T; its adjoint
    is P_right A.T P_left. With a self-adjoint A and identical left/right
    subspaces, these Euclidean projections preserve the self-adjoint solved
    subspace. They do not certify A or preconditioner definiteness. Weighted
    and pinned gauges only add right-nullspace components AFTER solving.

    Setup/validation, RHS policy selection and gauges may synchronize and must
    run outside CUDA capture. Ordinary applications use Torch's current stream
    without shared scratch. Coefficients stay owned by the original action;
    call validate() after changes, and keep the action fixed during a solve.
    Basis setup is nondifferentiable; ordinary vector projection/application
    preserves AD when the callbacks do. No implicit solve AD is supplied.
    """

    def __init__(self, operator, rightBasis, leftBasis, *, basisTolerance=None,
                 nullTolerance=None):
        if not isinstance(operator, LinearOperator) or operator.shape[0] != operator.shape[1]:
            raise ValueError('nullspace requires a square LinearOperator')
        if operator.device.type not in ('cpu', 'cuda'):
            raise ValueError('nullspace supports CPU/CUDA operators')
        self._full = operator
        self._outsideCapture()
        default = 64 * torch.finfo(operator.dtype).eps
        self._basisTolerance = default if basisTolerance is None else _tolerance('basisTolerance', basisTolerance)
        self._nullTolerance = default if nullTolerance is None else _tolerance('nullTolerance', nullTolerance)
        if self._basisTolerance >= 1:
            raise ValueError('basisTolerance must be less than 1')
        self._right = self._basis(rightBasis, 'rightBasis')
        self._left = self._basis(leftBasis, 'leftBasis')
        if self._right.shape != self._left.shape:
            raise ValueError('left/right bases must have the same number of modes')
        if self._right.shape[1] and operator.adjoint is None:
            raise ValueError('left nullspace validation requires an explicit adjoint')
        self.validate()
        self._operator = LinearOperator(
            operator.shape, operator.dtype, operator.device, self._apply,
            lambda v, out: out.copy_(self._apply(v)),
            self._applyAdjoint if operator.adjoint is not None else None)

    def _outsideCapture(self):
        if self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError('nullspace setup/validation/RHS policies/gauges must run outside CUDA capture')

    @property
    def shape(self):
        return self._full.shape

    @property
    def dtype(self):
        return self._full.dtype

    @property
    def device(self):
        return self._full.device

    @property
    def modes(self):
        return self._right.shape[1]

    @property
    def rightBasis(self):
        return self._right.clone()

    @property
    def leftBasis(self):
        return self._left.clone()

    def _basis(self, value, name):
        if (not isinstance(value, torch.Tensor) or value.ndim != 2
                or value.shape[0] != self.shape[0] or value.shape[1] > self.shape[0]
                or value.dtype != self.dtype or value.device != self.device):
            raise ValueError(f'{name} must have shape (n, k), k <= n, and operator dtype/device')
        if value.requires_grad or forward_ad.unpack_dual(value).tangent is not None:
            raise ValueError('nullspace basis setup does not differentiate coefficients')
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f'{name} must be finite')
        k = value.shape[1]
        gram = value.T @ value
        eye = torch.eye(k, dtype=self.dtype, device=self.device)
        if k and not bool(torch.linalg.matrix_norm(gram - eye, ord=float('inf')) <= self._basisTolerance):
            raise ValueError(f'{name} columns must be orthonormal (and independent)')
        return value.clone().contiguous()

    def validate(self):
        """Check each declared mode with absolute L2 tolerance nullTolerance.

        This tolerance has action units; no operator norm/scale is inferred.
        Both default tolerances are 64 * eps(dtype). Inexact modes can change
        the original residual after projection/gauging; verify() checks it.
        """
        self._outsideCapture()
        for basis, action, name in ((self._right, self._full, 'right'),
                                    (self._left, self._full.applyAdjoint, 'left')):
            for q in basis.unbind(1):
                residual = torch.linalg.vector_norm(action(q))
                if not bool(torch.isfinite(residual) & (residual <= self._nullTolerance)):
                    raise ValueError(f'declared {name} basis is not annihilated within nullTolerance')
        return self

    def _vector(self, value, name='input'):
        return self._full._validate(value, self.shape[0], name)

    def _project(self, v, basis, out):
        self._vector(v)
        if out is not None:
            self._vector(out, 'supplied output')
            if torch._C._overlaps(v, out):
                raise ValueError('supplied output must not overlap input')
        result = v - basis @ (basis.T @ v) if basis.shape[1] else v.clone()
        if out is None:
            return result
        out.copy_(result)
        return out

    def projectRight(self, v, out=None):
        """Euclidean orthogonal projection of solution/warm-start coordinates."""
        return self._project(v, self._right, out)

    def projectLeft(self, v, out=None):
        """Euclidean projection of source/residual coordinates."""
        return self._project(v, self._left, out)

    def _apply(self, v):
        return self.projectLeft(self._full(self.projectRight(v)))

    def _applyAdjoint(self, v):
        return self.projectRight(self._full.applyAdjoint(self.projectLeft(v)))

    def __call__(self, v):
        return self._operator(v)

    def apply(self, v, out=None):
        return self._operator.apply(v, out)

    def applyAdjoint(self, v):
        return self._operator.applyAdjoint(v)

    def asLinearOperator(self):
        return self._operator

    def prepareRhs(self, b, *, policy='error', rtol=None, atol=0.0):
        """Check compatibility; explicit policy='project' removes left modes.

        policy='error' returns an unchanged snapshot if compatible under
        max(atol, rtol * ||b||), otherwise raises. Default rtol is 64*eps.
        A tolerated tiny incompatible component is not silently removed.
        """
        self._outsideCapture()
        self._vector(b, 'RHS')
        if policy not in ('error', 'project'):
            raise ValueError("policy must be 'error' or 'project'")
        rtol = 64 * torch.finfo(self.dtype).eps if rtol is None else _tolerance('rtol', rtol)
        atol = _tolerance('atol', atol)
        if not bool(torch.isfinite(b).all()):
            raise ValueError('RHS must be finite')
        requested = b.clone()
        component = self._left @ (self._left.T @ requested)
        norm = torch.linalg.vector_norm(component)
        tolerance = absoluteTolerance(requested.norm(), 0, rtol, atol)
        if not bool(torch.isfinite(requested.norm()) & torch.isfinite(tolerance)):
            raise ValueError('RHS norm and compatibility tolerance must be finite')
        compatible = bool(converged(norm, tolerance))
        if policy == 'error' and not compatible:
            raise ValueError('RHS is incompatible with the declared left nullspace; use explicit policy=\'project\' to change it')
        rhs = self.projectLeft(requested) if policy == 'project' else requested.clone()
        return NullspaceRhs(requested, rhs, requested - rhs, norm, tolerance,
                            compatible, policy, self)

    def projectPreconditioner(self, inverse):
        """Return P_right M_inverse P_left, with an explicit optional adjoint.

        For identical left/right subspaces and an SPD inverse, symmetry and
        positivity are preserved on the solved subspace. No certificate of
        those assumptions is inferred, and the full action stays singular.
        """
        if (not isinstance(inverse, LinearOperator) or inverse.shape != self.shape
                or inverse.dtype != self.dtype or inverse.device != self.device):
            raise ValueError('inverse action must match operator shape, dtype and device')

        def action(v):
            return self.projectRight(inverse(self.projectLeft(v)))

        def adjoint(v):
            return self.projectLeft(inverse.applyAdjoint(self.projectRight(v)))

        return LinearOperator(self.shape, self.dtype, self.device, action,
                              lambda v, out: out.copy_(action(v)),
                              adjoint if inverse.adjoint is not None else None)

    def _gauge(self, v, functionals, values):
        self._outsideCapture()
        self._vector(v)
        if values is None:
            values = torch.zeros(self.modes, dtype=self.dtype, device=self.device)
        self._full._validate(values, self.modes, 'gauge values')
        if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(v).all()):
            raise ValueError('gauge input/values must be finite')
        if not self.modes:
            return v.clone()
        coupling = functionals.T @ self._right
        singular = torch.linalg.svdvals(coupling)
        if not bool(torch.isfinite(singular).all()) or not bool(singular[-1] > self._basisTolerance * singular[0]):
            raise ValueError('gauge constraints must independently determine every right-nullspace mode')
        delta = torch.linalg.solve(coupling, values - functionals.T @ v)
        result = v + self._right @ delta
        if not bool(torch.isfinite(result).all()):
            raise ValueError('gauge reconstruction is not finite')
        return result

    def gauge(self, v, coefficients=None):
        """Set Q_right.T x = coefficients (default zero).

        For normalized componentwise constants this selects zero arithmetic
        mean on each component. Other bases impose orthogonal coefficients,
        not a presumed physical mean.
        """
        return self._gauge(v, self._right, coefficients)

    def weightedGauge(self, v, weights, values=None):
        """Set k weighted means; weights has shape (n, k), nonnegative columns.

        Each positive column is normalized to sum one. The k functionals must
        independently fix the k right-nullspace modes. This only shifts the
        final solution; it is not the solver's residual/solution projector.
        """
        self._outsideCapture()
        if (not isinstance(weights, torch.Tensor) or weights.shape != self._right.shape
                or weights.dtype != self.dtype or weights.device != self.device):
            raise ValueError('weights must match rightBasis shape, dtype and device')
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError('gauge weights must be finite and nonnegative')
        # Scale before summing to avoid overflow for otherwise valid weights.
        if self.modes:
            scale = weights.amax(dim=0)
            if not bool((scale > 0).all()):
                raise ValueError('every gauge weight column must have positive total')
            scaled = weights / scale
            functionals = scaled / scaled.sum(dim=0)
        else:
            functionals = weights
        return self._gauge(v, functionals, values)

    def pinGauge(self, v, indices, values=None):
        """Pin k scalar coordinates, with independent coupling to the k modes."""
        self._outsideCapture()
        if (not isinstance(indices, torch.Tensor) or indices.shape != (self.modes,)
                or indices.dtype not in (torch.int32, torch.int64) or indices.device != self.device):
            raise ValueError('pin indices must be a flat int32/int64 device tensor of length modes')
        if bool(((indices < 0) | (indices >= self.shape[0])).any()) or indices.unique().numel() != self.modes:
            raise ValueError('pin indices must be unique and in range')
        functionals = torch.zeros_like(self._right)
        functionals[indices.long(), torch.arange(self.modes, device=self.device)] = 1
        return self._gauge(v, functionals, values)

    def verify(self, x, prepared, *, rtol=1e-5, atol=0.0):
        """Report requested/selected ORIGINAL residuals and projected residual.

        A successful solve of a changed source does not certify the requested
        equation. Check requestedConverged when the original source is required.
        """
        self._vector(x, 'solution')
        if not isinstance(prepared, NullspaceRhs) or prepared._system is not self:
            raise ValueError('prepared RHS must come from this NullspaceSystem')
        rtol, atol = _tolerance('rtol', rtol), _tolerance('atol', atol)
        for v in (prepared.requested, prepared.rhs, prepared.removed):
            self._vector(v, 'prepared RHS')
        action = self._full(x)
        requested = (prepared.requested - action).norm()
        solved = (prepared.rhs - action).norm()
        projected = (prepared.rhs - self(x)).norm()
        originalTarget = absoluteTolerance(prepared.requested.norm(), 0, rtol, atol)
        selectedTarget = absoluteTolerance(prepared.rhs.norm(), 0, rtol, atol)
        finite = torch.isfinite(x).all()
        return NullspaceResiduals(
            requested, solved, projected, (prepared.requested - prepared.rhs).norm(),
            finite & converged(requested, originalTarget),
            finite & converged(solved, selectedTarget),
            finite & converged(projected, selectedTarget))
