"""Coordinate restriction and Dirichlet lifting of a declared linear action."""

import torch

from .operator import LinearOperator

__all__ = ['RestrictedSystem']


class RestrictedSystem:
    """Own a free-coordinate map for a square ``LinearOperator``.

    ``freeIndices`` is a unique 1-D int32/int64 tensor on the operator device.
    Its order defines reduced coordinates; indices address individual scalar
    entries of a flat vector (particle-major blocks must be expanded explicitly).
    Indices are copied, but the full operator and its coefficients are retained
    by reference. They must define a fixed action throughout each solve.

    For free-coordinate injection E and R = E.T, this helper exposes R A E,
    ``liftRhs(b, g) = R (b - A g)`` and ``reconstruct(y, g) = E y + g``.
    Prescribed g is a full vector with exactly zero free entries. It is supplied
    per call, never cached or folded into the linear action. None means zero.
    All-free, no-free and empty full systems are supported.

    Ordinary outputs and diagnostic index properties own their storage. No
    shared application scratch is retained. Optional outputs must match shape,
    dtype/device and must not overlap any input. Supplied-output use is intended
    for inference. Setup and prescribed-value validation may synchronize and
    must run outside CUDA capture; reduced operator application uses Torch's
    current stream. Neither symmetry nor positive definiteness is inferred.
    """

    def __init__(self, operator: LinearOperator, freeIndices: torch.Tensor):
        if not isinstance(operator, LinearOperator):
            raise ValueError('operator must be a LinearOperator with explicit metadata')
        if operator.shape[0] != operator.shape[1]:
            raise ValueError('restriction requires a square operator')
        if operator.device.type not in ('cpu', 'cuda'):
            raise ValueError('restriction supports CPU/CUDA operators')
        self._outsideCapture(operator.device)
        if (not isinstance(freeIndices, torch.Tensor) or freeIndices.ndim != 1
                or freeIndices.dtype not in (torch.int32, torch.int64)
                or freeIndices.device != operator.device):
            raise ValueError('freeIndices must be a flat int32/int64 tensor on the operator device')
        n = operator.shape[0]
        if bool(((freeIndices < 0) | (freeIndices >= n)).any()):
            raise ValueError('freeIndices contains an out-of-range coordinate')
        if freeIndices.unique().numel() != freeIndices.numel():
            raise ValueError('freeIndices must not contain duplicates')
        self._full = operator
        self._free = freeIndices.to(dtype=torch.int64).clone().contiguous()
        self._operator = LinearOperator(
            (self._free.numel(), self._free.numel()), operator.dtype, operator.device,
            matvec=self._apply, matvecOut=self._applyOut,
            adjoint=self._applyAdjoint if operator.adjoint is not None else None)

    @staticmethod
    def _outsideCapture(device):
        if device.type == 'cuda':
            with torch.cuda.device(device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError('restriction setup/prescribed validation must run outside CUDA capture')

    @property
    def shape(self):
        return self._operator.shape

    @property
    def dtype(self):
        return self._full.dtype

    @property
    def device(self):
        return self._full.device

    @property
    def fullShape(self):
        return self._full.shape

    @property
    def freeIndices(self):
        """Owned snapshot, in reduced-coordinate order."""
        return self._free.clone()

    @property
    def constrainedIndices(self):
        """Owned snapshot of the complementary scalar coordinates, sorted."""
        mask = torch.ones(self.fullShape[0], dtype=torch.bool, device=self.device)
        mask[self._free] = False
        return mask.nonzero().flatten()

    def asLinearOperator(self):
        """Stable adapter with an adjoint only if the full operator supplied one."""
        return self._operator

    def _validate(self, value, length, name):
        return self._full._validate(value, length, name)

    def _output(self, out, length, *inputs):
        if out is not None:
            self._validate(out, length, 'supplied output')
            if any(torch._C._overlaps(out, value) for value in inputs if value is not None):
                raise ValueError('supplied output must not overlap any input')

    @staticmethod
    def _write(value, out):
        if out is None:
            return value
        out.copy_(value)
        return out

    def restrict(self, v, out=None):
        """Gather a full vector into the declared free-coordinate order."""
        self._validate(v, self.fullShape[0], 'full input')
        self._output(out, self.shape[0], v)
        return self._write(v.index_select(0, self._free), out)

    def prolong(self, v, out=None):
        """Inject a reduced vector into a new full vector, zero elsewhere."""
        self._validate(v, self.shape[1], 'reduced input')
        self._output(out, self.fullShape[0], v)
        full = torch.zeros(self.fullShape[0], dtype=self.dtype, device=self.device)
        full.index_copy_(0, self._free, v)
        return self._write(full, out)

    def _prescribed(self, prescribed):
        if prescribed is not None:
            self._validate(prescribed, self.fullShape[0], 'prescribed vector')
            self._outsideCapture(self.device)
            if not bool(torch.isfinite(prescribed).all()):
                raise ValueError('prescribed vector must be finite')
            if bool((prescribed.index_select(0, self._free) != 0).any()):
                raise ValueError('prescribed vector must be zero on free coordinates')

    def liftRhs(self, b, prescribed=None, out=None):
        """Return R(b - A g); g is a full prescribed vector, zero on free rows.

        No full operator call is needed for None (homogeneous values), or when
        there are no free rows. This method does not alter b or prescribed.
        """
        self._validate(b, self.fullShape[0], 'full RHS')
        self._prescribed(prescribed)
        self._output(out, self.shape[0], b, prescribed)
        if prescribed is None or self.shape[0] == 0:
            return self.restrict(b, out)
        return self.restrict(b - self._full(prescribed), out)

    def reconstruct(self, v, prescribed=None, out=None):
        """Return E v + g, preserving the prescribed coordinates exactly."""
        self._validate(v, self.shape[1], 'reduced input')
        self._prescribed(prescribed)
        self._output(out, self.fullShape[0], v, prescribed)
        # Copying free entries avoids adding zero to prescribed values and
        # preserves even their signed-zero bit pattern.
        full = (torch.zeros(self.fullShape[0], dtype=self.dtype, device=self.device)
                if prescribed is None else prescribed.clone())
        full.index_copy_(0, self._free, v)
        return self._write(full, out)

    def _apply(self, v):
        full = self.prolong(v)
        if self.shape[0] == 0:
            return v.clone()
        return self.restrict(self._full(full))

    def _applyOut(self, v, out):
        return self._write(self._apply(v), out)

    def _applyAdjoint(self, v):
        full = self.prolong(v)
        if self.shape[0] == 0:
            return v.clone()
        return self.restrict(self._full.applyAdjoint(full))

    def __call__(self, v):
        return self._operator(v)

    def apply(self, v, out=None):
        """Apply R A E into optional caller storage."""
        return self._operator.apply(v, out)

    def applyAdjoint(self, v):
        """Apply R A.T E, requiring an explicit full adjoint."""
        return self._operator.applyAdjoint(v)

    def restrictInverseAction(self, inverse: LinearOperator):
        """Explicitly form R P E from an existing full inverse action P.

        This is generally NOT inverse(R A E). To build reduced Jacobi/block
        Jacobi, restrict the appropriate coefficients first and then construct
        that preconditioner. This adapter owns its index map and retains inverse
        by reference; coefficient updates remain visible between solves.
        """
        if (not isinstance(inverse, LinearOperator) or inverse.shape != self.fullShape
                or inverse.dtype != self.dtype or inverse.device != self.device):
            raise ValueError('full inverse action must match full operator shape, dtype and device')
        return RestrictedSystem(inverse, self._free).asLinearOperator()
