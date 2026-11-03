"""Fixed-geometry field linear products for matrix-free native solvers."""
from typing import NamedTuple, Optional
from dataclasses import replace
import torch
from torch.autograd.function import once_differentiable
from ..enumTypes import WarpOperation
from .._models import OperationProperties
from ._checks import check_execution
from ._inputs import prepare_adjacency
from ._topology import CSRTopology, tensor_token


class FieldCotangents(NamedTuple):
    query: Optional[torch.Tensor]
    reference: Optional[torch.Tensor]


class _FieldApply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, linearization, query, reference):
        linearization._check()
        ctx.linearization = linearization
        ctx.save_for_backward(*linearization._fixed)
        return linearization._product(query, reference)

    @staticmethod
    @once_differentiable
    def backward(ctx, cotangent):
        # Access saved tensors for Torch's normal version checks. Primal fields
        # are not saved: this linear map's pullback does not depend on them.
        _ = ctx.saved_tensors
        q, r = ctx.linearization._pullback(cotangent.contiguous(),
                                           ctx.needs_input_grad[1], ctx.needs_input_grad[2])
        return None, q, r


class FieldLinearization:
    """A linear field map with borrowed, fixed geometry and correction arrays.

    apply() supports first-order field backwards. jvp()/vjp() are explicit
    numerical products and do not record a derivative graph. Query/reference
    roles are independent; pass both tangents for an aliased field, or sum the
    two returned cotangents. No geometry/correction/preparation derivatives are
    implied. Replacing external state attributes does not update this map;
    in-place changes to captured tensors invalidate it.
    """
    def __init__(self, inputs, corrections, properties, topology, consistent, query, reference, alias):
        from ._options import normalization_factor
        self._scale = normalization_factor(properties, len(inputs[7]))
        self.topology = topology
        self._topology_token = topology.token()
        self._inputs = [t.detach() if isinstance(t, torch.Tensor) else t for t in inputs]
        self._corrections = [t.detach() if isinstance(t, torch.Tensor) else t for t in corrections]
        self._fixed = [t for i, t in enumerate(self._inputs) if i not in (13, 14) and isinstance(t, torch.Tensor)]
        self._fixed += [t for t in self._corrections if isinstance(t, torch.Tensor)]
        self._tokens = tuple(tensor_token(t) for t in self._fixed)
        self._parameters = (properties.operation.value, properties.supportMode.value,
            properties.operationMode.value, properties.kernel.value, properties.gradientMode.value,
            properties.laplacianMode.value, properties.divergenceDotMode, consistent)
        self._alias = alias
        self.value = self.apply(query, reference)

    def _check(self):
        if self.topology.token() != self._topology_token:
            raise RuntimeError('CSR topology changed; create a new field linearization')
        if tuple(tensor_token(t) for t in self._fixed) != self._tokens:
            raise RuntimeError('Fixed native geometry/corrections changed; create a new field linearization')

    def _product(self, query, reference):
        from ._build import build
        self._check()
        value = build().field_product(self._inputs, self._corrections, query, reference, *self._parameters)
        return value if self._scale == 1. else value * self._scale

    def _pullback(self, cotangent, query, reference):
        from ._build import build
        self._check()
        transpose = []
        if reference:
            csc = self.topology.transpose()
            transpose = [csc.edgeOffsets, csc.numNeighbors, csc.paddedJ]
        cotangent = cotangent if self._scale == 1. else cotangent * self._scale
        return FieldCotangents(*build().field_pullback(self._inputs, self._corrections,
            cotangent, transpose, query, reference, *self._parameters))

    def apply(self, query=None, reference=None):
        """Evaluate new field values; first-order Torch backwards is available."""
        if reference is None and self._alias:
            reference = query
        check_execution(query, reference, allow_reverse=True)
        return _FieldApply.apply(self, query, reference)

    def jvp(self, *, query=None, reference=None):
        """Apply independent role tangents; an omitted role is exactly zero."""
        check_execution(query, reference)
        with torch.no_grad():
            return self._product(query, reference)

    def vjp(self, cotangent, *, query=True, reference=True):
        """Return independent role cotangents, gathering through CSR and CSC."""
        check_execution(cotangent)
        with torch.no_grad():
            return self._pullback(cotangent, query, reference)


def linearizeFields(queryParticles, operationProperties, domain, queryValues=None,
                    referenceValues=None, queryVolumes=None, referenceVolumes=None,
                    adjacency=None, referenceParticles=None, crkState=None,
                    gradHState=None, renormalizationState=None, consistentDivergence=False):
    """Bind interpolation/gradient/divergence/curl/Laplacian at fixed geometry.

    Pass CSRTopology as adjacency to share its lazy CSC across operators/solves.
    AdjacencyList or omitted adjacency also work. Construction requires fixed
    non-differentiable coefficients, or an explicit torch.no_grad() scope to
    freeze differentiable geometry/corrections. The ordinary corrected warpOperation API
    has a separate full first-order derivative path.
    """
    if not isinstance(operationProperties, OperationProperties):
        raise TypeError('operationProperties must be an OperationProperties')
    if operationProperties.operation not in (WarpOperation.Interpolate, WarpOperation.Gradient,
            WarpOperation.Divergence, WarpOperation.Curl, WarpOperation.Laplacian):
        raise NotImplementedError('Field linearization supports interpolation, gradient, divergence, curl and Laplacian')
    if operationProperties.operation == WarpOperation.Laplacian and operationProperties.positiveDivergence:
        raise ValueError('Positive-divergence Laplacian is nonlinear in fields; use ordinary Torch JVP/VJP')
    check_execution(queryValues, referenceValues, allow_reverse=True)
    alias = referenceValues is None
    referenceValues = queryValues if alias else referenceValues
    reference = queryParticles if referenceParticles is None else referenceParticles
    topology = adjacency if isinstance(adjacency, CSRTopology) else CSRTopology(
        prepare_adjacency(queryParticles, reference, domain, operationProperties, adjacency))
    from ..coreOperations.native.nt_corrected import _prepare, _check_primal
    inputs, corrections, used = _prepare(queryParticles, reference, domain, replace(operationProperties, calibrateNormalization=False),
        topology.adjacency, queryValues.detach() if queryValues is not None else None,
        referenceValues.detach() if referenceValues is not None else None,
        queryVolumes, referenceVolumes, crkState, gradHState, renormalizationState,
        consistent_divergence=consistentDivergence)
    _check_primal(used)
    return FieldLinearization(inputs, corrections, operationProperties, topology,
                              consistentDivergence, queryValues, referenceValues, alias)
