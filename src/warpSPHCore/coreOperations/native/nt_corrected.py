"""Corrected core primals with real nullable native inputs."""
import torch
from ..._models import CRKState, GradHState, RenormalizationState
from ...enumTypes import KernelFunctions, WarpOperation, GradientScheme, LaplacianScheme
from ...native._inputs import prepare_geometry, prepare_adjacency
from ...native._checks import check_execution


def compute(query, reference, domain, properties, adjacency, query_values, reference_values,
            query_volumes=None, reference_volumes=None, crk_state=None,
            grad_h_state=None, renormalization_state=None, *, consistent_divergence=False,
            return_neighbor_counts=False):
    operation = properties.operation
    interpolation = operation == WarpOperation.Interpolate
    covariance = operation == WarpOperation.Covariance
    laplacian = operation == WarpOperation.Laplacian
    if operation not in (WarpOperation.Interpolate, WarpOperation.Gradient, WarpOperation.Divergence,
                          WarpOperation.Curl, WarpOperation.Laplacian, WarpOperation.Covariance):
        raise NotImplementedError('Corrections do not apply to this native operation')
    if consistent_divergence and operation != WarpOperation.Divergence:
        raise ValueError('consistentDivergence requires divergence')
    if return_neighbor_counts and not covariance:
        raise ValueError('covarianceReturnNumNeighbors requires covariance')
    differential = not interpolation and not (laplacian and properties.laplacianMode == LaplacianScheme.Naive)
    # Naive Laplacian uses the raw second derivative. Covariance never reads
    # grad-h or supplied fields. Do not validate/load unused correction arrays.
    if laplacian and not differential:
        crk_state = renormalization_state = None
    if covariance:
        grad_h_state = None
    if not isinstance(properties.kernel, KernelFunctions):
        raise TypeError('kernel must be a KernelFunctions')
    if properties.calibrateNormalization or properties.positiveDivergence:
        raise NotImplementedError('Native corrected primals do not support normalization calibration or positive divergence')
    if not covariance and (reference_values is None or (not interpolation and query_values is None)):
        raise ValueError('Corrected field operations require reference values and differential operations require query values')
    if interpolation and (grad_h_state is not None or renormalization_state is not None):
        raise NotImplementedError('grad-h and renormalization are native gradient corrections')
    # Match the public Warp volume fallback. Only reference volumes enter these
    # directed operators; an independently supplied query volume is not loaded.
    reference_volumes = reference_volumes if reference_volumes is not None else query_volumes
    state = crk_state
    if isinstance(state, tuple) and len(state) == 2:
        state = state[0]  # These operators use query-side coefficients only.
    if state is not None and not isinstance(state, CRKState):
        raise TypeError('crkState must be a CRKState or a query/reference pair')
    A, B, gradA, gradB = (None,) * 4
    if state is not None:
        A, B = state.A, state.B
        if A is None or B is None:
            raise ValueError('CRK requires both A and B')
        if differential:
            gradA, gradB = state.gradA, state.gradB
            if gradA is None or gradB is None:
                raise ValueError('CRK gradients require gradA and gradB')
    omega = query_omega = None
    if grad_h_state is not None:
        if isinstance(grad_h_state, GradHState):
            left, right = grad_h_state.queryOmegas, grad_h_state.referenceOmegas
        elif isinstance(grad_h_state, tuple) and len(grad_h_state) == 2:
            left, right = grad_h_state
        elif isinstance(grad_h_state, torch.Tensor):
            left = right = grad_h_state
        else:
            raise TypeError('gradHState must be a GradHState, tensor or query/reference pair')
        omega = right if right is not None else left
        if operation in (WarpOperation.Divergence, WarpOperation.Curl, WarpOperation.Laplacian):
            query_omega = left if left is not None else right
        if omega is None:
            raise ValueError('grad-h requires reference omegas')
    renorm = (renormalization_state.renormalizationMatrices
              if isinstance(renormalization_state, RenormalizationState) else renormalization_state)
    if renorm is not None and not isinstance(renorm, torch.Tensor):
        raise TypeError('renormalizationState must be a RenormalizationState or tensor')

    corrections = [query_volumes, reference_volumes, A, B, gradA, gradB, omega, renorm, query_omega]
    used = [query.positions, reference.positions, query.supports, reference.supports,
            domain.min, domain.max, None if covariance else reference_values, reference_volumes, A, B, gradA, gradB, omega, renorm, query_omega]
    if not interpolation and not covariance:
        used.append(query_values)
    symmetric = not interpolation and not covariance and properties.gradientMode == GradientScheme.Symmetric
    if reference_volumes is None or symmetric:
        used.extend((reference.masses, reference.densities))
    if symmetric:
        used.append(query.densities)
    if consistent_divergence:
        used.extend((query.densities, reference.densities))
    check_execution(*used, allow_reverse=True)
    if torch.is_grad_enabled() and any(isinstance(value, torch.Tensor) and value.requires_grad for value in used):
        raise NotImplementedError('Corrected native primals do not support autograd yet; use torch.no_grad() for primal execution')
    geometry = prepare_geometry(query, reference, domain, allow_reverse=True)
    adjacency = prepare_adjacency(query, reference, domain, properties, adjacency)
    inputs = [*geometry[:4], reference.masses, query.kinds, reference.kinds,
              *geometry[4:], adjacency.edgeOffsets, adjacency.numNeighbors, adjacency.paddedJ,
              query_values if not interpolation and not covariance else None,
              None if covariance else reference_values, reference.densities, query.densities]
    from ...native._build import build
    name = operation.name.lower()
    options = (properties.supportMode.value, properties.operationMode.value,
               properties.kernel.value, properties.gradientMode.value)
    function = getattr(build(), name + '_corrected_forward')
    if operation in (WarpOperation.Interpolate, WarpOperation.Gradient):
        return function(inputs, corrections, *options)
    options += (properties.laplacianMode.value, properties.divergenceDotMode)
    if operation == WarpOperation.Divergence:
        return function(inputs, corrections, *options, consistent_divergence)
    if covariance:
        result = function(inputs, corrections, *options, return_neighbor_counts)
        return tuple(result) if return_neighbor_counts else result[0]
    return function(inputs, corrections, *options)
