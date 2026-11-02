"""Corrected interpolation/gradient primals with real nullable native inputs."""
import torch
from ..._models import CRKState, GradHState, RenormalizationState
from ...enumTypes import KernelFunctions, WarpOperation, GradientScheme
from ...native._inputs import prepare_geometry, prepare_adjacency
from ...native._checks import check_execution


def compute(query, reference, domain, properties, adjacency, query_values, reference_values,
            query_volumes=None, reference_volumes=None, crk_state=None,
            grad_h_state=None, renormalization_state=None):
    gradient = properties.operation == WarpOperation.Gradient
    if properties.operation not in (WarpOperation.Interpolate, WarpOperation.Gradient):
        raise NotImplementedError('Corrections currently support native interpolation and gradient primals only')
    if not isinstance(properties.kernel, KernelFunctions):
        raise TypeError('kernel must be a KernelFunctions')
    if properties.calibrateNormalization or properties.positiveDivergence:
        raise NotImplementedError('Native corrected primals do not support normalization calibration or positive divergence')
    if reference_values is None or (gradient and query_values is None):
        raise ValueError('Corrected operations require reference values and gradients require query values')
    if not gradient and (grad_h_state is not None or renormalization_state is not None):
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
        if gradient:
            gradA, gradB = state.gradA, state.gradB
            if gradA is None or gradB is None:
                raise ValueError('CRK gradients require gradA and gradB')
    omega = None
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
        if omega is None:
            raise ValueError('grad-h requires reference omegas')
    renorm = (renormalization_state.renormalizationMatrices
              if isinstance(renormalization_state, RenormalizationState) else renormalization_state)
    if renorm is not None and not isinstance(renorm, torch.Tensor):
        raise TypeError('renormalizationState must be a RenormalizationState or tensor')

    corrections = [query_volumes, reference_volumes, A, B, gradA, gradB, omega, renorm]
    used = [query.positions, reference.positions, query.supports, reference.supports,
            domain.min, domain.max, reference_values, reference_volumes, A, B, gradA, gradB, omega, renorm]
    if gradient:
        used.append(query_values)
    if reference_volumes is None or (gradient and properties.gradientMode == GradientScheme.Symmetric):
        used.extend((reference.masses, reference.densities))
    if gradient and properties.gradientMode == GradientScheme.Symmetric:
        used.append(query.densities)
    check_execution(*used, allow_reverse=True)
    if torch.is_grad_enabled() and any(isinstance(value, torch.Tensor) and value.requires_grad for value in used):
        raise NotImplementedError('Corrected native primals do not support autograd yet; use torch.no_grad() for primal execution')
    geometry = prepare_geometry(query, reference, domain, allow_reverse=True)
    adjacency = prepare_adjacency(query, reference, domain, properties, adjacency)
    inputs = [*geometry[:4], reference.masses, query.kinds, reference.kinds,
              *geometry[4:], adjacency.edgeOffsets, adjacency.numNeighbors, adjacency.paddedJ,
              query_values if gradient else None, reference_values, reference.densities, query.densities]
    from ...native._build import build
    name = 'gradient' if gradient else 'interpolate'
    return getattr(build(), name + '_corrected_forward')(inputs, corrections,
        properties.supportMode.value, properties.operationMode.value,
        properties.kernel.value, properties.gradientMode.value)
