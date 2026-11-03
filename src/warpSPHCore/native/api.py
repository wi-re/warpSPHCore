"""Compatibility entry points; operation policy lives with its owning module.

Explicit imports from native.api continue to request the native implementation,
including in reference-backend qualification running in a Warp process.
"""
from functools import partial
from ..enumTypes import KernelFunctions, WarpOperation, SupportScheme
from ._autograd import REVERSE_KERNELS
from ..coreOperations.native import sphOperation as _operation
from ..radiusSearch.native import radiusSearch as _radius

warpOperation = partial(_operation, backend='native')
sphOperation = warpOperation
radiusSearch = partial(_radius, backend='native')
from ..coreOperations.native.nt_jvp import warpOperationJVP

def nativeCapabilities():
    """Declared native scope; no runtime initialisation or compilation."""
    return dict(operations=tuple(op.name.lower() for op in WarpOperation if op != WarpOperation.Custom),
                kernels=tuple(k.name for k in KernelFunctions), dimensions=(1, 2, 3),
                precision=('float32', 'float64'), execution=('cpu', 'cuda', 'hip'),
                experimental_execution=('hip',), support_schemes=tuple(s.name for s in SupportScheme),
                autograd=True, autograd_order=1, forward_ad=True, higher_order=False,
                reverse_kernels=tuple(k.name for k in KernelFunctions if k.value in REVERSE_KERNELS),
                field_ranks=(0, 1, 2), deterministic_adjoint=False, radius_search='compact_hash', graph_capture=False,
                graph_capture_fixed_adjacency=('interpolate_primal',),
                warp_required=False, row_chunked_primal=True, row_chunked_ad=False,
                row_chunked_graph_capture=False, row_chunked_index_dtypes=('int32', 'int64'),
                verlet_lifecycle=True, verlet_validity_device_flags=True,
                normalization_calibration=True, positive_divergence_laplacian=True,
                renormalization_construction=True, renormalization_forward_ad=True,
                small_dense=True, native_krylov_workspaces=('cg', 'bicgstab'),
                krylov_chunk_capture=True, polynomial_reconstruction=True,
                polynomial_forward_ad=True, polynomial_reverse_ad_order=1,
                meshless_primal=True, meshless_forward_ad=False, meshless_fixed_geometry_capture=True,
                corrected_primals=('interpolate', 'gradient', 'divergence', 'curl', 'laplacian', 'covariance'), corrected_autograd=True,
                corrections=('crk', 'volume', 'grad_h_gradient', 'renormalization_gradient'),
                crk_factor_construction=True, crk_preparation_autograd=True,
                grad_h_primals=('gradient', 'divergence', 'curl', 'laplacian'),
                renormalized_primals=('gradient', 'divergence', 'curl', 'laplacian', 'covariance'),
                consistent_divergence_primal=True, covariance_neighbor_counts_primal=True,
                field_linearization_operations=('interpolate', 'gradient', 'divergence', 'curl', 'laplacian'),
                field_jvp=True, field_vjp=True, field_backward_order=1,
                field_geometry_derivatives=False, field_adjoint_ownership='csr_csc',
                field_adjoint_floating_atomics=False, field_adjoint_cached_topology=True,
                corrected_forward_ad=True, corrected_ad_order=1,
                corrected_geometry_kernels=tuple(k.name for k in KernelFunctions if k.value in REVERSE_KERNELS),
                corrected_adjoint_ownership='csr_csc', corrected_adjoint_floating_atomics=False,
                crk_preparation_forward_ad=True, crk_preparation_ad_order=1,
                explicit_jvp=True, density_position_hvp=True,
                density_position_hvp_geometry_ad=False, density_position_hvp_vector_mass_ad_order=1)
