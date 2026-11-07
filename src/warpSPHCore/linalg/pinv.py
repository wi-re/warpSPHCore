"""Tensor operators follow OPERATORS; kernel helpers follow runtime BACKEND."""
from .._backend import BACKEND, OPERATORS

if OPERATORS == "native":
    from ..pinv.native import pinvSymmetric as _tensor_inverse, applyPinv as _tensor_apply
    if BACKEND == "native":
        pinvSymmetric, applyPinv = _tensor_inverse, _tensor_apply
    else:
        # Preserve the Warp tensor API's public keyword names in hybrid mode.
        def pinvSymmetric(M, rtol=None):
            return _tensor_inverse(M, rtol)

        def applyPinv(V, M, rtol=None):
            return _tensor_apply(V, M, rtol)
else:
    from .wp_pinv import pinvSymmetric, applyPinv

# Warp frontends still compile functions which call these two helpers. Keep
# their Warp declarations even when the tensor entry points use native kernels.
if BACKEND == "native":
    from ..pinv.native import symmetricPinv3, loadPadded3x3
else:
    from .wp_pinv import symmetricPinv3, loadPadded3x3

__all__ = ["pinvSymmetric", "applyPinv", "symmetricPinv3", "loadPadded3x3"]
