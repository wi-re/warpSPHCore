"""Backend-neutral small symmetric dense operations."""
from .._backend import BACKEND
if BACKEND == "native":
    from ..pinv.native import pinvSymmetric,applyPinv,symmetricPinv3,loadPadded3x3
else:
    from .wp_pinv import pinvSymmetric,applyPinv,symmetricPinv3,loadPadded3x3

__all__ = ["pinvSymmetric","applyPinv","symmetricPinv3","loadPadded3x3"]
