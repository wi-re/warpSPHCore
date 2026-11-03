"""Compatibility import for the selected backend's prepared meshless passes."""
from .._backend import BACKEND
if BACKEND == 'native':
    from .native.nt_mfm import PreparedLauncher
else:
    from ..autograd.prepared import PreparedLauncher
__all__ = ['PreparedLauncher']
