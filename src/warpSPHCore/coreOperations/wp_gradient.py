"""Compatibility alias; implementations live beside the C++/CUDA sources."""
import sys
from .native import wp_gradient as _implementation
sys.modules[__name__] = _implementation
