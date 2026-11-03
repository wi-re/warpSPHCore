"""Compatibility alias; implementations live beside native operator sources."""
import sys
from .native import wp_curl as _implementation
sys.modules[__name__] = _implementation
