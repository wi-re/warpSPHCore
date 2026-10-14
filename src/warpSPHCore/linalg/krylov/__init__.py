"""Krylov solvers with device-resident stopping tests (see `common.py`)."""

from .cg import cg
from .common import SolveResult, Status, absoluteTolerance, makePsolve, thresholdExceeded

__all__ = ['cg', 'SolveResult', 'Status', 'absoluteTolerance', 'makePsolve', 'thresholdExceeded']
