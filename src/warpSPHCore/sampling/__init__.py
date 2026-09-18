"""Host-side particle sampling (numpy, float64).

Densest-packing lattices for periodic domains (1D uniform, 2D
hexagonal, 3D FCC) — see `lattice.py`. Shared by the D&A 2012
replication (higherOrderSPH/) and wrapped by the warpSPH frontend's
`sample/` package.
"""
from .lattice import *
