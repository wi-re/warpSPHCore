"""Pre-import configuration for sphWarpCore.

Usage:

    import sphWarpCore_config as swc
    swc.configure(precision="float64", dim=2)

    import sphWarpCore

This avoids environment variables while still configuring types before
`sphWarpCore` modules are imported.
"""

from __future__ import annotations

from typing import Optional


precision: Optional[str] = None
dim: Optional[int | str] = None


def configure(*, precision: Optional[str] = None, dim: Optional[int | str] = None) -> None:
    """Set pre-import precision and dimension overrides.

    Parameters are intentionally permissive and validated by
    `sphWarpCore.type_config` when the package is imported.
    """
    globals()["precision"] = precision
    globals()["dim"] = dim


__all__ = ["precision", "dim", "configure"]
