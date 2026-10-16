"""Locations for local development evidence outside the source checkout.

WARPSPH_DEVELOPMENT_ROOT names the shared archive root, not its core child.
Resolving paths does not create directories or require an archive to exist.
"""

import os
from pathlib import Path


def coreDevelopmentPath(*parts):
    default = Path(__file__).resolve().parents[2] / 'warpSPHDevelopment'
    root = Path(os.environ.get('WARPSPH_DEVELOPMENT_ROOT', str(default))).expanduser()
    return root.resolve() / 'core' / Path(*parts)


def artifactLabel(path):
    """Archive-relative manifest labels, or absolute paths for custom outputs."""
    path, root = Path(path).resolve(), coreDevelopmentPath()
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
