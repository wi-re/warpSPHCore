"""Version agreement protects package introspection and the release script."""

from pathlib import Path
import re

from warpSPHCore import __version__


def test_version_matches_project_metadata():
    root = Path(__file__).resolve().parents[2]
    project = (root / 'pyproject.toml').read_text()
    declared = re.search(r'^version\s*=\s*"([^"]+)"', project, re.MULTILINE)
    cmake_version = re.search(r'project\(warpSPHCore\s+VERSION\s+(\S+)', (root / 'CMakeLists.txt').read_text())
    assert declared is not None
    assert cmake_version is not None
    assert __version__ == declared.group(1) == cmake_version.group(1)
