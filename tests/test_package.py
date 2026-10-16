"""Version agreement protects package introspection and the release script."""

from pathlib import Path
import re

from warpSPHCore import __version__


def test_version_matches_project_metadata():
    project = (Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text()
    declared = re.search(r'^version\s*=\s*"([^"]+)"', project, re.MULTILINE)
    assert declared is not None
    assert __version__ == declared.group(1)
