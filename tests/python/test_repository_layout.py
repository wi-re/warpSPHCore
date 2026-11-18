"""Check the index that would be committed, including force-added outputs."""

import json
from pathlib import Path
import runpy
import subprocess

layout = runpy.run_path(Path(__file__).resolve().parents[2] / 'scripts/check_repository_layout.py')


def test_notebook_check_uses_index(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    path = tmp_path / 'probe.ipynb'

    def write(count):
        path.write_text(json.dumps({
            'cells': [{'cell_type': 'code', 'execution_count': count, 'outputs': []}],
        }))

    write(7)
    subprocess.run(['git', 'add', path.name], cwd=tmp_path, check=True)
    write(None)
    assert layout['check_index'](tmp_path) == ['probe.ipynb: notebook execution outputs or counts']

    subprocess.run(['git', 'add', path.name], cwd=tmp_path, check=True)
    write(7)
    assert layout['check_index'](tmp_path) == []


def test_build_outputs_are_rejected():
    paths = ['dist/core.whl', 'docs/build/404.html', 'scripts/native.so', 'docs/docs/guide.md']
    assert {problem.split(':', 1)[0] for problem in layout['violations'](paths)} == set(paths[:3])
