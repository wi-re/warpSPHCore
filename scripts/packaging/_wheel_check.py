"""Run wheel checks with installed dependencies and no site startup hooks."""

from pathlib import Path
import site
import subprocess
import sys


def run_wheel_python(directory, code, *args, env=None):
    root = Path(directory).resolve()
    # -I ignores PYTHONPATH and the user site, but still runs global .pth
    # files. -S also disables those startup hooks, including editable finders.
    bootstrap = f"""\
import sys
from pathlib import Path
wheel_root = Path(sys.argv[1])
sys.path.insert(0, str(wheel_root))
sys.path.extend({site.getsitepackages()!r})

def assert_from_wheel(module):
    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(wheel_root):
        raise AssertionError(
            f'{{module.__name__}} resolved to {{module_path}}; expected it under {{wheel_root}}'
        )
"""
    subprocess.run([sys.executable, '-I', '-S', '-c', bootstrap + code, str(root), *args],
                   cwd=root, env=env, check=True)
