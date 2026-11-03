"""Map changed files to test owners; docs-only changes do not launch numerical jobs."""
import argparse
import json
import os
from pathlib import Path
import subprocess

KEYS = ('python', 'warp', 'native', 'torch', 'torch_full', 'packaging', 'parity')


def select(paths):
    selected = set()
    for path in paths:
        if path.startswith(('docs/', 'higherOrderSPH/')) or path.endswith('.md'):
            continue
        if path in ('pyproject.toml', 'CMakeLists.txt') or path.startswith(('.github/', 'cmake/', 'scripts/testing/', 'tests/conftest.py')):
            selected.update(KEYS)
        elif path.startswith(('tests/python/', 'tests/verify/')):
            selected.add('python')
        elif path.startswith('tests/warp/'):
            selected.add('warp')
        elif path.startswith('tests/parity/'):
            selected.add('parity')
        elif path.startswith(('tests/native/cpp/', 'tests/native/standalone/', 'scripts/examples/standalone/')):
            selected.add('native')
        elif path.startswith('tests/native/torch/'):
            selected.update(('torch', 'torch_full'))
        elif path.startswith('tests/_support/'):
            selected.update(('native', 'torch', 'torch_full', 'parity'))
        elif path.startswith('scripts/packaging/'):
            selected.update(('python', 'packaging'))
        elif path.startswith('scripts/'):
            selected.add('warp')
        elif path.startswith('src/'):
            # Backend-owned numerical files share no Python/Warp execution path.
            if Path(path).name.startswith('wp_'):
                selected.update(('warp', 'parity'))
            elif '/native/csrc/runtime/' in path:
                selected.add('native')
            elif '/native/' in path and path.endswith(('.h', '.cuh', '.cpp', '.cu')):
                selected.update(('native', 'torch'))
                if any(part in path for part in ('/kernels/', '/csrc_', '/math/',
                                                  '/execution/compactTraversal.h', '/dataTypes/compactHash.h')):
                    selected.add('parity')
            elif '/native/' in path:
                selected.update(('python', 'torch', 'torch_full'))
                if Path(path).name.startswith(('_build', '_loader', '_sources', 'sources.json')):
                    selected.add('packaging')
            elif any(f'/{part}/' in path for part in ('math', 'kernels', 'radiusSearch', 'linalg', 'crk', 'renorm')):
                # Shared search, algebra and solver controllers serve both
                # backends; backend-owned wp_/native files were handled above.
                selected.update(('warp', 'torch', 'torch_full', 'parity'))
            else:
                # Shared models, enums and frontend dispatch can affect both adapters.
                selected.update(KEYS)
    return {key: key in selected for key in KEYS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='*', help='Inspect explicit paths, or read the GitHub event diff')
    args = parser.parse_args()
    paths = args.paths
    if not paths:
        event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
        base = event.get('pull_request', {}).get('base', {}).get('sha') or event.get('before')
        if not base or set(base) == {'0'}:
            result = dict.fromkeys(KEYS, True)
        else:
            paths = subprocess.check_output(['git', 'diff', '--name-only', base, 'HEAD'], text=True).splitlines()
            result = select(paths)
    else:
        result = select(paths)
    for key, enabled in result.items():
        print(f'{key}={str(enabled).lower()}')


if __name__ == '__main__':
    main()
