"""Reject tracked research/run artifacts without importing numerical packages.

This checks the Git index, so staged deletions are respected. Maintained small
fixtures are explicit exceptions; arbitrary JSON/CSV/PNG files are not forbidden
because documentation and tests legitimately use those formats.
"""
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

RESEARCH_ROOTS = {'experiments', 'scratch', 'tmp', '.tmp', 'runs', 'artifacts',
                  'results', 'logs', 'profiles'}
OUTPUT_DIRECTORIES = {'results', 'outputs', 'output', 'logs', 'profiles', '__pycache__',
                      '.pytest_cache', '.docusaurus', 'node_modules'}
BUILD_ROOTS = {('build',), ('dist',), ('docs', 'build')}
GENERATED_SUFFIXES = {'.h5', '.hdf5', '.pt', '.pth', '.npy', '.npz', '.vtk', '.vtu',
                      '.vtp', '.pvd', '.mp4', '.nvvp', '.nsys-rep', '.ncu-rep', '.pyc',
                      '.log', '.whl', '.so', '.dylib', '.dll', '.pyd', '.o', '.obj', '.a'}
# Frozen, maintained inputs consumed by the existing higher-order PDE harness.
CURATED_FIXTURES = {'higherOrderSPH/harness/data/tgv2d_fullshift_nx128.npz',
                    'higherOrderSPH/harness/data/tgv2d_noshift_nx128.npz'}


def violations(paths):
    problems = []
    for name in sorted(paths):
        path = PurePosixPath(name)
        if path.parts[0] in RESEARCH_ROOTS:
            reason = 'research or run directory'
        elif (any(part in OUTPUT_DIRECTORIES for part in path.parts[:-1])
              or any(path.parts[:len(prefix)] == prefix for prefix in BUILD_ROOTS)):
            reason = 'generated output directory'
        elif path.suffix.lower() in GENERATED_SUFFIXES and name not in CURATED_FIXTURES:
            reason = 'generated data, log or build artifact'
        elif name.startswith('docs/regression/') and path.suffix != '.md':
            reason = 'raw qualification output (retain a concise Markdown report only)'
        else:
            continue
        problems.append(f'{name}: {reason}')
    return problems


def check_index(root):
    indexed = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root)
    paths = sorted({os.fsdecode(path) for path in indexed.split(b'\0') if path})
    problems = violations(paths)
    for name in (path for path in paths if path.endswith('.ipynb')):
        try:
            notebook = json.loads(subprocess.check_output(
                ['git', 'cat-file', 'blob', ':' + name], cwd=root))
            cells = notebook['cells']
            if not isinstance(cells, list) or any(not isinstance(cell, dict) for cell in cells):
                raise ValueError('expected a list of cell objects')
        except (subprocess.CalledProcessError, ValueError, KeyError, TypeError) as error:
            problems.append(f'{name}: could not read indexed notebook ({error})')
            continue
        if any(cell.get('outputs') or cell.get('execution_count') is not None
               for cell in cells if cell.get('cell_type') == 'code'):
            problems.append(f'{name}: notebook execution outputs or counts')
    return problems


def main():
    root = Path(__file__).resolve().parents[1]
    problems = check_index(root)
    if problems:
        print('Keep research source and generated runs outside the core checkout:', file=sys.stderr)
        print('\n'.join(f'  {problem}' for problem in problems), file=sys.stderr)
        print('See docs/docs/development-artifacts.md.', file=sys.stderr)
        return 1
    print('Repository layout: no tracked research/run artifacts.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
