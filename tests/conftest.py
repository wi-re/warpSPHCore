"""Select one backend suite before imports and devices before native parametrization."""
from pathlib import Path
import ctypes
import os
import sys

import pytest

TEST_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_ROOT))
SUITES = ('python', 'warp', 'torch', 'standalone', 'parity')
DEVICE_IDS = {'cpu': 0, 'cuda': 1, 'hip': 2}


def suite_for(path):
    """Return the test owner without importing that suite or its backend."""
    try:
        parts = Path(path).resolve().relative_to(TEST_ROOT).parts
    except ValueError:
        return None
    if not parts:
        return None
    if parts[0] in ('python', 'warp', 'parity'):
        return parts[0]
    if len(parts) > 1 and parts[0] == 'native' and parts[1] in ('torch', 'standalone'):
        return parts[1]
    return None


def pytest_addoption(parser):
    group = parser.getgroup('backend suites')
    group.addoption(
        '--suite', choices=SUITES,
        help='One backend/layer per process; default: python, or infer from explicit path',
    )
    group.addoption(
        '--device', choices=(*DEVICE_IDS, 'all'), default='cpu',
        help='Execution device(s), default: cpu',
    )
    group.addoption(
        '--native-library', help='Standalone CMake bridge: libnative_standalone_test.so',
    )


def select_suite(config):
    # Strip pytest node IDs before inferring ownership from explicit test paths.
    owners = {suite_for(str(arg).split('::', 1)[0]) for arg in config.args}
    owners.discard(None)
    inferred = next(iter(owners)) if len(owners) == 1 else 'python'
    selected = config.getoption('suite') or inferred
    if owners - {selected}:
        raise pytest.UsageError(
            'Run each backend suite in its own process: '
            '--suite python|warp|torch|standalone|parity'
        )
    return selected


def select_devices(config):
    requested = config.getoption('device')
    available = ['cpu']
    if requested != 'cpu' and config.backend_suite != 'python':
        import torch
        if torch.cuda.is_available():
            available.append('hip' if torch.version.hip else 'cuda')
    if config.backend_suite == 'standalone':
        library = os.environ.get('WARPSPHCORE_STANDALONE_LIBRARY')
        if not library:
            raise pytest.UsageError(
                'Build the standalone target and pass '
                '--native-library=/path/to/libnative_standalone_test.so'
            )
        bridge = ctypes.CDLL(library)
        bridge.standalone_supports.argtypes = [ctypes.c_int]
        bridge.standalone_supports.restype = ctypes.c_int
        available = [name for name in available if bridge.standalone_supports(DEVICE_IDS[name])]
    if requested != 'all' and requested not in available:
        raise pytest.UsageError(
            f'Requested {requested} execution is unavailable for suite '
            f'{config.backend_suite}; available: {available}'
        )
    return available if requested == 'all' else [requested]


def pytest_configure(config):
    config.backend_suite = select_suite(config)
    backend = 'warp' if config.backend_suite in ('warp', 'parity') else 'native'
    imported = sys.modules.get('warpSPHCore')
    if imported is not None and imported.backend != backend:
        raise pytest.UsageError(
            f'{config.backend_suite} requires a fresh process with WARPSPHCORE_BACKEND={backend}'
        )
    os.environ['WARPSPHCORE_BACKEND'] = backend
    if config.backend_suite == 'parity':
        os.environ['WARPSPHCORE_NATIVE_PARITY'] = '1'
    if config.getoption('native_library'):
        os.environ['WARPSPHCORE_STANDALONE_LIBRARY'] = config.getoption('native_library')
    for marker in (*SUITES, *DEVICE_IDS, 'gpu', 'extended'):
        config.addinivalue_line('markers', f'{marker}: backend/layer or device ownership')
    config.backend_devices = select_devices(config)
    if config.backend_suite == 'torch':
        build = 'cpu' if config.backend_devices == ['cpu'] else 'gpu'
        os.environ.setdefault('WARPSPHCORE_NATIVE_BUILD', build)


def pytest_report_header(config):
    return f'suite: {config.backend_suite}; devices: {", ".join(config.backend_devices)}'


def pytest_ignore_collect(collection_path, config):
    """Exclude other suites before pytest imports their modules or conftests."""
    try:
        relative = collection_path.resolve().relative_to(TEST_ROOT)
    except ValueError:
        return False
    if relative.parts and relative.parts[0] == '_support':
        return True
    suite = suite_for(collection_path)
    if suite is not None:
        return suite != config.backend_suite
    return relative.parts[:2] == ('native', 'cpp')


def pytest_generate_tests(metafunc):
    if 'device' not in metafunc.fixturenames:
        return
    # Legacy Warp modules sometimes own parametrized fixtures. Let pytest
    # generate those once, then filter devices during collection below.
    fixture_defs = metafunc._arg2fixturedefs.get('device', ())
    if fixture_defs and fixture_defs[-1].params is not None:
        return
    for mark in metafunc.definition.iter_markers('parametrize'):
        names = mark.args[0]
        if isinstance(names, str):
            names = names.replace(' ', '').split(',')
        if 'device' in names:
            return
    # Native matrices generate only the requested devices, avoiding duplicate
    # work and hundreds of unavailable-device skips.
    cases = [
        pytest.param(name, id=name, marks=getattr(pytest.mark, name))
        for name in metafunc.config.backend_devices
    ]
    metafunc.parametrize('device', cases, indirect=True, scope='session')


@pytest.fixture(scope='session')
def device(request):
    # Torch uses the "cuda" device spelling for ROCm too.
    actual = 'cpu' if request.param == 'cpu' else 'cuda'
    if request.config.backend_suite == 'warp':
        import torch
        return torch.device(actual)
    return actual


def pytest_collection_modifyitems(config, items):
    selected, deselected = [], []
    devices = config.backend_devices
    for item in items:
        suite = suite_for(item.path) or config.backend_suite
        item.add_marker(getattr(pytest.mark, suite))
        keep = True
        if item.get_closest_marker('cpu') and 'cpu' not in devices:
            keep = False
        if item.get_closest_marker('gpu') and devices == ['cpu']:
            keep = False
        callspec = getattr(item, 'callspec', None)
        if callspec is not None and 'device' in callspec.params:
            name = str(callspec.params['device'])
            if name.startswith('cuda'):
                import torch
                name = 'hip' if torch.version.hip else 'cuda'
            keep &= name in devices
            item.add_marker(getattr(pytest.mark, name))
        (selected if keep else deselected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected
