"""Native setup helpers keep the existing host precision and storage contracts."""
import json
import os
import subprocess
import sys
import numpy as np
import pytest


GUARD = '''
import importlib.abc, sys
class NoWarp(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] == 'warp':
            raise AssertionError(name)
sys.meta_path.insert(0, NoWarp())
'''


def run(script, runtime, precision='float32'):
    env = os.environ | {'WARPSPHCORE_BACKEND': runtime,
                        'WARPSPHCORE_OPERATORS': runtime,
                        'warpSPHCore_PRECISION': precision,
                        'CUDA_VISIBLE_DEVICES': ''}
    result = subprocess.run([sys.executable, '-c',
                             (GUARD if runtime == 'native' else '') + script],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_host_kernel_scale_and_xi_match_reference_exactly(precision):
    script = '''
import json, math
from warpSPHCore import KernelFunctions, sphKernelScale, sphKernel_xi
rows = {k.name: [[float(sphKernelScale(k.value, d)), float(sphKernel_xi(k.value, d))]
                 for d in (1, 2, 3)] for k in KernelFunctions}
assert math.isnan(float(sphKernelScale(9999, 2)))
assert math.isnan(float(sphKernel_xi(9999, 2)))
print('RESULT ' + json.dumps(rows, sort_keys=True))
'''
    values = {}
    for runtime in ('warp', 'native'):
        output = run(script, runtime, precision)
        values[runtime] = json.loads(next(line[7:] for line in output.splitlines()
                                         if line.startswith('RESULT ')))
    assert values['native'] == values['warp']


def test_native_optional_argument_tensor_cache_without_warp():
    run('''
import torch
from warpSPHCore import getCachedDummyTensor, getCachedIdentityMatrices, clearDummyTensorCache
from warpSPHCore.autograd import getCachedDummyTensor as same
assert same is getCachedDummyTensor
for dtype in (torch.float32, torch.float64):
    a = getCachedDummyTensor((2, 3), dtype=dtype, device='cpu', fillValue=4.)
    assert a is getCachedDummyTensor([2, 3], dtype=dtype, device=torch.device('cpu'), fillValue=4.)
    assert torch.equal(a, torch.full_like(a, 4.))
    identity = getCachedIdentityMatrices(3, dtype=dtype, device='cpu')
    assert identity is getCachedIdentityMatrices(3, dtype=dtype, device='cpu')
    assert torch.equal(identity, torch.eye(3, dtype=dtype)[None])
    clearDummyTensorCache()
    assert a is not getCachedDummyTensor((2, 3), dtype=dtype, device='cpu', fillValue=4.)
assert 'warp' not in __import__('sys').modules
''', 'native')


def test_native_public_verlet_entry_accepts_backend_keyword():
    run('''
import torch
import warpSPHCore as core
from warpSPHCore.radiusSearch import buildVerletList
assert core.buildVerletList is buildVerletList
x = torch.tensor([[.1, .1], [.2, .1], [.9, .9]])
state = core.ParticleState(positions=x, supports=torch.full((3,), .25),
    masses=torch.ones(3), densities=torch.ones(3), kinds=torch.zeros(3, dtype=torch.int32))
domain = core.DomainDescription(torch.zeros(2), torch.ones(2), torch.zeros(2, dtype=torch.bool), 2)
adjacency = core.buildVerletList(state, domain, backend='native', storeI=False)
i, j = adjacency.livePairs(backend='native')
assert set(zip(i.tolist(), j.tolist())) == {(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)}
assert adjacency.searchBackend == 'native'
try:
    core.buildVerletList(state, domain, backend='warp')
except ValueError:
    pass
else:
    raise AssertionError('native-only process admitted a Warp reference')
assert 'warp' not in __import__('sys').modules
''', 'native')


def test_native_scalar_argument_keeps_storage_and_gradient_without_warp():
    run('''
import torch
from warpSPHCore import asScalarArg
from warpSPHCore.autograd import asScalarArg as same
assert same is asScalarArg
x = torch.tensor(2.5, dtype=torch.float64, requires_grad=True)
y = asScalarArg(x, device='cpu')
assert y.shape == (1,) and y.data_ptr() == x.data_ptr()
assert torch.autograd.grad((3 * y).sum(), x)[0].item() == 3
assert asScalarArg(1.25, device='cpu').tolist() == [1.25]
try:
    asScalarArg(torch.ones(2), device='cpu')
except ValueError:
    pass
else:
    raise AssertionError('non-scalar argument admitted')
''', 'native')


@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_host_wendland_lattice_calibration_matches_reference(precision):
    script = """
import json
from warpSPHCore.util.latticeDensity import latticeDensityFactor
from warpSPHCore import KernelFunctions
rows = [[latticeDensityFactor(k, h, dim) for h in (2., 3.5, 4.)]
        for k in (KernelFunctions.Wendland2, KernelFunctions.Wendland4, KernelFunctions.Wendland6)
        for dim in (1, 2, 3)]
print('RESULT ' + json.dumps(rows))
"""
    values = {}
    for runtime in ('warp', 'native'):
        output = run(script, runtime, precision)
        values[runtime] = json.loads(next(line[7:] for line in output.splitlines()
                                         if line.startswith('RESULT ')))
    if precision == 'float64':
        assert values['native'] == values['warp']
    else:
        # Warp's host float32 power can round differently from NumPy's
        # across platforms. The factors must agree within one float32 step.
        actual, expected = np.asarray(values['native']), np.asarray(values['warp'])
        step = np.spacing(expected.astype(np.float32))
        assert np.all(np.abs(actual - expected) <= step), (actual, expected)
