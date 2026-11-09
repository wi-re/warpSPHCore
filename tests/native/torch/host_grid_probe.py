"""Compact-grid layout/order probe for source and installed native runtimes."""
import argparse
import importlib.abc
import os
import sys

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--device', default='cpu')
a = p.parse_args()
if os.environ.get('WARPSPHCORE_BACKEND') == 'native':
    class NoWarp(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split('.')[0] == 'warp':
                raise AssertionError('Native grid attempted Warp import: ' + name)
    sys.meta_path.insert(0, NoWarp())
else:
    import warp as wp
    wp.init()
    if os.environ.get('WARPSPHCORE_OPERATORS') == 'native':
        def reject(*args, **kwargs):
            raise AssertionError('Native grid attempted Warp execution')
        wp.launch = wp.capture_launch = reject
        wp.Launch.launch = reject

import torch
import warpSPHCore as sph
from warpSPHCore.radiusSearch import buildCompactHashMap, makeHashMapSlots, buildFixed, GraphedBuild

dtype = sph.get_torch_precision()
values = {}

def snapshot(prefix, grid):
    for name, value in vars(grid).items():
        values[prefix + '.' + name] = value.detach().cpu().clone() if isinstance(value, torch.Tensor) else getattr(value, 'value', value)

for dim in (1, 2, 3):
    generator = torch.Generator().manual_seed(315 + dim)
    q = (torch.rand((9, dim), generator=generator, dtype=dtype) * 3 - 1).to(a.device)
    r = (torch.rand((13, dim), generator=generator, dtype=dtype) * 3 - 1).to(a.device)
    qh = torch.linspace(.12, .31, len(q), device=a.device, dtype=dtype)
    rh = torch.linspace(.15, .35, len(r), device=a.device, dtype=dtype)
    periodic = torch.tensor([axis % 2 == 0 for axis in range(dim)], device=a.device)
    domain = sph.DomainDescription(torch.zeros(dim, device=a.device, dtype=dtype),
        torch.ones(dim, device=a.device, dtype=dtype), periodic, dim)
    for mode in (sph.SupportScheme.Gather, sph.SupportScheme.Scatter,
                 sph.SupportScheme.MeanSymmetric, sph.SupportScheme.KernelMeanSymmetric,
                 sph.SupportScheme.SuperSymmetric, sph.SupportScheme.PartialSymmetric):
        for length in (1, 7, 4096):
            grid = buildCompactHashMap(q, r, qh, rh, periodic, domain, mode, length)
            snapshot(f'{dim}.{mode.name}.{length}', grid)
    for nq, nr in ((0, 0), (0, len(r)), (len(q), 0)):
        grid = buildCompactHashMap(q[:nq], r[:nr], qh[:nq], rh[:nr], periodic, domain)
        snapshot(f'{dim}.empty.{nq}.{nr}', grid)
    slots = makeHashMapSlots(len(r), domain, .4, device=a.device, dtype=dtype,
                            hashMapLength=7)
    slots.positions.copy_(r)
    slots.supports.copy_(rh)
    selected = os.environ['WARPSPHCORE_OPERATORS']
    grid = buildFixed(slots, backend=selected)
    snapshot(f'{dim}.fixed', grid)
    build = GraphedBuild(slots, backend=selected)
    build.replay()
    build.replay()
    slots.positions.add_(.013)
    replay = build.replay()
    saved = {key: value.clone() if isinstance(value, torch.Tensor) else value
             for key, value in vars(replay).items()}
    if a.device.startswith('cuda'):
        assert build.graph is not None
    expected = buildFixed(slots, backend=selected)
    for key, value in saved.items():
        reference = getattr(expected, key)
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
    snapshot(f'{dim}.replay', replay)
torch.save(values, a.output)
if os.environ.get('WARPSPHCORE_BACKEND') == 'native':
    assert not any(name.split('.')[0] == 'warp' for name in sys.modules)
print('Compact grid: dimension/support/collision/empty/fixed/replay layout passed', flush=True)
