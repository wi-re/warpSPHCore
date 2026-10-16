#!/usr/bin/env python3
"""Reproduce the field reconstruction guide's figures with production CPU APIs.

Run from the repository root with PYTHONPATH=src. No timing measurements.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=root / 'docs/docs/assets/reconstruction')
    args = parser.parse_args()
    os.environ['warpSPHCore_PRECISION'] = 'float64'
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/warpsphcore-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import warp as wp
    from warpSPHCore import (DomainDescription, GradientScheme, KernelFunctions,
                             OperationProperties, ParticleState, SupportScheme,
                             WarpOperation, warpOperation)
    from warpSPHCore.crk import computeCRKFactors
    from warpSPHCore.polyfit import PolyFit, Reconstructor
    from warpSPHCore.renorm import computeRenormalizationMatrices

    wp.init()
    torch.set_grad_enabled(False)
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    args.output.mkdir(parents=True, exist_ok=True)
    kernel = KernelFunctions.Wendland2
    dom = DomainDescription(torch.zeros(1), torch.ones(1),
                            torch.zeros(1, dtype=torch.bool), 1)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True,
                         'grid.alpha': .16, 'svg.fonttype': 'none'})
    colors = dict(sph='#b75124', shepard='#a58612', crk='#8752a1',
                  mls='#16718b', labfm='#28824a', left='#16718b', right='#c45631')

    def state(x, h, volume=None):
        return ParticleState(positions=x[:, None].contiguous(),
                             supports=torch.full_like(x, h),
                             masses=torch.ones_like(x) if volume is None else volume,
                             densities=torch.ones_like(x),
                             kinds=torch.zeros(len(x), dtype=torch.int32))

    def interpolate(query, refs, field, volume=None, crk=None):
        return warpOperation(query, OperationProperties(kernel=kernel,
            operation=WarpOperation.Interpolate, supportMode=SupportScheme.Scatter),
            dom, referenceParticles=refs, referenceValues=field.contiguous(),
            referenceVolumes=volume, crkState=crk)

    def save(fig, name):
        fig.savefig(args.output / (name + '.svg'), bbox_inches='tight')
        fig.savefig(args.output / (name + '.png'), dpi=170, bbox_inches='tight')
        plt.close(fig)

    data = {'precision': 'float64', 'device': 'cpu', 'kernel': 'Wendland2',
            'torch': torch.__version__, 'warp': wp.__version__,
            'matplotlib': matplotlib.__version__, 'cases': {}}
    # Open cloud: all four methods use the same reference apparent volumes.
    n = 25
    k = torch.arange(n)
    x = (k + .5 + .18 * torch.sin(7.3 * k)) / n
    h = 5.5 / n
    p = state(x, h)
    volume, _, _ = computeCRKFactors(p, dom, kernel)
    qx = torch.linspace(0, 1, 301)
    q = state(qx, h)
    _, _, crk = computeCRKFactors(q, dom, kernel, referenceState=p)
    fit = PolyFit.build(q, dom, kernel, 2, referenceParticles=p,
                        referenceVolumes=volume)
    assert not fit.deficient.any()
    partition = interpolate(q, p, torch.ones_like(x), volume)
    functions = [lambda a: torch.ones_like(a), lambda a: 1 + .7 * a,
                 lambda a: 1 + .7 * a + .8 * a.square()]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.8), layout='constrained')
    records = []
    for ax, name, func in zip(axes, ['Constant', 'Linear', 'Quadratic'], functions):
        f, exact = func(x), func(qx)
        raw = interpolate(q, p, f, volume)
        curves = {'sph': raw, 'shepard': raw / partition,
                  'crk': interpolate(q, p, f, volume, crk), 'mls': fit.values(f)}
        ax.plot(qx, exact, 'k--', lw=2, label='Exact field')
        for key, label in [('sph', 'SPH sum'), ('shepard', 'Normalized SPH'),
                           ('crk', 'CRK (linear correction)'), ('mls', 'MLS degree 2')]:
            ax.plot(qx, curves[key], color=colors[key], lw=1.7, label=label)
        ax.scatter(x, f, s=13, color='#30353b', zorder=5, label='Particle samples')
        ax.set(title=name, xlabel='Query position x')
        records.append(dict(name=name, exact=exact.tolist(),
                            values={key: val.tolist() for key, val in curves.items()},
                            max_error={key: float((val-exact).abs().max()) for key, val in curves.items()}))
    axes[0].set_ylabel('Reconstructed value')
    axes[1].legend(loc='lower center', fontsize=8)
    fig.suptitle('An open boundary removes kernel support; discrete moments decide reproduction')
    save(fig, 'boundary-consistency')
    for row in records[:2]:
        assert row['max_error']['crk'] < 1e-9
    assert max(row['max_error']['mls'] for row in records) < 1e-9
    data['cases']['boundary'] = dict(x=x.tolist(), query=qx.tolist(), support=h,
        reference_volumes=volume.tolist(), fields=records,
        weighting='same reference-to-reference CRK apparent volumes for every method')

    # Same stencil, different anchoring; a smooth, deliberately non-polynomial field.
    edges = torch.cat([torch.zeros(1), .5*(x[:-1]+x[1:]), torch.ones(1)])
    p = state(x, h, edges.diff())
    smooth = lambda a: 1 + .3 * torch.sin(2*torch.pi*a) + .15 * torch.cos(5*torch.pi*a)
    derivative = lambda a: .6*torch.pi*torch.cos(2*torch.pi*a) - .75*torch.pi*torch.sin(5*torch.pi*a)
    f = smooth(x)
    mls = PolyFit.build(p, dom, kernel, 3)
    labfm = PolyFit.build(p, dom, kernel, 3, constant=False)
    assert not mls.deficient.any() and not labfm.deficient.any()
    props = OperationProperties(kernel=kernel, operation=WarpOperation.Gradient,
                                gradientMode=GradientScheme.Difference)
    renorm = computeRenormalizationMatrices(p, props, dom, returnEigVals=False)
    grad = warpOperation(p, props, dom, queryValues=f)
    corrected = warpOperation(p, props, dom, queryValues=f, renormalizationState=renorm)
    corrected_linear = warpOperation(p, props, dom, queryValues=x,
                                     renormalizationState=renorm)
    torch.testing.assert_close(corrected_linear[:, 0], torch.ones_like(x),
                               rtol=1e-10, atol=1e-10)
    center = 2
    curve_x = torch.linspace(max(0., float(x[center]-h)), float(x[center]+h), 250)
    idx = torch.full((len(curve_x),), center, dtype=torch.int64)
    offset = (curve_x-x[center])[:, None]
    vm, vl = mls.evaluate(f, idx, offset), labfm.evaluate(f, idx, offset)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4), layout='constrained')
    axes[0].plot(curve_x, smooth(curve_x), 'k--', label='Exact smooth field')
    inside = (x-x[center]).abs() < h
    axes[0].scatter(x[inside], f[inside], s=24, color='#30353b', label='Stencil samples')
    axes[0].plot(curve_x, vm, color=colors['mls'], label='MLS degree 3')
    axes[0].plot(curve_x, vl, color=colors['labfm'], label='LABFM degree 3')
    axes[0].axvline(x[center], color='gray', lw=.8)
    axes[0].scatter([x[center]], [f[center]], s=80, facecolors='none', edgecolors='k',
                    label='Fit center / stored value', zorder=6)
    axes[0].set(title='One fixed polynomial patch near a boundary', xlabel='Position x', ylabel='Field value')
    axes[0].text(.49, .71,
                 f'At the fit center:\nMLS = {float(mls.values(f)[center]):.6f}\n'
                 f'LABFM = stored value = {float(f[center]):.6f}',
                 transform=axes[0].transAxes, fontsize=9,
                 bbox=dict(facecolor='white', edgecolor='#dddddd', alpha=.9))
    axes[0].legend(fontsize=8)
    gradients = dict(sph=grad[:, 0], renorm=corrected[:, 0],
                     mls=mls.gradient(f)[:, 0], labfm=labfm.gradient(f)[:, 0])
    axes[1].plot(qx, derivative(qx), 'k--', label='Exact derivative')
    for key, label, color in [('sph', 'SPH Difference', colors['sph']),
                              ('renorm', 'SPH Difference + renorm', colors['crk']),
                              ('mls', 'MLS degree 3', colors['mls']),
                              ('labfm', 'LABFM degree 3', colors['labfm'])]:
        axes[1].plot(x, gradients[key], '.-', label=label, color=color, ms=4)
    axes[1].set(title='Derivatives read at every particle center', xlabel='Particle position x', ylabel='df / dx')
    axes[1].legend(fontsize=8)
    save(fig, 'polynomial-patches')
    data['cases']['patches'] = dict(x=x.tolist(), support=h, volume=p.masses.tolist(),
        field=f.tolist(), center=center,
        query=curve_x.tolist(), mls=vm.tolist(), labfm=vl.tolist(),
        mls_center=float(mls.values(f)[center]), stored_center=float(f[center]),
        gradients={key: val.tolist() for key, val in gradients.items()})
    assert torch.equal(labfm.values(f), f)

    # A larger open 1D cloud keeps the displayed interior far from boundary fallbacks.
    n = 64
    k = torch.arange(n)
    x = (k + .5 + .12 * torch.sin(7.3 * k)) / n
    edges = torch.cat([torch.zeros(1), .5*(x[:-1]+x[1:]), torch.ones(1)])
    p = state(x, 5.5/n, edges.diff())
    i, j = torch.arange(n-1), torch.arange(1, n)
    d = p.positions[j]-p.positions[i]
    mid = .5*(x[i]+x[j])
    q = state(mid, 5.5/n)
    field = (x >= .5).to(x.dtype)
    central = PolyFit.build(p, dom, kernel, 3)
    teno = Reconstructor.teno(p, dom, kernel, 'O4')
    weno = Reconstructor.weno(p, dom, kernel, degree=2)
    pair_states = {'MLS degree 3': central.interfaceStates(field, i, j, d),
                   'TENO O4': teno.interfaceStates(field, i, j, d),
                   'WENO degree 2': weno.interfaceStates(field, i, j, d)}
    raw = interpolate(q, p, field)
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.2), layout='constrained', sharex=True, sharey=True)
    for ax, title in zip(axes.flat, ['SPH sum', *pair_states]):
        ax.plot([.27, .5, .5, .73], [0, 0, 1, 1], 'k--', lw=1.4, label='Exact jump')
        ax.scatter(x, field, s=20, color='#30353b', zorder=4, label='Stored samples')
        if title == 'SPH sum':
            ax.plot(mid, raw, 'o-', ms=3.5, color=colors['sph'], label='One value at interface')
        else:
            left, right = pair_states[title]
            ax.plot(mid, left, 'o-', ms=3.5, color=colors['left'], label='Left state (from i)')
            ax.plot(mid, right, 'x--', ms=4, color=colors['right'], label='Right state (from j)')
        ax.set(title=title, xlim=(.29, .71), ylim=(-.23, 1.23))
        ax.legend(fontsize=8, loc='upper left')
    for ax in axes[1]:
        ax.set_xlabel('Interface position x')
    for ax in axes[:, 0]:
        ax.set_ylabel('Reconstructed value')
    fig.suptitle('At the same interface, two one-sided states can preserve a jump')
    save(fig, 'jump-interfaces')
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.6), layout='constrained', sharey=True)
    nonlinear = {}
    interior = (x > .25) & (x < .75)
    straddle = field[i] != field[j]
    for ax, name, rec in zip(axes, ['TENO O4', 'WENO degree 2'], [teno, weno]):
        weights = rec.last_omega
        for s, label, color in [(0, 'Central stencil', colors['crk']),
                                 (1, 'Left-only stencil', colors['left']),
                                 (2, 'Right-only stencil', colors['right'])]:
            ax.plot(x, weights[:, s], '.-', ms=4, label=label, color=color)
        ax.axvline(.5, color='k', ls='--', lw=1)
        ax.set(title=name, xlim=(.29, .71), ylim=(-.04, 1.04), xlabel='Particle / fit-center position x')
        ax.legend(fontsize=8, loc='upper left')
        left, right = pair_states[name]
        nonlinear[name] = dict(weights=weights.tolist(), beta=rec.last_beta.tolist(),
            fallback=rec.last_fallback.tolist(), left=left.tolist(), right=right.tolist(),
            jump_left=float(left[straddle][0]), jump_right=float(right[straddle][0]))
        assert not rec.last_fallback[interior].any()
        assert float(left[straddle].abs().max()) < .02
        assert float((right[straddle]-1).abs().max()) < .02
    axes[0].set_ylabel('Nonlinear candidate weight')
    save(fig, 'nonlinear-weights')
    # JSON uses null for inactive infinite beta, so it is valid in strict parsers.
    for item in nonlinear.values():
        item['beta'] = [[v if np.isfinite(v) else None for v in row] for row in item['beta']]
    data['cases']['jump'] = dict(x=x.tolist(), field=field.tolist(), interface=mid.tolist(),
        support=5.5/n, volume=p.masses.tolist(), raw_sph=raw.tolist(),
        mls_left=pair_states['MLS degree 3'][0].tolist(),
        mls_right=pair_states['MLS degree 3'][1].tolist(), nonlinear=nonlinear)
    # Also check smooth interface accuracy on the illustrated geometry.
    smooth_errors = {}
    for name, obj in [('MLS degree 3', central), ('TENO O4', teno), ('WENO degree 2', weno)]:
        left, right = obj.interfaceStates(torch.sin(2*torch.pi*x), i, j, d)
        mask = (mid > .25) & (mid < .75)
        exact = torch.sin(2*torch.pi*mid)
        err = max(float((left[mask]-exact[mask]).abs().max()),
                  float((right[mask]-exact[mask]).abs().max()))
        assert err < .005
        smooth_errors[name] = err
    data['cases']['jump']['smooth_interface_max_errors'] = smooth_errors
    sources = ['scripts/diagnostics/illustrate_reconstruction.py',
               'src/warpSPHCore/polyfit/polyfit.py', 'src/warpSPHCore/polyfit/reconstruction.py',
               'src/warpSPHCore/polyfit/wp_polyfit.py', 'src/warpSPHCore/crk/crk_wrapper.py',
               'src/warpSPHCore/crk/crk_terms.py', 'src/warpSPHCore/coreOperations/wp_interpolate.py',
               'src/warpSPHCore/coreOperations/wp_gradient.py', 'src/warpSPHCore/renorm.py']
    data['source_sha256'] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources}
    (args.output / 'examples.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'boundary_max_errors': [r['max_error'] for r in records],
                      'smooth_interface_max_errors': smooth_errors,
                      'jump_states': {name: [v['jump_left'], v['jump_right']] for name, v in nonlinear.items()},
                      'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
