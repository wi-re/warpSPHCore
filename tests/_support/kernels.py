"""Kernel specifications and independent Torch shape formulas."""
from pathlib import Path
import math
import torch
import yaml

SPECS = yaml.safe_load((Path(__file__).parents[2] / 'scripts/kernels/kernel_specs.yaml').read_text())['kernels']
SPECS = {s['code_name']: s for s in SPECS.values()}


def shape(q, kernel, dim):
    name = kernel.name
    if name in SPECS:
        spec = SPECS[name]
        expr = spec.get('shape', {}).get('expr') or spec['shape_by_dim'][str(dim)]
        return eval(expr, {'__builtins__': {}, 'q': q, 'pos': lambda x: torch.clamp(x, min=0),
                           'step': lambda x: (x >= 0).to(q.dtype), 'pi': math.pi,
                           'exp': torch.exp, 'sqrt': torch.sqrt})
    if name == 'Poly6': return (1-q*q)**3
    if name == 'Spiky': return (1-q)**3
    if name == 'ViscosityKernel': return -.5*q**3+q*q+.5/q-1
    if name == 'CohesionKernel': return torch.where(q < .5, 2*(1-q)**3*q**3-1/64, (1-q)**3*q**3)
    if name == 'AdhesionKernel': return (-4*q*q+6*q-2)**.25
    raise AssertionError(name)
