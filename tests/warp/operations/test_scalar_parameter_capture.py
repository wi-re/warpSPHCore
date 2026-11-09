"""Delayed/composed VJPs use the scalar record from each forward call."""
from typing import Any
import torch
import warp as wp
from warpSPHCore import (ParticleState, DomainDescription, AdjacencyList,
    OperationProperties, KernelFunctions, SPHContext, Corrections, scalar_t)
from warpSPHCore.autograd.operator_spec import OperatorSpec, OutputSpec, ExtraSpec, ExtraKind, ShapeOf, launchOperator
from warpSPHCore.type_config import get_torch_precision, get_torch_index_dtype, get_torch_offset_dtype


@wp.struct
class Parameters:
    factor: scalar_t
    square: wp.bool


@wp.kernel
def map_kernel(query: Any, reference: Any, domain: Any, useAdjacency: wp.bool,
               adjacency: Any, grid: Any, correction: Any, properties: Any,
               values: wp.array(dtype=scalar_t), parameters: Parameters,
               output: wp.array(dtype=scalar_t)):
    i = wp.tid()
    result = values[i]
    if parameters.square:
        result = result * result
    output[i] = parameters.factor * result


def test_composed_map_captures_mutable_scalar_record(device):
    dtype = get_torch_precision()
    n = 4
    x = torch.linspace(.2, .8, n, device=device, dtype=dtype).requires_grad_()
    particle = ParticleState(positions=torch.zeros(n, 1, device=device, dtype=dtype),
        supports=torch.ones(n, device=device, dtype=dtype), masses=torch.ones_like(x),
        densities=torch.ones_like(x), kinds=torch.zeros(n, device=device, dtype=torch.int32))
    domain = DomainDescription(torch.zeros(1, device=device, dtype=dtype), torch.ones(1, device=device, dtype=dtype), torch.zeros(1, device=device, dtype=torch.bool), 1)
    adjacency = AdjacencyList(None, torch.arange(n, device=device, dtype=get_torch_index_dtype()),
        torch.ones(n, device=device, dtype=torch.int32), torch.arange(n, device=device, dtype=get_torch_offset_dtype()), n, n, numEdges=n)
    context = SPHContext(particle, OperationProperties(kernel=KernelFunctions.Wendland2), domain, adjacency, particle, Corrections())
    spec = OperatorSpec(kernel=map_kernel, outputs=(OutputSpec(dtype=lambda c,e: scalar_t, shape=ShapeOf.QUERY),),
        extras=(ExtraSpec('values', ExtraKind.TENSOR), ExtraSpec('parameters', ExtraKind.SCALAR)))
    parameters = Parameters()
    parameters.factor, parameters.square = 2., False
    first = launchOperator(spec, context, values=x, parameters=parameters)
    parameters.factor, parameters.square = 3., True
    second = launchOperator(spec, context, values=first, parameters=parameters)
    parameters.factor, parameters.square = 99., False
    expected = 24 * x.detach()
    actual = torch.autograd.grad(second.sum(), x)[0]
    torch.testing.assert_close(first, 2 * x.detach(), atol=0, rtol=0)
    torch.testing.assert_close(actual, expected)
