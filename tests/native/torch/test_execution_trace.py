"""Normal native dispatch contains no data-verification reductions or scalar reads."""
import torch
import warpSPHCore as sph


def test_operation_does_not_scan_contents(device):
    x = torch.tensor([[-.2, 0.], [.2, .1]], device=device, dtype=torch.float64)
    particles = sph.ParticleState(x, x.new_ones(2), x.new_ones(2),
        torch.zeros(2, device=device, dtype=torch.int32), x.new_ones(2))
    domain = sph.DomainDescription(x.new_full((2,), -1), x.new_ones(2),
        torch.ones(2, device=device, dtype=torch.bool), 2)
    adjacency = sph.radiusSearch(particles, domain)
    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2,
                                         operation=sph.WarpOperation.Interpolate)
    def call(values):
        return sph.warpOperation(particles, properties, domain, values, adjacency=adjacency)
    call(x)  # Compilation and warmup are outside the trace.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        actual = call(x)
    forbidden = {'aten::isfinite', 'aten::all', 'aten::cumsum', 'aten::sum',
                 'aten::equal', 'aten::_local_scalar_dense'}
    assert not (forbidden & {event.key for event in trace.key_averages()})
    assert torch.isfinite(actual).all()
    # Value diagnostics are separate: safe numerical NaNs propagate normally.
    assert torch.isnan(call(torch.full_like(x, float('nan')))).all()
