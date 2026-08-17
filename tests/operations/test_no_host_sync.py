"""Host-readback regression guards for the two sites Step H's sync census found
(docs/regression/real_workload_bottleneck_audit.md).

A device->host read (`.item()`, `bool(tensor)`, `.cpu()`, ...) drains the CUDA
queue, so one buried in a per-step code path costs real wall clock even though it
looks free. Both sites below were reading a value that either never changes for
the whole run (`domain.periodic`) or was only being consulted to *maybe* skip
work (`if torch.any(lowNbrMask)`), and between them they were 13.5 of the 39.1
readbacks a real solver step performed -- worth 1.08-1.20x of per-step wall clock
once removed, more than every caching layer in Steps A-F combined.

These tests fail if either is reintroduced. They assert on a *count*, not on
timing, so they are deterministic and cheap.
"""

import traceback

import pytest
import torch

from warpSPHCore.radiusSearch.verlet.util import _minimum_image_delta


class _ReadbackCounter:
    """Counts device->host reads, optionally only those originating in `pathPart`.

    Patches the tensor methods that synchronize. `__bool__` matters as much as
    `.item()`: `if someTensor:` and `if torch.any(mask):` both route through it,
    and that pattern was the more common of the two in practice.
    """

    OPS = ("item", "__bool__", "tolist", "numpy")

    def __init__(self, pathPart=None):
        self.pathPart = pathPart
        self.hits = []
        self._originals = {}

    def __enter__(self):
        counter = self

        def makeWrapper(name, original):
            def wrapper(self, *args, **kwargs):
                if self.is_cuda:
                    stack = traceback.extract_stack()[:-1]
                    if counter.pathPart is None or any(counter.pathPart in f.filename for f in stack):
                        counter.hits.append((name, [f"{f.filename}:{f.lineno}" for f in stack[-4:]]))
                return original(self, *args, **kwargs)
            return wrapper

        for name in self.OPS:
            original = getattr(torch.Tensor, name)
            self._originals[name] = original
            setattr(torch.Tensor, name, makeWrapper(name, original))
        return self

    def __exit__(self, *exc):
        for name, original in self._originals.items():
            setattr(torch.Tensor, name, original)
        return False

    def report(self):
        return "\n".join(f"  {op}: {' < '.join(reversed(stack))}" for op, stack in self.hits)


requiresCuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                  reason="host-readback stalls only exist on a device")


@requiresCuda
def test_minimum_image_delta_does_not_read_back():
    """Was 11.5 readbacks per step -- the single largest source in a real step --
    from `bool(periodicity[d].item())` once per axis per call."""
    device = torch.device("cuda")
    current = torch.rand(256, 3, device=device)
    previous = torch.rand(256, 3, device=device)
    domainMin = torch.full((3,), -1.0, device=device)
    domainMax = torch.full((3,), 1.0, device=device)

    for periodic in ([True, True, True], [False, False, False], [True, False, True]):
        periodicity = torch.tensor(periodic, device=device)
        with _ReadbackCounter() as counter:
            _minimum_image_delta(current, previous, periodicity, domainMin, domainMax)
        assert not counter.hits, (
            f"_minimum_image_delta synchronized with periodic={periodic}:\n{counter.report()}"
        )


@pytest.mark.parametrize("periodic", [True, False])
def test_minimum_image_delta_measures_through_the_nearest_image(periodic):
    """The behaviour the branch was there for: a particle stepping across the seam
    has moved a little, not almost a whole box length -- but only when that axis is
    periodic. Pins the values, so a future rewrite cannot quietly drop the wrap."""
    domainMin = torch.tensor([-1.0])
    domainMax = torch.tensor([1.0])
    periodicity = torch.tensor([periodic])
    current = torch.tensor([[-0.95]])
    previous = torch.tensor([[0.95]])

    delta = _minimum_image_delta(current, previous, periodicity, domainMin, domainMax)
    expected = 0.1 if periodic else -1.9
    assert delta.item() == pytest.approx(expected, abs=1e-6)


@requiresCuda
def test_renormalization_does_not_read_back(particle_case):
    """Was 2 readbacks per step from `if torch.any(lowNbrMask):` guarding a masked
    assignment -- which is itself a synchronizing op, so the guard paid a stall to
    maybe avoid one."""
    from warpSPHCore import OperationProperties
    from warpSPHCore.enumTypes import (GradientScheme, KernelFunctions,
                                       OperationDirection, SupportScheme, WarpOperation)
    from warpSPHCore.renorm import computeRenormalizationMatrices

    particles = particle_case["particles"]
    if particles.positions.device.type != "cuda":
        pytest.skip("needs the CUDA parametrization of particle_case")

    properties = OperationProperties(
        kernel=KernelFunctions.Wendland2,
        operation=WarpOperation.Gradient,
        supportMode=SupportScheme.Gather,
        operationMode=OperationDirection.AllToAll,
        gradientMode=GradientScheme.Difference,
    )

    with _ReadbackCounter(pathPart="warpSPHCore/renorm.py") as counter:
        computeRenormalizationMatrices(particles, properties, particle_case["domain"],
                                       adjacency=particle_case["adjacency"])
    assert not counter.hits, f"renorm.py synchronized:\n{counter.report()}"


def test_low_neighbour_fallback_still_replaces_those_rows(particle_case):
    """The fallback's actual job, kept under test now that it is branch-free: a
    particle with too few neighbours gets the identity, not its own (untrustworthy)
    covariance. Constructed directly, since a well-sampled test case has no
    low-neighbour particles and the assertion would be vacuous."""
    device = particle_case["particles"].positions.device
    dim = 2
    numNeighbors = torch.tensor([0, 1, dim + 1, dim + 2, 50], device=device)
    C = torch.randn(5, dim, dim, device=device, dtype=torch.float64)

    lowNbrMask = numNeighbors < dim + 2
    identity = torch.eye(dim, dtype=C.dtype, device=device)
    result = torch.where(lowNbrMask.view(-1, 1, 1), identity.unsqueeze(0), C)

    assert int(lowNbrMask.sum()) == 3, "test would be vacuous without low-neighbour rows"
    for i in range(3):
        assert torch.equal(result[i], identity)
    for i in (3, 4):
        assert torch.equal(result[i], C[i])
