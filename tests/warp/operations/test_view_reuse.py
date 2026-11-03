"""Step D: view reuse on the no-grad path (warpier_fields.md Step D, Section
5's reentrancy tests 1-3). Runs at the test suite's default precision
(float32) with bit-identical comparisons rather than gradcheck's numerical
Jacobian -- the float64/subprocess-per-script gradcheck machinery in
scripts/gradcheck_*.py already covers backward-mode correctness in isolation
and is re-run against Step D's changes as part of the same gate; what's
missing from that coverage, and what this file adds, is the specific
interaction Step D introduces: a tensor object that gets a cached Field
attached on a no-grad call, then reused with requires_grad toggled on and
back off in the same process.
"""

import os

import pytest
import torch

from warpSPHCore import DomainDescription, OperationProperties, ParticleState, radiusSearchCompactHashMap, warpOperation
from warpSPHCore.enumTypes import KernelFunctions, OperationDirection, SupportScheme, WarpOperation


def _line_particles(n=7, device="cpu"):
    device = torch.device(device)
    x = torch.linspace(-1.0, 1.0, n, device=device).unsqueeze(-1)
    positions = x.contiguous()
    supports = torch.full((n,), 0.6, device=device)
    masses = torch.full((n,), 1.0, device=device)
    kinds = torch.zeros(n, dtype=torch.int32, device=device)
    domain = DomainDescription(
        min=torch.tensor([-10.0], device=device),
        max=torch.tensor([10.0], device=device),
        periodic=torch.tensor([False], device=device),
        dim=1,
    )
    particles = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(particles, domain, mode=SupportScheme.Gather)
    return particles, domain, adjacency


def _density_props():
    return OperationProperties(
        kernel=KernelFunctions.Wendland2,
        operation=WarpOperation.Density,
        supportMode=SupportScheme.Gather,
        operationMode=OperationDirection.AllToAll,
    )


def test_repeated_nograd_call_hits_the_view_cache():
    particles, domain, adjacency = _line_particles()
    positions = particles.positions

    out1 = warpOperation(particles, _density_props(), domain, adjacency=adjacency)
    field1 = getattr(positions, "_wsc_field", None)
    assert field1 is not None, "a no-grad call should attach a cached Field to positions"
    view1 = field1.view()

    out2 = warpOperation(particles, _density_props(), domain, adjacency=adjacency)
    field2 = getattr(positions, "_wsc_field", None)
    assert field2 is field1
    assert field2.view() is view1, "second no-grad call should reuse the same wp.array view"
    assert torch.equal(out1, out2)


def test_disable_field_cache_env_var_prevents_reuse(monkeypatch):
    monkeypatch.setenv("WARPSPHCORE_DISABLE_FIELD_CACHE", "1")
    particles, domain, adjacency = _line_particles()
    positions = particles.positions

    warpOperation(particles, _density_props(), domain, adjacency=adjacency)
    assert getattr(positions, "_wsc_field", None) is None


def test_toggling_requires_grad_on_same_tensor_object_does_not_corrupt_gradient():
    """The interesting new interaction Step D introduces: `positions` gets a
    cached Field attached by a no-grad call, then the *same* tensor object
    (not a copy) has requires_grad toggled on for a differentiable call, then
    off again. None of the three calls should see a wrong answer."""
    particles, domain, adjacency = _line_particles()
    positions = particles.positions

    out_nograd_1 = warpOperation(particles, _density_props(), domain, adjacency=adjacency)
    assert getattr(positions, "_wsc_field", None) is not None  # cache warmed

    positions.requires_grad_(True)
    particles.supports.requires_grad_(True)
    particles.masses.requires_grad_(True)
    try:
        out_grad = warpOperation(particles, _density_props(), domain, adjacency=adjacency)
        assert torch.equal(out_nograd_1, out_grad.detach()), "forward value must not depend on requires_grad"

        out_grad.sum().backward()
        grad_pos = positions.grad.clone()
        grad_sup = particles.supports.grad.clone()
        grad_mas = particles.masses.grad.clone()
    finally:
        positions.requires_grad_(False)
        particles.supports.requires_grad_(False)
        particles.masses.requires_grad_(False)
        positions.grad = None
        particles.supports.grad = None
        particles.masses.grad = None

    out_nograd_2 = warpOperation(particles, _density_props(), domain, adjacency=adjacency)
    assert torch.equal(out_nograd_1, out_nograd_2), "no-grad path must be unaffected after the interleaved grad call"

    # Cross-check the gradient itself against a run with the Field cache
    # forced off end-to-end (WARPSPHCORE_DISABLE_FIELD_CACHE=1), on a fresh
    # set of leaf tensors so there is no cache interaction to speak of --
    # the ground truth Step D must reproduce.
    ref_particles, ref_domain, ref_adjacency = _line_particles()
    ref_particles.positions.requires_grad_(True)
    ref_particles.supports.requires_grad_(True)
    ref_particles.masses.requires_grad_(True)
    old_env = os.environ.get("WARPSPHCORE_DISABLE_FIELD_CACHE")
    os.environ["WARPSPHCORE_DISABLE_FIELD_CACHE"] = "1"
    try:
        ref_out = warpOperation(ref_particles, _density_props(), ref_domain, adjacency=ref_adjacency)
        ref_out.sum().backward()
    finally:
        if old_env is None:
            os.environ.pop("WARPSPHCORE_DISABLE_FIELD_CACHE", None)
        else:
            os.environ["WARPSPHCORE_DISABLE_FIELD_CACHE"] = old_env

    assert torch.allclose(grad_pos, ref_particles.positions.grad)
    assert torch.allclose(grad_sup, ref_particles.supports.grad)
    assert torch.allclose(grad_mas, ref_particles.masses.grad)
