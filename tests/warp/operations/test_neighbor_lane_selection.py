"""Explicit lane choices override an operator's qualified default."""

import pytest

from warpSPHCore.autograd import lanes


def test_operator_default_and_process_precedence(monkeypatch):
    monkeypatch.setattr(lanes, '_override', None)
    monkeypatch.setattr(lanes, '_envLanes', None)
    assert lanes.neighborLanes() == 32
    assert lanes.neighborLanes(default=1) == 1
    monkeypatch.setattr(lanes, '_envLanes', 8)
    assert lanes.neighborLanes(default=1) == 8
    lanes.setNeighborLanes(16)
    assert lanes.neighborLanes(default=1) == 16
    lanes.setNeighborLanes(None)
    assert lanes.neighborLanes(default=1) == 8


@pytest.mark.parametrize('raw, expected', [(None, None), ('', None), ('0', 1), ('1', 1), ('32', 32)])
def test_environment_distinguishes_unset_from_explicit(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv('WARPSPHCORE_NEIGHBOR_LANES', raising=False)
    else:
        monkeypatch.setenv('WARPSPHCORE_NEIGHBOR_LANES', raw)
    assert lanes._fromEnv() == expected


def test_trial_defaults_are_serial():
    from warpSPHCore.coreOperations.wp_curl import _CURL_SPEC
    from warpSPHCore.coreOperations.wp_laplacian import _LAPLACIAN_SPEC
    from warpSPHCore.coreOperations.wp_gradient import _GRADIENT_SPEC
    assert _CURL_SPEC.defaultNeighborLanes == _LAPLACIAN_SPEC.defaultNeighborLanes == 1
    assert _GRADIENT_SPEC.defaultNeighborLanes == 32
