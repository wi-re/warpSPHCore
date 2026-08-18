"""Unit tests for the Field abstraction (warpier_fields.md Step B).

Not wired into any hot path yet -- these test dataTypes/field_t.py and
util/fieldRegistry.py in isolation. See warpier_fields.md Section 5 for the
numbered test list this implements (4, 5, 6, 7, 9 here; 8 is covered by
test_particle_state_kinds.py; 10's full-matrix/gradcheck sweep is deferred to
the step that actually wires nullField into arg_extract.py).
"""

import copy
import gc
import pickle

import pytest
import torch
import warp as wp

from warpSPHCore.dataTypes import Field, Role, ExecutionMode, FieldKind
from warpSPHCore.util import acquireView, nullField, clearNullFieldRegistry, structFor
from warpSPHCore.dataTypes.particleData import particleDataSoA_1, particleDataSoA_2, particleDataSoA_3
from warpSPHCore.dataTypes.corrections_t import correctionData_1, correctionData_2, correctionData_3


@pytest.fixture(autouse=True)
def _clean_null_registry():
    clearNullFieldRegistry()
    yield
    clearNullFieldRegistry()


# --------------------------------------------------------------------------
# Test 4: in-place visibility
# --------------------------------------------------------------------------

def test_inplace_write_visible_through_cached_view():
    t = torch.zeros(8, dtype=torch.float32)
    view1 = acquireView(t)
    ptr_before = t.data_ptr()

    t.add_(1.0)  # in-place: same storage, same data_ptr

    view2 = acquireView(t)
    assert t.data_ptr() == ptr_before
    assert view1 is view2  # same cached wp.array object, not rebuilt
    assert wp.to_torch(view2).sum().item() == pytest.approx(8.0)


def test_copy_inplace_visible_through_cached_view():
    t = torch.zeros(8, dtype=torch.float32)
    acquireView(t)
    ptr_before = t.data_ptr()

    replacement = torch.full((8,), 3.0)
    t.copy_(replacement)

    view = acquireView(t)
    assert t.data_ptr() == ptr_before
    assert wp.to_torch(view).sum().item() == pytest.approx(24.0)


# --------------------------------------------------------------------------
# Test 5: non-contiguous refusal
# --------------------------------------------------------------------------

def test_noncontiguous_tensor_never_cached():
    base = torch.zeros(4, 4, dtype=torch.float32)
    strided = base[:, 0]  # non-contiguous view (stride 4)
    assert not strided.is_contiguous()

    acquireView(strided)
    assert getattr(strided, "_wsc_field", None) is None

    strided.fill_(5.0)
    view = acquireView(strided)
    assert wp.to_torch(view).sum().item() == pytest.approx(20.0)


# --------------------------------------------------------------------------
# Test 6: lifetime / no leak
# --------------------------------------------------------------------------

def test_attached_field_no_reference_cycle():
    import weakref

    gc.disable()
    try:
        t = torch.zeros(4, dtype=torch.float32)
        acquireView(t)
        assert getattr(t, "_wsc_field", None) is not None

        ref = weakref.ref(t)
        del t
        # With the cyclic GC disabled, only refcounting can collect t. If
        # Field held a strong reference back to it (tensor -> Field ->
        # wp.array -> tensor, the cycle the removed cache used to close),
        # this would stay alive until a gc.collect() -- which never runs
        # here. ref() is None proves refcounting alone was enough.
        assert ref() is None
    finally:
        gc.enable()


# --------------------------------------------------------------------------
# Test 7: copy / pickle safety
# --------------------------------------------------------------------------

def test_deepcopy_drops_attached_field():
    t = torch.zeros(8, dtype=torch.float32)
    acquireView(t)
    assert getattr(t, "_wsc_field", None) is not None

    t2 = copy.deepcopy(t)
    assert getattr(t2, "_wsc_field", None) is None

    # Rebuilds cleanly rather than reusing a stale/foreign pointer.
    view = acquireView(t2)
    assert wp.to_torch(view).shape[0] == 8


def test_pickle_roundtrip_drops_attached_field(tmp_path):
    t = torch.zeros(8, dtype=torch.float32)
    acquireView(t)

    path = tmp_path / "t.pt"
    torch.save(t, path)
    t2 = torch.load(path, weights_only=False)

    assert getattr(t2, "_wsc_field", None) is None
    view = acquireView(t2)
    assert wp.to_torch(view).shape[0] == 8


def test_field_copy_deepcopy_reduce_degrade_to_none():
    t = torch.zeros(4, dtype=torch.float32)
    view = acquireView(t)
    field = t._wsc_field
    assert isinstance(field, Field)

    assert copy.copy(field) is None
    assert copy.deepcopy(field) is None
    assert pickle.loads(pickle.dumps(field)) is None


# --------------------------------------------------------------------------
# Test 9: tangent-slot inertness
# --------------------------------------------------------------------------

def test_tangent_slot_inert_when_unset():
    t = torch.zeros(8, dtype=torch.float32)
    view_a = acquireView(t, role=Role.PRIMAL)
    field = t._wsc_field
    assert field.tangent is None
    assert set(field.views.keys()) == {Role.PRIMAL}

    # Re-acquiring PRIMAL is bit-identical (same cached object) regardless of
    # the tangent slot's presence.
    view_b = acquireView(t, role=Role.PRIMAL)
    assert view_a is view_b


# --------------------------------------------------------------------------
# WARPSPHCORE_DISABLE_FIELD_CACHE escape hatch
# --------------------------------------------------------------------------

def test_disable_field_cache_env_var(monkeypatch):
    monkeypatch.setenv("WARPSPHCORE_DISABLE_FIELD_CACHE", "1")
    t = torch.zeros(8, dtype=torch.float32)
    acquireView(t)
    assert getattr(t, "_wsc_field", None) is None


# --------------------------------------------------------------------------
# nullField
# --------------------------------------------------------------------------

def test_null_field_zero_fill_default():
    device = torch.device("cpu")
    field = nullField(FieldKind.SCALAR, dim=2, device=device)
    out = wp.to_torch(field.view())
    assert torch.isfinite(out).all()
    assert (out == 0).all()


def test_null_field_cached_by_key():
    device = torch.device("cpu")
    f1 = nullField(FieldKind.VECTOR, dim=3, device=device)
    f2 = nullField(FieldKind.VECTOR, dim=3, device=device)
    assert f1 is f2


def test_null_field_sentinel_fill(monkeypatch):
    monkeypatch.setenv("WARPSPHCORE_NULL_FILL", "sentinel")
    device = torch.device("cpu")

    scalar_field = nullField(FieldKind.SCALAR, dim=2, device=device)
    out = wp.to_torch(scalar_field.view())
    assert torch.isnan(out).all()

    int_field = nullField(FieldKind.INT32, dim=2, device=device)
    out_i = wp.to_torch(int_field.view())
    assert (out_i == -2147483648).all()

    bool_field = nullField(FieldKind.BOOL, dim=2, device=device)
    out_b = wp.to_torch(bool_field.view())
    assert (out_b == False).all()  # noqa: E712 -- bool has no sentinel; stays zero


def test_null_field_tangent_always_zero_even_under_sentinel(monkeypatch):
    monkeypatch.setenv("WARPSPHCORE_NULL_FILL", "sentinel")
    device = torch.device("cpu")

    field = nullField(FieldKind.SCALAR, dim=2, device=device, tangent=True)
    out = wp.to_torch(field.view())
    assert (out == 0).all()


# --------------------------------------------------------------------------
# structFor
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dim,expected", [(1, particleDataSoA_1), (2, particleDataSoA_2), (3, particleDataSoA_3)])
def test_struct_for_particle_soa(dim, expected):
    assert structFor("particleDataSoA", dim, ExecutionMode.REVERSE) is expected
    assert structFor("particleDataSoA", dim, ExecutionMode.NONE) is expected


@pytest.mark.parametrize("dim,expected", [(1, correctionData_1), (2, correctionData_2), (3, correctionData_3)])
def test_struct_for_correction_data(dim, expected):
    assert structFor("correctionData", dim, ExecutionMode.REVERSE) is expected


def test_struct_for_forward_mode_aliases_reverse():
    # warpier_forward_mode_plan.md Phase 2: Tier-1 forward mode needs no new
    # struct shape, so FORWARD's rows alias REVERSE's instead of raising.
    assert structFor("particleDataSoA", 2, ExecutionMode.FORWARD) is structFor(
        "particleDataSoA", 2, ExecutionMode.REVERSE)


def test_struct_for_unknown_kind_raises():
    with pytest.raises(ValueError):
        structFor("notARealKind", 2, ExecutionMode.REVERSE)
