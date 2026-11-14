# Typed Warp fixed-slot grid

`makeHashMapSlots`, `buildFixed` and `GraphedBuild` accept a selected-type device
specialization for the nonperiodic 2D producer used by FluxBridge. Pass a Warp
scalar `radius` array of shape `(1,)` and typed domain arrays. No floating dtype
configuration is changed. The existing host/Torch-input builder is separate and
still performs host numerical preparation and transient tensor allocation.

```python
import warp as wp
from warpSPHCore.dataTypes import DomainDescription
from warpSPHCore.radiusSearch.compactHash import makeHashMapSlots, buildFixed

# low/high: same-type scalar (2,) device arrays; periodic: bool (2,), all false.
# radius: positive scalar (1,), float32 or float64, on the same device.
domain = DomainDescription(low, high, periodic, 2)
slots = makeHashMapSlots(count, domain, radius, device=radius.device)
# positions: same-type (count, 2); supports: same-type (count,).
grid = slots.update_device(positions, supports)
# Or write into slots.positions/supports, then call buildFixed(slots).
slots.check()                 # explicit synchronizing diagnostic
```

Domain/radius are owned same-type snapshots. Cell width and integer cell counts
are calculated on the selected device using the retained finite Core grid
formula. Geometry/search quantities remain FP32 or FP64; integer sort keys,
counts and scans never share floating storage. Physical lengths keep the same
units: bounds, radius, supports and cell width all have units of length.

Rebuilds retain particle sorting, occupied-cell compaction and hash sorting
scratch. GPU uses stable parallel merge ranking; CPU uses eight byte-radix passes
inside a Warp CPU kernel. No NumPy, Torch numerical operation, hidden readback,
CUB allocation cache or synchronization is used in construction/rebuilds.
Constructor allocations are explicit and bounded by a 1 GiB workspace guard;
warmed updates allocate nothing. `slots.array_bytes` includes owned scratch.
The graph wrapper uses Warp capture for these slots without a Torch sync.

The output preserves the existing compact hash layout, sorted original particle
IDs and stable within-cell order. `grid.hCell` is a **device array** in this
specialization; `slots.radius`, `grid.numCells` and `grid.status` stay on device.
FluxBridge consumes those arrays directly. Older Core particle consumers that
expect a host `hCell` are **not qualified for this device-parameter contract**;
no scalar extraction or fallback is provided to make them work.

Nonzero periodicity, 3D typed producer support, changing domain/radius and native
execution of these slots are unsupported. Recreate slots after changing the
frame, radius, particle count or hash capacity. Per-axis cell counts must fit
2^24 for FP32 or int32 maximum for FP64; linear cell keys use int64.
Invalid domains, nonfinite positions/supports, nonpositive supports, or supports
above the declared radius set device failure status and `fits=False`. A later
valid rebuild resets dynamic failure status. Explicit `check()` raises; a
consumer must gate on `grid.status` before using an index. FluxBridge does so.

Producer writes and consumer reads must use an ordered stream or explicit event
dependency. Each concurrent stream/snapshot needs its own mutable slots. Pointer
reuse does not establish content validity; rebuild after changing particles.

Regression coverage is `tests/warp/radiusSearch/test_typedSlots.py` with
`--device cpu` or `--device cuda`. The maintained FluxBridge benchmark records
matched producer/search/action costs and source hashes in the external evidence
workspace; performance reports are not package dependencies.
