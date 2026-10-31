# Explicit input verification

These helpers are independent of operation execution. The runtime imports none
of them and has no verification flag or mode. They perform full tensor scans and
may synchronize the device; invoke them only when investigating particular input
data, outside simulation steps and benchmark timing.

From the repository root, with `tests` on `PYTHONPATH`:

```python
from verify.native_inputs import verify_geometry, verify_csr, verify_finite

verify_geometry(query, reference, domain)
verify_csr(adjacency.edgeOffsets, adjacency.numNeighbors,
           adjacency.paddedJ, len(reference.positions))
verify_finite(reference_values)
```

Run their tests separately with `pytest tests/verify`.
