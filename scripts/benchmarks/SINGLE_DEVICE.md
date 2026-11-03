# Single-device benchmark worker

`bench_single_device.py` adds CPU and CUDA coverage for the existing qualified
`_operator_cases.py` fixtures, standalone hash build/query/rebuild phases, and
preallocated Torch copy/triad/random-gather controls. It does not change library
runtime behavior or import the simulator to construct operator fixtures.

The coordinated sweep, request planner, accepted-step/RHS worker, analysis tool
and complete instructions live in the companion frontend checkout:
[warpSPH/benchmarks/performance](https://github.com/wi-re/warpSPH/tree/main/benchmarks/performance).
Use that runner to generate matching JSON requests without importing Torch/Warp:

```sh
cd ../warpSPH
python benchmarks/performance/run.py --device cuda:0 --suite operators search \
  --operators all --nx 32 --particle-order input morton --plan
```

Every actual run retains one request file per worker. An individual case can be
replayed directly or wrapped with Nsight without starting the whole frontend:

```sh
python scripts/benchmarks/bench_single_device.py \
  --request /path/to/requests/0000-operator-density.json \
  --output /path/to/density-replay.json
```

Configure precision, CPU threads and index widths through the recorded request,
before core import. Results include source digests/revisions, device/hardware,
actual directed edge counts, raw repetitions, completed public-call wall time,
submission time, CUDA stream intervals and timed-window allocator peaks. Stream
intervals include host launch gaps. Logical memory GB/s is not actual DRAM GB/s.
Profiling runs after base timing with core hooks enabled only for that pass.

The worker validates finite results; operators compare against lane-1 CSR on the
same field, and retained/rebuilt searches compare the exact live CSR topology.
Morton setup permutes particle fields and input values together; sorting is not
part of a core operator call. Public implicit-rebuild traversal uses the public
operator's default search settings, whereas explicit search workers apply the
recorded collection/query/COO options. Local matrix stages do not have a pair
denominator. Failed/unsupported requests remain explicit; no backend fallback
is allowed. The original worker still qualifies Warp operator/search fixtures
only. For matched native/Warp fixed-list chunking, use
[bench_chunking.py](CHUNKING.md); the simulator runner can also select native
core operators inside its Warp runtime explicitly.
