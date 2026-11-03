# Fixed-list chunking diagnostic

`bench_chunking.py` isolates operator dispatch and GPU launch geometry from
neighbour-list construction and simulator maintenance. It supports the Warp and
native core backends on CPU/CUDA, FP32/FP64, input/Morton storage, and explicit
index widths. Run each backend/device in a fresh process with the same arguments.
Use the normal installed core environment; native CUDA builds require its usual
compiler/Ninja setup. No new runtime dependency is introduced.

```sh
python scripts/benchmarks/bench_chunking.py --backend warp --device cuda:0 \
  --nx 48 --operations density gradient divergence laplacian covariance crk_factors \
  --chunks 0 1 2 4 8 --lanes 1 --output /path/to/evidence/warp.json

python scripts/benchmarks/bench_chunking.py --backend native --device cuda:0 \
  --nx 48 --operations density gradient divergence laplacian covariance crk_factors \
  --chunks 0 1 2 4 8 --output /path/to/evidence/native.json

# Cooperative Warp kernels; native kernels own their launch geometry.
python scripts/benchmarks/bench_chunking.py --backend warp --device cuda:0 \
  --nx 48 --operations gradient crk_factors --chunks 0 8 --lanes 32 \
  --profile --output /path/to/evidence/warp-cooperative.json
```

`--chunks 0` means ordinary CSR; `1` still uses the chunked container and dispatch.
The default fixture is a periodic 3D lattice, physical support `4/nx`, candidate
list support `1.24` times that radius, and deterministic random scalar/vector
fields. Candidate degree includes self and pairs outside the physical support.
The default avoids putting a lattice shell exactly on the search cutoff, where
backend strict/inclusive distance predicates can differ. Compare actual edge
counts as well as requested radii. The requested number of chunks sets a maximum row count; actual chunk counts
and row sizes are recorded. All requested topologies stay resident. `--budget-gib`
is a conservative capacity estimate, not an allocator limit. Raise it explicitly
for large fixtures; the periodic search radius must stay below half the box.

Setup checks identical counts and exact live reference-ID sequences against
that backend's CSR search. Native search normally emits int64 IDs; the frozen
operator fixture converts them to `--index-dtype` outside timing so storage widths
match. Each backend retains its own neighbour traversal order. Every timed variant
checks finite output and agreement with the same-backend, lane-1 CSR reference.
CRK comparisons scale volume, B, gradA and gradB to dimensionless quantities;
raw derivative magnitudes depend on resolution. Flat reference arrays are retained
as `.reference.npz` for independent cross-backend comparisons. Morton setup
permutes positions and all fields consistently, before building any topology.

Timings include normal public-call dispatch, output allocation/release and
completed work. They exclude search, planning, sorting and validation. CUDA stream
intervals include launch gaps; they are not pure kernel time. `--profile` adds
separate Torch traces and a Python profile after uninstrumented timing. Python
profile durations can include waiting for the GPU. Native `--lanes` is deliberately
rejected except for the placeholder default of 1; its cooperative width must be
read from native source or launch traces, not inferred from that argument.

Use fresh output paths. JSON contains raw repetitions, actual edge count, machine
and source metadata, Python and native source hashes, errors and effective backend.
An immutable script copy is retained beside the result. Keep run evidence outside
this checkout; see `docs/docs/development-artifacts.md`. Failed cases exit nonzero.

For full simulator timings, use warpSPH's `benchmarks/performance/run.py` and sweep
`--neighborhood csr chunked`, `--edge-budget`, `--max-rows` and `--lanes`. Its
`--operator-backend native` selects native operators inside the Warp runtime; it
does not select the native MFM controller or qualify a native-only simulator.
Fixed-list timings exclude CPU planning and repeated geometry/search work, which
can dominate a full step even when individual native operators scale well.

## Local qualification after the native merge

On the RTX PRO 6000 Blackwell (188 SMs), FP32/int32-ID 128³ fixtures with 485
candidate entries per row gave these completed-call medians, in milliseconds:

| Operation | Warp CSR | Warp 144 chunks | Native CSR | Native 144 chunks |
| --- | ---: | ---: | ---: | ---: |
| Gradient | 26.02 | 149.28 | 7.86 | 10.64 |
| CRK factors | 158.26 | 469.75 | 89.09 | 104.40 |

The small chunks contain at most 14,564 rows / about 7.06M live edges. Eight much
larger chunks gave Warp 28.48 / 175.23 ms and native 8.07 / 89.06 ms respectively.
Thus global particle count alone does not predict the chunking penalty. Separate
traces of comparable small chunks show Warp scalar gradient launches with only
54 blocks versus native cooperative launches with 863 blocks. These are launch
measurements, not a general proof of optimal native bandwidth use. Morton did not
improve the ordered lattice fixture. FP32/FP64 CPU checks and matched native/Warp
reference-array comparisons passed; this does not qualify developed turbulence.

The companion 48³ Warp simulator comparison measured CRKSPH at 74.72 ms CSR,
229.96 ms with the default 8M-edge budget, and 75.67 ms after raising both the
edge and row limits. MFM was 24.02, 138.07 and 124.53 ms: larger chunks leave its
CPU planning bottleneck. Synthetic planner-only counts took about 52 ms at 48³
and 1.04 s at 128³ on one P core of an Intel 285K. MFM's measured trajectory
builds two chunked searches per accepted step. The reference planner's Python
row loops/digest and geometry's repeated construction therefore need separate
attention from GPU kernel packing. These timings are single-machine evidence,
not a strong/weak scaling or GH200 result.
