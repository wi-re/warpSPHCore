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

## CPU planner optimization

The baseline above predates the optimized shared planner. Planning now computes
one exact live-edge bound from padding/alignment, scans counts in NumPy int64,
finds maximal chunk ends with binary searches, and hashes the prefix buffer in
one call. The public Python-integer prefix tuple, chunk boundaries, owners,
errors and v1 signature encoding are preserved. All callers use the faster path;
there is no new compute flag, device build or CUDA initialization requirement.
NumPy is already a core dependency.

On the same single P core and constant counts, the 48³ planner changed from
51.33 to 1.32 ms and the 128³ planner from 1037.12 to 53.72 ms. Heterogeneous
128³ counts changed from 1048.63 to 54.42 ms; signatures matched the old code.
The immutable prefix still requires O(N) Python integer storage/construction.

Fresh matched 48³ accepted-step runs (three warmup, five measured steps, five
repetitions, default chunk budgets) changed MFM from 138.17 to 44.53 ms and MFV
from 137.48 to 45.40 ms. CRKSPH remained about 227–228 ms, consistent with its
retained list during these measured steps and its GPU launch bottleneck.

The host-only maintained benchmark times completed plan construction and output
release, with count generation and validation outside the window:

```sh
python scripts/benchmarks/bench_row_planner.py --rows 110592 2097152 \
  --owners 1 4 --output /path/to/evidence/planner-constant.json
python scripts/benchmarks/bench_row_planner.py --rows 110592 2097152 \
  --distribution uniform --output /path/to/evidence/planner-variable.json
```

Default policy is an 8M-edge budget, 65,536 maximum rows, 10% headroom,
16 bytes/edge and a three-scalar extent per edge. Supply `--policy policy.json`
with `RowChunkPolicy` fields for other exact rational padding or byte budgets.
`--neighbors` sets the constant count or the inclusive upper bound of uniform
random counts. These are synthetic stored counts, not generated particle pairs.
Use external CPU affinity controls for matching comparisons. Results include
raw samples, counts hashes, signatures, policy, owner counts and source metadata.


## Cooperative CRK volume and compatible work

Warp CRK volume and simulator compatible-work sweeps now have deterministic
multi-lane variants. Each lane visits a contiguous part of a row; particle sums
use the existing fixed tile reduction, and pair-work writes remain disjoint.
The scalar kernels remain available and CPU dispatch stays scalar. Native core
operators keep their own implementation and launch geometry.

These two operators default to 32 lanes for `ChunkedAdjacency`, retaining their
scalar default for flat CSR/grid inputs. Explicit `setNeighborLanes` or
`WARPSPHCORE_NEIGHBOR_LANES` settings take precedence. This distinction matters:
forcing 32 lanes for both new variants made the 96³ flat trajectory about 6%
slower, although it substantially improved bounded chunks. Chunk budgets still
matter; this policy does not establish an optimum for every custom chunk size.
Different lane counts can change floating summation order.

Matched production-default CRKSPH accepted-step medians on the same Blackwell,
FP32, input-order fixture (three warmup steps, three repetitions):

| Size | CSR before | CSR after | Chunked before | Chunked after |
| --- | ---: | ---: | ---: | ---: |
| 48³ | 71.85 ms | 71.73 ms | 120.12 ms | 82.38 ms |
| 96³ | 526.74 ms | 526.95 ms | 920.09 ms | 627.23 ms |

The measured windows contain five and three accepted steps respectively, using
the default 8M-edge / 65,536-row budgets. These measurements already include the
CPU planner optimization above and the existing cooperative force kernels.
Within each before/after pair, live directed edge counts and accepted times
match. Separate 48³ five-step profiles reduced compatible work from 42.9 to
13.1 ms/step and volume from 11.7 to 3.3 ms/step. Those instrumented kernel sums
are separate evidence and must not replace completed-step wall timings.

FP32/FP64 checks cover volume geometry gradients on CSR/grid inputs, masked rows,
CUDA graph replay with changed supports, compatible work, stage rates, topology
rebuild/replay, default kernel selection and energy budgets. The differentiable
forcing route retains its expanded fallback. These are local Warp results;
3090/GH200 and multi-GPU comparisons still need matched runs.

The frontend benchmark retains `--lanes 1` as its scalar baseline. Use
`--lanes default 1 32` to compare production/environment selection with both
forced paths. The `default` request is recorded as JSON `null`, distinct from
1 or 32; pin or unset `WARPSPHCORE_NEIGHBOR_LANES` consistently across machines.
