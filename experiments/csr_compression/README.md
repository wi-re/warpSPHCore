# Lossless byte-compressed particle CSR

An isolated CUDA interpolation experiment, with production native/Torch CSR32
and CSR64 controls. The readable Warp kernels and all AD paths are unchanged.

Both formats preserve exact neighbor IDs in sorted CSR rows. A row is split
into independently decodable chunks of 4, 8, 16 or 32 entries. The last chunk is
padded with initialized zero codes; row counts exclude padding from computation.

- **Deltas:** uint32 first ID, byte differences, code 255 plus a separate uint32
  payload for large differences. Each chunk has a uint32 escape-payload offset.
- **Cell-relative:** byte offsets within contiguous source-cell ranges, a run
  boundary bitmask and uint32 run bases. Cells exceeding 256 particles are split
  into contiguous ranges of at most 256; there is no occupancy cap or truncation.
  Each chunk has a uint32 run-base offset.

Both retain int64 row-to-chunk offsets and int32 neighbor counts. Their uint32
payload directories require fewer than 2^32 payload entries. The codec supports particle IDs through INT32_MAX. Interpolation also inherits
the addressing limits of the existing packed-reference loader; these fixtures
remain well within those limits.

The scalar decoder assigns one lane per code. The packed decoder loads an
aligned uint32 containing four codes per thread. Delta reconstruction scans
quartet totals across a smaller subgroup; cell decoding reuses a run base until
a boundary bit changes it. Raw CSR controls support both schedules, so a change
in lane assignment is not silently attributed to compression.

`build_plan` explicitly converts an existing CSR. GPU count/fill passes and
Torch prefix sums allocate the final buffers; size readbacks happen only here.
Interpolation retains neither CSR IDs nor their offsets, nor the temporary
particle-to-cell map. Production search still creates CSR: this prototype does
not eliminate that intermediate or its planning peak memory. Reported storage
counts retained tensors, including row metadata, directories and code padding.

Build and verify:

```sh
nvcc -std=c++17 -O2 -arch=sm_120 -lineinfo -Xcompiler=-fPIC -shared \
  -Isrc/warpSPHCore/native/csrc experiments/csr_compression/kernels.cu \
  -o /tmp/csr-compression.so
python experiments/csr_compression/verify.py \
  --library /tmp/csr-compression.so --output /tmp/verification.json
```

The verifier uses independent scalar CPU readers, exact scalar/vector GPU ID
roundtrips, dense FP64 numerical references and bit-identical scheduled raw
controls. It covers empty/singleton/partial rows, 254/255 escapes, high int32 IDs,
oversized cells, zero supports, mixed periodicity, both precisions and nondefault
streams. CUDA memcheck/racecheck can wrap `verify.py --quick`.

Benchmark using the immutable `native-torch-final.so` and `exploration-v2.so`
artifacts from the preceding profiler study:

```sh
python experiments/csr_compression/benchmark.py \
  --library /tmp/csr-compression.so --artifacts /path/to/previous/artifacts \
  --precision float32 --screen --ns 262144 --repeat 7 --iterations 5 \
  --output /tmp/screen-float32.json
python experiments/csr_compression/benchmark.py \
  --library /tmp/csr-compression.so --artifacts /path/to/previous/artifacts \
  --precision float32 --selection /tmp/screen-float32.json \
  --ns 32768 131072 524288 1000000 --repeat 9 --iterations 5 \
  --output /tmp/final-float32.json
```

Repeat for float64. Both neighborhood sizes n_h=4/2.57 and dimensions 1/2/3 are
defaults. Screening chooses a fixed schedule at 262K, then held-out sizes use it
without retuning. Configuration tuples are `[chunk entries, batches per lane,
threads per CUDA block, packed-four-code decoder]`; packed batches process four
entries per lane instead of one. Screened modes are 1/2/4 scalar batches and
1/2 packed batches, at 128/256 CUDA threads.

GPU-event and wall samples rotate all paths within each batch. Matched private
controls share the C++ launch loop, arithmetic, field packing and resident
scratch/output. Public compressed prototypes allocate scratch/output each call;
public raw controls use native/Torch. Host wrapper costs differ there. All
validation, sorting, search, CSR32 conversion and compression are outside
interpolation timing. Conversion wall times are retained separately. Search and
Morton sorting time are not included in conversion measurements.

`analyze.py ARCHIVE` consumes `final-float32.json` and `final-float64.json`, plots
both matched and allocating calls, exports CSV, and compares actual adjacency
storage. Throughput axes start at zero; whiskers show observed batch ranges.

Measured results and profiling interpretation: [study report](../../docs/regression/native_backend_csr_compression_2026_10_31.md).
Additional `selection-chunk32-*` / `chunk32-*` archive files hold the compressed
chunk size at 32 for the 3D memory/performance comparison. Their batch/thread/vector
choices are selected from the same independent screen under that constraint.

## Large fixtures and comparison with published codecs

The [extended study](../../docs/regression/native_backend_csr_compression_scaling_2026_10_31.md)
measures 1M/2M/4M/8M/16M particles. Run each fixture in a fresh process to avoid
retaining a previous large topology. `--target-ms 30` selects 1..100 calls per
batch from a native64 pilot; selected iteration counts and pilot measurements
are recorded. For example:

```sh
python experiments/csr_compression/benchmark.py \
  --library /path/to/artifacts/compression-v2.so --artifacts /path/to/artifacts \
  --precision float32 --selection /path/to/screen-float32.json \
  --dims 3 --n-h 4 --ns 16000000 --repeat 9 --iterations 3 \
  --target-ms 30 --plan-repeat 1 --output /tmp/float32-d3-h4-n16000000.json
```

`aggregate_scale.py ARCHIVE` expects files named
`{precision}-d{dimension}-h{nh}-n{particles}.json` for the 60-fixture grid, then
writes aggregate inputs for `analyze.py`. Sorting in 1D uses exact coordinate
order, equivalent to the 1D Morton order, without the toy key encoder's 21-bit
limit. Dimensions 2/3 retain the previous refined Morton planner.

`codec_models.py` contains offline size formulas for row-stream byte escapes,
Stream VByte, Band-style controls and nibble coding of full lists. These models
include identical row metadata and do not implement or time GPU consumers.
`model_sizes.py` samples query rows against the full sorted reference state,
avoiding an unused full CSR for this separate analysis:

```sh
python experiments/csr_compression/model_sizes.py --artifacts /path/to/artifacts \
  --ns 1000000 16000000 --output /tmp/codec-models.json
```
