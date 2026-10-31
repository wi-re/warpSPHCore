# Unity and small-gap full-neighbor lists

Isolated primal interpolation experiment. Production dispatch, Warp kernels and
AD implementations are unchanged. Particles and each exact CSR row must already
be sorted. Compression is a separate planning step and the interpolation kernel
never reads or reconstructs a global CSR array.

Two actual GPU consumers are implemented:

* **Band:** two control bits per entry. Positive gaps 1 and 2 have no payload;
  3–256 have one shifted payload byte; larger gaps have four. This follows the
  shifted-delta scheme of Band et al., *Compressed Neighbour Lists for SPH*.
* **Nibble:** a unity mask per segment, one information nibble per nonunit gap,
  and variable payload nibbles for gaps larger than 9. This adapts the codec of
  Thaler/Keller to exact particle lists, rather than their shared cluster lists.

Both store a first absolute uint32 ID, then carry the last decoded ID and payload
cursor across segments. Every row starts on a four-byte boundary. Retained
metadata consists of int32 counts and uint64 byte offsets. The nibble segment
mask remains uint32 even for segments smaller than 32. The dummy first control
and row alignment make actual allocations slightly larger than the preceding
storage-only models. Empty rows occupy zero bytes. The prototype assumes
nonnegative IDs fitting int32 and a row stream smaller than 2 GiB; the aggregate
stream can exceed 4 GiB.

One subgroup owns a row. Each lane decodes 1, 2 or 4 adjacent entries; subgroup
scans locate payloads and recover absolute IDs. One or two segments can be
staged before gathering packed particle records. Controls in the same library
use exactly the same math and accumulation schedule with raw CSR32/64 IDs.
The encoder currently assigns a row to one thread, favoring readable packing
and race-free nibble writes over planning throughput.

## Reproduce

From this worktree, with a compatible CUDA/PyTorch environment:

```bash
nvcc -std=c++17 -O2 -arch=sm_120 -lineinfo -Xcompiler=-fPIC -shared --threads 4 \
  -Isrc/warpSPHCore/native/csrc experiments/csr_small_gaps/kernels.cu \
  -o /tmp/small-gaps.so
python experiments/csr_small_gaps/verify.py \
  --library /tmp/small-gaps.so --output /tmp/small-gaps-verify.json
```

`run_study.py --archive <output> --baseline <previous-scale-archive>` expects
`artifacts/small-gaps-v1.so` in the new archive and the previous archive's native
and planner libraries plus byte-format screen JSONs. It screens schedules at
262,144 particles, freezes selections, and starts a fresh process for each
held-out fixture. Defaults are 1M, 4M and 16M particles, 1D/2D/3D, both precisions,
and both support sizes (`n_h=4` and `2.57`). It refuses to start while another
GPU compute process is present and records activity before/after each fixture.

Sorting, search, conversion, compression, allocation of resident buffers and
all numerical checks occur outside interpolation timings. Reports distinguish
resident-buffer private calls from allocating Python wrappers, with native
Torch allocating calls as context. CUDA-event batches and wall times are both
retained. Planning costs and peak Torch allocation are reported separately;
the benchmark deliberately retains all compared representations simultaneously.

References:

* https://cg.informatik.uni-freiburg.de/publications/2019_CGF_CompressedNeighbors.pdf
* https://arxiv.org/html/2602.19873v1

The completed study and measured tradeoffs are documented in
[the regression report](../../docs/regression/native_backend_csr_small_gaps_2026_10_31.md).
