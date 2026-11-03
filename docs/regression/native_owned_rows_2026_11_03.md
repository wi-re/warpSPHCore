# Native core closure: bounded owned rows

Stage four ports compact-index count/collection and bounded Verlet filtering to
module-owned CPU/CUDA/HIP code. The row planner, ownership schedule, replica
checks, redistribution, leases and checkpoint protocol stay shared with upstream.
A native launch-record protocol allows the same dispatcher to call native kernels
in both a Warp-free process and a Warp process using native operators.

Standard particle operations, CRK preparation, renormalization, polynomial
assembly and meshless CG/MG closure and rates accept owned row chunks. Query
fields are sliced together; references retain global IDs. Extra tensors declare
their row role explicitly. Polynomial self/sector and LABFM terms carry the
global row base. Empty populations, separate/equal-sized populations, asymmetric
supports, masks and symmetric density factors retain flat-CSR behavior.

Search counts and collects directly into bounded chunk buffers, including 32-bit
and 64-bit J. A regression forbids the global-CSR search entry point during
chunked rebuilding. The compact index lives in Torch tensors, with stream-ready
and record-stream ownership. Initial index creation has a transient duplicate
compact-index allocation copied device-to-device; references and O(N) results
remain replicated. There is no claim of constant total memory or new timings.
No particle/neighbor CPU readback or CSR-content verification enters execution.

| Qualification | Result |
|---|---|
| Three bounded native row contracts | CPU and strict prebuilt CUDA pass |
| Native prior dense/solver, polynomial, meshless, options and Verlet contracts | 17 passed, 1 existing skip on CPU |
| Two-rank CPU Gloo worker, exact and contract replica modes | Pass with both 32-bit and 64-bit indices |
| Warp-free CPU/CUDA composite probes | Search, all standard operators, CRK/renorm, polynomial and CG/MG meshless pass |
| Warp-built chunk lists/native launch records | Pass on CPU |
| Affected upstream Warp CPU suites | 280 passed, 2 skipped; two allocation-interception failures fixed and focused rerun passes |
| Python scaffolding | 24 passed |
| Offline HIP gfx942 | Index, radius, Verlet and polynomial GPU translation units compile |

The Gloo worker also exercises filtering/rebudgeting across owners, dependent
reads, replay with a changed checkpoint budget, coordinated rank-local errors
and subsequent process-group recovery. CUDA probes include default-device and
non-default-stream execution. No NCCL/multi-node or AMD runtime claim is made.
Chunked AD and graph capture remain explicitly unsupported, matching upstream.
Four maintained contracts include one opt-in extended multi-process check; this
stage does not duplicate the upstream Cartesian campaign.

Logs, failed/superseded probes, source, binary, receipt and HIP objects are kept
outside core under
`/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-core-owned-rows-20261103`.
