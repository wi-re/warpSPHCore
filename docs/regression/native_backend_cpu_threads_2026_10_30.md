# CPU worker scaling at one million particles — 2026-10-30

The machine has **24 physical cores with one hardware thread per core**: eight
performance cores (OS CPUs 0–7) and sixteen efficiency cores (CPUs 8–23). All 24
are available to the benchmark process. The earlier particle-count sweep selected
four workers explicitly; the standalone runtime has no four-worker limit.
`Context.cpu_threads=0` uses the OpenMP default, which can be controlled through
its environment; a positive count explicitly requests that many workers.

This follow-up compares the actual serial Warp CPU implementation against native
multicore execution at **4, 8, 16 and 24 workers**. Multicore execution is a primary
benefit of the native backend. These results are useful end-to-end backend
comparisons without normalizing away that benefit.

The graph `cpu-thread-scaling.png` and raw measurements live in the external
development archive at `core/experiments/native-backend-20261030-cpu-threads/`.

- Density, standalone output reuse: 4 workers: **0.718 s**, 8 workers: **0.324 s**, 16 workers: **0.177 s**, 24 workers: **0.192 s**.
- Interpolation, standalone output reuse: 4 workers: **1.069 s**, 8 workers: **0.508 s**, 16 workers: **0.302 s**, 24 workers: **0.309 s**.
- Covariance, standalone output reuse: 4 workers: **0.910 s**, 8 workers: **0.409 s**, 16 workers: **0.233 s**, 24 workers: **0.236 s**.
- Fresh standalone search: 4 workers: **2.262 s**, 8 workers: **1.107 s**, 16 workers: **0.795 s**, 24 workers: **0.858 s**.

At 24 workers, standalone output reuse is between **8.23× and
18.28× faster** than the serial Warp implementation across the seven
SPH primals in this run. Native/Torch and both standalone allocation policies are
shown separately in the plot and retained in `summary.csv`.

## Workload, placement and interpretation

Every run uses the existing deterministic 2D million-particle FP64 fixture with
32,996,295 directed neighbor entries (32.996 per particle, including self).
Wendland2, Gather support, Difference gradients, Brookshaw Laplacian and identical
sorted CSR are retained from the earlier comparison. All seven primals and full
fresh search are measured. Search includes compact-index construction, count,
scan, allocation and collection; operators exclude search.

Each worker count runs in a separate process pinned to the first T CPUs in the
available affinity mask. `OMP_NUM_THREADS=T`, `OMP_PLACES=cores`,
`OMP_PROC_BIND=close` and `MKL_NUM_THREADS=T` are set before importing Torch.
Torch CPU workers and the standalone context both receive T. Warp retains its
serial CPU implementation, including when more CPUs are available to its process.
The four/eight-worker cases use performance cores; the sixteen/twenty-four-worker
cases mix performance and efficiency cores. CPU topology and exact commands are
recorded in `manifest.json`.

Standalone execution uses OpenMP static partitioning of query particles. Each
worker traverses its particles' neighbor rows sequentially and writes independent
primal outputs. Native/Torch schedules the same numerical row kernels through
Torch's parallel-for facility. No explicit SIMD implementation is present.
Equal static particle ranges do not balance unequal P/E core speeds. Search also
contains serial support-bound reduction, sorting, cell construction and CPU scan
stages. Thus adding all available cores need not produce proportional speedup.

Each case has one warmup batch and three single-call measured batches. Lines show
medians and shaded bands show min/max. All 124 timing records passed their
qualification: exact sorted neighbor membership for search and FP64 primal
agreement against Warp outside the measured intervals. No debug content or CSR
verification is in a timed execution path.

This remains a workstation measurement with desktop/background activity. Affinity
and worker count are controlled, but clocks and system load are not isolated.
Torch JIT uses `-O2 -march=native`; standalone uses portable Release `-O3` with
OpenMP. Small timing differences between native allocation policies should not
be interpreted as an allocation benefit without further measurement.

## Reproduction and retained evidence

Run `scripts/benchmarks/bench_cpu_threads.py` using the instructions in
`scripts/benchmarks/native/README.md`. The exact measured driver is archived as
`driver-measured.py`, whose SHA256 matches `manifest.json`; `driver-portable.py`
and the repository driver generalize its root path and plot annotation.
Per-worker JSON files retain source/build hashes, raw samples, activity snapshots
and correctness qualification. No numerical runtime changes were needed for this
sweep. The archive is
`core/experiments/native-backend-20261030-cpu-threads/` under the development root.

After this run, the general benchmark default was changed from four workers to
the number of CPUs available to the process. Explicit `--threads` remains
available. The measured benchmark source is retained as
`bench_backends-measured.py`; its hash matches every per-worker result. All
measured commands supplied `--threads` explicitly, so the default change does
not change their execution.
