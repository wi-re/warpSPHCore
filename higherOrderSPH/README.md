# Higher-Order SPH — paper replications

Purpose: before building the higher-order kernels in `../higher_order.md`,
validate the current `warpSPHCore` kernel stack by replicating published
results and evaluations. A replication proves two things at once:

1. the kernels we ship are *exactly* the kernels of the reference paper
   (functional form, normalization, scale convention), and
2. the kernels *behave* as the paper claims (density estimates, stability,
   sound speed, dynamics tests) when used the way the paper uses them.

Each replication lives in its own subfolder named after the PDF file name
(no extension) from the literature collection at
`/home/lu26029/dev/warpSPH/literature/`.

## Folders

| Subfolder | Paper | Status |
|---|---|---|
| `dehnen2012_convergence-without-pairing-instability/` | Dehnen & Aly 2012, MNRAS 425, 1068 (arXiv:1204.2471) — Wendland kernels, pairing instability, kernel-NH evaluation | planned (`PLAN.md`, `paper_notes.md`); no scripts yet |

The dehnen2012 replication also carries scope beyond the paper itself:
a repo-level **kernel onboarding/audit pipeline** (`scripts/kernels/` +
`tests/kernels/`), **Gaussian + HOCT4 kernels added to the core library**,
and the **C&D 2010 / R&H 2012 dissipation build-out in the `warpSPH`
frontend**. See the replication's `PLAN.md` for the phase structure.

## Conventions (all replication folders)

- Python: conda env `warp` — `/home/lu26029/miniconda3/envs/warp/bin/python`
  (Python 3.13, NumPy 2.x, warp, torch).
- The box is shared: cap BLAS threads before any numerics-heavy command:
  `export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4`
- Replication code imports kernels from the installed `warpSPHCore` package
  (`from warpSPHCore.kernels.eval_kernel import ...`,
  `from warpSPHCore.enumTypes import KernelFunctions`) — it must never
  re-implement a kernel that exists in `src/`. Kernels that exist only in
  the paper (Gaussian, HOCT4) are defined locally in the replication folder
  and clearly marked as such.
- Each paper folder contains:
  - `README.md` — paper identity, why we replicate it, status.
  - `paper_notes.md` — extracted content: definitions, tables, key equations,
    a figure-by-figure inventory of what each figure evaluates.
  - `PLAN.md` — the replication plan with phases and acceptance criteria.
  - `scripts/` — deterministic replication code, one per figure/table.
  - `notebooks/` — exploratory notebooks (optional per figure).
  - `data/` — reference values transcribed from the paper (single source of
    truth, machine-readable).
  - `figures/`, `results/` — outputs (git-ignored).
- Findings that contradict the shipped code are recorded in the plan's
  findings log, not fixed silently — kernel constants are only changed in
  `src/` as a deliberate, separate step after a discrepancy is confirmed.
- Kernels added to `src/` (by any replication) go through the onboarding
  pipeline: `kernelFunctions/<name>.py` + enum + dispatch + entry in
  `scripts/kernels/kernel_specs.yaml` + green `kernel_audit.py` + CI test.
  See `scripts/kernels/README.md` once it exists (dehnen2012 Phase 1).
