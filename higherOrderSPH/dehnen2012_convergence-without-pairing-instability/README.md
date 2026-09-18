# Dehnen & Aly 2012 — replication folder

- **Paper:** Improving convergence in smoothed particle hydrodynamics
  simulations without pairing instability. W. Dehnen & H. Aly, MNRAS
  425(2), 1068–1082 (2012). arXiv:1204.2471.
- **PDF:** `/home/lu26029/dev/warpSPH/literature/dehnen2012_convergence-without-pairing-instability.pdf`
- **Why we replicate it:** the origin paper of this codebase's kernel
  choices. `src/warpSPHCore/kernels/kernelFunctions/*.py` transcribes its
  Table 1 (`kernelScale` = H/h at h = 2σ) and Table 2 (`packingRatio` =
  h(ρ̂/m)^{1/3} at the kernel's reference N_H), and it is the standing
  reference for why the default kernel is Wendland2 rather than a
  B-spline. Replicating its evaluations validates that our kernels are
  exactly the paper's kernels and behave as claimed — the first milestone
  of the higher-order work in `../../higher_order.md`.

## Contents

| File | Purpose |
|---|---|
| `paper_notes.md` | Extracted paper content: notation map (paper h vs code h!), Tables 1–2, key equations, figure-by-figure inventory of what each figure evaluates, acceptance anchors |
| `PLAN.md` | Phased plan (0: scaffolding + literature sync → 1: kernel audit pipeline → 2: shipped-kernel audit → 3: Gaussian + HOCT4 in core → 4: density estimation → 5: linear stability → 6: C&D/R&H frontend build-out → 7: dynamic tests → 8: report), with acceptance criteria and a findings log |
| `data/`, `scripts/`, `notebooks/`, `sim/`, `figures/`, `results/` | Created as the plan proceeds (see `PLAN.md` layout) |

## Scope note (2026-09-17)

The replication also carries durable infrastructure scope:

- **Phase 1** builds the repo-level kernel onboarding/audit pipeline
  (`warpSPHCore/scripts/kernels/`, `tests/kernels/`) — the streamlined
  process for adding a kernel and proving it works (B7 verification,
  future general Wendland ψ_ℓk, …).
- **Phase 3** adds **Gaussian** (truncated at 16σ, the paper's convention)
  and **HOCT4** (from Read et al. 2010) to `warpSPHCore` as first-class
  kernels via that pipeline.
- **Phase 6** builds the **Cullen & Dehnen (2010) viscosity switch**
  (validating/cleaning the existing `warpSPH` module) and **Read &
  Hayfield (2012) artificial conductivity** into the `warpSPH`
  compressible scheme — prerequisite for the dynamic tests.

## Status

- 2026-09-17: paper extracted and plan written (Phases 0–8); scope
  extended as above. No scripts yet.
- Phase order is chosen so that the kernel-identity verdict
  (Phases 2–5, pure math / static evaluations) lands before any dynamic
  test work; Phase 0 first (it syncs `read2010` out of the literature
  dump, which blocks the HOCT4 work).

## Related

- Existing non-paper-specific kernel validation:
  `scripts/gradcheck/kernel_sanity_native.py` (normalisation, derivative
  chains, support boundary) — Phase 1 builds on it, does not duplicate it.
- Kernel constants in the code:
  `src/warpSPHCore/kernels/kernelFunctions/{wendland2,wendland4,wendland6,cubicSpline,quarticSpline,quinticSpline,B7,B8}.py`
  (`_C_d`, `_kernelScale`, `_packingRatio`).
- Lattice-normalisation correction (`calibrateNormalization` / n_h in
  `src/warpSPHCore/renorm.py` + `kernels/properties.py`) is a *different*
  correction from the paper's eq. (18) self-term correction — see
  `PLAN.md` Phase 5.
