# Development records and generated results

User-facing documentation, small regression fixtures, reproduction scripts and
curated guide figures stay in Git. Development logs, detailed qualification
reports, raw measurements and historical plot collections live outside the
checkout. Building these docs and running ordinary tests does not require
the development archive.

## Local archive

The default layout is a sibling of the source repository:

```text
dev/
  warpSPHCore/
  warpSPHDevelopment/
    README.md
    core/
      README.md
      REPOSITORY_PLAN.md       # detailed plan at the time of migration
      docs/development/        # reports and generated figures
      docs/regression/         # raw results and workload fixtures
      experiments/            # experiment results and reports
      higherOrderSPH/          # generated harness reports
      logs/2026-10-16/          # captured development runs
      manifests/              # migration inventory and SHA-256 checksums
```

Other projects can use separate children such as `plotting/` under the same
`warpSPHDevelopment` root. Backup of that shared folder is manual; no cloud
upload, synchronization or retention service is configured by this repository.

`scripts/development_paths.py` resolves the core artifact directory. To use a
different location, set the shared root (not its `core` child):

```bash
export WARPSPH_DEVELOPMENT_ROOT=/path/to/warpSPHDevelopment
```

This is also useful for worktrees whose immediate parent is not `dev/`.
The helper only resolves paths. Producers create directories when writing;
imports and `--help` do not create an archive.

## Regeneration

Run commands from the source checkout, with the appropriate package/runtime
installed. Set output paths explicitly for tools requiring them:

```bash
export WARPSPH_DEVELOPMENT_ROOT="$(cd .. && pwd)/warpSPHDevelopment"
CORE_DEV="$WARPSPH_DEVELOPMENT_ROOT/core"

python scripts/diagnostics/qualify_mfm.py --suite wave --n 32 --device cpu \
  --precision float64 --backend warp --modes MFM MFV --closures project \
  --tend 0.001 --output "$CORE_DEV/docs/regression/mfm-short-new.json"

python scripts/diagnostics/qualify_reconstruction.py --help
python scripts/diagnostics/qualify_mfm_geometry.py --help
python scripts/benchmarks/plot_work_scaling.py --help
```

Development plotting/frontend tools with default result paths now use the
external core directory. Explicit CLI output arguments remain caller-owned.
Historical commands in archived reports preserve the paths used for their
original runs; replace those output paths with the external location when
rerunning them. Source hashes, configurations and numerical assumptions remain
in the original records. Exact historical numbers require compatible sources
and dependencies; regeneration on a newer revision creates new evidence.
Source-hash guards are retained rather than silently relabeling old results.

The [field reconstruction guide](field-reconstruction-guide.md) is a curated
documentation artifact. Its small figures and example dataset intentionally
remain in `docs/docs/assets/reconstruction/`, and its regeneration command
continues to update those public assets.

## Core capability qualification

The bounded core pass is recorded by source commit `2034a4e`. Its final local
checks passed:

| Gate | Result |
|---|---|
| Full CPU suite | 2,568 passed, 2,429 skipped; includes 44 isolated float64 derivative scripts |
| Real-CUDA gate, float32 | 4,143 passed, 163 skipped |
| Real-CUDA gate, float64 | 4,163 passed, 143 skipped |
| Targeted operation matrices | All five passed, including 1D, 2D/both precisions/light jitter and 3D CUDA |
| Public examples | Fourteen configurations per precision on CPU/CUDA |
| Isolated wheel | Same public examples passed on both devices and precisions |
| Documentation | Production build passed |

The runtime was Python 3.13.14, Torch 2.13.0+cu130, Warp 1.17.0 and NumPy
2.5.2, with an RTX PRO 6000 Blackwell for CUDA. Python >=3.10 is the package
metadata range; hosted CPU CI is configured for Python 3.12. Those declarations
do not certify every dependency/OS/GPU combination. Skips and exclusions are
described in the archived reports. Hosted CUDA CI still requires GPU infrastructure.

Detailed evidence is in `core/docs/development/delivery-validation.md` and
`core/docs/regression/capability-delivery-2026-10-16.json` inside the shared
development folder. Historical reconstruction/MFM reports and raw ladders are
there too. The public [reconstruction](reconstruction.md), [MFM/MFV](mfm.md),
[autodiff](autograd.md) and [linalg](linalg.md) pages retain their supported
contracts and explicit limitations. The [runnable examples](examples.md) are
independent of the archive.
