# Neighbor-major experiment

Prototype neighbor-major storage and CSR-J comparison tools live here. They
retain source-transform checks for their historical ABI and do not change
production dispatch. Their primal performance measurements do not establish
AD or full-scheme support.

The original detailed report, results and figures are archived under
`../warpSPHDevelopment/core/experiments/neighbor_major/` relative to the repository
root. See the [development artifact policy](../../docs/docs/development-artifacts.md).
Reproduction requires the source state and runtime pinned by the report hashes.

Run from the repository root, directing results to the external archive:

```bash
export WARPSPH_DEVELOPMENT_ROOT="$(cd .. && pwd)/warpSPHDevelopment"
CORE_DEV="$WARPSPH_DEVELOPMENT_ROOT/core"
python experiments/neighbor_major/benchmark.py --help
python experiments/neighbor_major/benchmark.py --output "$CORE_DEV/experiments/neighbor_major/results/new-run.json"
python experiments/neighbor_major/plot.py --help
```

The benchmark requires CUDA. Plotting accepts explicit `--inputs`, `--output-dir`
and `--summary` paths; use the external directory for each. Preserve historical
runs under their original names, and consult the archived README for the full
matched-layout sweep and build-path commands.
