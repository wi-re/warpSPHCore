# Particle-index width experiment

These prototype scripts compare int32/int64 CSR particle IDs. They target the
historical ABI recorded in their source hashes; current production dtype
configuration is documented in the repository README. This experiment does not
select production defaults.

Detailed reports, raw results, audit and figures are in
`../warpSPHDevelopment/core/experiments/index_width/` relative to the repository
root. The original detailed README is preserved there. See the
[development artifact policy](../../docs/docs/development-artifacts.md).

Run from the repository root with the matching source/runtime and a CUDA device:

```bash
export WARPSPH_DEVELOPMENT_ROOT="$(cd .. && pwd)/warpSPHDevelopment"
CORE_DEV="$WARPSPH_DEVELOPMENT_ROOT/core"
python experiments/index_width/benchmark.py --help
python experiments/index_width/benchmark.py --output "$CORE_DEV/experiments/index_width/results.json"
python experiments/index_width/analyze.py
python experiments/index_width/plot.py
```

Analysis and plotting use the external folder by default. Analysis checks the
recorded source/generated-source hashes; newer sources do not silently qualify
historical runs. Preserve a previous result under a dated name before replacing
it. No benchmark is run by the ordinary documentation build.
