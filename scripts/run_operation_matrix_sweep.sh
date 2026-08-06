#!/usr/bin/env bash
#
# Wrapper around scripts/operation_matrix.py for two common invocation
# patterns:
#
#   --quick (default): the same single 2D/float32/non-jittered CPU sweep
#     that .github/workflows/tests.yml runs on every push (~15s). Safe to
#     run as often as you like while iterating on an operator.
#
#   --full: the precision x dim x jitter x device sweep documented in the
#     operation-matrix skill. Recompiles and re-launches every
#     operator/scheme/traversal/correction combination many times over --
#     several minutes, not something to run per-edit. Reserve it for
#     validating a change that touches kernel math broadly (a shared
#     @wp.func, a traversal path, an AD-bridge fix) before merging, not
#     for routine iteration -- use --quick for that.
#
# The gated part of --full only uses jitter in {0.0, 0.01} -- the only
# jitter level actually confirmed to stay under the MAE threshold on every
# cell. Heavier jitter (0.15-0.3, closer to the notebooks' examples and what
# actually stress-tests CRK/renorm) is real, expected-HIGH territory today:
# sound thresholds for it are an open Phase-0 item (see warpier_core.md),
# not a bug in this script. --full still runs one pass at --jitter 0.3
# afterward for human inspection, but WITHOUT --ci, so it reports rather
# than fails the sweep.
#
# Every gated run is passed --ci, so this exits non-zero on the first
# HIGH/ERR/NAN cell or fatal build error, with set -e stopping the sweep
# right there.
#
# Usage:
#   scripts/run_operation_matrix_sweep.sh
#   scripts/run_operation_matrix_sweep.sh --quick
#   scripts/run_operation_matrix_sweep.sh --full

set -euo pipefail

cd "$(dirname "$0")/.."

MODE="${1:---quick}"

run() {
  echo
  echo "=== operation_matrix.py $* ==="
  python scripts/operation_matrix.py "$@"
}

if [[ "$MODE" == "--quick" ]]; then
  run --device cpu --ci --verbose
  exit 0
fi

if [[ "$MODE" != "--full" ]]; then
  echo "Usage: $0 [--quick|--full]" >&2
  exit 2
fi

echo "Full sweep: this launches many operation_matrix.py runs and can take"
echo "several minutes. Reserve this for validating a broad kernel-level"
echo "change, not routine iteration (use --quick for that)."

HAVE_CUDA=$(python -c "import torch; print('1' if torch.cuda.is_available() else '0')")

DEVICES=(cpu)
if [[ "$HAVE_CUDA" == "1" ]]; then
  DEVICES+=(cuda)
fi

# dim=1/2: cheap enough to run on every available device/precision/jitter
# combination. dim=3 is handled separately below, CUDA-only (see
# warpier_core.md: Warp's CPU backend is single-core/unoptimized, so 3D is
# prohibitively slow on CPU). jitter is limited to {0.0, 0.01} here -- see
# the header comment for why heavier jitter isn't part of the gated sweep.
for device in "${DEVICES[@]}"; do
  for precision in float32 float64; do
    for dim in 1 2; do
      nx=32
      [[ "$dim" == "1" ]] && nx=64
      for jitter in 0.0 0.01; do
        run --device "$device" --precision "$precision" --dim "$dim" --nx "$nx" --jitter "$jitter" --ci
      done
    done
  done
done

if [[ "$HAVE_CUDA" == "1" ]]; then
  for precision in float32 float64; do
    for jitter in 0.0 0.01; do
      run --device cuda --precision "$precision" --dim 3 --nx 8 --jitter "$jitter" --ci
    done
  done
else
  echo
  echo "No CUDA device available -- skipping the dim=3 sweep (CPU is prohibitively slow for 3D, see warpier_core.md)."
fi

echo
echo "=== Diagnostic-only pass: heavier jitter (0.3), NOT gated on --ci ==="
echo "Expect real HIGH cells here -- sound thresholds for this jitter level"
echo "are an open Phase-0 item, not a regression. Eyeball the table instead"
echo "of the exit code."
python scripts/operation_matrix.py --device cpu --jitter 0.3 --verbose || true

echo
echo "Full sweep complete."
