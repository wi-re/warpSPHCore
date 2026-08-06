"""Runs the scripts/gradcheck_*.py canary scripts as pytest cases.

Phase 0's "gradient/finite checks" task (see warpier_core.md's Gradcheck
Script Rollout Plan) produced these as standalone scripts, one per operator,
rather than in-process pytest functions, because SPHWARPCORE_PRECISION is
baked into every compiled kernel at first sphWarpCore import and cannot
change mid-process (the same constraint operation_matrix.py's _configure()
docstring documents) -- importing several of these scripts' modules into one
pytest process would have the later ones silently reuse the first script's
precision. Subprocess isolation sidesteps that entirely: each script gets
its own fresh interpreter, exactly as if a user ran it by hand, so this file
just shells out and checks the exit code.

repro_warp_grad_reentrancy.py is deliberately not included here: it always
exits 0 and prints a PASS/FAIL matrix for illustration (naive/bug-1-only/
bug-2-only are *expected* to fail), so it isn't a pass/fail gate -- see its
own docstring and warpier_core.md for when to re-run it by hand.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"

GRADCHECK_SCRIPTS = [
    "gradcheck_density.py",
    "gradcheck_density_native.py",
    "gradcheck_interpolate_native.py",
    "gradcheck_gradient_native.py",
    "gradcheck_divergence_native.py",
    "gradcheck_curl_native.py",
    "gradcheck_laplacian_native.py",
    "gradcheck_covariance_native.py",
    "gradcheck_crk_native.py",
    "gradcheck_renorm_native.py",
]


@pytest.mark.parametrize("script_name", GRADCHECK_SCRIPTS)
def test_gradcheck_script(script_name):
    script_path = SCRIPTS_DIR / script_name
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"{script_name} exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
