"""Runs the scripts/gradcheck_*.py canary scripts as pytest cases.

Phase 0's "gradient/finite checks" task (see warpier_core.md's Gradcheck
Script Rollout Plan) produced these as standalone scripts, one per operator,
rather than in-process pytest functions, because warpSPHCore_PRECISION is
baked into every compiled kernel at first warpSPHCore import and cannot
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
    "gradcheck_crk_correction_native.py",
    "gradcheck_renorm_native.py",
    "gradcheck_renorm_uniform_grid_native.py",
    "gradcheck_pinv_native.py",
    "gradcheck_scalar_arg_native.py",
    "gradcheck_twice_in_process.py",
    "gradcheck_tier2_jvp_density.py",
    "gradcheck_tier2_jvp_interpolate.py",
    "gradcheck_tier2_jvp_gradient.py",
    "gradcheck_tier2_jvp_gradient_crk.py",
    "gradcheck_tier2_jvp_gradient_renorm.py",
    "gradcheck_tier2_jvp_divergence.py",
    "gradcheck_tier2_jvp_divergence_crk.py",
    "gradcheck_tier2_jvp_divergence_renorm.py",
    "gradcheck_tier2_jvp_curl.py",
    "gradcheck_tier2_jvp_curl_crk.py",
    "gradcheck_tier2_jvp_curl_renorm.py",
    "gradcheck_tier2_jvp_laplacian.py",
    "gradcheck_tier2_jvp_laplacian_brookshaw_crk.py",
    "gradcheck_tier2_jvp_laplacian_brookshaw_renorm.py",
    "gradcheck_tier2_jvp_laplacian_dot_crk.py",
    "gradcheck_tier2_jvp_laplacian_dot_renorm.py",
    "gradcheck_tier2_jvp_laplacian_default_crk.py",
    "gradcheck_tier2_jvp_laplacian_default_renorm.py",
    "gradcheck_tier2_jvp_gradient_crk_renorm_simultaneous.py",
    "gradcheck_tier2_jvp_chained_backprop.py",
]


# Not gradchecks (they check a forward-mode JVP identity, not a backward
# pass), but the same subprocess-per-script gate, for the same
# precision-baking reason. warpier_fields.md Step G called the Tier-1 spike
# "throwaway"; it is kept as a standing gate instead because what it pins is
# load-bearing for Phase 6's cost estimate -- that every operator is exactly
# linear in its field values, so a Tier-1 tangent is the existing kernel
# re-launched on the tangent array. If a future kernel change breaks that
# linearity (an affine term, a value-dependent correction), Phase 6's plan
# stops being valid and nothing else in the suite would notice. It also
# happens to be what found renorm.py's caller-properties mutation -- see
# test_renorm_no_caller_mutation.py.

# Also kept as a standing gate for a different reason: no exact in-process
# reference exists for computeSPHDensityPositionHVP (Phase 4 step 3, "Hess C
# . v is a JVP of that JVP") the way test_forward_mode_geometry_jvp_density.py has
# one for the first-order JVP -- torch.autograd.functional.hessian would
# need double-backward through StateAwareWarpFunction's own backward(),
# which reads a non-differentiable wp.Tape (see wp_densityHVP.py's module
# docstring). The reference here is finite-difference-of-the-first-order-JVP
# instead, which only agrees to round-off at float64 -- hence the subprocess
# isolation, same as every other script in this file.
SPIKE_SCRIPTS = [
    "spike_forward_mode_tier1.py",
    "spike_forward_mode_tier2_density_hvp.py",
    "spike_forward_mode_tier2_crk_extension.py",
    "spike_forward_mode_tier2_renorm_extension.py",
    "spike_forward_mode_tier2_laplacian_dot_default_extension.py",
    "spike_forward_mode_tier2_crk_renorm_simultaneous.py",
]


# Not a gradcheck either (no torch.autograd involved at all -- pure Warp
# wp.Tape against the raw @wp.func kernel math in src/warpSPHCore/kernels/,
# one level below every operator these other scripts exercise). Same
# subprocess-per-script gate, same precision-baking reason. See its own
# docstring: this is what caught the ViscosityKernel/CohesionKernel/
# AdhesionKernel KernelFunctions enum collision, the sphKernelC_d
# argument-order bug, the Poly6/Spiky 1D+2D normalization-constant bugs, and
# a wrong AdhesionKernel third-derivative formula -- all fixed alongside it.
KERNEL_SANITY_SCRIPTS = [
    "kernel_sanity_native.py",
]


@pytest.mark.parametrize("script_name", GRADCHECK_SCRIPTS + SPIKE_SCRIPTS + KERNEL_SANITY_SCRIPTS)
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
