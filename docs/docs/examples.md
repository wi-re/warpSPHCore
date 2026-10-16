# Runnable composition examples

These scripts are small external clients of the public package. Install
`pip install -e .` from the repository, then run:

```bash
python scripts/examples/run_all.py --device cpu --precision float32
python scripts/examples/run_all.py --device cpu --precision float64
# Requires an accessible CUDA device; absence is a failure:
python scripts/examples/run_all.py --device cuda --precision float64
```

Each invocation checks fourteen configurations and prints numerical records.
Precision is selected before importing the core; use a fresh process when
changing it. A `warpSPHCore_config` module takes precedence over environment
configuration. The records show the actual dtype. The files are in
`scripts/examples/` in the repository; they are not installed console commands.
No plotting, notebook or application package is required.

## What each example proves

| Script | Composition and checked result |
|---|---|
| `constrained_poisson.py` | Matrix-free one-dimensional diffusion, nonzero Dirichlet lifting, diagonal inverse and CG. Checks the analytic quadratic, exact prescribed values and reconstructed free-row residual. |
| `singular_graph.py` | Two disconnected graph Laplacians, declared nullspaces, projected preconditioner, CG and component gauges. With `--project-incompatible`, checks that the selected equation converges while the requested incompatible equation does not. |
| `solver_capabilities.py` | Singular symmetric saddle-point MINRES with an SPD projected inverse, plus warm-started Richardson on a translation-nullspace Hessian. Checks original-action residuals and the selected gauge. |
| `vector_gmres.py` | A two-component periodic transport/reaction action with nonsymmetric local coupling and unequal left/right transport coefficients. Block Jacobi and left/right restarted GMRES recover a manufactured discrete solution. |
| `corrected_autodiff.py` | A 2D density → apparent volume/CRK → renormalization → gradient chain. Rebuilds corrections and propagates their tangents, then checks explicit JVP/reverse dot agreement and centered finite differences. `--no-crk` checks density → renormalization alone. |
| `reconstruct_field.py` | Cubic 2D MLS/LABFM derivatives against an analytic polynomial, then TENO/WENO one-sided states at a resolved 1D jump using the illustrated guide's fixture. Reports fallback counts and extrema; nonlinear weights do not guarantee boundedness for arbitrary input. |
| `mfm_rates.py` | Both Torch and Warp backends, both MFM/MFV modes: conservative pair rates and four tiny periodic drift/update/rebuild steps. Checks total conserved quantities, positive raw mass/internal energy and exactly unchanged MFM particle masses. |

Individual commands, after installation:

```bash
python scripts/examples/constrained_poisson.py --dtype float64
python scripts/examples/singular_graph.py --project-incompatible --dtype float64
python scripts/examples/solver_capabilities.py --dtype float64
python scripts/examples/vector_gmres.py --dtype float64 --side right
warpSPHCore_PRECISION=float64 python scripts/examples/corrected_autodiff.py
warpSPHCore_PRECISION=float64 python scripts/examples/reconstruct_field.py
warpSPHCore_PRECISION=float64 python scripts/examples/mfm_rates.py --mode MFV --backend warp
```

The algebra-only examples accept `--dtype` directly. SPH/reconstruction/MFM
types follow the process configuration. All examples accept `--device cpu`
or `--device cuda`.

## Numerical and lifecycle assumptions

The diffusion example uses a finite-difference action. The vector GMRES
example uses a discrete manufactured source; it checks composition, rather
than a PDE convergence order. A corrected SPH Laplacian need not be symmetric:
the production composition test in `test_capability_composition.py` uses
restriction/lifting and GMRES and compares the reduced action with a dense
reference.

The derivative example freezes neighbor membership and selected numerical
branches. Its density, volumes, field and corrections depend on the perturbed
geometry. Updating positions while reusing old correction tensors would
differentiate a different function. It does not differentiate topology,
cutoff crossings, arbitrary iterative solves or a complete simulation.

Fits retain geometry-dependent matrices; rebuild after changes to positions,
supports, volumes or interactions. Reconstructors need adjacency covering all
candidate radii; this example lets the constructor build it. Rank-deficient
fits and first-order fallback have explicit policies in
[Reconstruction](reconstruction.md).

The MFM/MFV example fixes supports, takes `dt=1e-4`, disables half-step
prediction (`dt=0` in the rate call), and uses an illustrative explicit Euler
update followed by drift and a geometry rebuild. This is a short lifecycle
and conservation check, not a convergence certificate for that time policy.
It checks internal energy directly because primitive pressure is floored.
The qualified MUSCL-Hancock/support policy remains in the existing
`higherOrderSPH/harness/mfm_driver.py`; reuse it through:

```bash
python scripts/diagnostics/qualify_mfm.py --suite wave --n 32 --device cpu \
  --precision float64 --backend warp --modes MFM MFV --closures project \
  --tend 0.001 --output /tmp/mfm-short.json
```

See [MFM/MFV](mfm.md) for closure, first-moment limitations, geometric rank
guards and the Warp backend's forward-only contract. These examples are
eager CPU/CUDA programs; they do not promise complete CUDA graph capture.

## Installed-artifact check

```bash
python -m build --wheel --outdir /tmp/warpsphcore-wheel
python scripts/packaging/check_wheel.py /tmp/warpsphcore-wheel/*.whl
# On a GPU host, also exercise the public examples on CUDA:
python scripts/packaging/check_wheel.py /tmp/warpsphcore-wheel/*.whl --device cuda
```

The checker unpacks the wheel and uses isolated interpreters (`python -I`).
It verifies the import path and executes these same example files in fresh
float32/float64 processes against the artifact. An editable source checkout
cannot satisfy a missing wheel module. This checks artifact contents with
the active runtime dependencies; it is not a clean-environment dependency
resolver test. The supported/qualified environment distinction and complete
gate results are recorded in [development records](development-artifacts.md).
