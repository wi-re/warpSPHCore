# Higher-order SPH research

- [Convergence harness](harness/README.md) covers static operator accuracy,
  reference reconstruction and PDE comparisons. Its frozen SPH, CRKSPH and
  Bonet–Lok baselines are the comparison points for future runs.
- [Dehnen & Aly 2012 replication](dehnen2012_convergence-without-pairing-instability/README.md)
  covers kernel identity, density and linear stability. Pairing relaxation,
  3D Gresho and Sod, and relative step cost remain open.

The [reconstruction documentation](../docs/docs/reconstruction.md) covers the
production MLS/LABFM and TENO/WENO paths and their derivative contracts. Core
qualification includes cubic MLS/LABFM frozen advection. The maintained
[MFM/MFV documentation](../docs/docs/mfm.md) covers the Core kernels, and the
frontend has a registered meshless step.

Remaining research includes matched frozen diffusion and full-scheme
comparisons, LABFM boundary/disorder and hyperviscosity behavior, reconstruction
inside Riemann-SPH stages, and MFM/MFV face consistency and disorder. Keep
matched inputs, reference methods and evidence with each comparison. The
former study plan is available at `dcb14a7:higher_order.md`.

Use the installed kernels and the [kernel audit](../scripts/kernels/README.md)
when checking a reference paper. Record discrepancies and the exact source
revision before changing kernel data. Save new results under the shared
[development workspace](../docs/docs/development-artifacts.md).
