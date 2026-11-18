# Dehnen & Aly 2012 replication

*Improving convergence in smoothed particle hydrodynamics simulations without
pairing instability*, W. Dehnen and H. Aly, MNRAS 425(2), 1068–1082 (2012),
arXiv:1204.2471.

The static study and closeout are complete. The [report](REPORT.md) records
kernel identity, Fourier transforms, density estimates, linear stability and
sound speed, along with the frontend viscosity and conductivity work. The
earlier cubic long-wave discrepancy was traced to a phase reference using
raw wrapped coordinates. [Paper notes](paper_notes.md) map the notation and
figures, and `data/da2012_reference.yaml` stores reference values.

Use `scripts/` to reproduce the static figures and
[scripts/kernels](../../scripts/kernels/README.md) for the maintained library
audit. The [report's reproducibility section](REPORT.md#9-reproducibility)
lists the figure drivers and cached inputs. The lattice normalization in
`warpSPHCore` and the paper's equation 18 self-term correction are separate
operations.

The remaining dynamic comparisons are pairing relaxation (Figures 7–8),
the 3D Gresho–Chan vortex (Figures 9–10 and 13), 3D Sod with glass-like initial
conditions (Figure 11), and relative step cost (Figures 12–13). The
[report](REPORT.md#8-remaining-work--phase-7-dynamic-tests-figs-713) records
configurations and acceptance anchors. The frontend owns the simulation cases.
