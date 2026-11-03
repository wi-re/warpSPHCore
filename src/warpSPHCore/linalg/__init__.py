"""Shared linear algebra: sync-free small batched dense ops and checkpointed iterative solvers.

Small dense ops on `(n, d, d)` tensors (d <= 3) that torch only offers through
host-synchronising LAPACK / cuSOLVER calls (`torch.linalg.pinv` reads the host to check its info flag), with backend-owned kernels; and (growing, see `docs/linalg/SOLVER_INVENTORY.md` in warpSPH) the iterative
solvers the SPH schemes share.

* `pinv.py` -- the symmetric pseudo-inverse (`pinvSymmetric`, `applyPinv`, and `symmetricPinv3` for use inside a kernel).
* `krylov/` -- Krylov solvers with device-resident stopping tests (`cg`, ...; `Status`, `SolveResult`); optional `CGWorkspace`/`BiCGStabWorkspace` native updates.
* `stationary.py` -- relaxed-Jacobi style iterations with a device-resident stopping test (`runStationary`, `CheckSchedule`, `residualStatistic`).

In the native backend, dense kernels follow the tensor dtype and launch on
Torch's current stream; spectral AD uses the shared Torch derivative rule.
The Warp backend retains process precision and its inline `@wp.func` helpers. Krylov solvers follow
the tensor dtype, verify the original residual and perform host reads at
checkpoints and verification. `warpSPH.linalg` re-exports this package.
"""

from .pinv import applyPinv, loadPadded3x3, pinvSymmetric, symmetricPinv3

from .krylov import BiCGStabWorkspace, CGWorkspace, SolveResult, Status, bicg, bicgstab, cg, gmres, minres, richardson
from .operator import LinearOperator
from .preconditioner import BlockJacobiPreconditioner, DiagonalPreconditioner
from .restriction import RestrictedSystem
from .nullspace import NullspaceResiduals, NullspaceRhs, NullspaceSystem
from .stationary import CheckSchedule, RESIDUAL_STATISTICS, StationaryResult, residualStatistic, runStationary

__all__ = ['applyPinv', 'pinvSymmetric', 'symmetricPinv3', 'loadPadded3x3',
           'bicg', 'bicgstab', 'cg', 'gmres', 'minres', 'richardson', 'LinearOperator', 'RestrictedSystem', 'NullspaceSystem', 'NullspaceRhs', 'NullspaceResiduals', 'DiagonalPreconditioner', 'BlockJacobiPreconditioner', 'CGWorkspace', 'BiCGStabWorkspace', 'SolveResult', 'Status', 'CheckSchedule', 'RESIDUAL_STATISTICS', 'StationaryResult', 'residualStatistic', 'runStationary']
