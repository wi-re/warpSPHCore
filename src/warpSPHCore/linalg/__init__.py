"""Linear algebra on the GPU, sync-free: small batched dense ops on `(n, d, d)` tensors (d <= 3) that torch only offers through
host-synchronising LAPACK / cuSOLVER calls (`torch.linalg.pinv` reads the host to check its info flag), as warp kernels, plus the `@wp.func`s the
same ops are built from so other kernels can call them inline; and (growing, see `docs/linalg/SOLVER_INVENTORY.md` in warpSPH) the iterative
solvers the SPH schemes share.

* `pinv.py` -- the symmetric pseudo-inverse (`pinvSymmetric`, `applyPinv`, and `symmetricPinv3` for use inside a kernel).
* `krylov/` -- Krylov solvers with device-resident stopping tests (`cg`, ...; `Status`, `SolveResult`).
* `stationary.py` -- relaxed-Jacobi style iterations with a device-resident stopping test (`runStationary`, `CheckSchedule`, `residualStatistic`).

The rule for what goes in: batched or device-resident, no host read on the hot path, forward-checked against the float64 torch result, with the
torch call as the fallback for inputs that need an autograd tape. `warpSPH.linalg` re-exports this package.
"""

from .pinv import applyPinv, loadPadded3x3, pinvSymmetric, symmetricPinv3

from .krylov import SolveResult, Status, bicg, bicgstab, cg
from .stationary import CheckSchedule, RESIDUAL_STATISTICS, StationaryResult, residualStatistic, runStationary

__all__ = ['applyPinv', 'pinvSymmetric', 'symmetricPinv3', 'loadPadded3x3',
           'bicg', 'bicgstab', 'cg', 'SolveResult', 'Status', 'CheckSchedule', 'RESIDUAL_STATISTICS', 'StationaryResult', 'residualStatistic', 'runStationary']
