# Native core closure: dense helpers, renormalization and solvers

The second ordered core stage removes the remaining Warp dependency from the
renormalization construction and linear-algebra package. It does not close the
polynomial, meshless, chunked-traversal or final derivative/release stages.

Module-owned dense, block-application and Krylov-update kernels use shared
CPU/CUDA/HIP numerical headers. The symmetric Jacobi inverse keeps the dense
helper's caller-selected cutoff; legacy renormalization retains its distinct
1D positive-only inverse, 2D low-count/absolute/relative policies and signed
eigenvalue diagnostics. General 3D renormalization retains the existing Torch
inverse policy. Dense operations accept the tensor dtype, not process precision.

Renormalization uses native covariance and the shared Torch scaffold. The
explicit geometry JVP preserves independent query/reference roles: an omitted
reference tangent means a fixed reference cloud. Low-count replacements have
zero tangent. Retained/discarded spectral rotation includes nonzero truncated
eigenvalues; cutoff crossings and repeated diagnostic eigenvalues are nonsmooth.
The 2D native correction supplies first-order forward/reverse AD. General dense
spectral AD retains the existing differentiable Torch rule, including higher
reverse derivatives on a fixed selected branch.

CG/BiCGStab retain one shared controller, ownership and result contract. Only
array adaptation, update dispatch, reductions and capture differ by backend.
Native reductions use Torch dot products; native update rows use the compiled
library. Convergence still verifies the original residual and reads status at
checkpoints. Different reduction orders can change recurrence rounding and
iteration counts. Workspaces remain inference-only; ordinary Torch solvers
retain their existing AD behavior. Other iterative methods are shared Torch
code and now import without Warp.

Native capture uses the warmed current setup stream without a device-wide
synchronization. It preserves cyclic-GC restoration, callback-failure recovery,
fixed buffers, bounded graph caches and same-address coefficient updates.
Block inverse application handles strided vectors and its transpose directly;
vector AD retains Torch. Strided external diagonal preconditioners are also
supported without a persistent copied coefficient buffer.

## Qualification

| Check | Result |
|---|---|
| Five focused native contracts | CPU: 4 passed, 1 GPU-only skip; CUDA: 5 passed |
| Existing workspace/preconditioner contracts exercised with native execution | CPU: 124 passed; CUDA: 154 passed initially, with the one capture-sync failure fixed and its contract passing on rerun |
| Existing Warp workspace/preconditioner regressions after shared-controller refactor | 124 passed, 31 skipped, 130 device cases deselected |
| Existing native CPU suite, excluding extended selection | 3612 passed, 470 skipped, 7169 deselected before the final strided-diagonal addition; the affected focused contract passes afterward |
| Python scaffolding | 23 passed |
| Dense/legacy renormalization parity | FP32 and separate FP64 CPU processes pass |
| Offline HIP | Dense inverse, Krylov updates and block application compile for gfx942 |
| Strict prebuilt CUDA loader | Five focused contracts pass |
| Warp import blocker | Public example runs neighbors, calibrated density, corrected gradient, renormalization and CG with Warp imports rejected |

The new maintained coverage consists of five native contracts and one bounded
parity contract, rotating dimensions/dtypes within each. Existing solver tests
were exercised externally against native execution rather than copied into a
second maintained test matrix. Raw logs, binaries and probe scripts are outside
core under the development evidence workspace. No timings or AMD runtime claims
are made.
