# warp: Python constant arithmetic in traced code is evaluated in float32

Observed with **warp 1.17.0** (Python 3.13, CPU backend — the defect is in
tracing, so the device is irrelevant). Found 2026-09-18 while validating
SPH kernel shapes to machine precision in a `float64` build: a B-spline
knot written as `wp.float64(5.0/7.0)` inside a traced kernel came out as
`float32(5/7)`, corrupting the shape by ~1.2e-7.

The section "Upstream issue text" below is ready to paste as a GitHub
issue against `NVIDIA-Omniverse/warp`.

## Minimal reproduction

Self-contained (only `warp` + `numpy`), CPU, ~10 s including compile:

```python
"""Python constant arithmetic inside traced code is evaluated in float32.

warp 1.17.0, Python 3.13, CPU.
Key contrast: cases (1) and (2) pass the *same Python double* to
wp.float64 -- in plain Python, 5.0/7.0 == 0.7142857142857143 exactly.
"""
import numpy as np
import warp as wp

wp.init()

# Plain Python: the binop and the literal are the *same* double.
K = 5.0 / 7.0
assert K == 0.7142857142857143


@wp.kernel
def probe(out: wp.array(dtype=wp.float64)):
    out[0] = wp.float64(5.0 / 7.0)              # (1) binop in traced code
    out[1] = wp.float64(0.7142857142857143)     # (2) literal, same double
    out[2] = wp.float64(K)                      # (3) module-level constant


o = wp.zeros(3, dtype=wp.float64, device="cpu")
wp.launch(probe, dim=1, outputs=[o], device="cpu")
v = o.numpy()

labels = (
    "(1) binop in traced func  wp.float64(5.0 / 7.0)",
    "(2) literal, same double  wp.float64(0.7142857142857143)",
    "(3) module-level Name     wp.float64(K),  K = 5.0 / 7.0",
)
f32 = float(np.float32(K))
for i, lab in enumerate(labels):
    print(f"{lab}")
    print(f"    warp = {v[i]!r}   exact = {v[i] == K}   "
          f"diff = {v[i] - K:+.3e}   == float32(5/7): {v[i] == f32}")
```

Verified output (warp 1.17.0):

```text
(1) binop in traced func  wp.float64(5.0 / 7.0)
    warp = np.float64(0.7142857313156128)   exact = False   diff = +1.703e-08   == float32(5/7): True
(2) literal, same double  wp.float64(0.7142857142857143)
    warp = np.float64(0.7142857142857143)   exact = True   diff = +0.000e+00   == float32(5/7): False
(3) module-level Name     wp.float64(K),  K = 5.0 / 7.0
    warp = np.float64(0.7142857142857143)   exact = True   diff = +0.000e+00   == float32(5/7): False
```

## Upstream issue text

**Title**

> Traced code: Python float arithmetic is evaluated in float32, silently degrading float64 constants

**Body**

> **warp version:** 1.17.0 (latest at time of writing)
> **Python:** 3.13 · **Device:** CPU (defect is at trace/compile time, device-agnostic)
>
> **Description**
>
> Inside a traced `@wp.kernel` / `@wp.func`, Python float arithmetic is
> traced as a warp operation typed with warp's default float (float32),
> even when the result is immediately cast to `wp.float64`. Float64
> kernels therefore silently contain float32-precision constants.
>
> Python float *literals* and values loaded from module-level names are
> unaffected (they pass through as exact double constants), so only
> arithmetic *expressions* are affected — which makes the discrepancy
> hard to spot, since the source visibly looks like exact double math.
>
> **Minimal reproduction** (CPU, ~10 s):
>
> ```python
> import numpy as np
> import warp as wp
>
> wp.init()
>
> # Plain Python: the binop and the literal are the *same* double.
> K = 5.0 / 7.0
> assert K == 0.7142857142857143
>
> @wp.kernel
> def probe(out: wp.array(dtype=wp.float64)):
>     out[0] = wp.float64(5.0 / 7.0)              # (1) binop in traced code
>     out[1] = wp.float64(0.7142857142857143)     # (2) literal, same double
>     out[2] = wp.float64(K)                      # (3) module-level constant
>
> o = wp.zeros(3, dtype=wp.float64, device="cpu")
> wp.launch(probe, dim=1, outputs=[o], device="cpu")
> for i, v in enumerate(o.numpy()):
>     print(i, repr(v), "exact:", v == K, "is float32(5/7):", v == float(np.float32(K)))
> ```
>
> Actual output:
>
> ```text
> 0 np.float64(0.7142857313156128) exact: False is float32(5/7): True
> 1 np.float64(0.7142857142857143) exact: True  is float32(5/7): False
> 2 np.float64(0.7142857142857143) exact: True  is float32(5/7): False
> ```
>
> **Expected behavior**
>
> `wp.float64(5.0 / 7.0)` in traced code produces the same constant as
> `wp.float64(0.7142857142857143)` — both are the identical Python
> double before tracing — i.e. `0.7142857142857143`.
>
> **Actual behavior**
>
> The binop form produces `0.7142857313156128`, which is exactly
> `float32(5/7)` re-cast to float64: the constant sub-expression is
> evaluated in float32 during tracing, and the cast to float64 then
> preserves the already-lost precision. Relative error ≈ 1.2e-8
> (float32 eps).
>
> **Impact**
>
> Any float64 kernel that writes non-dyadic constants as arithmetic
> (fractions, π-related expressions, …) silently gets float32-precision
> constants. In an SPH code we hit this on B-spline knot positions:
> `wp.float64(5.0 / 7.0)` as a knot produced a shape error of ~1.2e-7
> against the exact reference and broke a machine-precision validation
> battery. Expressions involving numpy scalars show the same class of
> behavior (e.g. `wp.float64(np.pi * 3.0 / 4.0)`; `4 * np.pi / 3` traced
> as int32 × float32).
>
> **Workarounds**
>
> - Compute the constant outside the traced function (module level) and
>   reference it by name: `K = 5.0/7.0; wp.float64(K)` → exact.
> - Write the full-precision literal directly:
>   `wp.float64(0.7142857142857143)` → exact. (The literal must carry
>   the digits you want — the point is that literal doubles pass through
>   unharmed, binops do not.)
>
> **Suggested fix**
>
> Constant sub-expressions made purely of Python floats should be folded
> at trace time with Python (double) semantics — i.e. to the same value
> the interpreter computed — rather than being typed as default-float
> warp operations. If that is not desired, a warning when a
> float32-typed constant operation feeds a float64 cast would at least
> make the precision loss visible.

## Notes for this repo

- Safe patterns in `src/`: module-level Python-float constants referenced
  by name (see `kernels/kernelFunctions/B7.py` knot constants) or
  full-precision literals. Inexact literals such as `wp.float64(0.6)`
  are fine (0.6's double passes through exactly).
- Remaining unfixed instance of the same class:
  `src/warpSPHCore/util/support.py:86` (`wp.pow(... , scalar_t(1.0/3.0))`
  and `scalar_t(np.pi * 3.0 /4.0)`) — tracked in the D&A 2012 replication
  findings log (`higherOrderSPH/.../PLAN.md`), Phase 8 decision.
- Related earlier instance (fixed 2026-09-18): `sphKernelN_H` in
  `kernels/properties.py` would not even compile under float64 for the
  same reason (bare `np.pi` constant binops + `pow(float64, int32)`).
