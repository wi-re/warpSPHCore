# Kernel onboarding & audit

The machine half of "adding a kernel and checking that it works". A kernel
is **onboarded** when its spec in `kernel_specs.yaml` passes the audit
battery (`kernel_audit.py`), which is CI-gated by
`tests/kernels/test_kernel_audit.py`.

## Files

| file | role |
|---|---|
| `kernel_specs.yaml` | machine-readable spec per kernel: source citation, dimensionless shape `f(q)` (q = r/H), `C` per dim, `sigma2/H2` per dim (D&A eq.-8 convention), code `packingRatio` + reference `N_H`, `ft_expected`, documented `known_issues` |
| `kernel_audit.py` | the audit battery; evaluates the SHIPPED `eval_*` warp functions on CPU (float64) — never a re-transcription |
| `tests/kernels/test_kernel_audit.py` | CI wrapper: runs the audit in a float64 subprocess, checks the verdict, and runs a broken-spec canary |

## The checks (per spec'd dimension)

1. **form** — code `eval_k` vs the spec shape on a 4097-point grid plus
   outside the support.
2. **normalisation** — code `eval_C_d` vs the spec `C` vs the independent
   numerical `1/(V_ν ∫₀¹ f q^{ν−1} dq)`.
3. **moments** — `σ²/H² = C·V_ν/ν · ∫₀¹ f q^{ν+1} dq` (the **1/ν of D&A
   eq. 8 is real**; it is what makes `H/h = 1/(2√(σ²/H²))` agree with the
   tabulated values) vs the spec; code `kernelScale` vs
   `1/(2√(σ²/H²))` (6-decimal rounding allowed).
4. **derivatives** — code `_dkdq`/`_d2kdq2`/`_d3kdq3` vs `wp.Tape`
   automatic differentiation of the level below (same pattern as
   `scripts/gradcheck/kernel_sanity_native.py` Section C; the AD grid
   avoids the positive-part kinks where the subgradient is not the
   derivative).
5. **support** — exactly zero for q > 1; `f(1) = f′(1) = 0`.
6. **FT sign** (`--ft`, opt-in, ~15 s) — 3D FT per D&A eq. (14):
   `w̄(0) = 1`, the eq. (17) Taylor `w̄ = 1 − ½σ²k²` at small k, and
   non-negativity / first zero per the spec's `ft_expected`
   (`nonnegative` = pairing-stable in D&A's sense, `negative_lobe` =
   pairing-unstable).
7. **packing** — code `packingRatio` == spec value; `sphKernelN_H` at that
   packing reproduces the reference `N_H` (3-decimal packing rounding
   allowed); `sphKernel_xi == packing·scale == H·d_nn`.

## Run

```bash
python scripts/kernels/kernel_audit.py            # all kernels
python scripts/kernels/kernel_audit.py --ft       # + Fourier-sign check
python scripts/kernels/kernel_audit.py --only b7  # one kernel
pytest tests/kernels/                             # the CI gate
```

Exit code 0 iff no hard failures. `known_issues` entries in the spec
downgrade a matching failure to `KNOWN` (still reported, never silent);
a stale known issue (one that no longer reproduces) raises a `WARN` so
the spec gets cleaned up when the code is fixed.

## Onboarding a new kernel (checklist)

1. **Implement** the kernel in `src/warpSPHCore/kernels/` following the
   existing family files: `kernelFunctions/<name>.py` (`<name>_k`,
   `_dkdq`, `_d2kdq2`, `_d3kdq3`, `_C_d`, `_kernelScale`,
   `_packingRatio`) + a branch in each of the seven dispatch functions in
   `kernels/eval_kernel.py` + an enum member in `enumTypes.py`.
   Derivatives from the shape, `C_d` from normalisation, `kernelScale`
   from the moments (`H/h = 1/(2√(σ²/H²))`, eq.-8 convention),
   `packingRatio` from the intended `N_H`
   (`(V_ν N_H / scale^ν)^{1/ν}` at the reference `N_H`).
2. **Cite the source**: paper + table/eq. for the shape, `C`, `σ²`, and
   the reference `N_H`. If a quantity is not published anywhere, compute
   it from the shape and say so in the spec's `source` note (the B7
   entry is the model).
3. **Add the spec** to `kernel_specs.yaml` (shape expr in `q` with
   `pos(x) = (x₊)`; `shape_by_dim` when the shape differs per dimension,
   as for the Wendland family).
4. **Run the audit**; every check must PASS, or the deviation must be a
   documented `known_issues` entry with a reason (never silent).
5. **Run the CI tests** (`pytest tests/kernels/`) and the full suite.
6. **Commit** kernel + dispatch + spec together, so the audit of the
   committed code is the audit that passed.

## Current status (2026-09-18)

- cubic, quartic, quintic, wendland2/4/6: **all checks pass** — the
  shipped kernels are exactly the D&A 2012 Table 1 kernels (form to
  machine precision, C and σ² exact, derivatives AD-verified,
  pairing behaviour matches the paper: cubic/quartic/quintic have
  negative FT lobes with first zeros at κ̂ ≈ 12.57 / 22.46 / 18.87,
  Wendland C²/C⁴/C⁶ non-negative).
- b7, b8 (split 2026-09-18): the former code B7 shape was the D&A 2012
  eq.-11 family member b₈ (order 8, degree 7 — the old name "B7" is by
  degree, D&A index by order); it is now shipped as **B8** (enum 33,
  keeping the old shape's value so stored configs still resolve) with
  the shape-derived kernelScale 2.449/2.481/2.513 (σ²/H² = 1/24,
  531453/13085600, 19/480 — 1D exactly on the family pattern
  σ²/H² = 1/(3n); the previous scale copied the quintic row and is the
  only KNOWN deviation that was ever in the audit). The **genuine
  classical b₇** (order 7, degree 6, 4 terms, knots 1/7, 3/7, 5/7) was
  derived from the same family formula and added as **B7** (enum 34)
  with shape-derived C (823543/92160, 5764801/113149π,
  5764801/61440π) and scale 2.291/2.325/2.360 (σ²/H² = 1/21,
  7691281/166329030, 11/245). Both pass every check: b8's 3D FT is
  non-negative (pairing-stable); b7's 3D FT has a barely-negative lobe
  (min −3.2e-6, first zero κ̂ ≈ 21.96 — pairing-unstable in the paper's
  sense, as expected for a non-C² kernel). The family formula
  reproduces the shipped cubic/quartic/quintic exactly. Neither is in
  D&A Table 1.
- **Warp constant gotcha (found during the b7 derivation)**: a
  Python-float binop inside a traced `@wp.func` (e.g.
  `scalar_t(5.0/7.0)`) is evaluated in float32 by the tracer, silently
  corrupting the constant in float64 builds (~1.2e-7 shape error). Use
  full-precision float literals or module-level Python-float constants
  referenced by name (the safe pattern, verified). Affects any
  non-exactly-representable knot/constant; inexact literals like
  `scalar_t(0.6)` are fine. See `kernelFunctions/B7.py` header.
