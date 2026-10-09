#!/usr/bin/env python3
"""Harness adapter: registers the order-p MLS/RKPM operator (`rkpm.py`) as
the external modes ``rkpm1`` / ``rkpm2`` / ``rkpm3`` via
`operators.register_mode`, so the unchanged Phase 0 probes measure it next
to ``standard`` / ``crk`` / ``renorm`` / ``renormVal``.

The moment system is built once per case (cached on the `CorrectionCache`)
and reused for every field. Volumes are ``m / rho`` from the case's
particle state -- any positive weights reproduce polynomials exactly, this
is the SPH-natural choice and what `computeCRKFactors` starts from.
Rank-deficient rows (fewer neighbours than basis functions, or cond above
the cutoff) are *not* masked here: they surface as large errors in the
boundary-band rows, which is the information Phase 4's conditioning study
wants; the per-row flag is available as ``system.deficient``.
"""

from __future__ import annotations

import torch

from operators import register_mode
from rkpm import RKPMOperator, build_system

__all__ = ["rkpm_mode_name", "labfm_mode_name", "register_rkpm_modes",
           "register_labfm_modes", "get_operator"]


def rkpm_mode_name(order: int) -> str:
    return f"rkpm{order}"


def labfm_mode_name(order: int) -> str:
    return f"labfm{order}"


def get_operator(case, cache, order: int, labfm: bool = False) -> RKPMOperator:
    store = cache.__dict__.setdefault("_rkpm", {})
    if (order, labfm) not in store:
        parts = case.particles
        volumes = (parts.masses / parts.densities).to(case.positions.dtype)
        box = (torch.as_tensor(case.box, dtype=case.positions.dtype,
                               device=case.positions.device)
               if case.periodic else None)
        system = build_system(case.positions, volumes, case.h, order,
                              kernel=case.kernel.name.lower(), box=box,
                              constant=not labfm)
        store[(order, labfm)] = RKPMOperator(system)
    return store[(order, labfm)]


def _make(order: int, labfm: bool = False):
    def fn(case, cache, values, probe):
        op = get_operator(case, cache, order, labfm)
        if probe == "interpolate":
            return op.values(values)
        if probe == "gradient":
            return op.gradient(values)
        if probe == "laplacian":
            return op.laplacian(values)
        if probe == "hessian":
            return op.hessian(values)
        raise ValueError(probe)
    return fn


def register_rkpm_modes(orders=(1, 2, 3)) -> tuple[str, ...]:
    from operators import EXTERNAL_MODES
    names = []
    for p in orders:
        name = rkpm_mode_name(p)
        if name not in EXTERNAL_MODES:
            register_mode(name, _make(p))
        names.append(name)
    return tuple(names)


def register_labfm_modes(orders=(2, 4, 6, 8)) -> tuple[str, ...]:
    """LABFM variant (`build_system(constant=False)`): modes ``labfm<k>``
    with k the paper's order = the polynomial degree."""
    from operators import EXTERNAL_MODES
    names = []
    for p in orders:
        name = labfm_mode_name(p)
        if name not in EXTERNAL_MODES:
            register_mode(name, _make(p, labfm=True))
        names.append(name)
    return tuple(names)
