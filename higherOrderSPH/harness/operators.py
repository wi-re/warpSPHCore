#!/usr/bin/env python3
"""Operator probes for the convergence harness.

A thin adapter over the shipped `warpOperation` dispatch so every later
phase of the higher-order plan can probe its operators through one common
interface (the parent plan's Phase 0 deliverable). Four correction modes
are exposed:

* **standard** -- no correction state (uncorrected SPH operators);
* **crk** -- CRKSPH: `computeCRKFactors` state, with the CRK apparent
  volume passed as query/reference volumes (the validated usage in the
  `warpSPH` frontend's `schemes/crkSPH.py`);
* **renorm** -- covariance-matrix renormalization (Bonet--Lok-style
  corrected *gradient*): `computeRenormalizationMatrices` (its eigenvalues
  are the condition-number source; see `conditioning.py`). The value is
  left uncorrected (= standard) -- the parent plan's Phase 3 leaves the
  value-correction decision open.
* **renormVal** -- the full Phase 3 operator: the `renorm` corrected
  *gradient* plus a Randles--Libersky *value* renormalization,
  `fhat/S` where `S = Interpolate(ones)` is the (field-independent) 0th
  kernel moment. This makes constant reproduction exact (0th-order value)
  without a new `src/` operator -- it composes the shipped `Interpolate`
  twice. The Laplacian is left uncorrected (no Bonet--Lok Laplacian).

Probe operations: Interpolate, Gradient (Difference scheme -- the one the
CRK frontend uses), Laplacian (Brookshaw scheme, scalar fields only).
Correction states (and the kernel sum) are computed once per case and
cached (they do not depend on the probed field).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from warpSPHCore import OperationProperties, warpOperation
from warpSPHCore.crk import computeCRKFactors
from warpSPHCore.enumTypes import (
    GradientScheme,
    LaplacianScheme,
    OperationDirection,
    SupportScheme,
    WarpOperation,
)
from warpSPHCore.renorm import computeRenormalizationMatrices

from particle_sets import Case

__all__ = ["MODES", "PROBES", "CorrectionCache", "run_probe",
           "register_mode", "EXTERNAL_MODES"]

MODES = ("standard", "crk", "renorm", "renormVal")
# `hessian` is grad-of-grad through the mode's own gradient (the core has no
# Hessian operator); it is the baseline column a p >= 2 operator (Phase 4
# MLS/RKPM, Phase 5 LABFM) is compared against.
PROBES = ("interpolate", "gradient", "laplacian", "hessian")

# Operators from later phases plug in here instead of editing the dispatch:
# name -> fn(case, cache, values, probe) returning the operator output in the
# same layout as `run_probe` (interpolate (N,)/(N,D), gradient (N,dim) /
# (N,D,dim), laplacian (N,), hessian (N,dim,dim)), or None when that mode
# does not provide the probe (the caller skips the row). Registering is how
# a new operator's OWN Laplacian/Hessian is measured, rather than Brookshaw /
# grad-of-grad via `warpOperation`.
EXTERNAL_MODES: dict = {}


def register_mode(name: str, fn) -> None:
    """Register an external operator mode (see `EXTERNAL_MODES`). Built-in
    mode names cannot be shadowed."""
    if name in MODES:
        raise ValueError(f"{name!r} is a built-in mode")
    EXTERNAL_MODES[name] = fn

_OP_BY_PROBE = {
    "interpolate": WarpOperation.Interpolate,
    "gradient": WarpOperation.Gradient,
    "laplacian": WarpOperation.Laplacian,
}


@dataclass
class CorrectionCache:
    """Lazily computed correction states for one case (field-independent)."""
    case: Case
    crk_apparent_volume: torch.Tensor | None = field(default=None, repr=False)
    crk_state: object | None = field(default=None, repr=False)
    renorm_C: torch.Tensor | None = field(default=None, repr=False)
    renorm_eigvals: torch.Tensor | None = field(default=None, repr=False)
    renorm_state: object | None = field(default=None, repr=False)
    kernel_sum: torch.Tensor | None = field(default=None, repr=False)

    def kernelSum(self) -> torch.Tensor:
        """The 0th kernel moment S_i = sum_j V_j W_ij (field-independent).

        Computed as a standard (uncorrected) Interpolate of the constant
        field 1 -- the same operator the `renormVal` value renormalization
        divides by, so the two are consistent by construction.
        """
        if self.kernel_sum is None:
            pos = self.case.particles.positions
            ones = torch.ones(pos.shape[0], device=pos.device, dtype=pos.dtype)
            self.kernel_sum = warpOperation(
                self.case.particles,
                OperationProperties(
                    kernel=self.case.kernel,
                    operation=WarpOperation.Interpolate,
                    supportMode=SupportScheme.Gather,
                    operationMode=OperationDirection.AllToAll,
                ),
                self.case.domain,
                adjacency=self.case.adjacency,
                queryValues=ones,
                referenceValues=ones,
            )
        return self.kernel_sum

    def crk(self):
        if self.crk_state is None:
            apparent_volume, _crk_density, crk_state = computeCRKFactors(
                queryParticles=self.case.particles,
                domain=self.case.domain,
                kernel=self.case.kernel,
                operationMode=OperationDirection.AllToAll,
                adjacency=self.case.adjacency,
            )
            self.crk_apparent_volume = apparent_volume
            self.crk_state = crk_state
        return self.crk_apparent_volume, self.crk_state

    def renorm(self):
        if self.renorm_state is None:
            C, eigvals, state = computeRenormalizationMatrices(
                queryParticles=self.case.particles,
                operationProperties=OperationProperties(
                    kernel=self.case.kernel,
                    operation=WarpOperation.Gradient,
                    supportMode=SupportScheme.Gather,
                    operationMode=OperationDirection.AllToAll,
                    gradientMode=GradientScheme.Difference,
                ),
                domain=self.case.domain,
                adjacency=self.case.adjacency,
                returnEigVals=True,
            )
            self.renorm_C = C
            self.renorm_eigvals = eigvals
            self.renorm_state = state
        return self.renorm_C, self.renorm_eigvals, self.renorm_state


def _renormValue(raw: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Randles--Libersky value renormalization ``raw / S``, broadcasting the
    per-particle scalar 0th moment ``S`` over any trailing field-component
    axes (scalar field ``(N,)`` or vector field ``(N, D)``)."""
    if raw.dim() > 1:
        S = S.reshape(-1, *([1] * (raw.dim() - 1)))
    return raw / S


def run_probe(case: Case, cache: CorrectionCache,
              values: torch.Tensor, probe: str, mode: str) -> torch.Tensor:
    """Run one operator probe on `values` (the field at the particles) and
    return the operator output (torch tensor on the case device)."""
    if probe not in PROBES:
        raise ValueError(f"unknown probe {probe!r}; expected one of {PROBES}")
    if mode in EXTERNAL_MODES:
        return EXTERNAL_MODES[mode](case, cache, values, probe)
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES} "
                         f"or a registered external mode")
    if probe == "hessian":
        # grad-of-grad: the mode's gradient applied to its own gradient
        # (a vector field), giving out[:, a, b] = d/dx_b (d f / dx_a).
        g = run_probe(case, cache, values, "gradient", mode)
        return run_probe(case, cache, g.contiguous(), "gradient", mode)

    properties = OperationProperties(
        kernel=case.kernel,
        operation=_OP_BY_PROBE[probe],
        supportMode=SupportScheme.Gather,
        operationMode=OperationDirection.AllToAll,
        gradientMode=GradientScheme.Difference,
        laplacianMode=LaplacianScheme.Brookshaw,
    )
    # renormVal's value probe is a derived quantity (uncorrected interpolant
    # divided by the 0th kernel moment), so it returns before the common
    # dispatch. Its gradient probe uses the corrected L matrix below; its
    # laplacian falls through uncorrected (there is no Bonet--Lok Laplacian).
    if mode == "renormVal" and probe == "interpolate":
        raw = warpOperation(
            case.particles,
            properties,
            case.domain,
            adjacency=case.adjacency,
            queryValues=values,
            referenceValues=values,
        )
        return _renormValue(raw, cache.kernelSum())

    kwargs: dict = {}
    if mode == "crk":
        apparent_volume, crk_state = cache.crk()
        kwargs["crkState"] = crk_state
        kwargs["queryVolumes"] = apparent_volume
        kwargs["referenceVolumes"] = apparent_volume
    elif mode in ("renorm", "renormVal"):
        _C, _eigvals, state = cache.renorm()
        kwargs["renormalizationState"] = state

    return warpOperation(
        case.particles,
        properties,
        case.domain,
        adjacency=case.adjacency,
        queryValues=values,
        referenceValues=values,
        **kwargs,
    )
