#!/usr/bin/env python3
"""Operator probes for the convergence harness.

A thin adapter over the shipped `warpOperation` dispatch so every later
phase of the higher-order plan can probe its operators through one common
interface (the parent plan's Phase 0 deliverable). Three correction modes
are exposed:

* **standard** -- no correction state (uncorrected SPH operators);
* **crk** -- CRKSPH: `computeCRKFactors` state, with the CRK apparent
  volume passed as query/reference volumes (the validated usage in the
  `warpSPH` frontend's `schemes/crkSPH.py`);
* **renorm** -- covariance-matrix renormalization:
  `computeRenormalizationMatrices` (its eigenvalues are the condition-number
  source; see `conditioning.py`).

Probe operations: Interpolate, Gradient (Difference scheme -- the one the
CRK frontend uses), Laplacian (Brookshaw scheme, scalar fields only).
Correction states are computed once per case and cached (they do not depend
on the probed field).
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

__all__ = ["MODES", "PROBES", "CorrectionCache", "run_probe"]

MODES = ("standard", "crk", "renorm")
PROBES = ("interpolate", "gradient", "laplacian")

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


def run_probe(case: Case, cache: CorrectionCache,
              values: torch.Tensor, probe: str, mode: str) -> torch.Tensor:
    """Run one operator probe on `values` (the field at the particles) and
    return the operator output (torch tensor on the case device)."""
    if probe not in _OP_BY_PROBE:
        raise ValueError(f"unknown probe {probe!r}; expected one of {PROBES}")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

    properties = OperationProperties(
        kernel=case.kernel,
        operation=_OP_BY_PROBE[probe],
        supportMode=SupportScheme.Gather,
        operationMode=OperationDirection.AllToAll,
        gradientMode=GradientScheme.Difference,
        laplacianMode=LaplacianScheme.Brookshaw,
    )
    kwargs: dict = {}
    if mode == "crk":
        apparent_volume, crk_state = cache.crk()
        kwargs["crkState"] = crk_state
        kwargs["queryVolumes"] = apparent_volume
        kwargs["referenceVolumes"] = apparent_volume
    elif mode == "renorm":
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
