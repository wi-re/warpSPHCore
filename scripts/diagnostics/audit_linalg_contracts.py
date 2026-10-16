#!/usr/bin/env python3
"""Reproduce the 15 October 2026 linalg survey observations.

This reports behavior, including known defects; it is not a regression gate.
Run from the checkout with the development environment's Python. CUDA timing
and stream/capture qualification need a separate GPU run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
import warp as wp

from warpSPHCore import get_torch_precision
from warpSPHCore.linalg import (
    CheckSchedule, applyPinv, bicg, bicgstab, cg, pinvSymmetric, runStationary,
)


def survey():
    wp.init()
    observations = []

    def observe(name, expectation, call):
        row = {"case": name, "expectation": expectation}
        try:
            row["observed"] = call()
        except Exception as exc:
            row["observed"] = {"exception": type(exc).__name__, "message": str(exc)[:400]}
        observations.append(row)

    def solve_result(result, A, b):
        residual = float((b - A @ result.x).norm())
        return {
            "status": result.statusInt(),
            "finite": bool(result.x.isfinite().all()),
            "true_residual": residual if math.isfinite(residual) else None,
            "x": [v if math.isfinite(v) else None for v in result.x.tolist()],
        }

    I, b, z = torch.eye(2), torch.ones(2), torch.zeros(2)
    solvers = {
        "cg": lambda rhs, x0=None, **kw: cg(lambda v: I @ v, rhs, x0, **kw),
        "bicg": lambda rhs, x0=None, **kw: bicg(lambda v: I @ v, lambda v: I.T @ v, rhs, x0, **kw),
        "bicgstab": lambda rhs, x0=None, **kw: bicgstab(lambda v: I @ v, rhs, x0, **kw),
    }
    for name, solver in solvers.items():
        observe(name + "_zero_rhs_warm_start", "A converged result solves I x = 0.",
                lambda solver=solver: solve_result(solver(z, b), I, z))
        observe(name + "_zero_tolerance", "An exact zero residual can satisfy a zero tolerance.",
                lambda solver=solver: solve_result(solver(b, rtol=0, atol=0, maxiter=4), I, b))
        observe(name + "_zero_check_interval", "Validate the interval or define a no-host-check mode.",
                lambda solver=solver: solve_result(solver(b, checkEvery=0, maxiter=3), I, b))

    observe("bicg_scaled_preconditioner", "Success certifies the original system residual.",
            lambda: solve_result(solvers["bicg"](b, precond=torch.full((2,), 1e-12)), I, b))

    def recurrence_gap():
        g = torch.Generator().manual_seed(0)
        Q, _ = torch.linalg.qr(torch.randn(80, 80, dtype=torch.float64, generator=g))
        A = ((Q * torch.logspace(0, 5, 80, dtype=torch.float64)) @ Q.T).float()
        rhs = torch.randn(80, generator=g)
        result = cg(lambda v: A @ v, rhs, rtol=1e-5, maxiter=2000)
        return {
            "status": result.statusInt(), "requested_rtol": 1e-5,
            "true_relative_residual": float((rhs - A @ result.x).norm() / rhs.norm()),
            "last_recurrence_relative_residual": float(result.norms[result.valid][-1] / rhs.norm()),
        }
    observe("cg_recurrence_gap", "Success certifies the original system residual.", recurrence_gap)

    dtype = get_torch_precision()
    M = torch.eye(2, dtype=dtype)[None]
    observe("pinv_transposed_input", "A transposed batch of symmetric matrices is accepted or explicitly rejected.",
            lambda: {"shape": list(pinvSymmetric(M.repeat(3, 1, 1).transpose(1, 2)).shape)})
    other = torch.float64 if dtype == torch.float32 else torch.float32
    observe("pinv_other_dtype", "The process precision constraint is explicit at the public boundary.",
            lambda: {"dtype": str(pinvSymmetric(M.to(other)).dtype)})

    def tangent(fn, argument):
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(argument, torch.ones_like(argument))
            _, actual = torch.autograd.forward_ad.unpack_dual(fn(dual))
            return {"tangent": None if actual is None else actual.tolist()}
    observe("pinv_forward_ad", "At I, the tangent for an all-ones perturbation is minus ones.",
            lambda: tangent(pinvSymmetric, M))
    observe("apply_pinv_forward_ad", "For V @ pinv(I), the tangent with respect to V is preserved.",
            lambda: tangent(lambda V: applyPinv(V, M), torch.ones_like(M)))

    def nonmonotone():
        class Store:
            pass
        store = Store()
        store._jacobiIterationHistory = {"example": 8}
        schedule = CheckSchedule(store, "example", "adaptive", 0, 4)

        def step(x, i):
            return x + 1, torch.tensor(0.0 if i == 0 else 1.0), None

        result = runStationary(step, torch.zeros(1), maxIters=4, tolerance=0.5, schedule=schedule)
        return {"converged": result.converged, "iterations": result.iterations, "stats": result.stats,
                "remembered_first_convergence": store._jacobiIterationHistory["example"]}
    observe("stationary_nonmonotone_statistic", "Adaptive overshoot does not imply improved residual quality.", nonmonotone)

    root = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    return {
        "source_revision": revision, "python": sys.version.split()[0], "torch": torch.__version__,
        "tracked_source_changes": subprocess.run(["git", "diff", "--quiet"], cwd=root).returncode != 0,
        "warp": wp.__version__, "device": "cpu", "cuda_available": torch.cuda.is_available(),
        "process_precision": str(dtype), "torch_threads": torch.get_num_threads(),
        "observations": observations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    result = json.dumps(survey(), indent=2)
    if args.output:
        args.output.write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
