#!/usr/bin/env python3
"""Independent numerical and planner checks, never called by interpolation."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import torch
from runtime import Library, build_plan, call


@dataclass
class State:
    positions: torch.Tensor
    supports: torch.Tensor
    masses: torch.Tensor
    kinds: torch.Tensor
    densities: torch.Tensor


def dense(state, values):
    x = state.positions.double()
    dx = x[:, None, :] - x[None, :, :]
    dx -= torch.floor(dx + .5)
    distance = dx.norm(dim=-1)
    dimension = x.shape[1]
    q = distance / state.supports.double()[:, None]
    t = (1 - q).clamp_min(0)
    coefficient = [1.25, 2.228169203286535, 3.3422538049298023][dimension - 1]
    weight = coefficient * t.pow(3 if dimension == 1 else 4)
    weight *= 1 + (3 if dimension == 1 else 4) * q
    weight /= state.supports.double()[:, None].pow(dimension)
    field = values.double() * (state.masses.double() / state.densities.double())[:, None]
    return weight @ field


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    library = Library(args.library)
    results = []
    for dtype in [torch.float32, torch.float64]:
        for dimension in [1, 2, 3]:
            for max_support in [.140001, .310001]:
                generator = torch.Generator().manual_seed(5521 + dimension)
                # Exercise Morton bits beyond those used by this tiny geometry.
                key_x = torch.rand((17, dimension), generator=generator).to(device="cuda", dtype=dtype)
                grid = 2**20 + 17
                coordinate = (key_x.cpu() * grid).long()
                key_expected = torch.zeros(17, dtype=torch.int64)
                for bit in range(21):
                    for d in range(dimension):
                        key_expected |= ((coordinate[:,d] >> bit) & 1) << (dimension*bit+d)
                torch.testing.assert_close(library.keys(key_x,grid).cpu(),key_expected,rtol=0,atol=0)
                n = 257
                x = torch.rand((n, dimension), generator=generator, dtype=torch.float64)
                # Overfull cells require multiple bounded query tasks and reference
                # tiles; empty cells, periodic boundaries and partial warps coexist.
                x[:133] = .47 + .02 * x[:133]
                x[133:140, 0] = torch.linspace(0, .999999, 7)
                x = x.to(device="cuda", dtype=dtype)
                base, span = (.11, .03) if max_support < .2 else (.27, .04)
                h = base + span * torch.rand(n, generator=generator).to(device="cuda", dtype=dtype)
                state = State(x, h, h.new_full((n,), 1/n),
                              torch.zeros(n, device="cuda", dtype=torch.int32), h.new_ones(n))
                values = torch.randn((n, dimension), generator=generator).to(device="cuda", dtype=dtype)
                plan = build_plan(library, state, values, max_support)
                expected = dense(state, values)[plan.order]
                # Independent CPU Morton encoding checks all bit positions present
                # in this fixture; the physical cell table must match exactly.
                coordinates = (x.cpu().double() * plan.grid).long()
                keys = torch.zeros(n, dtype=torch.int64)
                for bit in range(21):
                    for d in range(dimension):
                        keys |= ((coordinates[:, d] >> bit) & 1) << (dimension * bit + d)
                torch.testing.assert_close(keys[plan.order.cpu()],
                    torch.repeat_interleave(plan.keys.cpu(), plan.counts.cpu()), rtol=0, atol=0)
                assert int(plan.counts.max()) > 128
                for block in [32, 128, 256]:
                    for width in [1, 4, 8]:
                        tasks = plan.tasks(block // width)
                        covered = torch.cat([torch.arange(a, b) for a,b in zip(tasks.begin.cpu(),tasks.end.cpu())])
                        torch.testing.assert_close(covered, torch.arange(n), rtol=0, atol=0)
                        for mode in [0, 1, 3, 4, 5, 6, 7, 8]:
                            actual = call(library, plan, tasks, mode, block, width).double()
                            tolerance = 3e-5 if dtype == torch.float32 else 3e-10
                            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                            results.append(dict(dtype=str(dtype), dimension=dimension, grid=plan.grid, max_support=max_support, block=block,
                                width=width, mode=mode, max_abs=float((actual-expected).abs().max())))
    # The preparation/scratch path must also obey the current nondefault stream.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = call(library, plan, tasks, 6, 256, 8)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(actual.double(), expected, rtol=3e-10, atol=3e-10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(passed=len(results), stream_passed=True, records=results), indent=2)+"\n")
    print(f"{len(results)} numerical/planner configurations and nondefault stream passed")


if __name__ == "__main__":
    main()
