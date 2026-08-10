#!/usr/bin/env python3
"""Find graded output values a model could hardcode, per task and per element.

Two questions, and the second is the one that has actually caught something.

**Whole output.** Are all six expected outputs of a task identical? Then one
constant answer passes every hidden case. This has always reported zero.

**Per element.** Does some *position* in the graded output take the same value on
all six cases, while the rest vary? Then that position is free: a model can
hardcode it and be graded only on the others. The whole-output check cannot see
this, and reported zero throughout.

That is not hypothetical. The 0.4.x `linsolve_tolerance` level 3 graded
``[x; norm(A*x-b)]`` while the generator drew ``b = A @ x0`` -- so ``b`` lay in
the range of ``A`` by construction and the residual was **1.33e-14 against a
1e-7 tolerance**, i.e. a constant zero, on every case for weeks. Every validator
stayed green, because the reference and the naive solution both compute the
residual correctly and both get zero. Nothing asked whether a graded position
varies. See PIPELINE_LOG, 2026-08-10.

**Constant within tolerance, not bit-identical.** The residual above was never
exactly 0.0; it was a different tiny number each time. A test for exact equality
would have missed it, which is precisely how it survived. Values here count as
the same when they agree inside the task's own comparison rule -- the identical
``abs(a - b) <= tol * max(1, abs(b))`` the grader applies.

A constant element is not automatically a defect: a task may legitimately have a
position that is always zero, and one *element* of a six-element answer being
free is a much smaller problem than a whole answer being free. What this script
produces is a list to read, ranked by how much of the output is free.

Usage:
    uv run python scripts/audit_constant_outputs.py
    uv run python scripts/audit_constant_outputs.py --num-tasks 500 --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "environments" / "octave_rl"))

from generators import build_tasks
from harness import _octave_flatten

# Report a task when at least this share of its graded positions never vary.
# Not a threshold with a theory behind it: a screen, set low enough that one
# free element of a six-element answer still surfaces.
FREE_SHARE_TO_REPORT = 0.05


def canonical(value: object) -> str:
    return json.dumps(value, allow_nan=True, separators=(",", ":"), sort_keys=True)


def _same(left: float, right: float, tolerance: float) -> bool:
    """The grader's own comparison, so 'constant' means what scoring means."""
    if math.isnan(left) or math.isnan(right):
        return math.isnan(left) and math.isnan(right)
    return abs(left - right) <= tolerance * max(1.0, abs(right))


def constant_positions(task: dict[str, Any]) -> tuple[int, int]:
    """(positions that never vary, positions compared) for one task.

    Positions are taken in the flattened, column-major order the grader uses,
    and anchored from **both ends**.

    Anchoring matters more than it looks. Most families here return a result
    whose length follows the input, so the six cases rarely share a length, and
    comparing only equal-length outputs skips almost the whole pool -- an
    earlier version of this script compared 148 positions across 1,500 tasks
    and was therefore incapable of finding anything. Anchoring only from the
    left is no better for the case that motivated the script: the 0.4.x
    `linsolve_tolerance` free element was the residual **appended after** a
    solution vector of varying length, so it sits at a fixed offset from the
    end and at no fixed offset from the start.

    A position is counted once even when both anchors reach it, so a task whose
    outputs all have the same length is not double-counted.
    """
    info = task["info"]
    tolerance = info.get("tolerance", 1e-9)
    flattened = []
    for case in info["cases"]:
        try:
            flattened.append([float(v) for v in _octave_flatten(case["expected"])])
        except (TypeError, ValueError):
            return (0, 0)  # non-numeric output; nothing to compare
    if not flattened:
        return (0, 0)
    shortest = min(len(row) for row in flattened)
    if shortest == 0:
        return (0, 0)
    longest = max(len(row) for row in flattened)

    def frozen_at(index: int) -> bool:
        first = flattened[0][index]
        return all(_same(row[index], first, tolerance) for row in flattened)

    # Offsets from the left over the shortest output, then -- only when the
    # lengths actually differ -- the same count of offsets from the right. When
    # every output has the same length the two passes would cover identical
    # positions, so the second is skipped rather than double-counted.
    frozen = sum(frozen_at(index) for index in range(shortest))
    compared = shortest
    if longest != shortest:
        def frozen_from_end(offset: int) -> bool:
            first = flattened[0][len(flattened[0]) - 1 - offset]
            return all(
                _same(row[len(row) - 1 - offset], first, tolerance)
                for row in flattened
            )

        frozen += sum(frozen_from_end(offset) for offset in range(shortest))
        compared += shortest
    return (frozen, compared)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tasks", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "artifacts" / "constant_output_audit.json",
    )
    args = parser.parse_args()

    levels: dict[str, Any] = {}
    flagged: list[dict[str, Any]] = []
    for level in (1, 2, 3):
        tasks = build_tasks(level, args.num_tasks, args.seed, False, True)
        by_cell: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"tasks": 0, "whole_constant": 0, "frozen": 0, "positions": 0}
        )
        for task in tasks:
            info = task["info"]
            cell = (info["family"], info.get("variant", ""))
            outputs = {canonical(case["expected"]) for case in info["cases"]}
            frozen, width = constant_positions(task)
            tally = by_cell[cell]
            tally["tasks"] += 1
            tally["whole_constant"] += len(outputs) == 1
            tally["frozen"] += frozen
            tally["positions"] += width
        for (family, variant), tally in sorted(by_cell.items()):
            share = tally["frozen"] / tally["positions"] if tally["positions"] else 0.0
            if share >= FREE_SHARE_TO_REPORT or tally["whole_constant"]:
                flagged.append(
                    {
                        "level": level,
                        "family": family,
                        "variant": variant,
                        "free_share": round(share, 4),
                        "whole_constant_tasks": tally["whole_constant"],
                        "tasks": tally["tasks"],
                    }
                )
        levels[str(level)] = {
            "tasks": len(tasks),
            "constant_solvable_tasks": sum(t["whole_constant"] for t in by_cell.values()),
            "frozen_positions": sum(t["frozen"] for t in by_cell.values()),
            "graded_positions": sum(t["positions"] for t in by_cell.values()),
            "by_cell": {
                f"{family}:{variant}" if variant else family: dict(tally)
                for (family, variant), tally in sorted(by_cell.items())
            },
        }

    payload = {
        "definition": {
            "whole_constant": (
                "all six expected outputs of a task are identical, so one "
                "constant answer passes every hidden case"
            ),
            "frozen_position": (
                "a position in the flattened graded output takes the same value "
                "on all six cases, within the task's own tolerance, so a model "
                "can hardcode that position and be graded only on the rest"
            ),
        },
        "seed": args.seed,
        "num_tasks_per_level": args.num_tasks,
        "flagged": flagged,
        "levels": levels,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n")

    total_frozen = sum(level["frozen_positions"] for level in levels.values())
    total_positions = sum(level["graded_positions"] for level in levels.values())
    print(f"graded positions : {total_positions}")
    print(
        f"never varying    : {total_frozen} "
        f"({total_frozen / total_positions:.2%})" if total_positions else "n/a"
    )
    if flagged:
        print(f"\n{len(flagged)} cells with a free share at or above {FREE_SHARE_TO_REPORT:.0%}:")
        for entry in sorted(flagged, key=lambda e: -e["free_share"]):
            name = f"{entry['family']}:{entry['variant']}" if entry["variant"] else entry["family"]
            print(
                f"  L{entry['level']} {name:<40} "
                f"{entry['free_share']:.1%} of graded positions never vary"
            )
        print(
            "\nRead these. A free position is not automatically a defect, but it "
            "is graded output\na model does not have to compute."
        )
    else:
        print("\nno cell has a free share above the reporting threshold")
    print(f"report           : {args.report}")


if __name__ == "__main__":
    main()
