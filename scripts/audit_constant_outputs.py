#!/usr/bin/env python3
"""Measure how often one constant output could pass every hidden case."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "environments" / "octave_rl"))

from generators import build_tasks


def canonical(value: object) -> str:
    return json.dumps(value, allow_nan=True, separators=(",", ":"), sort_keys=True)


def main() -> None:
    levels: dict[str, dict[str, object]] = {}
    for level in (1, 2, 3):
        tasks = build_tasks(level, 500, 0, False, True)
        by_family: dict[str, list[bool]] = defaultdict(list)
        for task in tasks:
            outputs = {
                canonical(case["expected"])
                for case in task["info"]["cases"]
            }
            by_family[task["info"]["family"]].append(len(outputs) == 1)
        levels[str(level)] = {
            "tasks": len(tasks),
            "constant_solvable_tasks": sum(
                sum(flags) for flags in by_family.values()
            ),
            "by_family": {
                family: {
                    "tasks": len(flags),
                    "constant_solvable_tasks": sum(flags),
                }
                for family, flags in sorted(by_family.items())
            },
        }
    payload = {
        "definition": (
            "A task is constant-solvable only if all six precomputed expected "
            "outputs are exactly identical."
        ),
        "seed": 0,
        "levels": levels,
    }
    output = ROOT / "artifacts" / "constant_output_audit.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
