"""Validate the local Octave runtime against the seeded reference pool.

This is the Prime-free counterpart to ``validate_reference_pool.py``. It runs
reference solutions through ``executors.execute_candidate_locally`` and asserts
that every hidden case passes, which is the same contract the Sandbox-backed
validation asserts. Divergence between the two means the local interpreter is
not a faithful stand-in for the pinned image, and the run should say so rather
than silently score against a different Octave.

Usage:
    uv run python scripts/validate_local_runtime.py --level 1 --num-tasks 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[1] / "environments" / "octave_rl"))

from executors import execute_candidate_locally, runtime_description
from generators import build_tasks


class _ReferenceTask:
    """The minimal surface ``execute_candidate_locally`` reads from a task."""

    def __init__(self, info: dict[str, Any]) -> None:
        self.fn_name = info["fn_name"]
        self.cases = info["cases"]
        self.tolerance = info["tolerance"]
        self._info = info

    def model_dump(self) -> dict[str, Any]:
        return self._info


async def validate_level(level: int, num_tasks: int, seed: int) -> dict[str, Any]:
    rows = build_tasks(
        level=level,
        num_tasks=num_tasks,
        seed=seed,
        require_vectorized=False,
        include_reference=True,
    )
    failures: list[dict[str, Any]] = []
    cases_passed = 0
    cases_total = 0
    for row in rows:
        task = _ReferenceTask(row["info"])
        record = await execute_candidate_locally(task, row["_reference"])
        cases_passed += record["passed"]
        cases_total += record["total"]
        if record["fraction"] != 1.0 or record["structured_result"] != 1.0:
            failures.append(
                {
                    "task": row["task"],
                    "family": row["info"]["family"],
                    "passed": record["passed"],
                    "total": record["total"],
                    "structured_result": record["structured_result"],
                    "exit_code": record["exit_code"],
                    "feedback": record["feedback"][-600:],
                }
            )
    return {
        "level": level,
        "tasks": len(rows),
        "cases_passed": cases_passed,
        "cases_total": cases_total,
        "failures": failures,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, action="append", choices=(1, 2, 3))
    parser.add_argument("--num-tasks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    levels = args.level or [1, 2, 3]

    summary: dict[str, Any] = {
        **runtime_description(),
        "seed": args.seed,
        "num_tasks_per_level": args.num_tasks,
        "levels": [],
    }
    print(f"interpreter : {summary['octave']}")
    print(f"source      : {summary['rootfs'] or 'host PATH'} -> {summary['interpreter']}")
    print(f"isolation   : {summary['isolation_prefix'] or 'NONE (host network reachable)'}"
          f"{' + chroot' if summary['filesystem_isolated'] else ''}")

    started = time.monotonic()
    for level in levels:
        result = await validate_level(level, args.num_tasks, args.seed)
        summary["levels"].append(result)
        status = "ok" if not result["failures"] else f"{len(result['failures'])} FAILED"
        print(
            f"level {level}    : {result['cases_passed']}/{result['cases_total']} "
            f"hidden cases across {result['tasks']} tasks -- {status}"
        )
        for failure in result["failures"][:5]:
            print(f"  ! {failure['task']} ({failure['family']}): "
                  f"{failure['passed']}/{failure['total']}")
    summary["elapsed_seconds"] = round(time.monotonic() - started, 1)

    total_failures = sum(len(level["failures"]) for level in summary["levels"])
    summary["ok"] = total_failures == 0
    print(f"elapsed     : {summary['elapsed_seconds']}s")
    print(f"result      : {'PASS' if summary['ok'] else f'FAIL ({total_failures} tasks)'}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"report      : {args.report}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
