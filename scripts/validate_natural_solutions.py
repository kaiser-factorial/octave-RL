"""Check that the *obvious* solution to each prompt passes, not just the reference.

``validate_reference_pool.py`` and ``validate_local_runtime.py`` both score the
generator's own reference solution. That check is necessary but it cannot fail
in the one way that matters most: if a family is only solvable through a
convention the prompt never states, the reference passes anyway, because the
reference contains the convention. Three families sat near a 0-3% pass rate for
weeks behind two green validators for exactly this reason -- see PIPELINE_LOG,
"Three task families are hard because of an undisclosed orientation
convention".

The solutions below are deliberately naive: what a competent Octave programmer
would write from the prompt alone, with no defensive reshaping, no `(:)`
coercion, and no transposes that the prompt does not ask for. If one of these
stops passing, a prompt and its grader have drifted apart again.

Usage:
    uv run python scripts/validate_natural_solutions.py
    uv run python scripts/validate_natural_solutions.py --num-tasks 40 --report out.json
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

# family -> level -> body of `function out = <family>(...)`. No coercion.
NATURAL: dict[str, dict[int, str]] = {
    "reduce_along_dim": {
        1: "out = mean(A);",
        2: "s = sort(A, 'descend'); out = s(k, :);",
        3: "s = sort(A, 'descend'); out = s(k, :);",
    },
    "logical_index": {
        1: "out = x(x > 0);",
        2: "out = x; out(x < lo | x > hi) = NaN;",
        3: "out = x; out(x < lo | x > hi) = NaN;",
    },
    "reshape_permute": {
        1: "out = x(:);",
        2: "y = permute(reshape(x, dims), [2 1 3]); out = y(:)';",
        3: "y = permute(reshape(x, dims), [3 1 2]); out = y(:)';",
    },
    "broadcast_arith": {
        1: "out = a + b;",
        2: "out = (a - b) .^ 2;",
        3: "out = (a - b) .^ 2;",
    },
    "sliding_window": {
        1: "out = conv(x, ones(1, w), 'valid');",
        2: "idx = (1:s:(numel(x)-w+1))' + (0:w-1); out = mean(x(idx), 2)';",
        3: "idx = (1:s:(numel(x)-w+1))' + (0:w-1); out = median(x(idx), 2)';",
    },
    "linsolve_tolerance": {
        1: "out = A \\ b;",
        2: "out = A \\ b;",
        3: "x = A \\ b; out = [x; norm(A*x - b)];",
    },
    "sequence_recurrence": {
        1: "out = a + d * (0:n-1);",
        2: "out = zeros(1, n); out(1:2) = [a b];\n"
           " for i = 3:n; out(i) = p*out(i-1) + q*out(i-2); endfor",
        3: "out = filter(1, [1 -p -q], [a, b - p*a, zeros(1, max(n-2, 0))]);",
    },
    "struct_cell_wrangle": {
        1: "out = a + b;",
        2: "out = [min(A); max(A)];",
        3: "out = [min(A); max(A)];",
    },
    "string_parse": {
        1: "out = sscanf(strrep(s, ',', ' '), '%f')';",
        2: "out = sscanf(strrep(s, ',', ' '), '%f')';",
        3: "out = sscanf(strrep(s, ',', ' '), '%f')';",
    },
    "signal_identity": {
        1: "out = circshift(x, k);",
        2: "out = real(ifft(abs(fft(x)) .^ 2));",
        3: "out = real(ifft(abs(fft(x)) .^ 2));",
    },
}


class _Task:
    """The minimal surface ``execute_candidate_locally`` reads from a task."""

    def __init__(self, info: dict[str, Any]) -> None:
        self.fn_name = info["fn_name"]
        self.cases = info["cases"]
        self.tolerance = info["tolerance"]
        self._info = info

    def model_dump(self) -> dict[str, Any]:
        return self._info


def natural_source(info: dict[str, Any]) -> str:
    body = NATURAL[info["family"]][info["level"]]
    signature = info["signature"].replace("function out = ", "").strip()
    return f"function out = {signature}\n {body}\nendfunction"


async def validate_level(level: int, num_tasks: int, seed: int) -> dict[str, Any]:
    rows = build_tasks(level, num_tasks, seed, False, True)
    per_family: dict[str, dict[str, int]] = {}
    failures: list[dict[str, Any]] = []
    for row in rows:
        info = row["info"]
        record = await execute_candidate_locally(_Task(info), natural_source(info))
        tally = per_family.setdefault(info["family"], {"passed": 0, "total": 0})
        tally["passed"] += record["passed"]
        tally["total"] += record["total"]
        if record["fraction"] != 1.0:
            failures.append(
                {
                    "task": row["task"],
                    "family": info["family"],
                    "passed": record["passed"],
                    "total": record["total"],
                    "feedback": record["feedback"][-400:],
                }
            )
    return {"level": level, "tasks": len(rows), "families": per_family, "failures": failures}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, action="append", choices=(1, 2, 3))
    parser.add_argument("--num-tasks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    summary: dict[str, Any] = {
        **runtime_description(),
        "seed": args.seed,
        "num_tasks_per_level": args.num_tasks,
        "levels": [],
    }
    print(f"interpreter : {summary['octave']}")
    started = time.monotonic()
    for level in args.level or [1, 2, 3]:
        result = await validate_level(level, args.num_tasks, args.seed)
        summary["levels"].append(result)
        status = "ok" if not result["failures"] else f"{len(result['failures'])} FAILED"
        print(f"level {level}    : {result['tasks']} tasks -- {status}")
        for family, tally in sorted(result["families"].items()):
            if tally["passed"] != tally["total"]:
                print(f"  ! {family:<22}{tally['passed']}/{tally['total']} hidden cases")
        for failure in result["failures"][:3]:
            print(f"    {failure['task']}: {failure['feedback'].strip()[:200]}")
    summary["elapsed_seconds"] = round(time.monotonic() - started, 1)
    summary["ok"] = not any(level["failures"] for level in summary["levels"])
    print(f"elapsed     : {summary['elapsed_seconds']}s")
    print(f"result      : {'PASS' if summary['ok'] else 'FAIL'}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"report      : {args.report}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
