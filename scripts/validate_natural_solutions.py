"""Check that the *obvious* solution to each prompt passes, not just the reference.

``validate_reference_pool.py`` and ``validate_local_runtime.py`` both score the
generator's own reference solution. That check is necessary but it cannot fail
in the one way that matters most: if a family is only solvable through a
convention the prompt never states, the reference passes anyway, because the
reference contains the convention. Three families sat near a 0-3% pass rate for
weeks behind two green validators for exactly this reason -- see PIPELINE_LOG,
"Three task families are hard because of an undisclosed orientation
convention".

The solution scored here is deliberately naive: what a competent Octave
programmer would write from the prompt alone, with no defensive reshaping, no
``(:)`` coercion, and no transposes that the prompt does not ask for. If one of
these stops passing, a prompt and its grader have drifted apart again.

## Why the naive solution is no longer written here

Before 0.5.0 this file carried a ``NATURAL`` table with one hardcoded body per
``(family, level)``. That worked only while a family had exactly one problem per
level. Under the variant form a family renders several distinct problems per
level, and a table keyed by ``(family, level)`` would silently score one of them
and report PASS for the other seven -- the same silence this validator exists to
break.

So the naive solution now comes from the task itself: ``specs.Variant`` carries
``description``, ``reference`` and ``natural`` written together from one
definition, and the generator threads ``natural`` through ``info``. Prompt,
reference and naive solution cannot drift apart, because there is only one
place they are written.

## What this validator refuses to do

It refuses to report a pass for a task it could not check. A family that has not
been converted to the variant form carries no ``natural``, and is reported as
UNVALIDATED -- named in the console output, listed in the JSON report, and
counted as a failure of the run. Every previous instance of the undisclosed
convention defect survived because a check was quietly absent rather than
loudly red.

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

# The task ``info`` keys this validator reads beyond the ones the executor needs.
# ``natural`` is the full function source from ``specs.Variant.natural``;
# ``variant`` is the ``Variant.key`` the task was rendered from.
NATURAL_KEY = "natural"
VARIANT_KEY = "variant"

# IN-FLIGHT CONTRACT NOTE (0.5.0 conversion)
# -----------------------------------------
# ``build_tasks`` does not yet thread ``variant``/``natural`` into ``info`` for
# every family: the conversion to the variant form lands family by family. This
# validator is written against the contract, not against the current state of
# ``generators.py``:
#   * ``info["natural"]``  absent -> the task is UNVALIDATED, never a pass;
#   * ``info["variant"]``  absent while ``natural`` is present -> the task is
#     still scored, and tallied under the placeholder variant below.
# Neither case is inferred, guessed, or filled in from a lookup table. When the
# last family is converted, ``UNVALIDATED`` and ``UNNAMED_VARIANT`` should both
# stop appearing in the report, and that is the signal the conversion is done.
UNNAMED_VARIANT = "(unnamed)"

# Why a task could not be checked. Recorded verbatim in the JSON report so a
# reader of the artefact does not have to guess what "unvalidated" meant.
NO_NATURAL_REASON = (
    "task info carries no 'natural' solution; this family has not been "
    "converted to the variant form, so its prompt/grader agreement is UNCHECKED"
)


class _Task:
    """The minimal surface ``execute_candidate_locally`` reads from a task."""

    def __init__(self, info: dict[str, Any]) -> None:
        self.fn_name = info["fn_name"]
        self.cases = info["cases"]
        self.tolerance = info["tolerance"]
        self._info = info

    def model_dump(self) -> dict[str, Any]:
        return self._info


def natural_source(info: dict[str, Any]) -> str | None:
    """The naive solution the task itself carries, or ``None`` if it carries none.

    ``Variant.natural`` is the complete function source, signature included, so
    there is nothing to assemble here. Returning ``None`` rather than raising
    keeps a half-converted pool checkable: the converted families are still
    scored, and the rest are reported as UNVALIDATED.
    """
    source = info.get(NATURAL_KEY)
    if not isinstance(source, str) or not source.strip():
        return None
    return source


def _new_family_tally() -> dict[str, Any]:
    return {
        # Per-family totals, kept from the pre-0.5.0 report shape.
        "passed": 0,
        "total": 0,
        # Added in 0.5.0: the per-variant breakdown. A family reporting 47/48
        # hidden cases hides *which* of its eight problems is broken, and which
        # one it is is the entire diagnostic value of this check.
        "variants": {},
        "tasks": 0,
        "validated_tasks": 0,
        "unvalidated_tasks": 0,
    }


def _new_variant_tally() -> dict[str, int]:
    return {"passed": 0, "total": 0, "tasks": 0, "failing_tasks": 0}


async def validate_level(level: int, num_tasks: int, seed: int) -> dict[str, Any]:
    # Keyword arguments only: the generator's signature is being extended for
    # the variant form while this runs, and positional calls would break on it.
    rows = build_tasks(level=level, num_tasks=num_tasks, seed=seed)
    per_family: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    unvalidated: list[dict[str, Any]] = []

    for row in rows:
        info = row["info"]
        family = info["family"]
        tally = per_family.setdefault(family, _new_family_tally())
        tally["tasks"] += 1

        source = natural_source(info)
        if source is None:
            tally["unvalidated_tasks"] += 1
            unvalidated.append(
                {
                    "task": row["task"],
                    "family": family,
                    "level": level,
                    "variant": info.get(VARIANT_KEY),
                    "reason": NO_NATURAL_REASON,
                }
            )
            continue

        variant = info.get(VARIANT_KEY) or UNNAMED_VARIANT
        record = await execute_candidate_locally(_Task(info), source)

        tally["validated_tasks"] += 1
        tally["passed"] += record["passed"]
        tally["total"] += record["total"]
        variant_tally = tally["variants"].setdefault(variant, _new_variant_tally())
        variant_tally["tasks"] += 1
        variant_tally["passed"] += record["passed"]
        variant_tally["total"] += record["total"]

        if record["fraction"] != 1.0:
            variant_tally["failing_tasks"] += 1
            failures.append(
                {
                    "task": row["task"],
                    "family": family,
                    "variant": variant,
                    "passed": record["passed"],
                    "total": record["total"],
                    "feedback": record["feedback"][-400:],
                }
            )

    return {
        "level": level,
        "tasks": len(rows),
        "validated_tasks": sum(t["validated_tasks"] for t in per_family.values()),
        "unvalidated_tasks": len(unvalidated),
        "families": per_family,
        "unvalidated_families": sorted(
            name for name, t in per_family.items() if t["unvalidated_tasks"]
        ),
        "failures": failures,
        "unvalidated": unvalidated,
        # A level is only ok when every task was checked *and* every checked
        # task passed every hidden case. An unchecked family is not a pass.
        "ok": not failures and not unvalidated,
    }


def _print_level(result: dict[str, Any]) -> None:
    level = result["level"]
    parts = [f"{result['tasks']} tasks"]
    if result["failures"]:
        parts.append(f"{len(result['failures'])} FAILED")
    if result["unvalidated_tasks"]:
        parts.append(f"{result['unvalidated_tasks']} UNVALIDATED")
    if result["ok"]:
        parts.append("ok")
    print(f"level {level}    : {' -- '.join(parts)}")

    for family, tally in sorted(result["families"].items()):
        if tally["unvalidated_tasks"]:
            print(
                f"  ? {family:<22}UNVALIDATED -- {tally['unvalidated_tasks']}"
                f"/{tally['tasks']} tasks carry no `natural` (family not yet"
                " on the variant form)"
            )
        if not tally["variants"]:
            continue
        broken = sum(1 for v in tally["variants"].values() if v["passed"] != v["total"])
        mark = "!" if broken else " "
        print(
            f"  {mark} {family:<22}{tally['passed']}/{tally['total']} hidden cases"
            f", {len(tally['variants'])} variants"
            + (f", {broken} BROKEN" if broken else "")
        )
        # Per-variant lines: every variant when something in the family is
        # broken, so a healthy variant next to a broken one is visible too.
        if broken:
            for variant, counts in sorted(tally["variants"].items()):
                bad = counts["passed"] != counts["total"]
                plural = "" if counts["tasks"] == 1 else "s"
                print(
                    f"      {'!' if bad else '.'} {variant:<20}"
                    f"{counts['passed']}/{counts['total']} hidden cases"
                    f" over {counts['tasks']} task{plural}"
                )

    for failure in result["failures"][:3]:
        print(
            f"    {failure['task']} [{failure['variant']}]: "
            f"{failure['feedback'].strip()[:200]}"
        )


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
        _print_level(result)
    summary["elapsed_seconds"] = round(time.monotonic() - started, 1)

    unvalidated_families = sorted(
        {
            family
            for level in summary["levels"]
            for family in level["unvalidated_families"]
        }
    )
    summary["unvalidated_families"] = unvalidated_families
    summary["unvalidated_tasks"] = sum(
        level["unvalidated_tasks"] for level in summary["levels"]
    )
    summary["failed_variants"] = sorted(
        {
            f"{family}:{variant}"
            for level in summary["levels"]
            for failure in level["failures"]
            for family, variant in [(failure["family"], failure["variant"])]
        }
    )
    # ``ok`` keeps its name and its meaning of "this pool is shippable", which
    # now requires that every task was actually checked. A family that cannot be
    # checked is reported, not absolved.
    summary["ok"] = all(level["ok"] for level in summary["levels"])

    print(f"elapsed     : {summary['elapsed_seconds']}s")
    if summary["failed_variants"]:
        print(f"broken      : {', '.join(summary['failed_variants'])}")
    if unvalidated_families:
        print(
            f"unvalidated : {summary['unvalidated_tasks']} tasks across "
            f"{len(unvalidated_families)} families -- "
            f"{', '.join(unvalidated_families)}"
        )
        print("              these are NOT passes; convert them to the variant form")
    print(f"result      : {'PASS' if summary['ok'] else 'FAIL'}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"report      : {args.report}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
