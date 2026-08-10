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

So for a converted family the naive solution now comes from the task itself:
``specs.Variant`` carries ``description``, ``reference`` and ``natural`` written
together from one definition, and the generator threads ``natural`` through
``info``. Prompt, reference and naive solution cannot drift apart, because there
is only one place they are written.

## The three paths a task can take here

The conversion is staged across more than one session, and this validator is the
only check that has ever caught the undisclosed-convention defect, so dropping
coverage of the not-yet-converted families for the duration is not acceptable.
Each task therefore resolves to exactly one of:

- ``variant`` -- ``info["natural"]`` came with the task. The real check.
- ``legacy``  -- the family is not in ``generators.VARIANT_MODULES`` and
  ``LEGACY_NATURAL`` still has an entry for it. Scored and counted, but marked
  ``LEGACY`` everywhere, because it is the weaker pre-0.5.0 check: one naive
  solution for the whole family at that level.
- ``unvalidated`` -- neither. Never a pass, always fails the run.

A converted family is **never** allowed to fall back to ``LEGACY_NATURAL``. If
it is in ``VARIANT_MODULES`` and a task arrives without ``natural``, that is a
threading bug in the generator, and scoring it against the one-size table would
report PASS for seven variants it never ran -- the exact defect this change
exists to remove. That case is UNVALIDATED and red.

## What this validator refuses to do

It refuses to report a pass for a task it could not check, and it refuses to let
a weaker check pass for a stronger one. UNVALIDATED tasks are named in the
console output, listed in the JSON report, and fail the run. Legacy-path
families pass, but the run says how many of them there are every time, because
"PASS with nine families on the legacy path" is a much smaller claim than
"PASS". Every previous instance of the undisclosed convention defect survived
because a check was quietly absent, or quietly weaker than it looked.

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
from generators import VARIANT_MODULES, build_tasks

# The task ``info`` keys this validator reads beyond the ones the executor needs.
# ``natural`` is the full function source from ``specs.Variant.natural``;
# ``variant`` is the ``Variant.key`` the task was rendered from.
NATURAL_KEY = "natural"
VARIANT_KEY = "variant"

# Placeholder variant buckets, so the per-variant breakdown stays well-formed
# whatever path a task took. Both are meant to disappear.
UNNAMED_VARIANT = "(unnamed)"  # converted family, `natural` present, no `variant`
LEGACY_VARIANT = "(legacy)"  # legacy path: one problem per level, by definition

# TRANSITIONAL -- DELETE ENTRIES AS FAMILIES CONVERT, THEN DELETE THIS TABLE.
# =========================================================================
# The pre-0.5.0 naive solutions: family -> level -> body of
# `function out = <family>(...)`, no coercion.
#
# This table holds **exactly one naive solution per (family, level)**. That is
# the flaw the 0.5.0 variant form removes: the moment a family generates several
# distinct problems per level, one entry here covers one of them and silently
# reports PASS for the rest. It is therefore WRONG for any converted family, and
# `resolve_natural` below refuses to consult it for a family in
# `generators.VARIANT_MODULES` -- a converted family missing its `natural` is a
# threading bug to be fixed, never a lookup to fall back on.
#
# It survives only because the conversion is staged across sessions and this
# validator is the only check that has ever caught the undisclosed-convention
# defect; dropping a family's coverage while it waits its turn would open exactly
# the hole this file exists to close. Eight families are converted and their
# entries are gone. Delete each family's entry in the same change that converts
# it; the `stale_legacy_entries` line flags it if you forget. When the last one
# goes, this table is empty, the `legacy` path is obviously dead, and both
# should be removed outright.
LEGACY_NATURAL: dict[str, dict[int, str]] = {
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
}

# Why a task could not be checked. Recorded verbatim in the JSON report so a
# reader of the artefact does not have to guess what "unvalidated" meant.
THREADING_BUG_REASON = (
    "family is in generators.VARIANT_MODULES but the task carries no 'natural'; "
    "this is a generator threading bug. Refusing to fall back to LEGACY_NATURAL: "
    "one naive solution cannot stand in for eight distinct problems"
)
NO_COVERAGE_REASON = (
    "task carries no 'natural', the family is not converted, and LEGACY_NATURAL "
    "has no entry for it, so its prompt/grader agreement is UNCHECKED"
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


def _legacy_source(info: dict[str, Any]) -> str | None:
    """Assemble the pre-0.5.0 naive solution for an unconverted family."""
    body = LEGACY_NATURAL.get(info["family"], {}).get(info["level"])
    if body is None:
        return None
    signature = info["signature"].replace("function out = ", "").strip()
    return f"function out = {signature}\n {body}\nendfunction"


def resolve_natural(info: dict[str, Any]) -> tuple[str | None, str, str]:
    """Pick the naive solution to score, and say which path it came from.

    Returns ``(source, path, reason)``. ``path`` is one of ``"variant"``,
    ``"legacy"`` or ``"unvalidated"``; ``reason`` is non-empty only when the
    task could not be checked at all.

    The order is the whole point. ``info["natural"]`` wins whenever it is
    present, because it was written together with the description and the
    reference and cannot have drifted from them. ``LEGACY_NATURAL`` is consulted
    only for a family that is *not* in ``VARIANT_MODULES``: a converted family
    with a missing ``natural`` gets no fallback, because the fallback would be
    one solution standing in for eight problems and would report PASS for the
    seven it never ran.
    """
    source = info.get(NATURAL_KEY)
    if isinstance(source, str) and source.strip():
        return source, "variant", ""
    if info["family"] in VARIANT_MODULES:
        return None, "unvalidated", THREADING_BUG_REASON
    legacy = _legacy_source(info)
    if legacy is not None:
        return legacy, "legacy", ""
    return None, "unvalidated", NO_COVERAGE_REASON


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
        # "variant" | "legacy" | "unvalidated" | "mixed". A family on "legacy"
        # is covered by the weaker pre-0.5.0 check, and every rendering of this
        # report says so rather than letting it read as a full pass.
        "path": None,
        "paths": {},
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

        source, path, reason = resolve_natural(info)
        tally["paths"][path] = tally["paths"].get(path, 0) + 1
        if source is None:
            tally["unvalidated_tasks"] += 1
            unvalidated.append(
                {
                    "task": row["task"],
                    "family": family,
                    "level": level,
                    "variant": info.get(VARIANT_KEY),
                    "converted": family in VARIANT_MODULES,
                    "reason": reason,
                }
            )
            continue

        if path == "legacy":
            variant = LEGACY_VARIANT
        else:
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
                    "path": path,
                    "passed": record["passed"],
                    "total": record["total"],
                    "feedback": record["feedback"][-400:],
                }
            )

    for tally in per_family.values():
        paths = [name for name, count in tally["paths"].items() if count]
        tally["path"] = paths[0] if len(paths) == 1 else "mixed"

    return {
        "level": level,
        "tasks": len(rows),
        "validated_tasks": sum(t["validated_tasks"] for t in per_family.values()),
        "unvalidated_tasks": len(unvalidated),
        "families": per_family,
        "unvalidated_families": sorted(
            name for name, t in per_family.items() if t["unvalidated_tasks"]
        ),
        # Which families got the real per-variant check and which got the weaker
        # transitional one. Recorded per level so a partial run is still honest.
        "variant_families": sorted(
            name for name, t in per_family.items() if t["paths"].get("variant")
        ),
        "legacy_families": sorted(
            name for name, t in per_family.items() if t["paths"].get("legacy")
        ),
        "legacy_tasks": sum(t["paths"].get("legacy", 0) for t in per_family.values()),
        "failures": failures,
        "unvalidated": unvalidated,
        # A level is only ok when every task was checked *and* every checked
        # task passed every hidden case. An unchecked family is not a pass; a
        # legacy-covered failure is still a failure.
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
    if result["legacy_families"]:
        parts.append(f"{len(result['legacy_families'])} families on LEGACY path")
    print(f"level {level}    : {' -- '.join(parts)}")

    for family, tally in sorted(result["families"].items()):
        for record in result["unvalidated"]:
            if record["family"] == family:
                kind = (
                    "converted family, `natural` NOT THREADED (generator bug)"
                    if record["converted"]
                    else "no `natural` and no legacy entry"
                )
                print(
                    f"  ? {family:<22}UNVALIDATED -- {tally['unvalidated_tasks']}"
                    f"/{tally['tasks']} tasks: {kind}"
                )
                break
        if not tally["variants"]:
            continue
        broken = sum(1 for v in tally["variants"].values() if v["passed"] != v["total"])
        legacy = tally["paths"].get("legacy", 0)
        mark = "!" if broken else ("~" if legacy else " ")
        coverage = (
            "LEGACY -- one naive solution for the whole family, not per-variant"
            if legacy
            else f"{len(tally['variants'])} variants"
        )
        print(
            f"  {mark} {family:<22}{tally['passed']}/{tally['total']} hidden cases"
            f", {coverage}" + (f", {broken} BROKEN" if broken else "")
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
    variant_families = sorted(
        {f for level in summary["levels"] for f in level["variant_families"]}
    )
    legacy_families = sorted(
        {f for level in summary["levels"] for f in level["legacy_families"]}
    )
    summary["variant_families"] = variant_families
    summary["legacy_families"] = legacy_families
    summary["legacy_tasks"] = sum(level["legacy_tasks"] for level in summary["levels"])
    # A converted family's legacy entry is inert -- `resolve_natural` never
    # reaches it -- but leaving it behind is how the transitional table stops
    # shrinking and starts looking permanent. Report it rather than trusting
    # everyone to remember rule 4 of the conversion.
    summary["stale_legacy_entries"] = sorted(
        set(LEGACY_NATURAL) & set(VARIANT_MODULES)
    )
    # ``ok`` keeps its name and its meaning of "this pool is shippable", which
    # now requires that every task was actually checked. A family that cannot be
    # checked is reported, not absolved. A legacy-covered family that fails is a
    # failure like any other -- the legacy path is a weaker check, not a softer
    # verdict.
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
    if legacy_families:
        # Printed on every run, pass or fail. A PASS carried mostly by the
        # legacy path is a much smaller claim than a PASS, and the difference
        # has to be visible without opening the JSON.
        print(
            f"legacy      : {len(legacy_families)} of "
            f"{len(legacy_families) + len(variant_families)} families still on "
            f"the transitional one-solution-per-level path "
            f"({summary['legacy_tasks']} tasks)"
        )
        print(f"              {', '.join(legacy_families)}")
        print(
            "              these are covered, but NOT per-variant; a PASS here "
            "does not mean\n              their variants are checked, because "
            "they have none yet"
        )
    if summary["stale_legacy_entries"]:
        print(
            "stale       : LEGACY_NATURAL still has entries for converted "
            f"families -- {', '.join(summary['stale_legacy_entries'])}"
        )
        print("              they are never consulted; delete them")
    print(f"result      : {'PASS' if summary['ok'] else 'FAIL'}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"report      : {args.report}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
