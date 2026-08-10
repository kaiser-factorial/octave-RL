"""Per-variant pass rates: the one check the naive-solution validator cannot do.

``validate_natural_solutions.py`` proves a prompt is **satisfiable** -- that a
competent reader's first attempt passes. It cannot prove the prompt is
**legible**, because it never asks a model to read it. A variant can be
perfectly well specified and still be phrased so that no model produces the
right function, and that is indistinguishable from the undisclosed-convention
defect this project has shipped three times.

**A variant sitting near 0.00 here is the finding.** Everything else this script
prints is context for interpreting that number.

Two columns separate the two ways a variant can look hard, following
``family_breakdown.py``:

- low ``exec`` -> the model cannot produce runnable code for this prompt. If it
  is near zero for every model, suspect the prompt, not the models.
- high ``exec``, low ``c|exec`` -> code runs and computes the wrong answer.
  That is what genuine difficulty looks like.

A variant with high ``exec`` and ``solved`` at 0.00 is the alarming shape: the
model understood it well enough to write running Octave and still never agreed
with the grader. That is what an undisclosed convention looks like from here.

**Standard errors are computed across tasks, not across rollouts.** Rollouts of
one task are correlated, so treating them as independent understates the error
-- `summarize_baseline_eval.py` did exactly that and reported +/-0.031 where the
truth was +/-0.056. Every earlier run had one rollout per task so nothing
historical is affected, but this script exists to be run at 8 rollouts a task.

Solve rate comes from the ``solved`` metric, never from ``rewards.case_fraction``
-- the reward is discounted by attempt, so thresholding it cannot count a
success after attempt 1. See PIPELINE_LOG, 2026-08-09.

Usage:
    uv run python scripts/variant_breakdown.py outputs/variant-sweep
    uv run python scripts/variant_breakdown.py NEW --against OLD --report out.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

# Below this, a variant is called out rather than merely listed. Not a
# threshold with a theory behind it: a screen, set where "no model ever solved
# this" stops being plausible difficulty and starts being a broken prompt.
SUSPECT_SOLVE_RATE = 0.02


def load(root: Path) -> dict[tuple[str, str, str, int], dict[str, list]]:
    """Map (run, family, variant, level) -> per-task lists of rollout metrics.

    Keyed per task so the standard error can be taken across tasks. Tasks with
    no ``variant`` are from families not yet on the variant form; they are kept
    under the key ``"(unconverted)"`` so their absence is visible rather than
    silent.
    """
    cells: dict[tuple[str, str, str, int], dict[str, list]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in glob.glob(str(root / "**" / "traces.jsonl"), recursive=True):
        run = Path(path).parent.parent.name
        with open(path) as handle:
            for line in handle:
                trace = json.loads(line)
                data = trace["task"]["data"]
                metrics = trace.get("metrics", {})
                key = (
                    run,
                    data["family"],
                    data.get("variant") or "(unconverted)",
                    data["level"],
                )
                cells[key][data["name"]].append(
                    {
                        # `solved` is undiscounted and identical in both reward
                        # modes. Fall back to thresholding raw_case_fraction for
                        # traces written before the metric existed -- never the
                        # reward, which is discounted.
                        "solved": metrics.get(
                            "solved",
                            float(metrics.get("raw_case_fraction", 0.0) == 1.0),
                        ),
                        "raw": metrics.get("raw_case_fraction"),
                        "exec": metrics.get("execution_fraction"),
                        "correct_given_executed": metrics.get("correct_given_executed"),
                        "format_ok": metrics.get("format_ok"),
                    }
                )
    return cells


def _mean(values) -> float:
    kept = [value for value in values if value is not None]
    return statistics.fmean(kept) if kept else float("nan")


def summarize(per_task: dict[str, list]) -> dict[str, Any]:
    """Collapse rollouts within a task, then average and error across tasks."""
    task_means = {
        name: {
            field: _mean(rollout[field] for rollout in rollouts)
            for field in ("solved", "raw", "exec", "correct_given_executed", "format_ok")
        }
        for name, rollouts in per_task.items()
    }
    solved = [task["solved"] for task in task_means.values()]
    stderr = (
        statistics.stdev(solved) / math.sqrt(len(solved)) if len(solved) > 1 else float("nan")
    )
    return {
        "tasks": len(task_means),
        "rollouts": sum(len(rollouts) for rollouts in per_task.values()),
        "solved": _mean(solved),
        "stderr": stderr,
        "raw": _mean(task["raw"] for task in task_means.values()),
        "exec": _mean(task["exec"] for task in task_means.values()),
        "correct_given_executed": _mean(
            task["correct_given_executed"] for task in task_means.values()
        ),
        "format_ok": _mean(task["format_ok"] for task in task_means.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--against", type=Path, help="an earlier run to diff against")
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--suspect-below",
        type=float,
        default=SUSPECT_SOLVE_RATE,
        help="call out variants at or below this solve rate",
    )
    args = parser.parse_args()

    cells = load(args.root)
    if not cells:
        raise SystemExit(f"no traces.jsonl found beneath {args.root}")
    baseline = load(args.against) if args.against else {}
    prior = {key: summarize(value) for key, value in baseline.items()}

    report: dict[str, Any] = {"root": str(args.root), "cells": {}, "suspect": []}
    suspect: list[tuple] = []

    for run in sorted({key[0] for key in cells}):
        print(f"\n===== {run} =====")
        header = (
            f"{'family':<20}{'variant':<22}{'lvl':>4}{'n':>5}"
            f"{'solved':>9}{'+/-':>7}{'exec':>7}{'c|exec':>8}{'fmt':>6}"
        )
        print(header + (f"{'was':>9}" if prior else ""))
        rows = [key for key in cells if key[0] == run]
        for key in sorted(rows, key=lambda k: (summarize(cells[k])["solved"], k)):
            _, family, variant, level = key
            stats = summarize(cells[key])
            report["cells"][f"{run}|{family}:{variant}|L{level}"] = stats
            flag = " <-- SUSPECT" if stats["solved"] <= args.suspect_below else ""
            if flag:
                suspect.append((run, family, variant, level, stats))
            line = (
                f"{family:<20}{variant:<22}{level:>4}{stats['rollouts']:>5}"
                f"{stats['solved']:>9.3f}{stats['stderr']:>7.3f}"
                f"{stats['exec']:>7.3f}{stats['correct_given_executed']:>8.3f}"
                f"{stats['format_ok']:>6.2f}"
            )
            if key in prior:
                line += f"{prior[key]['solved']:>9.3f}"
            print(line + flag)

    if suspect:
        print(f"\n{len(suspect)} variant-levels at or below {args.suspect_below:.2f} solve:")
        for run, family, variant, level, stats in suspect:
            # The distinction that decides what to do about it.
            shape = (
                "code runs and disagrees with the grader -- suspect an "
                "undisclosed convention"
                if stats["exec"] > 0.5
                else "code does not run -- suspect the prompt is unreadable"
            )
            print(f"  {family}:{variant} L{level} ({run}): {shape}")
        print(
            "\nA naive solution passing says the prompt is satisfiable, not that a\n"
            "model can read it. These are the ones to re-read by hand."
        )
        report["suspect"] = [
            {"run": r, "family": f, "variant": v, "level": lv, **s}
            for r, f, v, lv, s in suspect
        ]
    else:
        print(f"\nno variant-level at or below {args.suspect_below:.2f} solve")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
