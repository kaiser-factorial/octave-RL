"""Aggregate retained rollouts by task family, and diff two runs.

The level ladder is an average over families, and averaging is what hid three
separate defects: a family whose natural solution could not run, a family whose
prompt described a different signature, and a level whose description had
dropped its task definition. Each was obvious per family and invisible per
level. Run this before drawing any conclusion from a level number.

The columns that matter are `exec` and `c|exec`, because they separate the two
ways a family can look hard:

- low `exec` -> the model cannot produce runnable code for this family. If it
  is near zero for every model, suspect the task, not the models.
- high `exec`, low `c|exec` -> code runs and computes the wrong answer. That is
  what genuine difficulty looks like.

Usage:
    uv run python scripts/family_breakdown.py artifacts/group-spread-20260808
    uv run python scripts/family_breakdown.py NEW --against OLD
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def load(root: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Map (run, family) -> rollout records, over every traces.jsonl beneath root."""
    rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in glob.glob(str(root / "**" / "traces.jsonl"), recursive=True):
        run = Path(path).parent.parent.name
        with open(path) as handle:
            for line in handle:
                trace = json.loads(line)
                data = trace["task"]["data"]
                metrics = trace["metrics"]
                rows[(run, data["family"])].append(
                    {
                        "level": data["level"],
                        "reward": trace["rewards"]["case_fraction"],
                        "exec": metrics.get("execution_fraction"),
                        "correct_given_executed": metrics.get("correct_given_executed"),
                        "transposed": metrics.get("transposed_fraction"),
                    }
                )
    return rows


def mean(values) -> float:
    kept = [value for value in values if value is not None]
    return statistics.fmean(kept) if kept else float("nan")


def summarize(rows, run: str) -> dict[str, dict[str, float]]:
    return {
        family: {
            "n": len(records),
            "reward": mean(r["reward"] for r in records),
            "exec": mean(r["exec"] for r in records),
            "correct_given_executed": mean(r["correct_given_executed"] for r in records),
            "transposed": mean(r["transposed"] for r in records),
        }
        for (this_run, family), records in rows.items()
        if this_run == run
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--against", type=Path, help="an earlier run to diff against")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    rows = load(args.root)
    if not rows:
        raise SystemExit(f"no traces.jsonl found beneath {args.root}")
    baseline = load(args.against) if args.against else {}
    report: dict[str, Any] = {"root": str(args.root), "runs": {}}

    for run in sorted({key[0] for key in rows}):
        current = summarize(rows, run)
        prior: dict[str, dict[str, float]] = {}
        for other in sorted({key[0] for key in baseline}):
            if other == run:
                prior = summarize(baseline, other)
        report["runs"][run] = current
        print(f"\n===== {run} =====")
        header = f"{'family':<22}{'n':>5}{'reward':>9}{'exec':>8}{'c|exec':>8}{'transp':>8}"
        print(header + (f"{'was':>9}{'delta':>8}" if prior else ""))
        for family, stats in sorted(current.items(), key=lambda kv: kv[1]["reward"]):
            line = (
                f"{family:<22}{stats['n']:>5}{stats['reward']:>9.3f}"
                f"{stats['exec']:>8.3f}{stats['correct_given_executed']:>8.3f}"
                f"{stats['transposed']:>8.3f}"
            )
            if prior and family in prior:
                was = prior[family]["reward"]
                line += f"{was:>9.3f}{stats['reward'] - was:>+8.3f}"
            print(line)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nreport: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
