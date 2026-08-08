"""Summarize baseline evaluation traces into a comparable table.

Reports raw case fraction as the cross-run metric (per the project rule that
historical shaped rewards are not comparable to current ones), alongside the
things that decide whether a number means what it looks like: truncation rate,
format validity, and infrastructure error count.

The Wilson lower bound is on *fully solved* rate, which is a proportion. Mean
case fraction is not a proportion of trials, so it gets a standard error
instead.

Usage:
    uv run python scripts/summarize_baseline_eval.py --root /workspace/baseline-eval
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

Z = 1.96


def wilson_lower_bound(successes: int, total: int, z: float = Z) -> float:
    """One-sided-style lower bound on a proportion; 0.0 when there is no data."""
    if total == 0:
        return 0.0
    phat = successes / total
    denominator = 1 + z**2 / total
    centre = phat + z**2 / (2 * total)
    spread = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total)
    return max(0.0, (centre - spread) / denominator)


def summarize(path: Path) -> dict[str, Any] | None:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    if not rows:
        return None
    raw: list[float] = []
    solved = 0
    truncated = 0
    calls_seen = 0
    completion_tokens: list[int] = []
    format_ok: list[float] = []
    errors = 0
    for row in rows:
        if row.get("errors"):
            errors += 1
        metrics = row.get("metrics") or {}
        if "raw_case_fraction" in metrics:
            value = float(metrics["raw_case_fraction"])
            raw.append(value)
            solved += value == 1.0
        if "format_ok" in metrics:
            format_ok.append(float(metrics["format_ok"]))
        for call in row.get("calls") or []:
            calls_seen += 1
            if call.get("finish_reason") == "length":
                truncated += 1
            usage = call.get("usage") or {}
            if "completion_tokens" in usage:
                completion_tokens.append(int(usage["completion_tokens"]))
    if not raw:
        return {"cell": path.parent.name, "rollouts": len(rows), "errors": errors,
                "note": "no scored rollouts"}
    return {
        "cell": path.parent.name,
        "rollouts": len(raw),
        "raw_case_fraction": round(statistics.mean(raw), 4),
        "raw_case_fraction_stderr": round(
            statistics.stdev(raw) / math.sqrt(len(raw)), 4
        ) if len(raw) > 1 else 0.0,
        "solved": solved,
        "solve_rate": round(solved / len(raw), 4),
        "solve_rate_wilson_lb": round(wilson_lower_bound(solved, len(raw)), 4),
        "format_ok": round(statistics.mean(format_ok), 4) if format_ok else None,
        "truncation_rate": round(truncated / calls_seen, 4) if calls_seen else None,
        "mean_completion_tokens": round(statistics.mean(completion_tokens), 1)
        if completion_tokens else None,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cells = [
        summary
        for path in sorted(args.root.glob("*/traces.jsonl"))
        if (summary := summarize(path)) is not None
    ]

    header = (
        f"{'cell':<20}{'n':>5}{'raw':>9}{'+/-':>8}{'solved':>9}"
        f"{'wilson_lb':>11}{'trunc':>8}{'tokens':>9}{'fmt':>7}{'err':>5}"
    )
    print(header)
    print("-" * len(header))
    for cell in cells:
        if "raw_case_fraction" not in cell:
            print(f"{cell['cell']:<20}{cell['rollouts']:>5}   {cell.get('note','')}")
            continue
        print(
            f"{cell['cell']:<20}{cell['rollouts']:>5}"
            f"{cell['raw_case_fraction']:>9.4f}"
            f"{cell['raw_case_fraction_stderr']:>8.4f}"
            f"{str(cell['solved']) + '/' + str(cell['rollouts']):>9}"
            f"{cell['solve_rate_wilson_lb']:>11.4f}"
            f"{(cell['truncation_rate'] if cell['truncation_rate'] is not None else 0):>8.3f}"
            f"{(cell['mean_completion_tokens'] or 0):>9.1f}"
            f"{(cell['format_ok'] if cell['format_ok'] is not None else 0):>7.2f}"
            f"{cell['errors']:>5}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"cells": cells}, indent=2) + "\n")
        print(f"\nreport: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
