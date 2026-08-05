#!/usr/bin/env python3
"""Plot raw Octave case rewards from one or more verifiers trace files."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/reward_distribution.png"))
    parser.add_argument(
        "--summary", type=Path, default=Path("artifacts/reward_distribution.json")
    )
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    for path in args.traces:
        with path.open() as stream:
            for line in stream:
                trace = json.loads(line)
                errors = trace.get("errors") or []
                task = trace["task"]["data"]
                records.append(
                    {
                        "source": str(path),
                        "level": int(task["level"]),
                        "family": str(task["family"]),
                        "reward": float(trace["metrics"]["raw_case_fraction"]),
                        "error_count": len(errors),
                    }
                )

    valid = [record for record in records if record["error_count"] == 0]
    levels: dict[int, list[float]] = defaultdict(list)
    families: dict[str, list[float]] = defaultdict(list)
    for record in valid:
        levels[int(record["level"])].append(float(record["reward"]))
        families[str(record["family"])].append(float(record["reward"]))

    bins = [-1 / 12, 1 / 12, 3 / 12, 5 / 12, 7 / 12, 9 / 12, 11 / 12, 13 / 12]
    fig, axes = plt.subplots(
        1,
        len(levels),
        figsize=(5.2 * len(levels), 4.2),
        sharey=True,
        constrained_layout=True,
    )
    if len(levels) == 1:
        axes = [axes]
    colors = ["#4863A0", "#D2691E", "#348A6B"]
    for axis, (level, rewards), color in zip(axes, sorted(levels.items()), colors):
        axis.hist(rewards, bins=bins, color=color, edgecolor="white", linewidth=1.2)
        axis.axvline(
            sum(rewards) / len(rewards),
            color="#222222",
            linestyle="--",
            label=f"mean = {sum(rewards) / len(rewards):.3f}",
        )
        axis.set(
            title=f"Level {level} (n={len(rewards)})",
            xlabel="Raw case fraction",
            xticks=[0, 1 / 3, 2 / 3, 1],
            xlim=(-0.05, 1.05),
        )
        axis.legend(frameon=False)
    axes[0].set_ylabel("Rollouts")
    fig.suptitle("Qwen3.5-4B Octave reward distribution (seed 20260729)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)

    def summarize(values: list[float]) -> dict[str, object]:
        counts = Counter(f"{value:.6f}" for value in values)
        return {
            "n": len(values),
            "mean_raw_case_fraction": sum(values) / len(values),
            "fully_solved": sum(value == 1.0 for value in values),
            "partially_solved": sum(0.0 < value < 1.0 for value in values),
            "zero": sum(value == 0.0 for value in values),
            "distribution": dict(sorted(counts.items())),
        }

    payload = {
        "model": "Qwen/Qwen3.5-4B",
        "seed": 20260729,
        "valid_rollouts": len(valid),
        "infrastructure_errors": sum(int(record["error_count"]) for record in records),
        "overall": summarize([float(record["reward"]) for record in valid]),
        "by_level": {str(level): summarize(values) for level, values in sorted(levels.items())},
        "by_family": {
            family: summarize(values) for family, values in sorted(families.items())
        },
        "trace_files": [str(path) for path in args.traces],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
