#!/usr/bin/env python3
"""Plot accuracy, latency, attempts, and truncation across a prime-rl run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Rollout:
    step: int
    split: str
    level: int
    reward: float
    attempts: int
    generation_seconds: float
    total_seconds: float
    completion_tokens: int
    truncated: bool


def _level(trace: dict[str, Any]) -> int:
    task = trace.get("task", {}).get("data", {})
    if isinstance(task.get("level"), int):
        return int(task["level"])
    name = str(trace.get("info", {}).get("env_name", ""))
    for level in (1, 2, 3):
        if f"level{level}" in name or f"level-{level}" in name:
            return level
    return 0


def parse_trace(trace: dict[str, Any]) -> Rollout:
    calls = trace.get("calls") or []
    completion_tokens = sum(
        int((call.get("usage") or {}).get("completion_tokens") or 0) for call in calls
    )
    truncated = any(call.get("finish_reason") == "length" for call in calls)
    timing = trace.get("timing") or {}
    generation = timing.get("generation") or {}
    start = float(timing.get("start") or 0.0)
    scoring = timing.get("scoring") or {}
    end = float(scoring.get("end") or generation.get("end") or start)
    metrics = trace.get("metrics") or {}
    rewards = trace.get("rewards") or {}
    reward = float(
        metrics.get(
            "raw_case_fraction",
            rewards.get(
                "case_fraction", rewards.get("reward", rewards.get("score", 0.0))
            ),
        )
        or 0.0
    )
    return Rollout(
        step=int(trace.get("run", {}).get("step") or 0),
        split=str(trace.get("run", {}).get("type") or "unknown"),
        level=_level(trace),
        reward=reward,
        attempts=int(metrics.get("attempts_used") or len(calls) or 0),
        generation_seconds=float(
            generation.get("model", {}).get("duration")
            or max(
                float(generation.get("end") or start)
                - float(generation.get("start") or start),
                0.0,
            )
        ),
        total_seconds=max(end - start, 0.0),
        completion_tokens=completion_tokens,
        truncated=truncated,
    )


def read_rollouts(run_dir: Path) -> list[Rollout]:
    rollouts: list[Rollout] = []
    paths: list[Path] = []
    rollout_root = run_dir / "run_default" / "rollouts"
    for split_dir in sorted(rollout_root.glob("step_*/*")):
        effective = split_dir / "effective" / "traces.jsonl"
        all_traces = split_dir / "all" / "traces.jsonl"
        if effective.exists():
            paths.append(effective)
        elif split_dir.name == "eval" and all_traces.exists():
            paths.append(all_traces)
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    rollouts.append(parse_trace(json.loads(line)))
    return rollouts


def read_standalone(specification: str) -> list[Rollout]:
    """Read STEP:LEVEL:PATH as held-out evidence with explicit provenance."""
    fields = specification.split(":", 2)
    if len(fields) != 3:
        raise ValueError("Standalone input must be STEP:LEVEL:PATH")
    step, level, raw_path = fields
    path = Path(raw_path)
    rollouts: list[Rollout] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rollouts.append(
                    replace(
                        parse_trace(json.loads(line)),
                        step=int(step),
                        level=int(level),
                        split="eval",
                    )
                )
    return rollouts


def read_segment(specification: str) -> list[Rollout]:
    """Read OFFSET:PATH and map local checkpoint steps onto the global axis."""
    fields = specification.split(":", 1)
    if len(fields) != 2:
        raise ValueError("Segment input must be OFFSET:PATH")
    offset, raw_path = fields
    return [
        replace(rollout, step=int(offset) + rollout.step)
        for rollout in read_rollouts(Path(raw_path))
    ]


def _quantile(values: Iterable[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def summarize(rollouts: Iterable[Rollout]) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[int, str, int], list[Rollout]] = defaultdict(list)
    for rollout in rollouts:
        groups[(rollout.step, rollout.split, rollout.level)].append(rollout)
    rows: list[dict[str, float | int | str]] = []
    for (step, split, level), items in sorted(groups.items()):
        attempts = [item.attempts for item in items]
        generation = [item.generation_seconds for item in items]
        total = [item.total_seconds for item in items]
        rows.append(
            {
                "step": step,
                "split": split,
                "level": level,
                "n": len(items),
                "raw_case_fraction_mean": statistics.fmean(
                    item.reward for item in items
                ),
                "generation_seconds_median": statistics.median(generation),
                "generation_seconds_p90": _quantile(generation, 0.9),
                "total_seconds_median": statistics.median(total),
                "total_seconds_p90": _quantile(total, 0.9),
                "attempts_mean": statistics.fmean(attempts),
                "attempt_1_rate": sum(value == 1 for value in attempts) / len(items),
                "attempt_2_rate": sum(value == 2 for value in attempts) / len(items),
                "attempt_3_rate": sum(value >= 3 for value in attempts) / len(items),
                "truncation_rate": sum(item.truncated for item in items) / len(items),
                "completion_tokens_mean": statistics.fmean(
                    item.completion_tokens for item in items
                ),
            }
        )
    return rows


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def plot(
    rows: list[dict[str, float | int | str]],
    path: Path,
    transition_steps: Iterable[int] = (),
) -> None:
    import matplotlib.pyplot as plt

    eval_rows = [row for row in rows if row["split"] == "eval"]
    train_rows = [row for row in rows if row["split"] == "train"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    colors = {1: "#2563eb", 2: "#d97706", 3: "#7c3aed"}

    for level in (1, 2, 3):
        selected = [row for row in eval_rows if row["level"] == level]
        if selected:
            axes[0].plot(
                [row["step"] for row in selected],
                [row["raw_case_fraction_mean"] for row in selected],
                marker="o",
                color=colors[level],
                label=f"Held-out L{level}",
            )
    for level in (1, 2, 3):
        selected = [row for row in train_rows if row["level"] == level]
        if selected:
            axes[0].plot(
                [row["step"] for row in selected],
                [row["raw_case_fraction_mean"] for row in selected],
                marker=".",
                linestyle=":",
                color=colors[level],
                alpha=0.45,
                label=f"Train L{level}",
            )
    axes[0].set_ylabel("Raw case fraction")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(ncol=4, fontsize=9)

    for level in (1, 2, 3):
        selected = [row for row in eval_rows if row["level"] == level]
        if selected:
            axes[1].plot(
                [row["step"] for row in selected],
                [row["total_seconds_median"] for row in selected],
                marker="o",
                color=colors[level],
                label=f"L{level} median",
            )
            axes[1].plot(
                [row["step"] for row in selected],
                [row["total_seconds_p90"] for row in selected],
                linestyle="--",
                color=colors[level],
                alpha=0.65,
                label=f"L{level} p90",
            )
    for level in (1, 2, 3):
        selected = [row for row in train_rows if row["level"] == level]
        if selected:
            axes[1].plot(
                [row["step"] for row in selected],
                [row["total_seconds_median"] for row in selected],
                marker=".",
                linestyle=":",
                color=colors[level],
                alpha=0.45,
                label=f"Train L{level} median",
            )
    axes[1].set_ylabel("End-to-end return seconds")
    axes[1].legend(ncol=3, fontsize=8)

    for level in (1, 2, 3):
        selected = [row for row in eval_rows if row["level"] == level]
        if selected:
            axes[2].plot(
                [row["step"] for row in selected],
                [row["attempts_mean"] for row in selected],
                marker="o",
                color=colors[level],
                label=f"L{level}",
            )
    for level in (1, 2, 3):
        selected = [row for row in train_rows if row["level"] == level]
        if selected:
            axes[2].plot(
                [row["step"] for row in selected],
                [row["attempts_mean"] for row in selected],
                marker=".",
                linestyle=":",
                color=colors[level],
                alpha=0.45,
                label=f"Train L{level}",
            )
    axes[2].set_ylabel("Mean attempts")
    axes[2].set_ylim(0.8, 3.2)
    axes[2].legend(ncol=3, fontsize=9)

    for level in (1, 2, 3):
        selected = [row for row in eval_rows if row["level"] == level]
        if selected:
            axes[3].plot(
                [row["step"] for row in selected],
                [row["truncation_rate"] for row in selected],
                marker="o",
                color=colors[level],
                label=f"L{level}",
            )
    for level in (1, 2, 3):
        selected = [row for row in train_rows if row["level"] == level]
        if selected:
            axes[3].plot(
                [row["step"] for row in selected],
                [row["truncation_rate"] for row in selected],
                marker=".",
                linestyle=":",
                color=colors[level],
                alpha=0.45,
                label=f"Train L{level}",
            )
    axes[3].set_ylabel("Truncation rate")
    axes[3].set_ylim(-0.02, 1.02)
    axes[3].set_xlabel("Training step")
    axes[3].legend(ncol=3, fontsize=9)

    for step in transition_steps:
        for axis in axes:
            axis.axvline(step, color="#111827", linestyle=":", linewidth=1, alpha=0.6)
        axes[0].annotate(
            "level shift",
            xy=(step, 1.0),
            xytext=(4, -4),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8,
        )

    fig.suptitle("OCTAVE RL rollout dynamics: accuracy, latency, retries, truncation")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--standalone",
        action="append",
        default=[],
        metavar="STEP:LEVEL:PATH",
        help="append an externally served held-out trace set",
    )
    parser.add_argument(
        "--segment",
        action="append",
        default=[],
        metavar="OFFSET:PATH",
        help="append a rebased prime-rl segment on the global step axis",
    )
    parser.add_argument(
        "--replace-run-eval",
        action="store_true",
        help="discard in-run eval traces before appending standalone evidence",
    )
    parser.add_argument(
        "--transition-step",
        action="append",
        default=[],
        type=int,
        help="mark a global curriculum transition",
    )
    args = parser.parse_args()
    rollouts = read_rollouts(args.run_dir)
    for specification in args.segment:
        rollouts.extend(read_segment(specification))
    if args.replace_run_eval:
        rollouts = [rollout for rollout in rollouts if rollout.split != "eval"]
    for specification in args.standalone:
        rollouts.extend(read_standalone(specification))
    rows = summarize(rollouts)
    if not rows:
        raise SystemExit(f"No traces found below {args.run_dir}")
    write_csv(rows, args.csv)
    plot(rows, args.png, args.transition_step)
    print(f"Wrote {args.png} and {args.csv}")


if __name__ == "__main__":
    main()
