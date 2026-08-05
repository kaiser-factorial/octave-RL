#!/usr/bin/env python3
"""Summarize verifiers.v1 trace JSONL without counting infrastructure errors as zeros."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PASS_RE = re.compile(r"Hidden tests passed (\d+)/(\d+)")


def summarize(path: Path) -> dict[str, float | int | str]:
    traces = [json.loads(line) for line in path.read_text().splitlines() if line]
    valid = [trace for trace in traces if not trace.get("errors")]
    raw_final: list[float] = []
    raw_first: list[float] = []
    attempts: list[float] = []
    for trace in valid:
        rewards = trace.get("rewards", {})
        metrics = trace.get("metrics", {})
        final = float(metrics.get("raw_case_fraction", rewards.get("case_fraction", 0)))
        raw_final.append(final)
        attempts.append(float(metrics.get("attempts_used", 1)))
        feedback = [
            node["message"]["content"]
            for node in trace.get("nodes", [])
            if node.get("message", {}).get("role") == "user"
            and "Hidden tests passed" in node["message"].get("content", "")
        ]
        match = PASS_RE.search(feedback[0]) if feedback else None
        raw_first.append(
            int(match.group(1)) / int(match.group(2)) if match else final
        )
    n = len(valid)
    return {
        "run": str(path.parent),
        "rollouts": len(traces),
        "valid_rollouts": n,
        "infra_errors": len(traces) - n,
        "first_attempt_mean": sum(raw_first) / n if n else 0,
        "final_mean": sum(raw_final) / n if n else 0,
        "solved": sum(value == 1 for value in raw_final),
        "mean_attempts": sum(attempts) / n if n else 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    args = parser.parse_args()
    for trace_path in args.traces:
        print(json.dumps(summarize(trace_path), sort_keys=True))
